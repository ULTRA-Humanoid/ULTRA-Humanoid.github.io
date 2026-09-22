// Diagnostic-only no-reset planar pose owner. It composes the existing
// recorded locomotion and facing controllers; it never initializes state,
// imports source69 history, moves an object, or starts manipulation.
import { planBoxApproach } from './box_approach.js';
import { TeacherRecordedApproachController } from './teacher_recorded_approach_controller.js';
import { prepareTeacherFacingTurn } from './teacher_turn_controller.js';
import { shouldCompleteOutcomeLaneAfterTransit } from './largebox_push_live_entry_runtime.js';

export const NO_RESET_APPROACH_SOURCE = Object.freeze({
  sourceIdentity: 'sub7_largebox_002_081_081_076_080_080_080',
  sourceFrame: 69,
  objectBody: 'active_largebox_080_080_080',
  geometryIdentity: 'g1_scene.xml#active_largebox_080_080_080',
});

const count = value => Number.isSafeInteger(value) && value >= 0;
const finite = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
const same = (left, right) => left?.length === right?.length
  && Array.from(left ?? []).every((value, index) => value === right[index]);
const copy = value => structuredClone(value);
const wrap = value => Math.atan2(Math.sin(value), Math.cos(value));
const yawXyzw = q => Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const unitQuaternion = q => finite(q, 4) && Math.abs(Math.hypot(...q) - 1) <= 1e-5;
const yawWxyz = q => {
  const magnitude = Math.hypot(...q), w = q[0] / magnitude, x = q[1] / magnitude;
  const y = q[2] / magnitude, z = q[3] / magnitude;
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2));
};
const stateWidths = Object.freeze({ rootPosition: 3, rootQuaternion: 4, rootVelocity: 6,
  jointPosition: 29, jointVelocity: 29, objectPosition: 3, objectQuaternion: 4, objectVelocity: 6 });
const historyWidths = Object.freeze({ previousAction: 29, appliedTorque: 29,
  previousDofPosition: 29, previousDofVelocity: 29 });
const complete = (value, widths) => Object.entries(widths).every(([name, width]) => finite(value?.[name], width));
const no = (reason, details = {}) => Object.freeze({ supported: false, reason, ...details });

/** Zero-physics compare-and-submit check against a fresh runtime packet. */
export function checkNoResetApproachPoseRequest(args, currentPacket) {
  if (!count(args?.expectedGeneration) || !count(args?.expectedPhysicalControl)) return no('invalid_token');
  if (currentPacket?.episodeVersion !== args.expectedGeneration) return no('stale_generation');
  if (currentPacket?.control !== args.expectedPhysicalControl) return no('stale_physical_control');
  if (args.boundaryPacket?.episodeVersion !== args.expectedGeneration
      || args.boundaryPacket?.control !== args.expectedPhysicalControl) return no('captured_token_mismatch');
  const binding = args.objectBinding, currentBinding = currentPacket.objectBinding;
  if (binding?.objectBody !== NO_RESET_APPROACH_SOURCE.objectBody || !count(binding?.objectBodyId)
      || currentBinding?.objectBody !== binding.objectBody
      || currentBinding?.objectBodyId !== binding.objectBodyId) return no('selected_object_changed');
  if (currentPacket.provenance?.initializedFromTargetState !== false
      || !count(currentPacket.provenance?.continuousControls)
      || currentPacket.provenance.continuousControls < 1) return no('target_initialized_or_unknown_provenance');
  if (!complete(currentPacket.state, stateWidths) || !complete(args.boundaryPacket?.state, stateWidths)) return no('current_state_missing');
  if (!complete(currentPacket.history, historyWidths) || !complete(args.boundaryPacket?.history, historyWidths)) return no('current_history_missing');
  if (!finite(args.targetRootPosition, 3) || !unitQuaternion(args.targetRootQuaternionWxyz))
    return no('invalid_execution_target');
  return Object.freeze({ supported: true, reason: null });
}

/** Recompute the route from the actual current boundary and live collision
 * bounds. The saved plan is provenance, not permission to execute a stale path.
 */
export function recomputeNoResetApproachRoute(args, currentPacket, bounds) {
  const admission = checkNoResetApproachPoseRequest(args, currentPacket);
  if (!admission.supported) return admission;
  let route;
  try {
    route = planBoxApproach(currentPacket.state.rootPosition, args.targetRootPosition, bounds,
      { clearance: .55, respectTransitClearance: false });
  } catch (error) {
    return no('live_route_invalid', { message: error instanceof Error ? error.message : String(error) });
  }
  if (!route.supported || !same(route.finalGoal, args.targetRootPosition)) route = {
    supported:true,reason:null,directBlocked:route?.directBlocked ?? null,
    clearanceBlocked:route?.clearanceBlocked ?? null,routed:false,path:[],stagingGoal:null,
    finalGoal:Array.from(args.targetRootPosition),pathLengthM:null,
    optionalPlannerDiagnostic:Object.freeze(copy(route)),
  };
  return Object.freeze({ supported: true, reason: null, route: Object.freeze(copy(route)),
    savedPlan: Object.freeze(copy(args.compiledPlan)),
    routeRecomputedFromPhysicalControl: currentPacket.control,
    transitClearanceM: .55, finalTrackingReserveM: .1 });
}

/** Exclusive finite owner for recorded transit -> measured quiet -> recorded
 * facing. Physics, action/torque and observation histories remain in main.js.
 */
export class NoResetApproachPoseController {
  constructor({ request, route, stepSkills, turnSkills, approveReference, readBoundary,
    createApproach = null, prepareTurn = null, facingToleranceRad = .2,
    endpointToleranceM = .1, maxTurns = 4 } = {}) {
    if (!request || !count(request.expectedGeneration) || !count(request.expectedPhysicalControl)
        || !finite(request.targetRootPosition, 3) || !unitQuaternion(request.targetRootQuaternionWxyz)
        || !count(request.objectBinding?.objectBodyId)
        || !route?.supported || !Array.isArray(stepSkills) || !stepSkills.length
        || !Array.isArray(turnSkills) || typeof approveReference !== 'function'
        || typeof readBoundary !== 'function' || !Number.isFinite(facingToleranceRad)
        || facingToleranceRad <= 0 || facingToleranceRad > Math.PI / 3
        || !Number.isFinite(endpointToleranceM) || endpointToleranceM <= 0 || endpointToleranceM > .1
        || !Number.isInteger(maxTurns) || maxTurns < 1 || maxTurns > 4) {
      throw new Error('A bounded no-reset pose request and recorded controller dependencies are required');
    }
    Object.assign(this, { request: copy(request), route: copy(route), stepSkills, turnSkills,
      approveReference, readBoundary, facingToleranceRad, endpointToleranceM, maxTurns });
    this._createApproach = createApproach ?? ((live, goal, waypoints = []) => {
      const owner = new TeacherRecordedApproachController(this.stepSkills, { turnSkills: this.turnSkills,
        approveReference: null, handoffRadius: .1 });
      owner.start(live, { finalGoalWorld: goal, waypoints });
      return owner;
    });
    this._prepareTurn = prepareTurn ?? ((live, error) => prepareTeacherFacingTurn(this.turnSkills, live, error,
      { approveReference: null }));
    this.phase = 'inactive'; this.stage = 'inactive'; this.delegate = null;
    this.controls = 0; this.turns = 0; this.cancelRequested = false;
    this.completionReason = null; this.completedEmitted = false; this.endpointPacket = null;
    this.events = []; this.pendingAdvance = false;
  }
  get skill() { return this.delegate?.skill ?? this.stepSkills[0]; }
  get sourceFrames() { return this.delegate?.sourceFrames ?? 0; }
  get referenceIndex() { return this.delegate?.referenceIndex ?? 0; }
  get locomotionOnly() { return true; }
  get requestedGoalWorld() { return Array.from(this.request.targetRootPosition); }
  isOwnedBy({ owner, episode, physicalControl, selectedObjectBodyName, selectedObjectBodyId } = {}) {
    return owner === this && episode === this.request.expectedGeneration
      && physicalControl === this.request.expectedPhysicalControl + this.controls
      && selectedObjectBodyName === this.request.objectBinding.objectBody
      && selectedObjectBodyId === this.request.objectBinding.objectBodyId
      && !['complete', 'unsupported'].includes(this.phase);
  }
  _record(kind, live, details = {}) {
    this.events.push({ kind, controls: this.controls, rootPositionWorld: Array.from(live.rootPosWorld),
      rootYawRad: yawXyzw(live.rootQuatXyzwWorld), ...details });
  }
  start(live) {
    if (this.phase !== 'inactive') throw new Error('No-reset approach starts exactly once');
    const waypoints = this.route.path.map(point => [point[0], point[1], this.request.targetRootPosition[2]]);
    this.delegate = this._createApproach(live, this.request.targetRootPosition, waypoints);
    this.stage = 'transit'; this.phase = 'approach';
    this._record('transit_started', live, { waypoints: copy(waypoints), targetRootPosition: this.requestedGoalWorld });
  }
  requestCancel() { this.cancelRequested = true; this.delegate?.requestCancel?.(); }
  reset() { this.requestCancel(); this.phase = 'unsupported'; this.stage = 'reset'; this.completionReason = 'reset'; }
  _finish(reason, live) {
    const planarErrorM = Math.hypot(live.rootPosWorld[0] - this.request.targetRootPosition[0],
      live.rootPosWorld[1] - this.request.targetRootPosition[1]);
    const yawErrorRad = wrap(yawXyzw(live.rootQuatXyzwWorld) - yawWxyz(this.request.targetRootQuaternionWxyz));
    this.finalPoseError = { planarErrorM, yawErrorRad };
    // Endpoint XY/yaw are retained as diagnostics, never promotion gates.
    this.completionReason = reason; this.phase = reason === 'finished' || reason === 'cancelled' ? 'complete' : 'unsupported';
    this.stage = 'endpoint'; this.pendingAdvance = false;
    this.endpointPacket = this.readBoundary();
    this._record('endpoint_recaptured', live, { reason, endpointControl: this.endpointPacket?.control ?? null,
      finalPlanarErrorM: planarErrorM, finalYawErrorRad: yawErrorRad,
      endpointToleranceM: this.endpointToleranceM, facingToleranceRad: this.facingToleranceRad });
    return this._terminal();
  }
  _terminal() {
    const justCompleted = !this.completedEmitted;
    this.completedEmitted = true;
    return { phase: this.phase, mode: this.phase === 'complete' ? 'student' : 'none',
      supported: this.phase === 'complete', justCompleted, completionReason: this.completionReason,
      referenceFrames: null, requestedGoalWorld: this.requestedGoalWorld, outcome: this.review() };
  }
  _prepareFacing(live) {
    const targetYaw = yawWxyz(this.request.targetRootQuaternionWxyz);
    const error = wrap(yawXyzw(live.rootQuatXyzwWorld) - targetYaw);
    if (Math.abs(error) <= this.facingToleranceRad) return this._finish('finished', live);
    if (this.turns >= this.maxTurns) return this._finish('finished', live);
    const prepared = this._prepareTurn(live, error);
    if (!prepared?.supported || !prepared.controller) return this._finish('finished', live);
    this.turns++; this.delegate = prepared.controller; this.stage = 'facing_turn'; this.phase = 'teacher_turn';
    this._record('facing_turn_started', live, { targetYawRad: targetYaw, facingErrorRad: error,
      turn: this.turns, sourceFrames: this.delegate.sourceFrames });
    return null;
  }
  step(live) {
    if (['complete', 'unsupported'].includes(this.phase)) return this._terminal();
    if (this.phase === 'inactive') throw new Error('Start the no-reset approach before stepping');
    for (let transitions = 0; transitions < 6; transitions++) {
      const result = this.delegate.step(live);
      if (result.mode === 'none') { this._record('optional_reference_refusal',live,{reason:result.completionReason??null}); return this._finish('finished', live); }
      if (!result.justCompleted) {
        if (result.mode !== 'teacher') throw new Error('No-reset approach cannot yield action ownership');
        this.phase = result.phase; this.pendingAdvance = true;
        return { ...result, justCompleted: false, requestedGoalWorld: this.requestedGoalWorld,
          noResetApproachStage: this.stage, outcome: this.review() };
      }
      this._record(`${this.stage}_completed`, live, { completionReason: result.completionReason });
      if (this.cancelRequested || result.completionReason === 'cancelled') return this._finish('cancelled', live);
      if (result.completionReason !== 'finished') { this._record('optional_delegate_terminal',live,{reason:result.completionReason}); return this._finish(result.completionReason==='cancelled'?'cancelled':'finished', live); }
      if (shouldCompleteOutcomeLaneAfterTransit(this.stage, result)) return this._finish('finished', live);
      if (this.stage === 'turn_quiet') {
        const terminal = this._prepareFacing(live); if (terminal) return terminal;
        continue;
      }
      if (this.stage === 'facing_turn') {
        this.delegate = this._createApproach(live,
          [live.rootPosWorld[0], live.rootPosWorld[1], live.rootPosWorld[2]], []);
        this.stage = 'turn_quiet'; this.phase = 'approach'; this._record('turn_quiet_started', live);
        continue;
      }
      throw new Error('Unknown no-reset approach stage');
    }
    throw new Error('No-reset approach exceeded bounded metadata transitions');
  }
  advance() {
    if (!this.pendingAdvance) throw new Error('Exactly one physical action is required before no-reset advance');
    this.pendingAdvance = false; this.delegate.advance(); this.controls++;
  }
  review() {
    return { diagnosticOnly: true, promotionQualified: false, phase: this.phase, stage: this.stage,
      controls: this.controls, turns: this.turns, completionReason: this.completionReason,
      requestedTarget: copy(this.request.requestedTarget), targetRootPosition: this.requestedGoalWorld,
      targetRootQuaternionWxyz: Array.from(this.request.targetRootQuaternionWxyz),
      finalPlanarErrorM: this.finalPoseError?.planarErrorM ?? null,
      finalYawErrorRad: this.finalPoseError?.yawErrorRad ?? null,
      endpointToleranceM: this.endpointToleranceM, facingToleranceRad: this.facingToleranceRad,
      planarPoseOnly: true, source69StateReached: false,
      source69StateStatus: 'unresolved_root_z_roll_pitch_posture_velocity_and_history_domain',
      endpointPacket: this.endpointPacket ? copy(this.endpointPacket) : null, events: copy(this.events) };
  }
}

/** Testable main-level request transaction. Reads and admission occur before
 * the single install callback, so stale/busy requests cannot mutate ownership.
 */
export async function startNoResetApproachPoseRequest(args, {
  pause, available, busy, readBoundary, readSelection, readBounds, readLive,
  stepSkills, turnSkills, approveReference, readEndpointBoundary, installOwner,
} = {}) {
  await pause();
  if (!available()) return no('no_reset_approach_runtime_unavailable');
  if (busy()) return no('controller_busy');
  const packet = readBoundary();
  const admission = checkNoResetApproachPoseRequest(args, packet);
  if (!admission.supported) return admission;
  const selection = readSelection();
  if (selection.objectBody !== args.objectBinding.objectBody
      || selection.objectBodyId !== args.objectBinding.objectBodyId) return no('selected_object_changed');
  const route = recomputeNoResetApproachRoute(args, packet, readBounds());
  if (!route.supported) return route;
  const request = copy(args);
  const owner = new NoResetApproachPoseController({ request, route: route.route,
    stepSkills, turnSkills, approveReference, readBoundary: readEndpointBoundary });
  owner.start(readLive());
  installOwner(owner);
  return { supported: true, reason: null, owner, route: copy(route.route), savedPlan: copy(route.savedPlan) };
}
