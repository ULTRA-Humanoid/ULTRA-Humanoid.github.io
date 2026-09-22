// A finite student goal after a completed inter-segment box exit. The carry
// sequence remains the action owner and supplies its existing arrival decision.
// The caller owns inference, history, physical previews and teacher handoff.
import { BoundedStageGoalWindow, packLegacyStageStudentInput } from './stage_goal.js';

export const STAGED_STUDENT_APPROACH_LIMITS = Object.freeze({
  horizonControls: 180, maxDistanceM: .5, minRootHeightM: .7,
  minUpright: .95, maxPlanarSpeedMps: .05,
});

const finiteVector = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
const physicalCount = value => Number.isSafeInteger(value) && value >= 0;
const refusal = reason => ({ supported: false, reason, controller: null });
function completeExit(exit) {
  const clock = exit?.recordClock;
  return exit?.completionReason === 'finished' && Number.isSafeInteger(clock?.totalControls)
    && clock.totalControls > 0 && clock.totalControls === clock.expectedTotalControls;
}

/** Initial scope only: an untouched next segment, its physically completed
 * previous exit, stable live standing, and a supported unobstructed short path.
 * Refusal leaves the caller's recorded approach available. Admission bypasses
 * the recorded planner's 55 cm transit staging reserve for a physically clear
 * direct student path. Numerical collision checks remain unchanged, and every
 * student action still requires the caller's all-box physical preview.
 */
export function checkStagedStudentApproachEntry({ parent, live, route, exit, episode, physicalControl }) {
  if (!physicalCount(episode) || !physicalCount(physicalControl)) return { supported: false, reason: 'invalid_physical_clock' };
  if (!parent || parent.phase !== 'approach' || parent.finishRequested
      || !Number.isSafeInteger(parent.segmentIndex) || parent.segmentIndex < 1
      || parent.child?.approachSteps !== 0 || parent.referenceIndex !== 0) {
    return { supported: false, reason: 'segment_already_started_or_inactive' };
  }
  const previousExit = parent.segmentExitResults?.at(-1);
  if (!completeExit(exit) || !completeExit(previousExit)
      || previousExit.segmentIndex !== parent.segmentIndex - 1
      || parent.exitedSegmentIndex !== parent.segmentIndex - 1
      || previousExit.recordClock.totalControls !== exit.recordClock.totalControls) {
    return { supported: false, reason: 'previous_segment_exit_incomplete' };
  }
  const objectBodyName = parent.skill?.objectBodyName;
  if (!objectBodyName || live?.objectBodyName !== objectBodyName) return { supported: false, reason: 'placed_object_changed' };
  const goal = parent.approachGoalWorld, first = parent.referencePlan?.first;
  if (!finiteVector(live.rootPosWorld, 3) || !finiteVector(live.rootQuatXyzwWorld, 4)
      || !finiteVector(live.rootVelWorld, 3) || !finiteVector(live.objPosWorld, 3)
      || !finiteVector(goal, 3) || !finiteVector(parent.requestedGoalWorld, 3)
      || !first || first.length !== 747 || !Array.from(first.slice(0, 7)).every(Number.isFinite)
      || Math.abs(Math.hypot(...live.rootQuatXyzwWorld) - 1) > 1e-5
      || Math.abs(Math.hypot(...first.slice(3, 7)) - 1) > 1e-5) {
    return { supported: false, reason: 'invalid_live_state_or_fixed_goal' };
  }
  const q = live.rootQuatXyzwWorld;
  const upright = 1 - 2 * (q[0] ** 2 + q[1] ** 2);
  const planarSpeedMps = Math.hypot(live.rootVelWorld[0], live.rootVelWorld[1]);
  const distanceM = Math.hypot(goal[0] - live.rootPosWorld[0], goal[1] - live.rootPosWorld[1]);
  const limits = STAGED_STUDENT_APPROACH_LIMITS;
  if (live.rootPosWorld[2] < limits.minRootHeightM || upright < limits.minUpright
      || planarSpeedMps > limits.maxPlanarSpeedMps) {
    return { supported: false, reason: 'exit_not_stably_standing', distanceM, planarSpeedMps, upright };
  }
  if (route?.supported !== true || route.directBlocked !== false) return { supported: false, reason: 'direct_approach_not_supported' };
  if (distanceM > limits.maxDistanceM) return { supported: false, reason: 'student_approach_distance', distanceM };
  return { supported: true, reason: null, distanceM, planarSpeedMps, upright,
    objectBodyName, segmentIndex: parent.segmentIndex,
    plan: { stage: 'approach', mode: 'LOCO', humanGoalWorld: [goal[0], goal[1], first[2]],
      humanGoalRotationWorld: Array.from(first.slice(3, 7)), objectGoalWorld: Array.from(live.objPosWorld),
      finalDestinationWorld: Array.from(parent.requestedGoalWorld) } };
}


function checkFirstPickupOrOrdinaryEntry(options) {
  const lease=options.firstPickupLease;
  if(!lease)return checkStagedStudentApproachEntry(options);
  if(!lease.isOwnedBy(options)||options.physicalControl!==lease.initialControl)
    return {supported:false,reason:'first_pickup_lease_not_current'};
  return lease.metadata;
}
export class StagedStudentApproachController {
  #parent;
  #firstPickupLease = null;
  #window;
  #entry;
  #lastControl;
  #records = [];
  #refusedPreview = null;
  #ended = null;

  static tryStart(options) {
    const entry = checkFirstPickupOrOrdinaryEntry(options);
    if (!entry.supported) return { ...refusal(entry.reason), entry };
    return { supported: true, reason: null, entry,
      controller: new StagedStudentApproachController(options) };
  }

  constructor(options) {
    const entry = checkFirstPickupOrOrdinaryEntry(options);
    if (!entry.supported) throw new Error(`Student approach entry refused: ${entry.reason}`);
    this.#parent = options.parent;
    this.#firstPickupLease = options.firstPickupLease ?? null;
    this.#entry = Object.freeze({ ...entry, plan: undefined,
      route: structuredClone(options.route), exit: structuredClone(options.exit) });
    this.#window = new BoundedStageGoalWindow(entry.plan, { episode: options.episode,
      physicalControl: options.physicalControl, horizonControls: STAGED_STUDENT_APPROACH_LIMITS.horizonControls });
    this.#lastControl = options.physicalControl;
  }

  get firstPickupOrigin() { return this.#firstPickupLease !== null; }
  get episode() { return this.#window.episode; }
  get startControl() { return this.#window.startControl; }
  get controls() { return this.#lastControl - this.startControl; }
  get horizonControls() { return this.#window.horizonControls; }
  get plan() { return this.#window.plan; }
  get ended() { return this.#ended ? structuredClone(this.#ended) : null; }
  get active() { return this.#ended === null; }

  isOwnedBy({ parent, episode, requestId }) {
    if(this.#firstPickupLease&&!this.#firstPickupLease.isOwnedBy({parent,episode,requestId}))return false;
    return parent === this.#parent && episode === this.episode
      && parent?.segmentIndex === this.#entry.segmentIndex
      && parent?.skill?.objectBodyName === this.#entry.objectBodyName;
  }

  #clock(context) {
    if (!physicalCount(context?.episode) || !physicalCount(context?.physicalControl)) throw new Error('An actual physical episode and control are required');
    if (!this.isOwnedBy(context)) throw new Error('Student approach belongs to a different episode, parent, segment or object');
    if (context.physicalControl !== this.#lastControl) throw new Error('Student approach reads must use its last committed physical control');
  }

  #finish(reason, context, { arrived = false, fallback = false } = {}) {
    if (!this.#ended) this.#ended = Object.freeze({ reason, atControl: this.#lastControl,
      observedEpisode: context.episode, observedPhysicalControl: context.physicalControl,
      controls: this.controls, arrived, fallback });
    return { active: false, ...this.ended };
  }

  /** Call after parent.step and before inference. Only the parent's existing
   * quiet-arrival samples authorize teacher handoff. Expiry returns an explicit
   * request for the caller to plan a recorded approach from the actual state.
   */
  observeParent(skillStep, context) {
    if (!this.active) return { active: false, ...this.ended };
    if (!this.isOwnedBy(context)) return this.#finish(context.episode !== this.episode
      ? 'episode_changed' : 'approach_owner_changed', context);
    this.#clock(context);
    if (context.parent.phase === 'teacher' && skillStep?.mode === 'teacher' && skillStep.justEnteredTeacher) {
      return this.#finish('parent_teacher_handoff', context, { arrived: true });
    }
    if (context.parent.phase !== 'approach' || skillStep?.phase !== 'approach'
        || context.parent.finishRequested || skillStep?.mode === 'none') {
      return this.#finish(skillStep?.completionReason === 'cancelled' || context.parent.finishRequested
        ? 'cancelled' : 'parent_approach_ended', context);
    }
    if (this.controls >= this.horizonControls) return this.#finish('stage_goal_window_elapsed', context, { fallback: true });
    return { active: true, remainingControls: this.horizonControls - this.controls, arrived: false, fallback: false };
  }

  /** Read-only; duplicate reads and refused previews consume no physical time. */
  sample(live, context) {
    this.#clock(context);
    if (!this.active) return { expired: this.#ended.fallback, command: null, active: false, ...this.ended };
    if (context.parent.phase !== 'approach' || context.parent.finishRequested) throw new Error('Observe the parent transition before sampling a student goal');
    return this.#window.sample({ rootPositionWorld: live.rootPosWorld,
      rootQuaternionWorld: live.rootQuatXyzwWorld, objectPositionWorld: live.objPosWorld }, context);
  }

  buildObservation(encoded, bodyObservation) {
    return packLegacyStageStudentInput(encoded, bodyObservation, null);
  }

  /** Call only after the real action and all physics substeps have committed.
   * This records the caller's successful all-box preview; it runs no simulator.
   * physicalControl is the new post-action control, not the inference-call count.
   */
  commit({ record, ...context }) {
    if(this.#firstPickupLease&&context.physicsSubsteps!==17)throw Error('First pickup student requires all17 actual substeps');
    if (!this.active || !this.isOwnedBy(context)) throw new Error('Cannot commit an inactive or differently owned student approach');
    if (!physicalCount(context.physicalControl) || context.physicalControl !== this.#lastControl + 1) throw new Error('Student approach commits exactly one real control at a time');
    if (this.controls >= this.horizonControls) throw new Error('Student approach physical window is exhausted');
    if (record?.preview?.supported !== true || !finiteVector(record.command, 13)
        || !finiteVector(record.rawAction, 29)) throw new Error('A successful physical preview, 13-value command and 29-value action are required');
    this.#records.push({ ...structuredClone(record), preControl: this.#lastControl, physicalControl: context.physicalControl });
    this.#lastControl = context.physicalControl;
  }

  /** Explicit cancellation/reset never modifies the carry parent or its history. */
  cancel(reason, context) {
    if (typeof reason !== 'string' || !reason) throw new Error('Explicit cancellation reason required');
    if (this.isOwnedBy(context)) this.#clock(context);
    return this.#finish(reason, context);
  }

  /** End this window after an inferred action was refused, before physics.
   * The caller restores speculative history and replans recorded locomotion.
   * The unexecuted candidate stays separate from actual action records.
   */
  requestPreviewFallback({ record, ...context }) {
    this.#clock(context);
    if (!this.active || context.parent.phase !== 'approach' || context.parent.finishRequested)
      throw new Error('Preview fallback requires the active, unchanged student approach');
    if (record?.preview?.supported !== false || typeof record.preview.reason !== 'string'
        || !record.preview.reason || !finiteVector(record.command, 13) || !finiteVector(record.rawAction, 29))
      throw new Error('A refused preview and its unexecuted command/action are required');
    this.#refusedPreview = { ...structuredClone(record), physicalControl: this.#lastControl,
      executed: false };
    return this.#finish('student_preview_refused', context, { fallback: true });
  }

  review() {
    return { episode: this.episode, startControl: this.startControl, controls: this.controls,
      horizonControls: this.horizonControls, ended: this.ended, plan: structuredClone(this.plan),
      entry: structuredClone(this.#entry), records: structuredClone(this.#records),
      refusedPreview: structuredClone(this.#refusedPreview) };
  }
}
