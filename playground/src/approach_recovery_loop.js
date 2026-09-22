// Bounded approach recovery: retreat -> stance -> (error-reducing turn -> stance),
// then the caller retries the parent's segment from the reached pose. This module
// owns reference clocks, admission geometry and its own bounds only. The caller
// retains physics, inference, the per-action physical preview, policy history,
// request state and the parent carry sequence. Nothing here writes physics.
import { transformTeacherReference } from './teacher_reference.js';
import { turnReferenceTransform } from './teacher_turn_controller.js';
import { planTeacherStandingReference } from './teacher_standing_reference.js';
import { checkRestrictedReferenceSweep, restrictedSweepClearance } from './restricted_motion_geometry.js';
import { planBoxApproach } from './box_approach.js';
import { TEACHER_HUMAN_BODY_NAMES } from './teacher_obs.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const finite = (values, count) => values?.length === count && Array.from(values).every(Number.isFinite);
const planar = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const copy = value => value === undefined ? null : structuredClone(value);

/** URL flags. Every default is ON; `=0` disables (hullAdmissionFallback=off). */
export function readApproachRecoveryFlags(params) {
  const raw = name => (typeof params?.get === 'function' ? params.get(name) : params?.[name]) ?? null;
  const on = name => raw(name) !== '0';
  const number = (name, fallback) => { const value = Number.parseFloat(raw(name) ?? ''); return Number.isFinite(value) ? value : fallback; };
  const fallback = raw('hullAdmissionFallback');
  return Object.freeze({
    neverSuspend: on('neverSuspend'),
    approachRecoveryLoop: on('approachRecoveryLoop'),
    hullAdmissionFallback: fallback === null || fallback === '' || fallback === 'preview' ? 'preview' : 'off',
    // 'recovery' (default, also `=1`): convex mesh footprints replace the AABB in
    // the recovery loop's own admissions (retreat/turn/stance). 'all': also in
    // the ordinary recorded-approach admissions (changes approach trajectories;
    // regressed the J04 fixture). '0': AABB everywhere.
    convexObstacles: raw('convexObstacles') === '0' ? 'off' : raw('convexObstacles') === 'all' ? 'all' : 'recovery',
    // Deadline = control at which the box request was received + this budget
    // (the joint100 panel allows 5820 post-click controls).
    recoveryBudgetControls: Math.max(0, Math.round(number('recoveryBudgetControls', 5820))),
    // Controls reserved after the carry source: 180 settling + 439 exit + 739 ending.
    recoveryEndingControls: Math.max(0, Math.round(number('recoveryEndingControls', 180 + 439 + 739))),
    hullFallbackMaxDeficitM: Math.max(0, number('hullFallbackMaxDeficitM', .05)),
  });
}

export const APPROACH_RECOVERY_LIMITS = Object.freeze({
  maxRetreatsPerSegment: 2, repeatedRootM: .05, repeatedYawRad: 5 * Math.PI / 180,
  maxBoxDistanceM: 1.2, maxEntrySpeedMps: .25, minEntryRootHeightM: .7, minEntryUpright: .95,
  retreatReserveM: .1, turnReserveM: .15, stanceReserveM: .1, verticalReserveM: .1,
  stanceControls: 60, lookAheadFrames: 16, fallbackBodyRadiusM: .12, maxBodyRadiusM: .35,
  minRootHeightM: .45, minUpright: .5,
  progressClearanceM: .02, progressHeadingRad: 10 * Math.PI / 180,
  approachStepControls: 254 + 60, approachStepTravelM: .6, approachTurnControls: 273 + 60, approachSettleControls: 180,
  retreatControls: 199, turnControls: 273,
});

export const RECOVERY_REFUSAL_MESSAGES = Object.freeze({
  insufficient_budget_for_recovery: 'Not enough time is left to step back and retry. Choose another box destination.',
  repeated_refusal_state: 'Stepping back did not open a route to the box. Choose another box destination or walking direction.',
  no_recovery_progress: 'Stepping back did not improve the approach. Choose another box destination or walking direction.',
  retreat_limit: 'The robot already stepped back twice for this carry. Choose another box destination.',
  retreat_not_admissible: 'There is not enough room behind the robot to step back. Choose another box destination or walking direction.',
  recovery_preview_refused: 'The robot could not step back safely from here. Choose another box destination or walking direction.',
  recovery_lookahead_refused: 'Stepping back would come too close to a box. Choose another box destination or walking direction.',
  recovery_retry_unavailable: 'The carry could not resume after stepping back. Choose another box destination.',
  box_too_far: 'The approach stopped too far from the box to recover. Choose another box destination.',
  no_clear_approach_route: 'There is no clear route to the box from here. Choose another box destination or walking direction.',
  occupied_pregrasp_target: 'Another box blocks the pickup position. Move it or choose another destination.',
  route_unavailable: 'There is no clear route to the box from here. Choose another box destination or walking direction.',
  unsettled: 'The robot could not settle into a supported stance. Choose another walking direction or destination.',
  standing_geometry_unsupported: 'There is not enough room for a supported stance here. Choose another walking direction.',
});

/** Recorded-approach cost from the reached pose (informal, for the budget only). */
export function estimateApproachControls(distanceM, headingErrorRad, limits = APPROACH_RECOVERY_LIMITS) {
  const steps = Math.max(1, Math.ceil(Math.max(0, distanceM) / limits.approachStepTravelM));
  const turns = Math.abs(headingErrorRad) > Math.PI / 3 ? 1 : 0;
  return steps * limits.approachStepControls + turns * limits.approachTurnControls + limits.approachSettleControls;
}

/** retreat 199 + stances 120 + turn 273 (if needed) + re-approach estimate + source
 * + ending (180 settling + 439 exit + 739) must fit the controls left. */
export function recoveryBudgetFits({ controlsLeft, needsTurn, secondRetreat = false, approachEstimateControls, sourceControls,
  endingControls = 180 + 439 + 739, limits = APPROACH_RECOVERY_LIMITS }) {
  if (![controlsLeft, approachEstimateControls, sourceControls, endingControls].every(Number.isFinite)) {
    return { fits: false, reason: 'invalid_budget_inputs', required: NaN, controlsLeft, breakdown: null };
  }
  const breakdown = { retreat: limits.retreatControls, stances: 2 * limits.stanceControls,
    turn: needsTurn ? limits.turnControls : 0,
    // Facing refusals need a second retreat after the turn (behind the pickup
    // root along its yaw) so the re-approach arrives facing the pickup pose.
    secondRetreat: secondRetreat ? limits.retreatControls + limits.stanceControls : 0,
    approach: Math.ceil(approachEstimateControls),
    source: Math.ceil(sourceControls), ending: Math.ceil(endingControls) };
  const required = Object.values(breakdown).reduce((sum, value) => sum + value, 0);
  return { fits: controlsLeft >= required, reason: controlsLeft >= required ? null : 'insufficient_budget_for_recovery',
    required, controlsLeft, breakdown };
}

/** Refusal poses already seen for a request. A new refusal within 5 cm and 5 deg
 * of a previous one means retreating again would only repeat the loop. */
export class RefusalStateMemory {
  constructor({ repeatedRootM = APPROACH_RECOVERY_LIMITS.repeatedRootM, repeatedYawRad = APPROACH_RECOVERY_LIMITS.repeatedYawRad } = {}) {
    this.rootM = repeatedRootM; this.yawRad = repeatedYawRad; this.reset();
  }
  reset() { this.poses = []; }
  remember({ episode, requestId, rootXY, yawRad, physicalControl = null, reason = null }) {
    if (!finite(rootXY, 2) || !Number.isFinite(yawRad)) throw new Error('A finite refusal pose is required');
    this.poses.push({ episode, requestId, rootXY: [rootXY[0], rootXY[1]], yawRad, physicalControl, reason });
  }
  repeated({ episode, requestId, rootXY, yawRad }) {
    if (!finite(rootXY, 2) || !Number.isFinite(yawRad)) return null;
    return this.poses.find(p => p.episode === episode && p.requestId === requestId
      && planar(p.rootXY, rootXY) <= this.rootM && Math.abs(wrap(p.yawRad - yawRad)) <= this.yawRad) ?? null;
  }
  review() { return copy(this.poses); }
}

const asPolygon = rect => Array.isArray(rect.hull) && rect.hull.length >= 3 ? rect.hull
  : [[rect.minX, rect.minY], [rect.maxX, rect.minY], [rect.maxX, rect.maxY], [rect.minX, rect.maxY]];
/** Distance from an XY point to a convex obstacle footprint (0 inside). */
export function distanceToObstacle(point, rect) {
  const polygon = asPolygon(rect);
  let inside = true, best = Infinity, sign = 0;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length];
    const ex = b[0] - a[0], ey = b[1] - a[1], length = Math.hypot(ex, ey);
    if (length < 1e-12) continue;
    const cross = ex * (point[1] - a[1]) - ey * (point[0] - a[0]);
    if (Math.abs(cross) > 1e-12) { if (sign && Math.sign(cross) !== sign) inside = false; sign ||= Math.sign(cross); }
    const t = Math.max(0, Math.min(1, ((point[0] - a[0]) * ex + (point[1] - a[1]) * ey) / (length * length)));
    best = Math.min(best, Math.hypot(point[0] - a[0] - t * ex, point[1] - a[1] - t * ey));
  }
  return inside && sign ? 0 : best;
}
export function rootClearance(rootXY, obstacles) {
  let clearance = Infinity, obstacleIndex = null;
  obstacles.forEach((rect, index) => { const d = distanceToObstacle(rootXY, rect); if (d < clearance) { clearance = d; obstacleIndex = index; } });
  return { clearanceM: clearance, obstacleIndex };
}

/** Per-body disc radii and source Z ranges from the sweep's collisionParts (the
 * terminal hull of each part around that body's terminal position). Bodies
 * without a part keep a fallback radius. Cached per sweep. */
const discCache = new WeakMap();
export function bodyDiscsFromSweep(sweep, skill, limits = APPROACH_RECOVERY_LIMITS) {
  if (discCache.has(sweep)) return discCache.get(sweep);
  const terminal = skill.frames[skill.sourceFrames - 1], pose = sweep.terminalRootPose;
  const angle = yaw(pose.slice(3, 7)), c = Math.cos(-angle), s = Math.sin(-angle);
  const local = body => { const dx = terminal[84 + 3 * body] - pose[0], dy = terminal[85 + 3 * body] - pose[1]; return [c * dx - s * dy, s * dx + c * dy]; };
  const discs = TEACHER_HUMAN_BODY_NAMES.map((name, body) => ({ name, radiusM: limits.fallbackBodyRadiusM, fromPart: false,
    minZ: terminal[86 + 3 * body] - .15, maxZ: terminal[86 + 3 * body] + .15, parts: 0 }));
  for (const part of sweep.collisionParts ?? []) {
    const body = TEACHER_HUMAN_BODY_NAMES.indexOf(part.body);
    if (body < 0 || !Array.isArray(part.terminal?.xyHull)) continue;
    const anchor = local(body), disc = discs[body];
    const radius = Math.max(...part.terminal.xyHull.map(p => Math.hypot(p[0] - anchor[0], p[1] - anchor[1])));
    disc.radiusM = Math.min(limits.maxBodyRadiusM, disc.parts ? Math.max(disc.radiusM, radius) : radius);
    if (Number.isFinite(part.motion?.minZ) && Number.isFinite(part.motion?.maxZ)) {
      disc.minZ = disc.parts ? Math.min(disc.minZ, part.motion.minZ) : part.motion.minZ;
      disc.maxZ = disc.parts ? Math.max(disc.maxZ, part.motion.maxZ) : part.motion.maxZ;
    }
    disc.fromPart = true; disc.parts++;
  }
  const result = Object.freeze(discs.map(d => Object.freeze({ name: d.name, radiusM: d.radiusM, fromPart: d.fromPart, minZ: d.minZ, maxZ: d.maxZ })));
  discCache.set(sweep, result);
  return result;
}

/** Look ahead over the next `count` aligned reference frames: every body whose
 * source Z range overlaps an obstacle's Z range must keep its disc radius plus the
 * reserve away from that obstacle footprint. Reference geometry only. */
export function lookAheadClear({ worldFrames, fromIndex, count, obstacles, discs, reserve, verticalReserve = .1 }) {
  for (let k = 1; k <= count; k++) {
    const frame = worldFrames[fromIndex + k];
    if (!frame) break;
    for (let index = 0; index < obstacles.length; index++) {
      const rect = obstacles[index], lowZ = Number.isFinite(rect.minZ) ? rect.minZ : -Infinity, highZ = Number.isFinite(rect.maxZ) ? rect.maxZ : Infinity;
      for (let body = 0; body < 39; body++) {
        const disc = discs[body], z = frame[86 + 3 * body];
        if (Math.min(disc.maxZ, z + .05) < lowZ - verticalReserve || Math.max(disc.minZ, z - .05) > highZ + verticalReserve) continue;
        const distance = distanceToObstacle([frame[84 + 3 * body], frame[85 + 3 * body]], rect);
        if (distance < disc.radiusM + reserve - 1e-9) {
          return { clear: false, frameIndex: fromIndex + k, body: disc.name, obstacleIndex: index,
            obstacleName: rect.name ?? null, distanceM: distance, requiredM: disc.radiusM + reserve };
        }
      }
    }
  }
  return { clear: true };
}

function validSkill(skill, sourceFrames, label) {
  if (!skill || skill.locomotionOnly !== true || skill.sourceFrames !== sourceFrames || !Array.isArray(skill.frames)
      || skill.frames.length < sourceFrames + 16 || !skill.frames.every(frame => finite(frame, 747))) {
    throw new Error(`${label} requires a complete ${sourceFrames}-control locomotion record with +16 lookahead`);
  }
}
const turnDelta = skill => wrap(yaw(skill.frames[skill.sourceFrames - 1].slice(3, 7)) - yaw(skill.frames[0].slice(3, 7)));

/** Whole-clip hull admission with the preview-owned fallback for small deficits. */
function admitClip({ sweep, sourceFrames, worldFrames, obstacles, reserve, flags }) {
  const geometry = checkRestrictedReferenceSweep({ sweep, sourceFrames, alignedReferenceFrames: worldFrames, obstacles, trackingReserve: reserve });
  const clearance = geometry.worldHull ? restrictedSweepClearance(geometry.worldHull, obstacles, reserve) : { marginM: NaN, obstacleIndex: null };
  if (geometry.supported) return { admitted: true, mode: 'hull', geometry, marginM: clearance.marginM, obstacleIndex: clearance.obstacleIndex };
  const deficit = Number.isFinite(clearance.marginM) ? -clearance.marginM : Infinity;
  if (geometry.reason === 'reference_sweep_clearance' && flags.hullAdmissionFallback === 'preview' && deficit <= flags.hullFallbackMaxDeficitM) {
    return { admitted: true, mode: 'preview_owned', geometry, marginM: clearance.marginM, obstacleIndex: clearance.obstacleIndex, deficitM: deficit };
  }
  return { admitted: false, mode: null, geometry, marginM: clearance.marginM, obstacleIndex: clearance.obstacleIndex, deficitM: deficit,
    reason: geometry.reason === 'reference_sweep_clearance' ? 'reference_sweep_clearance' : geometry.reason };
}

const ACTIVE = new Set(['recovery_retreat', 'recovery_stance', 'recovery_turn', 'recovery_stance2', 'recovery_retreat2', 'recovery_stance3']);
export const RECOVERY_PHASES = Object.freeze([...ACTIVE]);
const STATUS = Object.freeze({ recovery_retreat: 'Stepping back to make room for the approach…',
  recovery_stance: 'Settling after stepping back…', recovery_turn: 'Turning toward the box…',
  recovery_stance2: 'Settling after turning toward the box…', recovery_retreat2: 'Stepping back behind the pickup position…',
  recovery_stance3: 'Settling before retrying the approach…' });

/** One retreat -> stance -> (turn -> stance) cycle. Behaves as a locomotion-only
 * teacher skill controller for the caller's control loop. Every issued action
 * must be previewed by the caller; commit() verifies that preview. */
export class ApproachRecoveryCycle {
  constructor({ coordinator, kind, parent, decision, retreatSkill, turnSkills, sweeps, readObstacles, flags, limits,
    objectBodyName, objectPointsLocal, requestedGoalWorld, turnTarget, entryMetrics, admission, worldFrames, episode, requestId, startControl }) {
    this.coordinator = coordinator; this.kind = kind; this.parent = parent; this.decision = decision;
    this.retreatSkill = retreatSkill; this.turnSkills = turnSkills; this.sweeps = sweeps; this.readObstacles = readObstacles;
    this.flags = flags; this.limits = limits; this.objectBodyName = objectBodyName; this.objectPointsLocal = objectPointsLocal;
    this._requestedGoalWorld = Array.from(requestedGoalWorld); this.turnTarget = turnTarget; this.entryMetrics = entryMetrics;
    this.episode = episode; this.requestId = requestId; this.startControl = startControl;
    this.stages = [
      { name: 'recovery_retreat', kind: 'clip', skill: retreatSkill, sweep: sweeps.get(retreatSkill), worldFrames,
        duration: retreatSkill.sourceFrames, admission, controls: 0, prepared: true, reserve: limits.retreatReserveM },
      { name: 'recovery_stance', kind: 'stance', duration: limits.stanceControls, controls: 0, prepared: false, reserve: limits.stanceReserveM },
      { name: 'recovery_turn', kind: 'clip', duration: 0, controls: 0, prepared: false, reserve: limits.turnReserveM },
      { name: 'recovery_stance2', kind: 'stance', duration: limits.stanceControls, controls: 0, prepared: false, reserve: limits.stanceReserveM },
      { name: 'recovery_retreat2', kind: 'clip', duration: 0, controls: 0, prepared: false, reserve: limits.retreatReserveM, second: true },
      { name: 'recovery_stance3', kind: 'stance', duration: limits.stanceControls, controls: 0, prepared: false, reserve: limits.stanceReserveM },
    ];
    this.stageIndex = 0; this.phase = 'recovery_retreat'; this.completionReason = null;
    this.totalControls = 0; this.records = []; this.events = []; this.refusedPreview = null; this.turnDecision = null; this.lookAheadRefusal = null;
    this.cancelRequested = false; this.locomotionOnly = true; this.secondRetreatDecision = null;
    this._entered = true; this._issued = false; this._completedPending = false; this._completed = false;
    this.outcome = { kind, startControl, controls: 0, stages: [], minRootHeightM: Infinity, minUpright: Infinity };
    this.events.push({ event: 'cycle_started', physicalControl: startControl, kind, admission: admission.mode, retreatMarginM: admission.marginM });
  }
  get stage() { return this.stages[this.stageIndex] ?? null; }
  get skill() {
    const base = this.stage?.kind === 'clip' && this.stage.skill ? this.stage.skill : this.retreatSkill;
    return { ...base, objectBodyName: this.objectBodyName, objectPointsLocal: this.objectPointsLocal, locomotionOnly: true };
  }
  get sourceFrames() { return this.stage?.kind === 'clip' ? this.stage.skill?.sourceFrames ?? 0 : 0; }
  get referenceIndex() { return this.stage?.controls ?? 0; }
  get requestedGoalWorld() { return [...this._requestedGoalWorld]; }
  get finishRequested() { return this.cancelRequested; }
  get statusMessage() { return STATUS[this.phase] ?? 'Recovering the approach…'; }
  get active() { return ACTIVE.has(this.phase); }
  isActive() { return this.active || this._completedPending; }

  requestCancel() { if (this.active) this.cancelRequested = true; }

  _observe(proprio) {
    if (!finite(proprio?.rootPosWorld, 3) || !finite(proprio?.rootQuatXyzwWorld, 4)) throw new Error('Finite live root pose is required');
    const q = proprio.rootQuatXyzwWorld, upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    this.outcome.minRootHeightM = Math.min(this.outcome.minRootHeightM, proprio.rootPosWorld[2]);
    this.outcome.minUpright = Math.min(this.outcome.minUpright, upright);
    this.outcome.finalRootPositionWorld = Array.from(proprio.rootPosWorld); this.outcome.finalRootYawRad = yaw(q);
    if (this.active && (proprio.rootPosWorld[2] < this.limits.minRootHeightM || upright < this.limits.minUpright)) this._abort('lost_balance');
  }

  _abort(reason) {
    if (!this.active) return;
    this.events.push({ event: 'cycle_aborted', reason, stage: this.phase, stageControl: this.stage?.controls ?? null,
      physicalControl: this.startControl + this.totalControls });
    this.phase = 'complete'; this.completionReason = reason; this._completedPending = true; this._issued = false;
  }

  _terminalOfPreviousClip() {
    for (let index = this.stageIndex - 1; index >= 0; index--) {
      const stage = this.stages[index];
      if (stage.kind === 'clip' && stage.worldFrames && !stage.skipped) return stage.worldFrames[stage.skill.sourceFrames - 1];
    }
    return null;
  }

  _prepareStance(stage, proprio) {
    const terminal = this._terminalOfPreviousClip();
    if (!terminal) throw new Error('A stance follows an executed clip');
    if (!finite(proprio.objPosWorld, 3) || !finite(proprio.objQuatXyzwWorld, 4)) throw new Error('Live object pose is required for a stance');
    const plan = planTeacherStandingReference(terminal, { alignment: 'original', rootPosition: proprio.rootPosWorld,
      rootQuaternion: proprio.rootQuatXyzwWorld, objectPosition: proprio.objPosWorld, objectQuaternion: proprio.objQuatXyzwWorld,
      objectPointsLocal: this.objectPointsLocal });
    const previous = this.stages.slice(0, this.stageIndex).reverse().find(s => s.kind === 'clip' && !s.skipped);
    const sweep = this.sweeps.get(previous?.skill ?? this.retreatSkill);
    const obstacles = this.readObstacles();
    const geometry = checkRestrictedReferenceSweep({ sweep, sourceFrames: 1, alignedReferenceFrames: [plan.frame], obstacles, trackingReserve: stage.reserve });
    const clearance = geometry.worldHull ? restrictedSweepClearance(geometry.worldHull, obstacles, stage.reserve) : { marginM: NaN };
    // Standing still is the safest owned action: a refused stance hull is
    // executed under the per-control preview instead of refusing the cycle.
    stage.admission = { admitted: true, mode: geometry.supported ? 'hull' : 'preview_owned', geometry, marginM: clearance.marginM };
    stage.frame = plan.frame; stage.prepared = true;
    this.events.push({ event: 'stance_prepared', stage: stage.name, mode: stage.admission.mode, marginM: clearance.marginM,
      physicalControl: this.startControl + this.totalControls });
  }

  _turnError(proprio) {
    const heading = yaw(proprio.rootQuatXyzwWorld), target = this.turnTarget;
    if (target?.kind === 'pickup_yaw') {
      let required = target.yawRad;
      try {
        const plan = this.parent?.child?._referencePlan?.(proprio);
        if (plan?.first && finite(plan.first.slice(3, 7), 4)) required = yaw(plan.first.slice(3, 7));
      } catch { /* keep the yaw measured at admission */ }
      return { errorRad: wrap(heading - required), targetKind: 'pickup_yaw', targetYawRad: required };
    }
    const goal = target?.goalWorld ?? this.parent?.approachGoalWorld ?? this.parent?.child?.approachGoalWorld;
    if (!goal) return { errorRad: 0, targetKind: 'none', targetYawRad: null };
    const bearing = Math.atan2(goal[1] - proprio.rootPosWorld[1], goal[0] - proprio.rootPosWorld[0]);
    return { errorRad: wrap(heading - bearing), targetKind: 'approach_bearing', targetYawRad: bearing };
  }

  _prepareTurn(stage, proprio) {
    const error = this._turnError(proprio), obstacles = this.readObstacles();
    const candidates = this.turnSkills.map(skill => ({ skill, delta: turnDelta(skill), remainingError: Math.abs(wrap(error.errorRad + turnDelta(skill))) }))
      .filter(candidate => candidate.delta * error.errorRad < 0 && candidate.remainingError < Math.abs(error.errorRad))
      .sort((a, b) => a.remainingError - b.remainingError);
    const attempts = [];
    let chosen = null;
    for (const preferred of ['hull', 'preview_owned']) {
      for (const candidate of candidates) {
        const transform = turnReferenceTransform(candidate.skill.frames[0], proprio.rootPosWorld, proprio.rootQuatXyzwWorld);
        const worldFrames = candidate.skill.frames.map(frame => transformTeacherReference(frame, transform));
        const admission = admitClip({ sweep: this.sweeps.get(candidate.skill), sourceFrames: candidate.skill.sourceFrames, worldFrames,
          obstacles, reserve: stage.reserve, flags: this.flags });
        if (preferred === 'hull') attempts.push({ skill: candidate.skill.name ?? null, deltaRad: candidate.delta, remainingErrorRad: candidate.remainingError,
          admitted: admission.admitted, mode: admission.mode, marginM: admission.marginM, reason: admission.reason ?? null });
        if (admission.admitted && admission.mode === preferred) { chosen = { candidate, worldFrames, admission }; break; }
      }
      if (chosen) break;
    }
    this.turnDecision = { errorRad: error.errorRad, targetKind: error.targetKind, targetYawRad: error.targetYawRad,
      candidates: attempts, chosen: chosen ? { skill: chosen.candidate.skill.name ?? null, deltaRad: chosen.candidate.delta, mode: chosen.admission.mode,
        marginM: chosen.admission.marginM } : null, skipped: !chosen, skipReason: chosen ? null : candidates.length ? 'turn_not_admissible' : 'no_error_reducing_turn' };
    this.events.push({ event: 'turn_prepared', ...this.turnDecision, physicalControl: this.startControl + this.totalControls });
    if (!chosen) {
      // No turn: skip the turn and its stance; the approach retries directly.
      for (const later of this.stages.slice(3)) { later.duration = 0; later.prepared = true; later.skipped = true; }
      this.secondRetreatDecision = { planned: false, reason: 'no_turn_executed' };
      return false;
    }
    Object.assign(stage, { skill: chosen.candidate.skill, sweep: this.sweeps.get(chosen.candidate.skill), worldFrames: chosen.worldFrames,
      duration: chosen.candidate.skill.sourceFrames, admission: chosen.admission, prepared: true });
    return true;
  }

  /** Facing refusals: after the turn the re-approach would turn back toward the
   * pickup root and arrive with the old facing error unless the robot first
   * steps back behind that root along its new heading. Counts as the second
   * retreat of the segment; skipped when the approach heading is already fine,
   * when no turn ran, when the segment has no retreat left or when the clip is
   * not admissible. */
  _prepareSecondRetreat(stage, proprio) {
    const skip = reason => { this.secondRetreatDecision = { planned: false, reason }; stage.duration = 0; stage.prepared = true; stage.skipped = true;
      this.stages[this.stageIndex + 1].duration = 0; this.stages[this.stageIndex + 1].prepared = true; this.stages[this.stageIndex + 1].skipped = true;
      this.events.push({ event: 'second_retreat_skipped', reason, physicalControl: this.startControl + this.totalControls }); return false; };
    if (this.kind !== 'facing') return skip('not_a_facing_refusal');
    if (!this.turnDecision?.chosen) return skip('no_turn_executed');
    if (!this.coordinator?.retreatAvailable?.(this)) return skip('retreat_limit');
    const heading = yaw(proprio.rootQuatXyzwWorld), pickup = this.entryMetrics?.pickupRootXY;
    if (!pickup) return skip('no_pickup_root');
    const bearing = Math.atan2(pickup[1] - proprio.rootPosWorld[1], pickup[0] - proprio.rootPosWorld[0]);
    const bearingErrorRad = wrap(heading - bearing);
    if (Math.abs(bearingErrorRad) <= Math.PI / 3) return skip('approach_heading_already_aligned');
    const transform = turnReferenceTransform(this.retreatSkill.frames[0], proprio.rootPosWorld, proprio.rootQuatXyzwWorld);
    const worldFrames = this.retreatSkill.frames.map(frame => transformTeacherReference(frame, transform));
    const admission = admitClip({ sweep: this.sweeps.get(this.retreatSkill), sourceFrames: this.retreatSkill.sourceFrames, worldFrames,
      obstacles: this.readObstacles(), reserve: stage.reserve, flags: this.flags });
    if (!admission.admitted) return skip('retreat_not_admissible');
    this.coordinator.countRetreat(this);
    Object.assign(stage, { skill: this.retreatSkill, sweep: this.sweeps.get(this.retreatSkill), worldFrames, duration: this.retreatSkill.sourceFrames, admission, prepared: true });
    this.secondRetreatDecision = { planned: true, bearingErrorRad, mode: admission.mode, marginM: admission.marginM };
    this.events.push({ event: 'second_retreat_prepared', ...this.secondRetreatDecision, physicalControl: this.startControl + this.totalControls });
    return true;
  }

  _enterStage(proprio) {
    // Prepare the current stage; skip zero-duration stages until one issues actions.
    for (let guard = 0; guard < 8 && this.stageIndex < this.stages.length; guard++) {
      const stage = this.stage;
      if (!stage.prepared) {
        if (stage.kind === 'stance') this._prepareStance(stage, proprio);
        else if (stage.second) { if (!this._prepareSecondRetreat(stage, proprio)) { this.stageIndex++; continue; } }
        else if (!this._prepareTurn(stage, proprio)) { this.stageIndex++; continue; }
      }
      if (stage.duration === 0) { this.stageIndex++; continue; }
      this.phase = stage.name; return true;
    }
    this.phase = 'complete'; this.completionReason = this.cancelRequested ? 'cancelled' : 'finished';
    this._completedPending = true;
    return false;
  }

  step(proprio) {
    if (this.active || this._completedPending) this._observe(proprio);
    if (this._completedPending) {
      this._completedPending = false; this._completed = true;
      this.outcome.controls = this.totalControls; this.outcome.completionReason = this.completionReason;
      this.events.push({ event: 'cycle_completed', reason: this.completionReason, physicalControl: this.startControl + this.totalControls });
      return { phase: 'complete', mode: 'student', justCompleted: true, justEnteredTeacher: false, completionReason: this.completionReason,
        outcome: copy(this.outcome), requestedGoalWorld: this.requestedGoalWorld, recoveryCycle: true, kind: this.kind };
    }
    if (!this.active) return { phase: this.phase, mode: 'student', justCompleted: false, justEnteredTeacher: false,
      completionReason: this.completionReason, outcome: copy(this.outcome), requestedGoalWorld: this.requestedGoalWorld, recoveryCycle: true, kind: this.kind };
    if (this._entered && !this._enterStage(proprio)) return this.step(proprio);
    const stage = this.stage;
    let referenceFrames, lookAhead = null;
    if (stage.kind === 'clip') {
      const index = stage.controls;
      referenceFrames = [stage.worldFrames[index + 1], stage.worldFrames[index + 16]];
      if (stage.admission.mode === 'preview_owned') {
        // The whole hull already missed the reserve by at most the admitted deficit,
        // so the look-ahead keeps (reserve - max deficit) around each relevant body;
        // closer than that the per-control physical preview decides.
        lookAhead = lookAheadClear({ worldFrames: stage.worldFrames, fromIndex: index, count: this.limits.lookAheadFrames,
          obstacles: this.readObstacles(), discs: bodyDiscsFromSweep(stage.sweep, stage.skill, this.limits),
          reserve: Math.max(0, stage.reserve - this.flags.hullFallbackMaxDeficitM), verticalReserve: this.limits.verticalReserveM });
        if (!lookAhead.clear) {
          this.lookAheadRefusal = { ...lookAhead, stage: stage.name, stageControl: index };
          this._abort('recovery_lookahead_refused');
          return this.step(proprio);
        }
      }
    } else referenceFrames = [stage.frame, stage.frame];
    const entered = this._entered; this._entered = false; this._issued = true;
    return { phase: this.phase, mode: 'teacher', locomotionOnly: true, skill: this.skill, sourceFrames: this.sourceFrames,
      referenceIndex: this.referenceIndex, referenceFrames, justEnteredTeacher: entered, justCompleted: false,
      completionReason: null, supported: true, requestedGoalWorld: this.requestedGoalWorld, outcome: copy(this.outcome),
      stage: stage.name, admissionMode: stage.admission?.mode ?? null, previewOwned: stage.admission?.mode === 'preview_owned',
      lookAhead, recoveryCycle: true, kind: this.kind };
  }

  /** Verify the caller's all-box preview of the action that was just executed. */
  commit(preview, physicalControl) {
    if (!this._issued) throw new Error('A recovery action must be issued before it is committed');
    if (preview?.supported !== true || preview.completedSubsteps !== 17 || preview.allowedContactCount !== 0 || preview.unwantedContactCount !== 0) {
      throw new Error('Every recovery action requires a complete all-box contact-free preview');
    }
    this.records.push({ stage: this.phase, stageControl: this.stage.controls, physicalControl, previewOwned: this.stage.admission?.mode === 'preview_owned' });
  }

  /** The caller's preview refused the issued action: abort, nothing executed. */
  refusePreview(preview, physicalControl) {
    this.refusedPreview = { stage: this.phase, stageControl: this.stage?.controls ?? null, physicalControl, preview: copy(preview) };
    this._issued = false;
    this._abort('recovery_preview_refused');
  }

  /** Advance exactly once after the issued action executed. */
  advance() {
    if (!this._issued || !this.active) return;
    this._issued = false;
    const stage = this.stage;
    stage.controls++; this.totalControls++;
    if (stage.controls < stage.duration) return;
    this.outcome.stages.push({ stage: stage.name, controls: stage.controls, admission: stage.admission?.mode ?? null, marginM: stage.admission?.marginM ?? null });
    this.stageIndex++; this._entered = true;
    if (this.cancelRequested || this.stageIndex >= this.stages.length) {
      this.phase = 'complete'; this.completionReason = this.cancelRequested ? 'cancelled' : 'finished'; this._completedPending = true;
    }
  }

  review() {
    return { kind: this.kind, episode: this.episode, requestId: this.requestId, startControl: this.startControl, controls: this.totalControls,
      phase: this.phase, completionReason: this.completionReason, decision: copy(this.decision), entryMetrics: copy(this.entryMetrics),
      stages: this.stages.map(stage => ({ name: stage.name, kind: stage.kind, duration: stage.duration, controls: stage.controls,
        prepared: stage.prepared, skipped: Boolean(stage.skipped), admission: stage.admission ? { mode: stage.admission.mode,
          marginM: stage.admission.marginM ?? null, deficitM: stage.admission.deficitM ?? null,
          geometry: stage.admission.geometry ? { supported: stage.admission.geometry.supported, reason: stage.admission.geometry.reason,
            obstacleIndex: stage.admission.geometry.obstacleIndex ?? null, trackingReserve: stage.admission.geometry.trackingReserve ?? null } : null } : null,
        skill: stage.skill?.name ?? null })),
      turnDecision: copy(this.turnDecision), secondRetreatDecision: copy(this.secondRetreatDecision),
      lookAheadRefusal: copy(this.lookAheadRefusal), refusedPreview: copy(this.refusedPreview),
      records: this.records.length, events: copy(this.events), outcome: copy(this.outcome) };
  }
}

/** Per-page owner of every recovery cycle: bounds, memory, admission and progress. */
export class ApproachRecoveryCoordinator {
  constructor({ retreatSkill, turnSkills, sweeps, readObstacles, flags = readApproachRecoveryFlags(null), limits = APPROACH_RECOVERY_LIMITS }) {
    validSkill(retreatSkill, limits.retreatControls, 'Approach recovery retreat');
    if (!Array.isArray(turnSkills) || !turnSkills.length) throw new Error('Approach recovery requires the complete 273-control turns');
    for (const skill of turnSkills) validSkill(skill, limits.turnControls, 'Approach recovery turn');
    if (!(sweeps instanceof Map) || !sweeps.get(retreatSkill) || turnSkills.some(skill => !sweeps.get(skill))) throw new Error('Bound collision sweeps are required for the retreat and turns');
    if (typeof readObstacles !== 'function') throw new Error('An obstacle reader is required');
    this.retreatSkill = retreatSkill; this.turnSkills = turnSkills; this.sweeps = sweeps; this.readObstacles = readObstacles;
    this.flags = flags; this.limits = limits; this.generation = 0; this.reset();
  }
  reset() {
    this.generation++; this.memory = new RefusalStateMemory(this.limits); this.segments = new Map();
    this.decisions = []; this.cycles = []; this.completions = [];
  }
  segmentKey(context, parent) { return `${context.episode}:${context.requestId}:${parent?.segmentIndex ?? 0}`; }
  segmentState(key) {
    if (!this.segments.has(key)) this.segments.set(key, { retreats: 0, noProgress: false, cycles: [] });
    return this.segments.get(key);
  }

  metrics(parent, proprio, obstacles, kind) {
    const root = proprio.rootPosWorld, heading = yaw(proprio.rootQuatXyzwWorld);
    let first = null;
    try { first = parent?.child?._referencePlan?.(proprio)?.first ?? parent?.referencePlan?.first ?? null; } catch { first = parent?.referencePlan?.first ?? null; }
    const pickupRoot = first ? [first[0], first[1]] : null, pickupYaw = first ? yaw(first.slice(3, 7)) : null;
    const goal = parent?.approachGoalWorld ?? parent?.child?.approachGoalWorld ?? (pickupRoot ? [...pickupRoot, 0] : null);
    const bearing = goal ? Math.atan2(goal[1] - root[1], goal[0] - root[0]) : null;
    const clearance = rootClearance([root[0], root[1]], obstacles);
    let route = null;
    try { route = goal ? planBoxApproach(root, goal, obstacles.map(r => ({ minX: r.minX, maxX: r.maxX, minY: r.minY, maxY: r.maxY })), { clearance: .55 }) : null; }
    catch { route = null; }
    return { rootXY: [root[0], root[1]], yawRad: heading, clearanceM: clearance.clearanceM, clearanceObstacleIndex: clearance.obstacleIndex,
      pickupRootXY: pickupRoot, pickupDistanceM: pickupRoot ? planar(root, pickupRoot) : null,
      facingErrorRad: pickupYaw === null ? null : wrap(heading - pickupYaw), pickupYawRad: pickupYaw,
      bearingErrorRad: bearing === null ? null : wrap(heading - bearing),
      headingErrorRad: kind === 'facing' && pickupYaw !== null ? wrap(heading - pickupYaw) : bearing === null ? 0 : wrap(heading - bearing),
      routeSupported: route?.supported ?? null, directBlocked: route?.directBlocked ?? null, routed: route?.routed ?? null };
  }

  /** Decide whether one more retreat cycle may start at this refusal. */
  tryStart({ kind, parent, live, context, refusal = {}, unloaded }) {
    const decision = { kind, episode: context?.episode ?? null, requestId: context?.requestId ?? null, physicalControl: context?.physicalControl ?? null,
      segmentIndex: parent?.segmentIndex ?? null, refusalReason: refusal?.reason ?? null, supported: false, reason: null };
    const refuse = (reason, extra = {}) => { Object.assign(decision, { reason, ...extra }); this.decisions.push(decision); return { supported: false, reason, decision }; };
    if (!this.flags.approachRecoveryLoop) return refuse('recovery_loop_disabled');
    if (!['facing', 'approach'].includes(kind)) return refuse('unknown_refusal_kind');
    if (!parent || !live || !context || ![context.episode, context.requestId, context.physicalControl].every(Number.isSafeInteger)) return refuse('missing_parent_or_context');
    if (unloaded !== true) return refuse('robot_loaded');
    if (!finite(live.rootPosWorld, 3) || !finite(live.rootQuatXyzwWorld, 4) || !finite(live.rootVelWorld, 3) || !finite(live.objPosWorld, 3) || !finite(live.objQuatXyzwWorld, 4)) return refuse('invalid_live_state');
    const objectBodyName = parent.skill?.objectBodyName ?? parent.rawSkill?.objectBodyName;
    if (!objectBodyName || live.objectBodyName !== objectBodyName || !Array.isArray(parent.skill?.objectPointsLocal)) return refuse('object_mismatch');
    if (!finite(parent.requestedGoalWorld, 3)) return refuse('missing_parent_goal');
    if (parent.finishRequested) return refuse('parent_cancelled');
    const q = live.rootQuatXyzwWorld, upright = Number.isFinite(live.uprightScore) ? live.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    const speed = Math.hypot(live.rootVelWorld[0], live.rootVelWorld[1]);
    decision.entry = { rootHeightM: live.rootPosWorld[2], upright, planarSpeedMps: speed };
    if (live.rootPosWorld[2] < this.limits.minEntryRootHeightM || upright < this.limits.minEntryUpright || speed > this.limits.maxEntrySpeedMps) return refuse('entry_not_standing');
    const boxDistance = planar(live.rootPosWorld, live.objPosWorld);
    decision.boxDistanceM = boxDistance;
    if (boxDistance > this.limits.maxBoxDistanceM) return refuse('box_too_far');
    const key = this.segmentKey(context, parent), segment = this.segmentState(key);
    decision.segmentKey = key; decision.retreatsBefore = segment.retreats;
    if (segment.retreats >= this.limits.maxRetreatsPerSegment) return refuse('retreat_limit');
    if (segment.noProgress) return refuse('no_recovery_progress');
    const pose = { episode: context.episode, requestId: context.requestId, rootXY: [live.rootPosWorld[0], live.rootPosWorld[1]], yawRad: yaw(q) };
    const repeated = this.memory.repeated(pose);
    this.memory.remember({ ...pose, physicalControl: context.physicalControl, reason: refusal?.reason ?? null });
    if (repeated) return refuse('repeated_refusal_state', { repeated: copy(repeated) });
    const obstacles = this.readObstacles();
    const metrics = this.metrics(parent, live, obstacles, kind);
    decision.entryMetrics = metrics;
    const needsTurn = kind === 'facing' ? true : Math.abs(metrics.headingErrorRad) > Math.PI / 3;
    const deadline = Number.isFinite(context.deadlineControl) ? context.deadlineControl
      : Number.isFinite(context.requestReceivedControl) ? context.requestReceivedControl + this.flags.recoveryBudgetControls : null;
    if (deadline === null) return refuse('unknown_deadline');
    const budget = recoveryBudgetFits({ controlsLeft: deadline - context.physicalControl, needsTurn,
      secondRetreat: kind === 'facing' && segment.retreats + 1 < this.limits.maxRetreatsPerSegment,
      approachEstimateControls: estimateApproachControls(metrics.pickupDistanceM ?? boxDistance, metrics.headingErrorRad, this.limits),
      sourceControls: parent.skill?.sourceFrames ?? 0, endingControls: this.flags.recoveryEndingControls, limits: this.limits });
    decision.budget = { ...budget, deadlineControl: deadline };
    if (!budget.fits) return refuse('insufficient_budget_for_recovery');
    const transform = turnReferenceTransform(this.retreatSkill.frames[0], live.rootPosWorld, live.rootQuatXyzwWorld);
    const worldFrames = this.retreatSkill.frames.map(frame => transformTeacherReference(frame, transform));
    const admission = admitClip({ sweep: this.sweeps.get(this.retreatSkill), sourceFrames: this.retreatSkill.sourceFrames, worldFrames,
      obstacles, reserve: this.limits.retreatReserveM, flags: this.flags });
    decision.retreatAdmission = { admitted: admission.admitted, mode: admission.mode, marginM: admission.marginM, deficitM: admission.deficitM ?? null,
      obstacleIndex: admission.obstacleIndex, geometryReason: admission.geometry?.reason ?? null };
    if (!admission.admitted) return refuse('retreat_not_admissible');
    segment.retreats++;
    const controller = new ApproachRecoveryCycle({ coordinator: this, kind, parent, decision, retreatSkill: this.retreatSkill, turnSkills: this.turnSkills,
      sweeps: this.sweeps, readObstacles: this.readObstacles, flags: this.flags, limits: this.limits, objectBodyName,
      objectPointsLocal: parent.skill.objectPointsLocal, requestedGoalWorld: parent.requestedGoalWorld,
      turnTarget: kind === 'facing' ? { kind: 'pickup_yaw', yawRad: metrics.pickupYawRad ?? yaw(q) } : { kind: 'approach_bearing', goalWorld: metrics.pickupRootXY ? [...metrics.pickupRootXY, 0] : null },
      entryMetrics: metrics, admission, worldFrames, episode: context.episode, requestId: context.requestId, startControl: context.physicalControl });
    Object.assign(decision, { supported: true, reason: null, segmentRetreats: segment.retreats });
    this.decisions.push(decision); this.cycles.push(controller); segment.cycles.push(controller);
    if (this.cycles.length > 16) this.cycles.shift();
    return { supported: true, reason: null, controller, decision };
  }

  /** After a cycle reports completion: measure progress, decide the retry. */
  finish(controller, step, live, obstacles = this.readObstacles()) {
    if (!(controller instanceof ApproachRecoveryCycle)) throw new Error('A recovery cycle is required');
    const key = this.segmentKey({ episode: controller.episode, requestId: controller.requestId }, controller.parent), segment = this.segmentState(key);
    const before = controller.entryMetrics, after = this.metrics(controller.parent, live, obstacles, controller.kind);
    const progress = {
      clearanceGainM: after.clearanceM - before.clearanceM,
      headingErrorReductionRad: Math.abs(before.headingErrorRad ?? 0) - Math.abs(after.headingErrorRad ?? 0),
      routeImproved: (before.routeSupported === false && after.routeSupported === true) || (before.directBlocked === true && after.directBlocked === false),
      pickupDistanceChangeM: (after.pickupDistanceM ?? 0) - (before.pickupDistanceM ?? 0),
    };
    progress.made = progress.clearanceGainM >= this.limits.progressClearanceM || progress.headingErrorReductionRad >= this.limits.progressHeadingRad || progress.routeImproved;
    const retry = step?.completionReason === 'finished';
    if (retry && !progress.made) segment.noProgress = true;
    const completion = { episode: controller.episode, requestId: controller.requestId, segmentKey: key, kind: controller.kind,
      completionReason: step?.completionReason ?? controller.completionReason, retry, controls: controller.totalControls, before, after, progress };
    this.completions.push(completion); if (this.completions.length > 16) this.completions.shift();
    return { retry, reason: retry ? null : completion.completionReason, progress, completion };
  }

  ownsController(controller) { return controller instanceof ApproachRecoveryCycle && controller.coordinator === this; }
  retreatAvailable(controller) {
    const segment = this.segmentState(this.segmentKey({ episode: controller.episode, requestId: controller.requestId }, controller.parent));
    return segment.retreats < this.limits.maxRetreatsPerSegment;
  }
  countRetreat(controller) {
    const segment = this.segmentState(this.segmentKey({ episode: controller.episode, requestId: controller.requestId }, controller.parent));
    segment.retreats++; return segment.retreats;
  }
  review() {
    return { flags: { ...this.flags }, limits: { ...this.limits }, generation: this.generation,
      segments: Object.fromEntries([...this.segments].map(([key, s]) => [key, { retreats: s.retreats, noProgress: s.noProgress, cycles: s.cycles.length }])),
      decisions: copy(this.decisions), completions: copy(this.completions), memory: this.memory.review(),
      cycles: this.cycles.map(cycle => cycle.review()) };
  }
  snapshot() {
    const cycle = this.cycles.at(-1);
    return { generation: this.generation, decisions: this.decisions.length, completions: this.completions.length,
      lastDecision: copy(this.decisions.at(-1) ?? null),
      lastCycle: cycle ? { kind: cycle.kind, phase: cycle.phase, completionReason: cycle.completionReason, controls: cycle.totalControls,
        stage: cycle.stage?.name ?? null, admissionMode: cycle.stage?.admission?.mode ?? null } : null };
  }
}
