// A short sequence of measured carry skills. Planning retains absolute targets;
// orchestration advances only when the caller advances real simulation physics.
import { CarryGoalController } from './teacher_carry_controller.js';
import { quatRotateOne } from './math.js';
import { segmentEntersRectangle } from './navigation_planner.js';
import { QuietHoldMonitor, quietHoldComplete, validateQuietOptions, measuredObjectTiltDeg } from './quiet_ending.js';

function finitePosition(value) {
  return value && value.length === 3 && Array.from(value).every(Number.isFinite);
}

// Task accuracy is separate from each reference's permitted warp and the
// live-range residual allowed before another complete source can start.
export const CARRY_PLACEMENT_TOLERANCE_M = 0.10;
const withinPlacementTolerance = distance => distance <= CARRY_PLACEMENT_TOLERANCE_M + 1e-12;
const completeExit = exit => exit?.completionReason === 'finished'
  && Number.isInteger(exit?.recordClock?.totalControls) && exit.recordClock.totalControls > 0
  && exit.recordClock.totalControls === exit.recordClock.expectedTotalControls;

/** Divide a straight planar goal into at most three supported carry lengths.
 * This checks reference travel only, not obstacles or the approach heading.
 * Unsupported gaps remain explicit; the goal and segment lengths are not clipped.
 */
export function planCarrySegments(objectPosition, goalWorld, skill,
  { maxSegments = 3, maxCorrection = 0.25, preferredDistance = 1 } = {}) {
  if (!finitePosition(objectPosition) || !finitePosition(goalWorld)
      || !Number.isInteger(maxSegments) || maxSegments < 1 || maxSegments > 3
      || !Number.isFinite(maxCorrection) || maxCorrection <= 0 || maxCorrection > 0.25
      || !Number.isFinite(preferredDistance) || preferredDistance <= 0
      || !skill || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 2
      || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames) {
    throw new Error('Finite goals, a carry skill, and one to three bounded segments are required');
  }
  const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
  if (first.length !== 747 || last.length !== 747
      || ![first[71], first[72], last[71], last[72]].every(Number.isFinite)) throw new Error('Expected full747 carry reference frames');
  const sourceTravelM = Math.hypot(last[71] - first[71], last[72] - first[72]);
  if (sourceTravelM <= maxCorrection) throw new Error('Carry reference must travel farther than its correction range');
  const minSegmentDistanceM = sourceTravelM - maxCorrection;
  const maxSegmentDistanceM = sourceTravelM + maxCorrection;
  const dx = goalWorld[0] - objectPosition[0], dy = goalWorld[1] - objectPosition[1];
  const distanceM = Math.hypot(dx, dy);
  const candidates = Array.from({ length: maxSegments }, (_, index) => index + 1)
    .filter(count => distanceM / count >= minSegmentDistanceM && distanceM / count <= maxSegmentDistanceM)
    .sort((a, b) => Math.abs(distanceM / a - preferredDistance) - Math.abs(distanceM / b - preferredDistance) || a - b);
  const count = candidates[0] || 0;
  const goals = Array.from({ length: count }, (_, index) => Object.freeze(index + 1 === count
    ? Array.from(goalWorld)
    : [objectPosition[0] + dx * (index + 1) / count, objectPosition[1] + dy * (index + 1) / count, goalWorld[2]]));
  return Object.freeze({ supported: count > 0, reason: count ? null : 'unsupported_distance',
    requestedGoalWorld: Object.freeze(Array.from(goalWorld)), initialObjectPositionWorld: Object.freeze(Array.from(objectPosition)),
    goals: Object.freeze(goals), distanceM, sourceTravelM, minSegmentDistanceM, maxSegmentDistanceM,
    supportedDistanceIntervalsM: Object.freeze(Array.from({ length: maxSegments }, (_, index) =>
      Object.freeze([minSegmentDistanceM * (index + 1), maxSegmentDistanceM * (index + 1)]))) });
}

export class CarryGoalSequenceController {
  constructor(skill, goalWorld, { maxSegments = 3, settlingSteps = 180, preferredDistance = 1,
    maxLiveGoalResidualM = 0.05, alternativeSkills = [], checkEntryReference = null,
    requireSegmentExit = false, quietSettling = null, ...carryOptions } = {}) {
    if (!finitePosition(goalWorld) || !Number.isInteger(settlingSteps) || settlingSteps < 1
        || !Number.isFinite(maxLiveGoalResidualM) || maxLiveGoalResidualM < 0 || maxLiveGoalResidualM > 0.05) {
      throw new Error('A finite goal, positive settling duration and live residual tolerance within 0–0.05m are required');
    }
    this.rawSkill = skill;
    this.requestedGoalWorld = Object.freeze(Array.from(goalWorld));
    this.carryOptions = { ...carryOptions };
    this.planOptions = { maxSegments, preferredDistance, maxCorrection: carryOptions.maxCorrection ?? 0.25 };
    this.settlingMinSteps = settlingSteps;
    // Quiet-terminated settling (WS-G): `settlingSteps` becomes the tracked
    // fixed minimum and the remainder up to the original maximum runs as the
    // separate 'settling_quiet' phase, ending once the live measurement has
    // been quiet for `window` consecutive controls. Placement is measured
    // exactly as before, after the last settling control.
    this.quietSettling = validateQuietOptions(quietSettling, 'Quiet settling');
    this.settlingMaxSteps = settlingSteps;
    if (this.quietSettling) {
      const minimum = this.quietSettling.minControls ?? 60;
      if (!Number.isInteger(minimum) || minimum < 1 || minimum > settlingSteps)
        throw new Error('Quiet settling minimum must be a whole count within the settling maximum');
      this.settlingMinSteps = minimum;
      // `limits` undefined -> the monitor's frozen QUIET_ENDING_LIMITS default (largebox, unchanged); a profile may supply
      // per-object limits (B5: + tippedObjectTiltDeg/objectUpAxisLocal, validated by validateQuietOptions).
      this.quietMonitor = new QuietHoldMonitor({ window: this.quietSettling.window, limits: this.quietSettling.limits });
    } else this.quietMonitor = null;
    this.maxLiveGoalResidualM = maxLiveGoalResidualM;
    if (checkEntryReference !== null && typeof checkEntryReference !== 'function') {
      throw new Error('Carry entry reference checking must be a synchronous function');
    }
    this.checkEntryReference = checkEntryReference;
    if (typeof requireSegmentExit !== 'boolean') throw new Error('Segment exit option must be boolean');
    this.requireSegmentExit = requireSegmentExit;
    if (!Array.isArray(alternativeSkills) || alternativeSkills.length > 3) throw new Error('At most three alternative carry references are supported');
    this.references = [{ skill, options: { ...carryOptions } }, ...alternativeSkills.map(entry => {
      if (!entry.skill || entry.skill.objectBodyName !== skill.objectBodyName) throw new Error('Alternative carries must use the same scene object');
      return { skill: entry.skill, options: { ...carryOptions, warpStartFrame: entry.warpStartFrame, warpEndFrame: entry.warpEndFrame } };
    })];
    for (const entry of this.references) {
      new CarryGoalController(entry.skill, goalWorld, entry.options);
      entry.range = planCarrySegments([0, 0, 0], [1, 0, 0], entry.skill, this.planOptions);
    }
    // Validate the complete skill and shared carry options before an interaction.
    this.child = new CarryGoalController(skill, goalWorld, this.carryOptions);
    this.reset();
  }

  get skill() { return this.child.skill; }
  get referencePlan() { return this.child.referencePlan; }
  get worldFrames() { return this.child.worldFrames; }
  get approachGoalWorld() { return this.phase === 'approach' ? this.child.approachGoalWorld : null; }
  get referenceIndex() { return this.child.referenceIndex; }
  get maxReferenceStartDistance() { return this.child.maxReferenceStartDistance; }
  get maxFacingError() { return this.child.maxFacingError; }
  get facingErrorRad() { return this.child.facingErrorRad; }
  get arrivalRadius() { return this.child.arrivalRadius; }
  get finishRequested() { return this._cancelRequested; }
  get outcome() { return this._lastOutcome ? { ...this._lastOutcome } : null; }
  get descentSagHoldSnapshot() { return this.child.descentSagHoldSnapshot; }

  reset() {
    this.child.reset(); this.phase = 'inactive'; this.plan = null;
    this.segmentIndex = 0; this.settlingCount = 0; this.segmentResults = [];
    this.settlingOutcome = null;
    this.quietMonitor?.reset(); this.quietSampledCount = -1; this.settlingQuietSummary = null;
    this._lastOutcome = null;
    this.liveGoalResidualM = null;
    this.activeReferenceIndex = 0; this.triedReferences = new Set(); this.referenceAttempts = [];
    this.completionReason = null; this._completed = false; this._cancelRequested = false;
    this.segmentExitResults = []; this.exitedSegmentIndex = null;
    this.segmentPreparations = [];
    this.placementChecks = []; this.placementStatus = null;
  }

  start(proprio) {
    if (['approach', 'teacher', 'settling', 'settling_quiet', 'awaiting_exit'].includes(this.phase)) throw new Error('A carry sequence is already active');
    if (proprio.objectBodyName !== this.rawSkill.objectBodyName) throw new Error(`This skill requires ${this.rawSkill.objectBodyName}`);
    const plan = planCarrySegments(proprio.objPosWorld, this.requestedGoalWorld, this.rawSkill, this.planOptions);
    this.reset(); this.plan = plan;
    if (!plan.supported) this._complete(plan.reason);
    else this._startSegment(proprio);
  }

  _liveDistanceSupported(proprio) {
    const target = this.plan.goals[this.segmentIndex], object = proprio.objPosWorld;
    if (!finitePosition(object)) throw new Error('Finite live object position is required');
    const distance = Math.hypot(target[0] - object[0], target[1] - object[1]);
    const range = this.references[this.activeReferenceIndex].range;
    this.liveGoalResidualM = Math.max(0, range.minSegmentDistanceM - distance, distance - range.maxSegmentDistanceM);
    return this.liveGoalResidualM <= this.maxLiveGoalResidualM;
  }

  _startSegment(proprio) {
    // Preserve the exact target and the original warp limit. A small placement
    // error may leave an explicit endpoint residual; larger errors stop here.
    if (!this._liveDistanceSupported(proprio)) { this._complete('unsupported_live_distance'); return; }
    const target = this.plan.goals[this.segmentIndex];
    const entry = this.references[this.activeReferenceIndex];
    this.triedReferences = new Set([this.activeReferenceIndex]);
    this.child = new CarryGoalController(entry.skill, target, this._segmentOptions(entry));
    this.child.start(proprio); this.phase = 'approach'; this.settlingCount = 0;
  }

  _segmentOptions(entry) {
    const preparation = this.segmentPreparations.find(row => row.segmentIndex === this.segmentIndex);
    return preparation ? { ...entry.options, initialStanceFrames: preparation.initialStanceFrames } : entry.options;
  }

  /** Prepare a longer initial teacher stance before a student approaches the
   * next grasp. Only an untouched segment after a completed exit can change.
   * The original source and its warp interval remain complete. */
  prepareStudentHandoff(proprio) {
    if (this.phase !== 'approach' || this.child.approachSteps !== 0 || this.referenceIndex !== 0
        || this._cancelRequested || this.exitedSegmentIndex !== this.segmentIndex - 1
        || this.segmentPreparations.some(row => row.segmentIndex === this.segmentIndex)) {
      throw new Error('Student handoff preparation requires an unstarted segment after its completed exit');
    }
    if (proprio.objectBodyName !== this.rawSkill.objectBodyName || !finitePosition(proprio.objPosWorld)) {
      throw new Error('Finite live state of the same placed object is required');
    }
    this.segmentPreparations.push({ segmentIndex: this.segmentIndex, initialStanceFrames: 90 });
    this._startSegment(proprio);
    return this.phase === 'approach';
  }

  _tryAlternative(proprio) {
    const target = this.plan.goals[this.segmentIndex];
    const distance = Math.hypot(target[0] - proprio.objPosWorld[0], target[1] - proprio.objPosWorld[1]);
    const forward = quatRotateOne(proprio.rootQuatXyzwWorld, [1, 0, 0]);
    const heading = Math.atan2(forward[1], forward[0]);
    let best = null;
    for (const [index, entry] of this.references.entries()) {
      if (this.triedReferences.has(index)) continue;
      const residual = Math.max(0, entry.range.minSegmentDistanceM - distance, distance - entry.range.maxSegmentDistanceM);
      if (residual > this.maxLiveGoalResidualM) continue;
      const child = new CarryGoalController(entry.skill, target, this._segmentOptions(entry));
      child.start(proprio);
      if (entry.skill.objectPointsLocal && proprio.objQuatXyzwWorld) {
        const points = entry.skill.objectPointsLocal.map(point => {
          const rotated = quatRotateOne(proprio.objQuatXyzwWorld, point);
          return [rotated[0] + proprio.objPosWorld[0], rotated[1] + proprio.objPosWorld[1]];
        });
        const bounds = { minX: Math.min(...points.map(point => point[0])), maxX: Math.max(...points.map(point => point[0])),
          minY: Math.min(...points.map(point => point[1])), maxY: Math.max(...points.map(point => point[1])) };
        // Switching reference can move the approach to the opposite side.
        // A straight approach through the physical box needs a separate route.
        if (segmentEntersRectangle(proprio.rootPosWorld, child.approachGoalWorld, bounds)) continue;
      }
      const desired = quatRotateOne(child.referencePlan.first.slice(3, 7), [1, 0, 0]);
      const angle = Math.atan2(desired[1], desired[0]) - heading;
      const error = Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle)));
      if (error <= child.maxFacingError && (!best || error < best.error)) best = { index, child, error };
    }
    if (!best) return false;
    this.referenceAttempts.push({ segmentIndex: this.segmentIndex, referenceIndex: this.activeReferenceIndex,
      completionReason: 'needs_facing', facingErrorRad: this.child.facingErrorRad,
      nextReferenceIndex: best.index, requestedGoalWorld: Array.from(target) });
    this.activeReferenceIndex = best.index; this.triedReferences.add(best.index);
    this.child = best.child; this.phase = 'approach';
    this._liveDistanceSupported(proprio);
    return true;
  }

  _complete(reason) {
    this.phase = 'complete'; this.completionReason = reason; this._completed = true;
  }

  /** Optional (B5) object-tilt placement check. null when the child's requirements carry no tilt limit (largebox);
   * else {objectTiltDeg, objectTipped, tippedObjectTiltDeg} with an unmeasurable tilt counted as tipped (fail closed). */
  _measureObjectTilt(proprio) {
    const tilt = this.child.objectTiltRequirement;
    if (!tilt) return null;
    const objectTiltDeg = measuredObjectTiltDeg(proprio.objQuatXyzwWorld, tilt.objectUpAxisLocal);
    return { objectTiltDeg, objectTipped: !(objectTiltDeg < tilt.tippedObjectTiltDeg), tippedObjectTiltDeg: tilt.tippedObjectTiltDeg };
  }

  _measurePlacement(proprio, { afterExit = false } = {}) {
    const position = proprio.objPosWorld;
    const remainingDistanceM = Math.hypot(position[0] - this.requestedGoalWorld[0], position[1] - this.requestedGoalWorld[1]);
    const tilt = this._measureObjectTilt(proprio);
    // A tipped object (on its side: height passes) is NOT set down when a tilt requirement is active.
    const setDown = position[2] <= this.child.outcomeRequirements.maxFinalObjectHeightM && !(tilt?.objectTipped === true);
    const status = Object.freeze({ segmentIndex: this.segmentIndex, checkedAfterSettling: true, checkedAfterExit: afterExit,
      settlingControls: this.settlingCount, completeSourceControls: this.child.skill.sourceFrames,
      settlingFixedControls: this._settlingRequiredSteps(), settlingMaxControls: this.settlingMaxSteps,
      settlingQuietControls: Math.max(0, this.settlingCount - this.settlingMinSteps),
      quietSettling: this.settlingQuietSummary ? structuredClone(this.settlingQuietSummary) : null,
      requestedGoalWorld: this.requestedGoalWorld, measuredObjectPositionWorld: Object.freeze(Array.from(position)),
      toleranceM: CARRY_PLACEMENT_TOLERANCE_M, remainingDistanceM, setDown,
      goalReached: setDown && withinPlacementTolerance(remainingDistanceM),
      skippedSegments: setDown && withinPlacementTolerance(remainingDistanceM)
        ? this.plan.goals.length - this.segmentIndex - 1 : 0,
      ...(tilt ?? {}) });
    this.placementStatus = status;
    this.placementChecks.push(status);
    return status;
  }

  /** A fresh planning input after measured placement and a complete physical
   * exit. The caller can feed it to its current skill library; this method
   * neither replaces the accepted plan nor applies a target to live state.
   * An external final exit may be supplied because that lifecycle belongs to
   * the caller. Recorded inter-segment exits already provide the same proof. */
  remainingGoalRequest(proprio, { exit = null } = {}) {
    if (proprio.objectBodyName !== this.rawSkill.objectBodyName || !finitePosition(proprio.objPosWorld))
      throw new Error('The same placed object and its current position are required');
    if (exit !== null && !completeExit(exit)) throw new Error('A complete physical exit is required for remaining-goal planning');
    const settled = this.placementStatus;
    const exitCompleted = Boolean(settled && (this.exitedSegmentIndex === settled.segmentIndex || completeExit(exit)));
    const position = Array.from(proprio.objPosWorld);
    const remainingDistanceM = Math.hypot(position[0] - this.requestedGoalWorld[0], position[1] - this.requestedGoalWorld[1]);
    const root = proprio.rootPosWorld, q = proprio.rootQuatXyzwWorld;
    const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore
      : q?.length === 4 && Array.from(q).every(Number.isFinite) ? 1 - 2 * (q[0] ** 2 + q[1] ** 2) : NaN;
    const requirement = this.child.outcomeRequirements;
    const stableSetdown = finitePosition(root) && root[2] >= requirement.minRootHeightM
      && Number.isFinite(upright) && upright >= requirement.minUpright
      && position[2] <= requirement.maxFinalObjectHeightM
      && !(this._measureObjectTilt(proprio)?.objectTipped === true);
    const cancelled = this._cancelRequested || this.completionReason === 'cancelled';
    const eligibleCompletion = this.phase === 'complete'
      && ['finished', 'placement_missed', 'unsupported_live_distance'].includes(this.completionReason);
    const ready = Boolean(settled?.setDown && exitCompleted && stableSetdown && !cancelled && eligibleCompletion);
    const goalReached = ready && withinPlacementTolerance(remainingDistanceM);
    const reason = cancelled ? 'cancelled' : !settled ? 'placement_not_checked'
      : !exitCompleted ? 'exit_required' : !stableSetdown ? 'placement_not_stable'
      : !eligibleCompletion ? 'sequence_not_complete' : goalReached ? 'within_placement_tolerance' : 'remaining_goal';
    return Object.freeze({ ready, needsPlan: ready && !goalReached, goalReached, reason,
      objectBodyName: proprio.objectBodyName, objectPositionWorld: Object.freeze(position),
      requestedGoalWorld: this.requestedGoalWorld, remainingDistanceM, toleranceM: CARRY_PLACEMENT_TOLERANCE_M,
      completedSegments: this.segmentResults.filter(segment => segment.completionReason === 'finished').length,
      completedCandidateCounts: Object.freeze(this.segmentResults.reduce((counts, segment) => {
        if (segment.completionReason === 'finished' && segment.candidateId)
          counts[segment.candidateId] = (counts[segment.candidateId] ?? 0) + 1;
        return counts;
      }, {})),
      previousSegmentIndex: settled?.segmentIndex ?? null, exitCompleted });
  }

  requestCancel() {
    if (this.phase === 'awaiting_exit') { this._cancelRequested = true; return; }
    if (this.phase === 'teacher') { this._cancelRequested = true; this.child.requestCancel(); }
    else if (['approach', 'settling', 'settling_quiet'].includes(this.phase)) this._complete('cancelled');
    else if (this.phase === 'complete' && this.completionReason === 'needs_facing') {
      this._cancelRequested = true; this._complete('cancelled');
    }
  }

  /** Retry the same unstarted segment after a caller has completed a preparation
   * such as a teacher turn. Preserve every original destination and prior carry.
   * The caller owns preparation limits and must verify its measured outcome.
   * Two entry states are supported: a completed `needs_facing` segment (the
   * original facing-turn retry) and an approach whose recorded walk was refused
   * before any source control (referenceIndex 0, phase still `approach`) — the
   * approach restarts from the measured live pose with a fresh child. */
  retryCurrentSegment(proprio) {
    const failed = this.segmentResults.at(-1);
    const facingRetry = this.phase === 'complete' && this.completionReason === 'needs_facing'
      && failed && failed.completionReason === 'needs_facing' && failed.teacherFrames === 0
      && failed.segmentIndex === this.segmentIndex && failed.referenceIndex === this.activeReferenceIndex;
    const approachRestart = this.phase === 'approach' && this.referenceIndex === 0
      && this.child?.phase === 'approach' && this.child.worldFrames === null;
    if (this._cancelRequested || !(facingRetry || approachRestart)) {
      throw new Error('Only an unstarted needs_facing segment or an unstarted refused approach may retry after preparation');
    }
    if (proprio.objectBodyName !== this.rawSkill.objectBodyName || !finitePosition(proprio.objPosWorld)
        || !finitePosition(proprio.rootPosWorld) || !proprio.rootQuatXyzwWorld
        || proprio.rootQuatXyzwWorld.length !== 4 || !Array.from(proprio.rootQuatXyzwWorld).every(Number.isFinite)) {
      throw new Error('Finite live state for the same scene object is required');
    }
    if (facingRetry) {
      this.referenceAttempts.push({ ...failed, requestedGoalWorld: [...failed.requestedGoalWorld],
        facingErrorRad: this.child.facingErrorRad, preparationCompleted: true });
      this.segmentResults.pop();
      this.completionReason = null; this._completed = false;
    } else {
      this.referenceAttempts.push({ segmentIndex: this.segmentIndex, referenceIndex: this.activeReferenceIndex,
        completionReason: 'approach_restart', approachSteps: this.child.approachSteps,
        requestedGoalWorld: Array.from(this.plan.goals[this.segmentIndex]), preparationCompleted: true });
    }
    this._startSegment(proprio);
    return this.phase === 'approach';
  }

  /** Retire an unstarted approach with an owned reason (route or clearance
   * refusal measured by the caller). The sequence completes; nothing is warped. */
  refuseApproach(reason) {
    if (this.phase !== 'approach' || this.referenceIndex !== 0 || typeof reason !== 'string' || !reason) {
      throw new Error('Only an unstarted approach may be refused with a reason');
    }
    this.segmentResults.push({ segmentIndex: this.segmentIndex, requestedGoalWorld: Array.from(this.plan.goals[this.segmentIndex]),
      referenceIndex: this.activeReferenceIndex, teacherFrames: 0, completionReason: reason, outcome: null });
    this._complete(reason);
  }

  /** The caller could not complete the physical exit it owns (its preview was
   * refused). Complete the sequence with that reason instead of waiting. */
  abandonAwaitingExit(reason) {
    if (this.phase !== 'awaiting_exit' || typeof reason !== 'string' || !reason) throw new Error('Only an awaited exit may be abandoned');
    this._complete(reason);
  }

  /** Resume only after the caller has physically completed the intervening
   * box exit. The original destinations and completed carry records survive. */
  resumeAfterSegmentExit(proprio, exit) {
    if (this.phase !== 'awaiting_exit' || !this.requireSegmentExit
        || this.segmentIndex + 1 >= this.plan.goals.length
        || !completeExit(exit)) {
      throw new Error('A complete physical exit is required before the next carry segment');
    }
    if (proprio.objectBodyName !== this.rawSkill.objectBodyName || !finitePosition(proprio.objPosWorld)) {
      throw new Error('The same placed object and its live position are required');
    }
    this.exitedSegmentIndex = this.segmentIndex;
    this.segmentExitResults.push({ segmentIndex: this.segmentIndex,
      requestedGoalWorld: Array.from(this.plan.goals[this.segmentIndex]),
      completionReason: exit.completionReason, recordClock: structuredClone(exit.recordClock),
      outcome: exit.outcome ? structuredClone(exit.outcome) : null });
    if (this._cancelRequested) this._complete('cancelled');
    else if (this._measurePlacement(proprio, { afterExit: true }).goalReached) this._complete('finished');
    else if (!this.placementStatus.setDown) this._complete(this.placementStatus.objectTipped === true ? 'object_tipped' : 'failed_setdown');
    else { this.segmentIndex++; this._startSegment(proprio); }
  }

  step(proprio) {
    if (!finitePosition(proprio.objPosWorld)) throw new Error('Finite live object position is required');
    let segmentFinished = false, segmentResult = null, childResult = null, referenceChanged = false;
    if (this.phase === 'settling' || this.phase === 'settling_quiet') {
      const root = proprio.rootPosWorld, q = proprio.rootQuatXyzwWorld;
      if (!finitePosition(root) || !q || q.length !== 4 || !Array.from(q).every(Number.isFinite)) throw new Error('Finite settling physical state is required');
      const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
      this.settlingOutcome ||= { minRootHeightM: root[2], minUpright: upright };
      this.settlingOutcome.minRootHeightM = Math.min(this.settlingOutcome.minRootHeightM, root[2]);
      this.settlingOutcome.minUpright = Math.min(this.settlingOutcome.minUpright, upright);
      const requirement = this.child.outcomeRequirements;
      if (root[2] < requirement.minRootHeightM || upright < requirement.minUpright) this._complete('lost_balance');
      else if (this.quietSettling) this._observeQuietSettling();
    }
    // The tracked fixed minimum has run. Continue as 'settling_quiet' until the
    // hold has been quiet for a full window or the original maximum is reached.
    if (this.phase === 'settling' && this.settlingCount >= this.settlingMinSteps && this._quietSettlingAllowed()
        && !this._settlingComplete()) this.phase = 'settling_quiet';
    if ((this.phase === 'settling' && this.settlingCount >= this._settlingRequiredSteps())
        || (this.phase === 'settling_quiet' && this._settlingComplete())) {
      const placement = this._measurePlacement(proprio);
      if (!placement.setDown) this._complete(placement.objectTipped === true ? 'object_tipped' : 'failed_setdown');
      else if (placement.goalReached) this._complete('finished');
      else if (this.segmentIndex + 1 === this.plan.goals.length) this._complete('placement_missed');
      else if (this.requireSegmentExit) this.phase = 'awaiting_exit';
      else { this.segmentIndex++; this._startSegment(proprio); }
    }
    if (this.phase === 'approach' || this.phase === 'teacher') {
      const wasTeacher = this.phase === 'teacher';
      childResult = this.child.step(proprio); this.phase = childResult.phase;
      if (childResult.justCompleted && childResult.completionReason === 'needs_facing' && this._tryAlternative(proprio)) {
        referenceChanged = true;
        childResult = this.child.step(proprio); this.phase = childResult.phase;
      }
      if (childResult.justEnteredTeacher && !this._liveDistanceSupported(proprio)) {
        // Approach contacts may move the object. Check again before issuing
        // even the first teacher action, while retaining the requested goal.
        childResult.justEnteredTeacher = false; childResult.outcome = null;
        this._complete('unsupported_live_distance');
      }
      if (childResult.justEnteredTeacher && this.checkEntryReference) {
        // The box may have moved since the original click. Check the complete
        // actual entry plan before applying any of its teacher actions.
        const check = this.checkEntryReference({ referenceFrames: this.worldFrames,
          sourceFrames: this.skill.sourceFrames, objectBodyName: this.skill.objectBodyName,
          requestedGoalWorld: this.plan.goals[this.segmentIndex], segmentIndex: this.segmentIndex });
        if (check?.supported !== true) childResult = { ...childResult, justEnteredTeacher: false,
          justCompleted: true, completionReason: check?.reason ?? 'carry_reference_clearance',
          referenceFrames: null, outcome: null };
      }
      if (childResult.justCompleted) {
        segmentResult = { segmentIndex: this.segmentIndex, requestedGoalWorld: Array.from(this.plan.goals[this.segmentIndex]),
          referenceIndex: this.activeReferenceIndex, teacherFrames: wasTeacher ? this.child.skill.sourceFrames : 0,
          completionReason: childResult.completionReason, outcome: childResult.outcome,
          referenceGoalWorld: childResult.referenceGoalWorld, referenceGoalResidualM: childResult.referenceGoalResidualM,
          carryDescentSagHold: this.child.descentSagHoldSnapshot };
        this.segmentResults.push(segmentResult);
        segmentFinished = wasTeacher;
        if (childResult.completionReason !== 'finished') this._complete(childResult.completionReason);
        else if (this._cancelRequested) this._complete('cancelled');
        else { this.phase = 'settling'; this.settlingCount = 0; }
      }
    }
    const endpoint = this.referencePlan?.referenceGoalWorld || null;
    const remaining = Math.hypot(proprio.objPosWorld[0] - this.requestedGoalWorld[0], proprio.objPosWorld[1] - this.requestedGoalWorld[1]);
    const lastOutcome = childResult?.outcome || this.segmentResults.at(-1)?.outcome || null;
    const result = {
      phase: this.phase, mode: this.phase === 'teacher' ? 'teacher' : this.phase === 'awaiting_exit' ? 'none' : 'student',
      awaitingSegmentExit: this.phase === 'awaiting_exit',
      approachGoalWorld: this.approachGoalWorld,
      referenceFrames: this.phase === 'teacher' ? childResult.referenceFrames : null,
      referenceIndex: this.referenceIndex, justEnteredTeacher: childResult?.justEnteredTeacher || false,
      justCompleted: this._completed, completionReason: this.completionReason,
      segmentFinished, segmentResult, segmentIndex: this.segmentIndex, segmentCount: this.plan?.goals.length || 0,
      referenceChanged, activeReferenceIndex: this.activeReferenceIndex,
      segmentGoalWorld: this.plan?.goals[this.segmentIndex] || null,
      completedSegments: this.segmentResults.filter(segment => segment.completionReason === 'finished').length,
      settlingStepsRemaining: this.phase === 'settling' || this.phase === 'settling_quiet' ? this.settlingMaxSteps - this.settlingCount : 0,
      settlingQuiet: this.settlingQuietSummary ? { ...this.settlingQuietSummary } : null,
      requestedGoalWorld: this.requestedGoalWorld, referenceGoalWorld: endpoint,
      referenceGoalResidualM: this.referencePlan?.remainingDistance ?? null,
      liveGoalResidualM: this.liveGoalResidualM, maxLiveGoalResidualM: this.maxLiveGoalResidualM,
      remainingDistanceM: remaining, facingErrorRad: this.child.facingErrorRad,
      carryDescentSagHold: this.child.descentSagHoldSnapshot,
      placementToleranceM: CARRY_PLACEMENT_TOLERANCE_M, placementStatus: this.placementStatus,
      goalReached: this.completionReason === 'finished' && Boolean(this.placementStatus?.goalReached)
        && withinPlacementTolerance(remaining) && proprio.objPosWorld[2] <= this.child.outcomeRequirements.maxFinalObjectHeightM,
      needsRemainingGoalPlan: this.phase === 'complete' && !this._cancelRequested
        && ['finished', 'placement_missed', 'unsupported_live_distance'].includes(this.completionReason)
        && Boolean(this.placementStatus?.setDown) && !withinPlacementTolerance(remaining),
      requiresFinalExit: this.phase === 'complete' && ['finished', 'placement_missed', 'cancelled'].includes(this.completionReason)
        && this.referenceIndex >= this.skill.sourceFrames && this.exitedSegmentIndex !== this.segmentIndex,
      settlingOutcome: this.settlingOutcome ? { ...this.settlingOutcome } : null,
      outcome: lastOutcome ? { ...lastOutcome, finalObjectPositionWorld: Array.from(proprio.objPosWorld), requestedGoalErrorM: remaining,
        referenceGoalErrorM: endpoint ? Math.hypot(proprio.objPosWorld[0] - endpoint[0], proprio.objPosWorld[1] - endpoint[1]) : null } : null,
    };
    this._lastOutcome = result.outcome;
    this._completed = false;
    return result;
  }

  /** One live quiet sample per settling control (the sample describes the
   * state after the previous control). Never steps physics. */
  _observeQuietSettling() {
    if (this.quietSampledCount === this.settlingCount) return;
    this.quietSampledCount = this.settlingCount;
    this.quietMonitor.observe(this.quietSettling.measure());
    this.settlingQuietSummary = { ...this.quietMonitor.summary(), minControls: this.settlingMinSteps,
      maxControls: this.settlingMaxSteps, controls: this.settlingCount,
      terminatedQuiet: this.quietMonitor.satisfied && this.settlingCount < this.settlingMaxSteps };
  }

  _settlingComplete() {
    if (!this.quietSettling) return this.settlingCount >= this.settlingMinSteps;
    return quietHoldComplete({ controls: this.settlingCount, minControls: this.settlingMinSteps,
      maxControls: this.settlingMaxSteps, satisfied: this.quietMonitor.satisfied });
  }

  /** Call once after actual physics, including each student settling step. */
  /** WS-G: quiet-terminated settling only on the LAST segment. Intermediate segments run the
   * full fixed settling — H083 evidence (int5/int6 arms): any shortened intermediate hold moved the
   * next pickup's approach start and it was refused on sweep clearance. */
  _quietSettlingAllowed() { return Boolean(this.quietSettling) && (!this.plan || this.segmentIndex === this.plan.goals.length - 1); }
  _settlingRequiredSteps() { return this.quietSettling && !this._quietSettlingAllowed() ? this.settlingMaxSteps : this.settlingMinSteps; }
  /** Exposed to the evaluator (native progress `settlingSteps`): the fixed settling length the current segment actually tracks. */
  get settlingSteps() { return this._settlingRequiredSteps(); }

  advance(options) {
    if (this.phase === 'settling' || this.phase === 'settling_quiet') this.settlingCount++;
    else if (this.phase === 'approach' || this.phase === 'teacher') this.child.advance(options);
  }

  /** WS-C externally owned held source clock: delegated to the active child carry controller (teacher phase only). */
  holdReferenceClock(reason, fields) {
    if (this.phase !== 'teacher') throw new Error('Reference clock can only be held while the teacher owns the carry');
    return this.child.holdReferenceClock(reason, fields);
  }
  get externalHoldSnapshot() { return this.child?.externalHoldSnapshot ?? null; }
}
