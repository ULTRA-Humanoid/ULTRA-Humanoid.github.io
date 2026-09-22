// Private recorded-step experiment. This controller owns reference clocks only;
// the caller retains physics, policy actions, previous DOFs and body history.
import { transformTeacherReference } from './teacher_reference.js';
import { turnReferenceTransform } from './teacher_turn_controller.js';
import { yawQuat } from './math.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function finite(values, count, label) {
  if (!values || values.length !== count || !Array.from(values).every(Number.isFinite)) {
    throw new Error(`${label} requires ${count} finite values`);
  }
}
function point(value, label) {
  if (!value || ![2, 3].includes(value.length)) throw new Error(`${label} requires finite XY or XYZ`);
  finite(value, value.length, label);
  return [value[0], value[1], value.length === 3 ? value[2] : 0];
}
function routePoints({ waypoints = [], finalGoalWorld } = {}) {
  if (!Array.isArray(waypoints)) throw new Error('Waypoints must be an array');
  return { final: point(finalGoalWorld, 'Final waypoint goal'),
    intermediate: waypoints.map(value => point(value, 'Waypoint')) };
}
function describeSkill(skill, turn) {
  if (!skill || skill.locomotionOnly !== true || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 2
      || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames + 16) {
    throw new Error('A complete explicitly masked locomotion skill is required');
  }
  for (const frame of skill.frames) finite(frame, 747, 'Locomotion reference');
  const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
  const travelM = distance(first, last), sourceYawRad = yaw(first.slice(3, 7));
  const yawChangeRad = wrap(yaw(last.slice(3, 7)) - sourceYawRad);
  if (turn ? Math.abs(yawChangeRad) < .1 : travelM < .1) throw new Error('Locomotion skill has no usable recorded motion');
  const measuredTravelDirectionRad = Math.atan2(last[1] - first[1], last[0] - first[0]);
  const alignmentTravelDirectionRad = skill.periodicTeacher?.continuous
    ? skill.periodicTeacher.alignmentTravelDirectionRad ?? null : null;
  if (alignmentTravelDirectionRad !== null && !Number.isFinite(alignmentTravelDirectionRad)) {
    throw new Error('Continuous periodic alignment direction must be finite');
  }
  return { skill, travelM, sourceYawRad, yawChangeRad,
    measuredTravelDirectionRad,
    travelDirectionRad: alignmentTravelDirectionRad ?? measuredTravelDirectionRad };
}

/** Choose complete records without changing their displacement. Look ahead up
 * to three clips; replan after every actual clip. Prefer fewer clips that reach
 * the waypoint tolerance, then the closest endpoint and the larger first step.
 */
export function planRecordedSteps(remainingM, travelLengthsM, radiusM, maxLookahead = 3) {
  if (!Number.isFinite(remainingM) || remainingM < 0 || !Number.isFinite(radiusM) || radiusM <= 0
      || !Array.isArray(travelLengthsM) || !travelLengthsM.length
      || !travelLengthsM.every(value => Number.isFinite(value) && value >= .1)
      || !Number.isInteger(maxLookahead) || maxLookahead < 1 || maxLookahead > 3) {
    throw new Error('Finite travel lengths, distance, radius and one to three lookahead steps are required');
  }
  if (remainingM <= radiusM) return { indices: [], referenceTravelM: 0, residualM: remainingM };
  let candidates = [{ indices: [], total: 0 }];
  for (let count = 1; count <= maxLookahead; count++) {
    candidates = candidates.flatMap(candidate => travelLengthsM.map((length, index) => ({
      indices: [...candidate.indices, index], total: candidate.total + length,
    }))).filter(candidate => candidate.total <= remainingM + radiusM);
    const feasible = candidates.filter(candidate => Math.abs(candidate.total - remainingM) <= radiusM);
    feasible.sort((a, b) => Math.abs(a.total - remainingM) - Math.abs(b.total - remainingM)
      || travelLengthsM[b.indices[0]] - travelLengthsM[a.indices[0]]);
    if (feasible.length) return { indices: feasible[0].indices, referenceTravelM: feasible[0].total,
      residualM: remainingM - feasible[0].total };
  }
  // A distant waypoint can need more than the bounded lookahead. Take a whole
  // forward step only when it makes progress without overshooting the radius.
  const index = travelLengthsM.map((length, index) => ({ length, index }))
    .filter(item => item.length <= remainingM + radiusM && Math.abs(remainingM - item.length) < remainingM)
    .sort((a, b) => b.length - a.length)[0]?.index;
  return index === undefined ? null : { indices: [index], referenceTravelM: travelLengthsM[index],
    residualM: remainingM - travelLengthsM[index] };
}

export class TeacherWaypointController {
  constructor(stepSkills, { turnSkills = [], settlingSteps = 60, stableSteps = 12, maxSettlingSteps = 180,
    maxSettlingSpeedMps = .25, waypointRadius = .28, arrivalRadius = .25,
    maxHeadingOffsetRad = Math.PI / 3, maxSteps = 8, maxTurns = 4,
    minRootHeightM = .45, minUpright = .5, settlingPolicy = 'student', preferAlignedSteps = false,
    continuousCompletion = false, entryReferenceRootPose = null } = {}) {
    if (!Array.isArray(stepSkills) || !stepSkills.length || !Array.isArray(turnSkills)) throw new Error('Recorded step and turn libraries are required');
    this.steps = stepSkills.map(skill => describeSkill(skill, false));
    this.turns = turnSkills.map(skill => describeSkill(skill, true));
    if (![settlingSteps, stableSteps, maxSettlingSteps, maxSteps, maxTurns].every(Number.isInteger)
        || settlingSteps < 1 || stableSteps < 1 || maxSettlingSteps < Math.max(settlingSteps, stableSteps)
        || maxSteps < 1 || maxTurns < 0
        || ![maxSettlingSpeedMps, waypointRadius, arrivalRadius, maxHeadingOffsetRad, minRootHeightM, minUpright].every(Number.isFinite)
        || maxSettlingSpeedMps <= 0 || waypointRadius <= 0 || arrivalRadius <= 0
        || maxHeadingOffsetRad <= 0 || maxHeadingOffsetRad > Math.PI || minRootHeightM < 0 || minUpright < 0 || minUpright > 1
        || !['student', 'teacher'].includes(settlingPolicy) || typeof preferAlignedSteps !== 'boolean'
        || typeof continuousCompletion !== 'boolean'
        || entryReferenceRootPose !== null && (!entryReferenceRootPose || entryReferenceRootPose.length !== 7
          || !Array.from(entryReferenceRootPose).every(Number.isFinite))) {
      throw new Error('Valid waypoint, heading, settling and outcome requirements are required');
    }
    Object.assign(this, { settlingSteps, stableSteps, maxSettlingSteps, maxSettlingSpeedMps, waypointRadius,
      arrivalRadius, maxHeadingOffsetRad, maxSteps, maxTurns, minRootHeightM, minUpright,
      settlingPolicy, preferAlignedSteps, continuousCompletion,
      entryReferenceRootPose: entryReferenceRootPose === null ? null : Array.from(entryReferenceRootPose) });
    this.reset();
  }

  get skill() { return this._clip?.descriptor.skill ?? null; }
  get sourceFrames() { return this.skill?.sourceFrames ?? 0; }
  get locomotionOnly() { return true; }
  get settlingCount() { return this._settlingCount; }

  reset() {
    this.phase = 'inactive'; this.waypointIndex = 0; this.referenceIndex = 0;
    this.worldFrames = null; this.referencePlan = null; this.completionReason = null; this.outcome = null;
    this.requestedGoalWorld = null; this.waypoints = []; this.requestedWaypoints = [];
    this.finishRequested = false; this._clip = null; this._clipComplete = false;
    this._entered = false; this._completed = false; this._pendingSettling = false;
    this._settlingCount = 0; this._stableCount = 0; this._stepCount = 0; this._turnCount = 0;
    this._pendingRoute = null; this._goalRevision = 0;
    this._cancelAfterSettling = false;
  }

  start(proprio, { waypoints = [], finalGoalWorld } = {}) {
    if (['teacher_step', 'teacher_turn', 'settling'].includes(this.phase)) throw new Error('A waypoint approach is already active');
    const route = routePoints({ waypoints, finalGoalWorld });
    this.reset();
    this._setRoute(route);
    this._observe(proprio); this._plan(proprio);
  }

  _setRoute({ final, intermediate }) {
    this.requestedGoalWorld = [...final];
    this.requestedWaypoints = [...intermediate.map(value => [...value]), [...final]];
    // Match real UI command precision, while retaining original destinations
    // separately for measured errors and caller-facing reporting.
    this.waypoints = this.requestedWaypoints.map(value => Array.from(Float32Array.from(value)));
    this.waypointIndex = 0;
  }

  /** Queue the latest explicit route change. Finish the active full record and
   * its measured settling interval before changing the reference destination.
   * The caller continues owning physics/actions/history throughout. */
  requestRetarget(route) {
    const next = routePoints(route);
    if (!['teacher_step', 'teacher_turn', 'settling'].includes(this.phase) || this.finishRequested) return false;
    if (this._pendingRoute) this._pendingRoute.request.status = 'superseded';
    const request = { revision: ++this._goalRevision, status: 'pending',
      goalWorld: [...next.final], waypoints: next.intermediate.map(value => [...value]),
      acceptedTeacherSteps: this.outcome.teacherSteps,
      acceptedStudentSettlingSteps: this.outcome.studentSettlingSteps,
      acceptedTeacherSettlingSteps: this.outcome.teacherSettlingSteps };
    this.outcome.goalRequests.push(request);
    this._pendingRoute = { ...next, request };
    return true;
  }

  _observe(proprio) {
    finite(proprio.rootPosWorld, 3, 'Live root position');
    finite(proprio.rootQuatXyzwWorld, 4, 'Live root quaternion');
    finite(proprio.rootVelWorld, 3, 'Live root linear velocity');
    const root = proprio.rootPosWorld, q = proprio.rootQuatXyzwWorld;
    const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    this.outcome ||= { initialRootPositionWorld: Array.from(root), minRootHeightM: root[2], minUpright: upright,
      teacherSteps: 0, studentSettlingSteps: 0, teacherSettlingSteps: 0, clips: [],
      initialRequestedGoalWorld: [...this.requestedGoalWorld], goalRequests: [] };
    Object.assign(this.outcome, { minRootHeightM: Math.min(this.outcome.minRootHeightM, root[2]),
      minUpright: Math.min(this.outcome.minUpright, upright), finalRootPositionWorld: Array.from(root),
      finalRootHeadingRad: yaw(q), finalPlanarSpeedMps: Math.hypot(...proprio.rootVelWorld.slice(0, 2)),
      finalGoalErrorM: distance(root, this.requestedGoalWorld), waypointIndex: this.waypointIndex });
  }

  _unsafe() { return this.outcome.minRootHeightM < this.minRootHeightM || this.outcome.minUpright < this.minUpright; }

  _finish(reason) {
    if (this._pendingRoute) {
      Object.assign(this._pendingRoute.request, { status: 'abandoned', reason });
      this._pendingRoute = null;
    }
    this.phase = 'complete'; this.completionReason = reason; this._completed = true;
  }

  _plan(proprio) {
    if (this._unsafe()) return this._finish('lost_balance');
    if (this.finishRequested) return this._finish('cancelled');
    if (this._pendingRoute) {
      const next = this._pendingRoute;
      this._setRoute(next);
      Object.assign(next.request, { status: 'applied', appliedTeacherSteps: this.outcome.teacherSteps,
        appliedStudentSettlingSteps: this.outcome.studentSettlingSteps,
        appliedTeacherSettlingSteps: this.outcome.teacherSettlingSteps,
        appliedRootPositionWorld: Array.from(proprio.rootPosWorld) });
      this._pendingRoute = null;
      this._observe(proprio);
    }
    const root = proprio.rootPosWorld;
    while (this.waypointIndex < this.waypoints.length) {
      const radius = this.waypointIndex === this.waypoints.length - 1 ? this.arrivalRadius : this.waypointRadius;
      if (distance(root, this.waypoints[this.waypointIndex]) > radius) break;
      this.waypointIndex++;
    }
    this.outcome.waypointIndex = this.waypointIndex;
    if (this.waypointIndex === this.waypoints.length) return this._finish('finished');
    if (this._stepCount >= this.maxSteps) return this._finish('step_limit');
    const goal = this.waypoints[this.waypointIndex], radius = this.waypointIndex === this.waypoints.length - 1
      ? this.arrivalRadius : this.waypointRadius;
    const plan = planRecordedSteps(distance(root, goal), this.steps.map(item => item.travelM), radius,
      Math.min(3, this.maxSteps - this._stepCount));
    if (!plan) return this._finish('unsupported_step_distance');
    let descriptor = this.steps[plan.indices[0]];
    const direction = Math.atan2(goal[1] - root[1], goal[0] - root[0]);
    const actualHeading = yaw(proprio.rootQuatXyzwWorld);
    const headingFor = item => wrap(direction + item.sourceYawRad - item.travelDirectionRad);
    let desiredHeading = headingFor(descriptor), headingOffset = wrap(desiredHeading - actualHeading);
    if (this.preferAlignedSteps && Math.abs(headingOffset) > this.maxHeadingOffsetRad) {
      // Opposite-direction recordings can have almost identical travel. Before
      // preparing a turn, apply the same bounded distance planner to complete
      // records supported by the current heading. Replan only after execution.
      const aligned = this.steps.filter(item => Math.abs(wrap(headingFor(item) - actualHeading)) <= this.maxHeadingOffsetRad);
      const alignedPlan = aligned.length ? planRecordedSteps(distance(root, goal), aligned.map(item => item.travelM), radius,
        Math.min(3, this.maxSteps - this._stepCount)) : null;
      if (alignedPlan) {
        const selected = aligned[alignedPlan.indices[0]];
        (this.outcome.alignedStepChoices ??= []).push({ distancePreferredSource: descriptor.skill.name ?? null,
          selectedSource: selected.skill.name ?? null, preferredHeadingOffsetRad: headingOffset,
          selectedHeadingOffsetRad: wrap(headingFor(selected) - actualHeading),
          sourceFrames: selected.skill.sourceFrames, sourceTravelM: selected.travelM,
          rootPositionWorld: Array.from(root), targetWorld: [...goal], waypointIndex: this.waypointIndex,
          teacherSteps: this.outcome.teacherSteps,
          settlingSteps: this.outcome.studentSettlingSteps + this.outcome.teacherSettlingSteps });
        descriptor = selected; desiredHeading = headingFor(descriptor); headingOffset = wrap(desiredHeading - actualHeading);
      }
    }
    if (Math.abs(headingOffset) > this.maxHeadingOffsetRad) {
      const candidate = this.turns.map(item => ({ descriptor: item, residual: Math.abs(wrap(headingOffset - item.yawChangeRad)) }))
        .filter(item => item.residual < Math.abs(headingOffset) - 1e-6).sort((a, b) => a.residual - b.residual)[0];
      if (!candidate) return this._finish('needs_heading');
      if (this._turnCount >= this.maxTurns) return this._finish('turn_limit');
      this._beginClip(candidate.descriptor, proprio, actualHeading, 'teacher_turn');
    } else {
      this._beginClip(descriptor, proprio, desiredHeading, 'teacher_step');
    }
  }

  _alignedClip(descriptor, proprio, heading, phase) {
    const continued = this.entryReferenceRootPose;
    const transform = continued
      ? turnReferenceTransform(descriptor.skill.frames[0], continued.slice(0, 3), continued.slice(3, 7))
      : turnReferenceTransform(descriptor.skill.frames[0], proprio.rootPosWorld, yawQuat(heading));
    const worldFrames = descriptor.skill.frames.map(frame => transformTeacherReference(frame, transform));
    const referencePlan = { transform, waypointIndex: this.waypointIndex, goalWorld: [...this.waypoints[this.waypointIndex]],
      sourceTravelM: descriptor.travelM,
      headingOffsetRad: wrap((continued ? yaw(continued.slice(3, 7)) : heading) - yaw(proprio.rootQuatXyzwWorld)),
      continuedReferenceRootPose: continued ? [...continued] : null };
    const clip = { descriptor, phase, initialRootPositionWorld: Array.from(proprio.rootPosWorld),
      initialRootHeadingRad: yaw(proprio.rootQuatXyzwWorld), waypointIndex: this.waypointIndex };
    return { worldFrames, referencePlan, clip };
  }

  _beginClip(descriptor, proprio, heading, phase) {
    const candidate = this._alignedClip(descriptor, proprio, heading, phase);
    this.worldFrames = candidate.worldFrames; this.referencePlan = candidate.referencePlan; this._clip = candidate.clip;
    this.phase = phase; this.referenceIndex = 0; this._entered = true;
    if (phase === 'teacher_step') this._stepCount++; else this._turnCount++;
  }

  /** After the caller refuses an unexecuted step, preview smaller complete
   * records at the exact same live state. Only the approved reference changes;
   * rejected candidates do not advance any control, route or motion count. */
  trySmallerStep(proprio, approveCandidate) {
    if (this.phase !== 'teacher_step' || this.referenceIndex !== 0 || this._clipComplete
        || typeof approveCandidate !== 'function') throw new Error('Only an unexecuted step may preflight smaller complete records');
    finite(proprio.rootPosWorld, 3, 'Live root position'); finite(proprio.rootQuatXyzwWorld, 4, 'Live root quaternion');
    const actualHeading = yaw(proprio.rootQuatXyzwWorld), original = this._clip.descriptor;
    if (distance(proprio.rootPosWorld, this._clip.initialRootPositionWorld) > 1e-9
        || Math.abs(proprio.rootPosWorld[2] - this._clip.initialRootPositionWorld[2]) > 1e-9
        || Math.abs(wrap(actualHeading - this._clip.initialRootHeadingRad)) > 1e-9) {
      throw new Error('Smaller-step preflight must use the same measured entry state');
    }
    const goal = this.waypoints[this.waypointIndex], remaining = distance(proprio.rootPosWorld, goal);
    const radius = this.waypointIndex === this.waypoints.length - 1 ? this.arrivalRadius : this.waypointRadius;
    const direction = Math.atan2(goal[1] - proprio.rootPosWorld[1], goal[0] - proprio.rootPosWorld[0]);
    const alternatives = this.steps.filter(item => item.travelM < original.travelM - 1e-6
      && item.travelM <= remaining + radius && Math.abs(remaining - item.travelM) < remaining)
      .sort((a, b) => b.travelM - a.travelM);
    for (const descriptor of alternatives) {
      const heading = wrap(direction + descriptor.sourceYawRad - descriptor.travelDirectionRad);
      if (Math.abs(wrap(heading - actualHeading)) > this.maxHeadingOffsetRad) continue;
      const candidate = this._alignedClip(descriptor, proprio, heading, 'teacher_step');
      const approval = approveCandidate({ phase: 'teacher_step', skill: descriptor.skill,
        sourceFrames: descriptor.skill.sourceFrames, alignedReferenceFrames: candidate.worldFrames,
        referencePlan: candidate.referencePlan, alternativeToSkill: original.skill });
      const approved = approval !== false && approval?.supported !== false;
      (this.outcome.stepAlternatives ??= []).push({ originalSource: original.skill.name ?? null,
        candidateSource: descriptor.skill.name ?? null, sourceFrames: descriptor.skill.sourceFrames,
        sourceTravelM: descriptor.travelM, waypointIndex: this.waypointIndex,
        rootPositionWorld: Array.from(proprio.rootPosWorld), targetWorld: [...goal],
        headingOffsetRad: candidate.referencePlan.headingOffsetRad,
        approved, reason: approved ? null : approval?.reason ?? 'reference_geometry_unsupported',
        teacherSteps: this.outcome.teacherSteps,
        settlingSteps: this.outcome.studentSettlingSteps + this.outcome.teacherSettlingSteps });
      if (!approved) continue;
      this.worldFrames = candidate.worldFrames; this.referencePlan = candidate.referencePlan; this._clip = candidate.clip;
      return { skill: this.skill, sourceFrames: this.sourceFrames, referenceIndex: 0,
        referenceFrames: [this.worldFrames[1], this.worldFrames[16]], outcome: structuredClone(this.outcome) };
    }
    return null;
  }

  requestCancel({ afterSettling = false } = {}) {
    this._cancelAfterSettling = afterSettling;
    if (this._pendingRoute) {
      this._pendingRoute.request.status = 'cancelled';
      this._pendingRoute = null;
    }
    if (['teacher_step', 'teacher_turn'].includes(this.phase) || this._clipComplete) this.finishRequested = true;
    else if (this.phase === 'settling' && afterSettling) this.finishRequested = true;
    else if (this.phase === 'settling' || (this.phase === 'complete' && this._completed)) {
      this.finishRequested = true; this._finish('cancelled');
    }
  }

  step(proprio) {
    let stepFinished = false, turnFinished = false;
    if (this.phase !== 'inactive' && (this.phase !== 'complete' || this._completed)) this._observe(proprio);
    if (this._clipComplete) {
      this._clipComplete = false;
      stepFinished = this._clip.phase === 'teacher_step'; turnFinished = !stepFinished;
      const last = this.worldFrames[this.sourceFrames - 1];
      this.outcome.clips.push({ phase: this._clip.phase, name: this.skill.name ?? null,
        sourceFrames: this.sourceFrames, waypointIndex: this._clip.waypointIndex,
        initialRootPositionWorld: [...this._clip.initialRootPositionWorld],
        initialRootHeadingRad: this._clip.initialRootHeadingRad,
        referenceEndpointWorld: Array.from(last.slice(0, 3)), finalRootPositionWorld: Array.from(proprio.rootPosWorld),
        endpointErrorM: distance(proprio.rootPosWorld, last), finalHeadingErrorRad: wrap(yaw(proprio.rootQuatXyzwWorld) - yaw(last.slice(3, 7))),
        finalPlanarSpeedMps: this.outcome.finalPlanarSpeedMps, headingOffsetRad: this.referencePlan.headingOffsetRad });
      if (this._unsafe()) this._finish('lost_balance');
      else if (this.finishRequested && !this._cancelAfterSettling) this._finish('cancelled');
      else if (this.continuousCompletion && !this.finishRequested) this._finish('finished');
      else {
        this.phase = 'settling'; this._settlingCount = 0; this._stableCount = 0; this._pendingSettling = false;
      }
    } else if (this.phase === 'settling' && this._pendingSettling) {
      this._pendingSettling = false;
      this._stableCount = this.outcome.finalPlanarSpeedMps <= this.maxSettlingSpeedMps ? this._stableCount + 1 : 0;
      if (this._unsafe()) this._finish('lost_balance');
      else if (this._settlingCount >= this.settlingSteps && this._stableCount >= this.stableSteps) this._plan(proprio);
      else if (this._settlingCount >= this.maxSettlingSteps) this._finish('unsettled');
    }
    if (this._completed && this._unsafe()) this.completionReason = 'lost_balance';
    const teacher = ['teacher_step', 'teacher_turn'].includes(this.phase);
    const result = { phase: this.phase, mode: teacher ? 'teacher' : 'student', skill: this.skill,
      locomotionOnly: true, referenceIndex: this.referenceIndex, sourceFrames: this.sourceFrames,
      referenceFrames: teacher ? [this.worldFrames[this.referenceIndex + 1], this.worldFrames[this.referenceIndex + 16]] : null,
      justEnteredTeacher: this._entered, stepFinished, turnFinished, justCompleted: this._completed,
      completionReason: this.completionReason, finishRequested: this.finishRequested,
      waypointIndex: this.waypointIndex, requestedGoalWorld: this.requestedGoalWorld ? [...this.requestedGoalWorld] : null,
      pendingGoalWorld: this._pendingRoute ? [...this._pendingRoute.final] : null,
      waypointWorld: this.waypoints[this.waypointIndex]?.slice() ?? null,
      settlingCount: this._settlingCount, stableCount: this._stableCount,
      outcome: this.outcome ? structuredClone(this.outcome) : null };
    this._entered = false; this._completed = false;
    return result;
  }

  /** Call once after real physics. Read completion on the next step(), so its
   * outcome includes the final action. No simulated state is ever changed. */
  advance() {
    if (['teacher_step', 'teacher_turn'].includes(this.phase) && !this._clipComplete) {
      this.referenceIndex++; this.outcome.teacherSteps++;
      if (this.referenceIndex === this.sourceFrames) this._clipComplete = true;
    } else if (this.phase === 'settling') {
      this._settlingCount++;
      this.outcome[this.settlingPolicy === 'teacher' ? 'teacherSettlingSteps' : 'studentSettlingSteps']++;
      this._pendingSettling = true;
    }
  }
}
