// Bounded pickup/setdown orchestration shared by UI and quantitative rollouts.
// Owns reference timing and approach intent; never writes physics state.
import { rootHeadingAlignedReferenceTransform, transformTeacherReference } from './teacher_reference.js';
import { quatRotateOne } from './math.js';
import { measuredObjectTiltDeg, objectTiltRequirement } from './quiet_ending.js';

export class TeacherSkillController {
  constructor(skill, { settleFrames = 12, arrivalRadius = 0.12, settleSpeed = 0.25, maxApproachSteps = 360,
    maxFacingError = 0.30, alignmentMode = 'approach-heading', maxReferenceStartDistance = arrivalRadius,
    outcomeRequirements = null } = {}) {
    if (!skill || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames + 16
        || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 1 || !skill.objectBodyName) {
      throw new Error('A complete teacher skill is required');
    }
    this.skill = skill;
    this.settleFrames = settleFrames; this.arrivalRadius = arrivalRadius;
    this.settleSpeed = settleSpeed; this.maxApproachSteps = maxApproachSteps;
    this.maxFacingError = maxFacingError;
    this.maxReferenceStartDistance = maxReferenceStartDistance;
    if (outcomeRequirements !== null) {
      for (const key of ['minLiftM', 'maxFinalObjectHeightM', 'minRootHeightM', 'minUpright']) {
        if (!Number.isFinite(outcomeRequirements[key]) || outcomeRequirements[key] < 0) throw new Error(`Invalid outcome requirement ${key}`);
      }
      if (outcomeRequirements.minUpright > 1) throw new Error('Outcome upright requirement must be at most one');
    }
    this.outcomeRequirements = outcomeRequirements === null ? null : { ...outcomeRequirements };
    // Optional (B5): {tippedObjectTiltDeg, objectUpAxisLocal} inside the requirements adds the fail-closed 'object_tipped'
    // outcome. Absent for the frozen largebox requirements -> null -> the outcome record and reasons are unchanged.
    this.objectTiltRequirement = outcomeRequirements === null ? null : objectTiltRequirement(outcomeRequirements);
    if (!['approach-heading', 'preserve-heading'].includes(alignmentMode)) throw new Error('Unknown skill alignment mode');
    this.alignmentMode = alignmentMode;
    this.reset();
  }

  reset() {
    this.phase = 'inactive'; this.referenceIndex = 0; this.approachSteps = 0;
    this.settled = 0; this.approachGoalWorld = null; this.worldFrames = null;
    this.referencePlan = null;
    this.finishRequested = false; this.completionReason = null;
    this.facingErrorRad = null;
    this.outcome = null;
    this._entered = false; this._completed = false;
    this._lastApproachSample = -1;
  }

  _alignment(proprio) {
    const root = proprio.rootPosWorld, object = proprio.objPosWorld;
    if (!root || !object || ![...root, ...object].every(Number.isFinite)) throw new Error('Live root and object positions are required');
    let heading = Math.atan2(object[1] - root[1], object[0] - root[0]);
    if (this.alignmentMode === 'preserve-heading') {
      const rotation = proprio.rootQuatXyzwWorld;
      if (!rotation || rotation.length !== 4 || !Array.from(rotation).every(Number.isFinite)) throw new Error('Live root quaternion is required');
      const forward = quatRotateOne(rotation, [1, 0, 0]); heading = Math.atan2(forward[1], forward[0]);
    }
    return rootHeadingAlignedReferenceTransform(this.skill.frames[0], object, heading);
  }

  _referencePlan(proprio) {
    const transform = this._alignment(proprio);
    return { first: transformTeacherReference(this.skill.frames[0], transform), transform };
  }

  start(proprio) {
    if (this.phase === 'approach' || this.phase === 'teacher') throw new Error('A teacher skill is already active');
    if (proprio.objectBodyName !== this.skill.objectBodyName) throw new Error(`This skill requires ${this.skill.objectBodyName}`);
    const plan = this._referencePlan(proprio), first = plan.first;
    this.reset(); this.phase = 'approach';
    this.referencePlan = plan;
    this.approachGoalWorld = Float32Array.from([first[0], first[1], 0]);
  }

  requestCancel() {
    if (this.phase === 'teacher') this.finishRequested = true;
    else if (this.phase === 'approach') this._complete('cancelled');
  }

  _complete(reason) {
    this.phase = 'complete'; this.completionReason = reason; this._completed = true;
  }

  _observeOutcome(proprio) {
    const root = proprio.rootPosWorld, object = proprio.objPosWorld, q = proprio.rootQuatXyzwWorld;
    if (!root || !object || !q || root.length !== 3 || object.length !== 3 || q.length !== 4
        || ![...root, ...object, ...q].every(Number.isFinite)) throw new Error('Finite physical state is required to measure skill outcome');
    const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    this.outcome ||= { initialObjectHeightM: object[2], peakObjectHeightM: object[2], liftM: 0,
      minRootHeightM: root[2], minUpright: upright, finalObjectHeightM: object[2] };
    this.outcome.peakObjectHeightM = Math.max(this.outcome.peakObjectHeightM, object[2]);
    this.outcome.liftM = this.outcome.peakObjectHeightM - this.outcome.initialObjectHeightM;
    this.outcome.minRootHeightM = Math.min(this.outcome.minRootHeightM, root[2]);
    this.outcome.minUpright = Math.min(this.outcome.minUpright, upright);
    this.outcome.finalObjectHeightM = object[2];
    if (this.objectTiltRequirement) {
      // NaN when the live object quaternion is unusable: the tipped test below then fails closed.
      const tilt = measuredObjectTiltDeg(proprio.objQuatXyzwWorld, this.objectTiltRequirement.objectUpAxisLocal);
      const previous = this.outcome.maxObjectTiltDeg;
      this.outcome.maxObjectTiltDeg = previous === undefined ? tilt : Number.isFinite(previous) && Number.isFinite(tilt) ? Math.max(previous, tilt) : NaN;
      this.outcome.finalObjectTiltDeg = tilt;
    }
  }

  _measuredCompletionReason() {
    const requirement = this.outcomeRequirements, outcome = this.outcome;
    if (outcome.minRootHeightM < requirement.minRootHeightM || outcome.minUpright < requirement.minUpright) return 'lost_balance';
    if (outcome.liftM < requirement.minLiftM) return 'failed_lift';
    if (outcome.finalObjectHeightM > requirement.maxFinalObjectHeightM) return 'failed_setdown';
    // Tilt requirement present: a set-down object lying on its side (height passes) or an unmeasured tilt is 'object_tipped'.
    if (this.objectTiltRequirement && !(outcome.finalObjectTiltDeg < this.objectTiltRequirement.tippedObjectTiltDeg)) return 'object_tipped';
    return 'finished';
  }

  step(proprio) {
    if (this.phase === 'approach' && this._lastApproachSample !== this.approachSteps) {
      this._lastApproachSample = this.approachSteps;
      const root = proprio.rootPosWorld, velocity = proprio.rootVelWorld;
      if (!root || !velocity || ![...root, ...velocity].every(Number.isFinite)) throw new Error('Finite root position and velocity are required');
      const distance = Math.hypot(root[0] - this.approachGoalWorld[0], root[1] - this.approachGoalWorld[1]);
      const speed = Math.hypot(velocity[0], velocity[1]);
      this.settled = distance <= this.arrivalRadius && speed <= this.settleSpeed ? this.settled + 1 : 0;
      if (this.settled >= this.settleFrames) {
        const plan = this._referencePlan(proprio), first = plan.first;
        this.referencePlan = plan;
        // Collisions may move the box during approach. Arriving at a stale
        // floor target must not switch to a now-distant reference pose.
        if (Math.hypot(root[0] - first[0], root[1] - first[1]) > this.maxReferenceStartDistance) {
          this.approachGoalWorld = Float32Array.from([first[0], first[1], 0]); this.settled = 0;
        } else {
          const current = proprio.rootQuatXyzwWorld;
          if (!current || current.length !== 4 || !Array.from(current).every(Number.isFinite)) throw new Error('Live root quaternion is required');
          const forward = quatRotateOne(current, [1, 0, 0]);
          const targetForward = quatRotateOne(first.slice(3, 7), [1, 0, 0]);
          const angle = Math.atan2(forward[1], forward[0]) - Math.atan2(targetForward[1], targetForward[0]);
          this.facingErrorRad = Math.atan2(Math.sin(angle), Math.cos(angle));
          // Each skill supplies its measured initial-heading range. Outside
          // that range, preserve student standing for a later facing attempt.
          if (Math.abs(this.facingErrorRad) > this.maxFacingError) this._complete('needs_facing');
          else {
            this.worldFrames = plan.frames || this.skill.frames.map(frame => transformTeacherReference(frame, plan.transform));
            this.phase = 'teacher'; this.referenceIndex = 0; this._entered = true;
          }
        }
      }
      if (this.phase === 'approach' && this.approachSteps >= this.maxApproachSteps) this._complete('approach_timeout');
    }
    if (this.outcomeRequirements && (this.phase === 'teacher' || (this._completed && this.completionReason === 'finished'))) {
      // advance() completes the clock after physics. The following step()
      // sees that final physical state before exposing completion to the UI.
      this._observeOutcome(proprio);
      if (this._completed) this.completionReason = this._measuredCompletionReason();
    }
    const result = {
      phase: this.phase, mode: this.phase === 'teacher' ? 'teacher' : 'student',
      approachGoalWorld: this.phase === 'approach' ? this.approachGoalWorld : null,
      referenceFrames: this.phase === 'teacher'
        ? [this.worldFrames[this.referenceIndex + 1], this.worldFrames[this.referenceIndex + 16]] : null,
      referenceIndex: this.referenceIndex, justEnteredTeacher: this._entered,
      justCompleted: this._completed, completionReason: this.completionReason,
      facingErrorRad: this.facingErrorRad,
      outcome: this.outcome ? { ...this.outcome } : null,
      requestedGoalWorld: this.referencePlan?.requestedGoalWorld || null,
      referenceGoalWorld: this.referencePlan?.referenceGoalWorld || null,
      referenceGoalResidualM: this.referencePlan?.remainingDistance ?? null,
    };
    this._entered = false; this._completed = false;
    return result;
  }

  /** Call only after the chosen action has advanced real physics. */
  advance() {
    if (this.phase === 'approach') this.approachSteps++;
    else if (this.phase === 'teacher') {
      this.referenceIndex++;
      if (this.referenceIndex >= this.skill.sourceFrames) this._complete('finished');
    }
  }
}
