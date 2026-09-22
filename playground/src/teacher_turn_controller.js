// Recorded locomotion turn orchestration. Owns references and their clock only;
// the caller keeps physics, previous actions/DOFs, and student history intact.
import { transformTeacherReference } from './teacher_reference.js';

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
function finite(values, count, name) {
  if (!values || values.length !== count || !Array.from(values).every(Number.isFinite)) {
    throw new Error(`${name} requires ${count} finite values`);
  }
}

/** Anchor source root XY and yaw to live root XY and yaw. Preserve floor Z. */
export function turnReferenceTransform(first, rootPosition, rootQuaternion) {
  finite(first, 747, 'Turn reference');
  finite(rootPosition, 3, 'Live root position');
  finite(rootQuaternion, 4, 'Live root quaternion');
  const rotation = wrap(yaw(rootQuaternion) - yaw(first.slice(3, 7)));
  const c = Math.cos(rotation), s = Math.sin(rotation);
  return { yawRadians: rotation, translation: [rootPosition[0] - c * first[0] + s * first[1],
    rootPosition[1] - s * first[0] - c * first[1], 0] };
}

/** Choose a complete recorded turn that reduces the carry's facing error.
 * Check its full aligned geometry before exposing a teacher reference. The
 * caller handles refusal while retaining ownership of the original box task. */
export function prepareTeacherFacingTurn(skills, proprio, facingErrorRad, { approveReference = null } = {}) {
  if (!Array.isArray(skills) || !Number.isFinite(facingErrorRad)
      || (approveReference !== null && typeof approveReference !== 'function')) {
    throw new Error('Facing preparation requires skills, a finite error and an optional reference check');
  }
  const candidates = skills.map(skill => {
    const delta = wrap(yaw(skill.frames[skill.sourceFrames - 1].slice(3, 7)) - yaw(skill.frames[0].slice(3, 7)));
    return { skill, delta, remainingError: Math.abs(wrap(facingErrorRad + delta)) };
  }).filter(candidate => candidate.delta * facingErrorRad < 0 && candidate.remainingError < Math.abs(facingErrorRad))
    .sort((a, b) => a.remainingError - b.remainingError);
  let refusal = null;
  const refusedCandidates = [];
  for (const candidate of candidates) {
    const controller = new TeacherTurnController(candidate.skill);
    controller.start(proprio);
    const geometry = approveReference?.(proprio, {
      phase: 'teacher_turn', skill: candidate.skill, sourceFrames: candidate.skill.sourceFrames,
      alignedReferenceFrames: controller.worldFrames,
    });
    if (approveReference && geometry?.supported !== true) {
      refusal = geometry?.reason || 'reference_sweep_clearance';
      refusedCandidates.push({ skillName: candidate.skill.name ?? null, sourceFrames: candidate.skill.sourceFrames,
        turnDeltaRad: candidate.delta, remainingErrorRad: candidate.remainingError, reason: refusal,
        obstacleName: geometry?.obstacleName ?? null });
      continue;
    }
    return { supported: true, reason: null, controller, turnAdmissible: true, facingErrorRad };
  }
  // `reason` keeps its historical value (a geometry reason when a reducing turn
  // was refused, otherwise needs_facing). The added fields let the owner keep
  // needs_facing as the task reason while reporting why the turn was not admitted.
  return { supported: false, reason: refusal || 'needs_facing', controller: null, turnAdmissible: false,
    facingErrorRad, turnRefusalReason: refusal, refusedCandidates, candidateCount: candidates.length };
}

export class TeacherTurnController {
  constructor(skill, { minRootHeightM = 0.45, minUpright = 0.5 } = {}) {
    if (!skill || skill.locomotionOnly !== true || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 1
        || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames + 16) {
      throw new Error('A complete explicitly masked locomotion skill is required');
    }
    for (const frame of skill.frames) finite(frame, 747, 'Turn reference');
    if (!Number.isFinite(minRootHeightM) || minRootHeightM < 0 || !Number.isFinite(minUpright) || minUpright < 0 || minUpright > 1) {
      throw new Error('Valid turn root-height and upright requirements are required');
    }
    this.skill = skill; this.minRootHeightM = minRootHeightM; this.minUpright = minUpright;
    this.locomotionOnly = true; this.reset();
  }

  get sourceFrames() { return this.skill.sourceFrames; }

  reset() {
    this.phase = 'inactive'; this.referenceIndex = 0; this.worldFrames = null; this.referencePlan = null;
    this.finishRequested = false; this.completionReason = null; this.outcome = null;
    this._entered = false; this._completed = false;
  }

  start(proprio) {
    if (this.phase === 'teacher_turn') throw new Error('A teacher turn is already active');
    const transform = turnReferenceTransform(this.skill.frames[0], proprio.rootPosWorld, proprio.rootQuatXyzwWorld);
    this.reset();
    this.referencePlan = { transform };
    this.worldFrames = this.skill.frames.map(frame => transformTeacherReference(frame, transform));
    this.phase = 'teacher_turn'; this._entered = true;
    this._observe(proprio);
  }

  _observe(proprio) {
    finite(proprio.rootPosWorld, 3, 'Live root position');
    finite(proprio.rootQuatXyzwWorld, 4, 'Live root quaternion');
    const root = proprio.rootPosWorld, q = proprio.rootQuatXyzwWorld;
    const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    const currentYaw = yaw(q);
    this.outcome ||= { initialRootPositionWorld: Array.from(root), initialRootHeadingRad: currentYaw,
      minRootHeightM: root[2], minUpright: upright,
      referenceYawChangeRad: wrap(yaw(this.worldFrames[this.skill.sourceFrames - 1].slice(3, 7)) - currentYaw) };
    this.outcome.minRootHeightM = Math.min(this.outcome.minRootHeightM, root[2]);
    this.outcome.minUpright = Math.min(this.outcome.minUpright, upright);
    this.outcome.finalRootPositionWorld = Array.from(root);
    this.outcome.finalRootHeadingRad = currentYaw;
    this.outcome.actualYawChangeRad = wrap(currentYaw - this.outcome.initialRootHeadingRad);
    this.outcome.finalHeadingErrorRad = wrap(currentYaw - yaw(this.worldFrames[this.skill.sourceFrames - 1].slice(3, 7)));
    this.outcome.rootPlanarDisplacementM = Math.hypot(root[0] - this.outcome.initialRootPositionWorld[0],
      root[1] - this.outcome.initialRootPositionWorld[1]);
  }

  requestCancel() {
    // Complete the recorded motion before handing back control. No later carry
    // should resume after this cancellation; the caller owns that decision.
    if (this.phase === 'teacher_turn' || (this.phase === 'complete' && this._completed)) this.finishRequested = true;
  }

  step(proprio) {
    if (this.phase === 'teacher_turn' || this._completed) this._observe(proprio);
    if (this._completed) {
      this.completionReason = this.outcome.minRootHeightM < this.minRootHeightM || this.outcome.minUpright < this.minUpright
        ? 'lost_balance' : this.finishRequested ? 'cancelled' : 'finished';
    }
    const result = { phase: this.phase, mode: this.phase === 'teacher_turn' ? 'teacher' : 'student',
      locomotionOnly: true, referenceIndex: this.referenceIndex,
      referenceFrames: this.phase === 'teacher_turn'
        ? [this.worldFrames[this.referenceIndex + 1], this.worldFrames[this.referenceIndex + 16]] : null,
      justEnteredTeacher: this._entered, justCompleted: this._completed,
      completionReason: this.completionReason, finishRequested: this.finishRequested,
      outcome: this.outcome ? { ...this.outcome, initialRootPositionWorld: [...this.outcome.initialRootPositionWorld],
        finalRootPositionWorld: [...this.outcome.finalRootPositionWorld] } : null };
    this._entered = false; this._completed = false;
    return result;
  }

  /** Advance exactly once after a real control step. Final outcome is read by
   * the following step(), which sees the last action's post-physics state. */
  advance() {
    if (this.phase !== 'teacher_turn') return;
    this.referenceIndex++;
    if (this.referenceIndex >= this.skill.sourceFrames) { this.phase = 'complete'; this._completed = true; }
  }
}
