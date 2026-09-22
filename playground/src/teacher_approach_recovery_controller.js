// Recover a refused approach using the terminal of its last complete step.
// The caller supplies the original world frame and the unchanged parent goal.
// Every proposed action still requires the live all-box physics preview.
export class TeacherApproachRecoveryController {
  constructor({ skill, plan, goalWorld, radius, approveReference, maxControls = 180 }) {
    if (!skill?.locomotionOnly || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 1
        || !plan?.frame || plan.frame.length !== 747 || !Array.from(plan.frame).every(Number.isFinite)
        || !Array.isArray(goalWorld) || goalWorld.length !== 3 || !Array.from(goalWorld).every(Number.isFinite)
        || !Number.isFinite(radius) || radius <= 0 || radius > .25
        || !Number.isInteger(maxControls) || maxControls < 1 || maxControls > 180
        || typeof approveReference !== 'function') {
      throw new Error('A complete executed locomotion terminal and original parent region are required');
    }
    const frame = Float32Array.from(plan.frame);
    if (!frame.every(Number.isFinite)) throw new Error('The executed terminal must fit finite policy observations');
    Object.assign(this, { skill, plan: { ...plan, frame }, goalWorld: [...goalWorld],
      radius, approveReference, maxControls });
    this.reset();
    this.phase = 'teacher_settling';
  }

  reset() {
    this.phase = 'inactive';
    this.controls = this.referenceIndex = 0;
    this.pending = this.issued = this.cancelled = this.completed = false;
    this.regionEntry = this.geometry = this.completionReason = null;
    this.quiet = 0;
    this.records = [];
  }
  get sourceFrames() { return 0; }
  get requestedGoalWorld() { return [...this.goalWorld]; }
  get outcome() {
    return { controls: this.controls, regionEntryControl: this.regionEntry,
      additionalRegionControls: this.regionEntry === null ? 0 : this.controls - this.regionEntry,
      quietControls: this.quiet, completionReason: this.completionReason };
  }
  requestCancel() {
    if (this.phase === 'teacher_settling') this.cancelled = true;
  }

  step(proprio) {
    if (this.phase === 'inactive') throw new Error('A reset recovery cannot resume');
    const distance = Math.hypot(proprio.rootPosWorld[0] - this.goalWorld[0],
      proprio.rootPosWorld[1] - this.goalWorld[1]);
    const speed = Math.hypot(...proprio.rootVelWorld.slice(0, 2));
    const q = proprio.rootQuatXyzwWorld;
    const upright = proprio.uprightScore ?? 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    // Only feedback following a committed physical action earns arrival credit.
    if (this.pending) {
      this.pending = false;
      if (distance <= this.radius) {
        this.regionEntry ??= this.controls;
        this.quiet = speed <= .05 && proprio.rootPosWorld[2] >= .7 && upright >= .95 ? this.quiet + 1 : 0;
      } else {
        this.regionEntry = null;
        this.quiet = 0;
      }
      this.records.push({ control: this.controls, distanceM: distance, rootPlanarSpeedMps: speed,
        rootHeightM: proprio.rootPosWorld[2], upright, regionEntryControl: this.regionEntry,
        quietControls: this.quiet });
    }
    if (this.phase === 'complete') return { mode: 'student', phase: this.phase, justCompleted: false,
      completionReason: this.completionReason, outcome: this.outcome };
    if (this.cancelled) {
      this.phase = 'complete';
      this.completionReason = 'cancelled';
      this.issued = false;
    } else if (this.regionEntry !== null && this.controls - this.regionEntry >= 60 && this.quiet >= 12) {
      this.phase = 'complete';
      this.completionReason = 'finished';
    } else if (this.controls >= this.maxControls) {
      this.phase = 'unsupported';
      this.completionReason = 'approach_recovery_timeout';
    }
    if (this.phase === 'teacher_settling' && !this.geometry) {
      this.geometry = this.approveReference(proprio, { phase: this.phase, skill: this.skill, sourceFrames: 1,
        alignedReferenceFrames: [this.plan.frame], referencePlan: this.plan });
      if (!this.geometry?.supported) {
        this.phase = 'unsupported';
        this.completionReason = this.geometry?.reason ?? 'approach_recovery_geometry';
      }
    }
    if (this.phase === 'unsupported') return { mode: 'none', phase: this.phase, supported: false,
      justCompleted: false, completionReason: this.completionReason, referenceFrames: null,
      outcome: this.outcome };
    if (this.phase === 'complete') {
      const justCompleted = !this.completed;
      this.completed = true;
      return { mode: 'student', phase: this.phase, justCompleted, completionReason: this.completionReason,
        outcome: this.outcome, ...(this.completionReason === 'finished'
          ? { handoffReferenceFrames: [this.plan.frame, this.plan.frame] } : {}) };
    }
    this.issued = true;
    return { mode: 'teacher', phase: this.phase, justEnteredTeacher: false, justCompleted: false,
      referenceFrames: [this.plan.frame, this.plan.frame], referenceIndex: this.controls,
      approachRecovery: true, outcome: this.outcome };
  }

  advance() {
    if (this.phase !== 'teacher_settling' || this.pending || !this.issued) {
      throw new Error('A recovery action may commit exactly once');
    }
    this.controls++;
    this.referenceIndex = this.controls;
    this.pending = true;
    this.issued = false;
  }
}
