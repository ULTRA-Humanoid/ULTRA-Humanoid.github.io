// Adapt the measured locomotion supervisor to a finite box-approach task.
// Completion requires a reached destination and a measured quiet stance. The
// caller resumes its paused box task using the actual resulting physical state.
import { RestrictedLocomotionController } from './restricted_locomotion_controller.js';

export class TeacherRecordedApproachController {
  constructor(stepSkills, options = {}) {
    this.handoffRadius = options.handoffRadius ?? .1;
    if (!Number.isFinite(this.handoffRadius) || this.handoffRadius <= 0) throw new Error('A finite positive parent handoff radius is required');
    const retainTerminal = options.retainTerminalInHandoffRegion ?? false;
    if (typeof retainTerminal !== 'boolean') throw new Error('Approach terminal retention must be explicitly enabled or disabled');
    const preserveLiveRoot = options.preserveLiveRootInHandoffRegion ?? false;
    if (typeof preserveLiveRoot !== 'boolean') throw new Error('Live-root handoff preservation must be explicitly enabled or disabled');
    this.controller = new RestrictedLocomotionController(stepSkills, { retainStandingOnRefusal: true, ...options,
      approachTerminalRadius: retainTerminal ? this.handoffRadius : null,
      liveRootHandoffRadius: preserveLiveRoot ? this.handoffRadius : null });
    this.maxQuietControls = options.maxSettlingSteps ?? 180;
    this.reset();
  }
  get skill() { return this.controller.skill; }
  get sourceFrames() { return this.controller.sourceFrames; }
  get referenceIndex() { return this.controller.referenceIndex; }
  get locomotionOnly() { return true; }
  get requestedGoalWorld() { return this.goal ? [...this.goal] : null; }

  // Called after physical execution, before advance(), to retain N−1 rather
  // than a lookahead or the next canonical stance. The caller owns the copy.
  terminalFrameForCurrentStep() {
    if (this.phase !== 'teacher_step' || this.referenceIndex !== this.skill.sourceFrames - 1) return null;
    const frame = this.controller._route?.worldFrames[this.skill.sourceFrames - 1];
    return frame ? Float32Array.from(frame) : null;
  }

  reset() {
    this.controller.reset(); this.phase = 'inactive'; this.cancelRequested = false;
    this.goal = null; this.completionReason = null; this.outcome = null;
    this.motionControls = this.settlingControls = 0;
    this.minRootHeightM = this.minUpright = Infinity;
    this.unsupportedControl = null;
    this.parentHandoff = null; this.arrivalKind = null;
  }
  start(proprio, { finalGoalWorld, waypoints = [] }) {
    if (!['inactive', 'complete'].includes(this.phase)) throw new Error('A recorded approach is already active');
    this.reset();
    this.controller.requestFloorGoal(finalGoalWorld, { waypoints });
    this.goal = Array.from(finalGoalWorld); this.phase = 'approach';
  }
  requestCancel() {
    if (['inactive', 'complete', 'unsupported'].includes(this.phase)) return;
    this.cancelRequested = true; this.parentHandoff = null; this.controller.requestCancel();
  }
  step(proprio) {
    if (this.unsupportedControl) return this.unsupportedControl;
    if (this.phase === 'complete') return { phase: 'complete', mode: 'student', justCompleted: false,
      completionReason: this.completionReason, outcome: this.outcome, requestedGoalWorld: this.requestedGoalWorld };
    if (this.phase === 'inactive') throw new Error('Start the recorded approach before stepping it');
    const control = this.controller.step(proprio);
    this.minRootHeightM = Math.min(this.minRootHeightM, proprio.rootPosWorld[2]);
    const q = proprio.rootQuatXyzwWorld;
    this.minUpright = Math.min(this.minUpright, proprio.uprightScore ?? 1 - 2 * (q[0] ** 2 + q[1] ** 2));
    const goalError = Math.hypot(proprio.rootPosWorld[0] - this.goal[0], proprio.rootPosWorld[1] - this.goal[1]);
    const approvedFallback = control.mode === 'teacher' && control.phase === 'teacher_standing'
      && control.supported === false && control.referenceFrames?.length === 2;
    if (!this.parentHandoff && !this.cancelRequested && approvedFallback && goalError <= this.handoffRadius) {
      this.parentHandoff = { frame: control.referenceFrames[0], startControls: control.standingControls,
        maxGoalErrorM: goalError, refusalReason: control.completionReason };
    }
    if (this.parentHandoff) {
      this.parentHandoff.maxGoalErrorM = Math.max(this.parentHandoff.maxGoalErrorM, goalError);
      // The approved reference and every observed root must remain in the
      // parent's original acceptance region throughout the physical hold.
      if (!approvedFallback || control.referenceFrames[0] !== this.parentHandoff.frame
          || goalError > this.handoffRadius) this.parentHandoff = null;
    }
    const handoffControls = this.parentHandoff ? control.standingControls - this.parentHandoff.startControls : 0;
    const regionReady = this.parentHandoff && handoffControls >= Math.max(60, this.controller.options.settlingSteps)
      && control.standingStableControls >= Math.max(12, this.controller.options.stableSteps);
    this.outcome = { ...control.outcome, teacherSteps: this.motionControls,
      teacherSettlingSteps: this.settlingControls, studentSettlingSteps: 0,
      minRootHeightM: this.minRootHeightM, minUpright: this.minUpright,
      finalGoalErrorM: goalError, arrivalKind: this.arrivalKind,
      handoffRadiusM: this.handoffRadius, parentHandoffControls: handoffControls,
      maxParentHandoffGoalErrorM: this.parentHandoff?.maxGoalErrorM ?? null,
      referenceRefusalReason: this.parentHandoff?.refusalReason ?? null,
      finalRootPositionWorld: Array.from(proprio.rootPosWorld),
      finalPlanarSpeedMps: Math.hypot(...proprio.rootVelWorld.slice(0, 2)) };
    const quietExpired = control.phase === 'teacher_standing' && !this.controller.isSettled && !regionReady
      && (control.supported !== false || this.parentHandoff)
      && (this.parentHandoff ? handoffControls : control.standingControls) >= this.maxQuietControls;
    if (control.mode === 'none' || quietExpired) {
      this.phase = 'unsupported'; this.completionReason = quietExpired ? 'unsettled' : control.completionReason;
      this.unsupportedControl = { ...control, phase: this.phase, mode: 'none', supported: false,
        completionReason: this.completionReason, referenceFrames: null, justCompleted: false,
        outcome: this.outcome, requestedGoalWorld: this.requestedGoalWorld };
      return this.unsupportedControl;
    }
    if ((control.supported === false && !this.parentHandoff) || this.controller.isSettled || regionReady) {
      this.phase = 'complete';
      this.completionReason = control.supported === false && !regionReady ? control.completionReason
        : this.cancelRequested ? 'cancelled' : 'finished';
      if (this.completionReason === 'finished') {
        this.arrivalKind = regionReady ? 'parent_handoff_region' : 'floor_goal';
        this.outcome.arrivalKind = this.arrivalKind;
      }
      return { phase: this.phase, mode: 'student', justCompleted: true,
        completionReason: this.completionReason, outcome: this.outcome, requestedGoalWorld: this.requestedGoalWorld,
        // The caller may continue this exact approved quiet stance while its
        // paused parent samples arrival/facing through real physics controls.
        // Never expose a failed or cancelled task as a successful handoff.
        ...(this.completionReason === 'finished' ? {
          handoffReferenceFrames: control.referenceFrames.map(frame => Float32Array.from(frame)),
        } : {}) };
    }
    // This remains a teacher-controlled hold; a distinct phase ensures callers
    // advance its clock instead of treating it as an indefinite post-task hold.
    this.phase = control.phase === 'teacher_standing' ? 'teacher_settling' : control.phase;
    return { ...control, phase: this.phase, justCompleted: false, outcome: this.outcome,
      ...(this.parentHandoff ? { supported: true, completionReason: null, parentHandoffRegion: true } : {}) };
  }
  advance() {
    if (['inactive', 'complete', 'unsupported'].includes(this.phase)) return;
    const before = this.controller.controls;
    this.controller.advance();
    const executed = this.controller.controls - before;
    if (['teacher_step', 'teacher_turn'].includes(this.phase)) this.motionControls += executed;
    else this.settlingControls += executed;
  }
}
