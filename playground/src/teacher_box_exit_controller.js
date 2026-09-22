// Finite recorded exit after a placed-box task. Owns references and clocks only;
// the caller retains physics, policy history and queued user intentions.
import { planTeacherStandingReference } from './teacher_standing_reference.js';
import { turnReferenceTransform } from './teacher_turn_controller.js';
import { transformTeacherReference } from './teacher_reference.js';
import { QuietHoldMonitor, validateQuietOptions } from './quiet_ending.js';
import { QUIET_EXIT_HOLD_MODES, createQuietExitHoldState, observeQuietExitHold, decideQuietExitHold,
  latchQuietExitHoldEnd } from './quiet_exit_hold_policy.js';
import { planExitReleaseFrame, validateExitRelease, EXIT_RELEASE_PHASE, planExitHandClampFrame, validateExitHandClamp } from './object_exit_release.js';

// ACTIVE is the recorded exit programme: fixed (tracked) holds plus the complete
// retreat. Its total length is fixed at start() so the frozen execution proof,
// which reads expectedTotalControls at the first exit control, stays valid.
// QUIET phases are the optional quiet-terminated hold extensions (WS-G): the
// same fixed hold reference, ended once the live measurement has been quiet
// for a full window, never past the original hold maximum. They are counted
// separately (quietHoldControls / quietSettlingControls) and never hidden.
const ACTIVE = new Set(['teacher_exit_hold', 'teacher_exit_retreat', 'teacher_exit_settling']);
const QUIET = new Set(['teacher_exit_quiet_hold', 'teacher_exit_quiet_settling']);
// RELEASE (v6g, 2026-09-21): the optional release-and-lift micro-phase that precedes the first hold when the placed object's profile declares it.
// Counted separately (releaseControls), never in totalControls / expectedTotalControls (the frozen execution proof of the recorded programme).
const RELEASE = new Set([EXIT_RELEASE_PHASE]);
const HOLDING = new Set([...ACTIVE, ...QUIET, ...RELEASE]);
function vector(value, length, label) {
  if (!value || value.length !== length || !Array.from(value).every(Number.isFinite)) {
    throw new Error(`${label} requires ${length} finite values`);
  }
}
function quaternion(value, label) {
  vector(value, 4, label);
  if (Math.abs(Math.hypot(...value) - 1) > .01) throw new Error(`${label} must be normalized`);
}
function reference(frame, label) {
  vector(frame, 747, label); quaternion(frame.slice(3, 7), `${label} root quaternion`);
  for (let body = 0; body < 39; body++) quaternion(frame.slice(201 + 4 * body, 205 + 4 * body), `${label} body quaternion`);
}
function proprioception(proprio) {
  vector(proprio?.rootPosWorld, 3, 'Root position');
  quaternion(proprio.rootQuatXyzwWorld, 'Root quaternion');
  vector(proprio.objPosWorld, 3, 'Placed object position');
  quaternion(proprio.objQuatXyzwWorld, 'Placed object quaternion');
  if (proprio.uprightScore !== undefined && !Number.isFinite(proprio.uprightScore)) throw new Error('Finite upright score required');
}

export class TeacherBoxExitController {
  constructor(retreatSkill, { initialHoldControls = 60, finalHoldControls = 180,
    minRootHeightM = .45, minUpright = .5, quietHold = null } = {}) {
    if (!retreatSkill || retreatSkill.locomotionOnly !== true || ![199, 249].includes(retreatSkill.sourceFrames)
        || !Array.isArray(retreatSkill.frames) || retreatSkill.frames.length < retreatSkill.sourceFrames + 16) {
      throw new Error('A vetted complete 199-control or 249-control locomotion retreat with +16 lookahead is required');
    }
    for (const frame of retreatSkill.frames) reference(frame, 'Retreat frame');
    for (const count of [initialHoldControls, finalHoldControls]) {
      if (!Number.isInteger(count) || count < 1) throw new Error('Hold durations must be positive integer controls');
    }
    if (!Number.isFinite(minRootHeightM) || minRootHeightM <= 0 || !Number.isFinite(minUpright)
        || minUpright < 0 || minUpright > 1) throw new Error('Valid balance limits required');
    this.retreatSkill = { ...retreatSkill, frames: retreatSkill.frames.map(frame => Float32Array.from(frame)) };
    this.quietHold = validateQuietOptions(quietHold, 'Quiet exit holds');
    // B2 (Phase B): an explicit `mode` selects the request-terminated settling policy for the FINAL
    // settling hold only (quiet_exit_hold_policy.js). Without `mode` the legacy WS-G quiet holds run
    // exactly as before (quietEndings=1 reproduction path). Fail closed on an unknown mode or a
    // quiet_on_request without a synchronous pending-request reader.
    this.quietPolicyMode = this.quietHold?.mode ?? null;
    if (this.quietPolicyMode !== null) {
      if (!QUIET_EXIT_HOLD_MODES.includes(this.quietPolicyMode) || this.quietPolicyMode === 'fixed')
        throw new Error(`Quiet exit settling mode must be one of quiet | quiet_on_request (got ${this.quietPolicyMode})`);
      if (this.quietPolicyMode === 'quiet_on_request' && typeof this.quietHold.requestQueued !== 'function')
        throw new Error('quiet_on_request exit settling requires a synchronous requestQueued() reader');
    }
    const initialTracked = this.quietHold ? this.quietHold.initialHoldMinControls ?? 30 : initialHoldControls;
    const finalTracked = this.quietHold ? this.quietHold.finalHoldMinControls ?? 30 : finalHoldControls;
    for (const [minimum, maximum] of [[initialTracked, initialHoldControls], [finalTracked, finalHoldControls]]) {
      if (!Number.isInteger(minimum) || minimum < 1 || minimum > maximum) throw new Error('Quiet hold minimums must be whole controls within the hold maximums');
    }
    this.options = Object.freeze({ initialHoldControls, finalHoldControls, minRootHeightM, minUpright,
      initialHoldTrackedControls: initialTracked, finalHoldTrackedControls: finalTracked,
      quietHold: this.quietHold ? Object.freeze({ window: this.quietHold.window,
        initialHoldMinControls: initialTracked, finalHoldMinControls: finalTracked,
        ...(this.quietPolicyMode ? { mode: this.quietPolicyMode } : {}) }) : null });
    this.quietMonitor = this.quietHold ? new QuietHoldMonitor({ window: this.quietHold.window }) : null;
    this.reset();
  }

  get skill() {
    return { ...this.retreatSkill, objectBodyName: this.objectBodyName,
      objectPointsLocal: this.objectPointsLocal, locomotionOnly: this.locomotionOnly };
  }
  get locomotionOnly() { return this.phase !== 'teacher_exit_hold' && this.phase !== 'teacher_exit_quiet_hold' && !RELEASE.has(this.phase); }
  get sourceFrames() { return this.retreatSkill.sourceFrames; }
  get referenceIndex() { return this.phase === 'teacher_exit_retreat' ? this.phaseControls : null; }
  get requestedRootGoalWorld() { return this.goal ? [...this.goal] : null; }
  _phaseDuration(phase = this.phase) {
    const o = this.options;
    return phase === 'teacher_exit_hold' ? (this.quietHolds === false ? o.initialHoldControls : o.initialHoldTrackedControls)
      : phase === 'teacher_exit_retreat' ? this.sourceFrames
      : phase === 'teacher_exit_settling' ? (this.quietFinalHold === false ? o.finalHoldControls : o.finalHoldTrackedControls)
      : phase === 'teacher_exit_quiet_hold' ? (this.quietHolds === false ? 0 : o.initialHoldControls - o.initialHoldTrackedControls)
      : phase === 'teacher_exit_quiet_settling' ? (this.quietFinalHold === false ? 0 : o.finalHoldControls - o.finalHoldTrackedControls)
      : RELEASE.has(phase) ? (this.releaseLift ? this.releaseLift.maxControls : 0) : 0;
  }
  get recordClock() {
    const o = this.options, quiet = QUIET.has(this.phase);
    return { phase: this.phase,
      phaseControls: quiet ? (this.phase === 'teacher_exit_quiet_hold' ? this.quietHoldControls : this.quietSettlingControls) : RELEASE.has(this.phase) ? this.releaseControls : this.phaseControls,
      phaseDurationControls: this._phaseDuration(),
      referenceIndex: this.referenceIndex, retreatControls: this.retreatControls,
      totalControls: this.totalControls,
      expectedTotalControls: (this.quietHolds === false ? o.initialHoldControls : o.initialHoldTrackedControls) + this.sourceFrames + (this.quietFinalHold === false ? o.finalHoldControls : o.finalHoldTrackedControls),
      quietFinalHold: this.quietFinalHold !== false, quietHolds: this.quietHolds !== false,
      quietHoldControls: this.quietHoldControls, quietSettlingControls: this.quietSettlingControls,
      totalControlsIncludingQuiet: this.totalControls + this.quietHoldControls + this.quietSettlingControls,
      ...(this.releaseLift ? { releaseControls: this.releaseControls, releaseWaiting: RELEASE.has(this.phase), release: this.releaseReview ?? this.releaseCurrent ?? null } : {}),   // v6g: absent without the option (v5 clock shape)
      quietHold: this.quietHold ? { ...o.quietHold, initialHold: this.quietReviews.initialHold ?? null,
        finalHold: this.quietReviews.finalHold ?? null, current: this.quietSummary } : null,
      quietExitPolicy: this.quietPolicyMode ? { mode: this.quietPolicyMode,
        requestQueuedAtStart: this.settlingPolicy?.requestQueuedAtStart ?? null,
        lastDecision: this.settlingDecision ? { decision: this.settlingDecision.decision, reason: this.settlingDecision.reason,
          controls: this.settlingDecision.controls } : null } : null };
  }
  /** The recorded exit programme (fixed holds + complete retreat) is running. */
  isActive() { return ACTIVE.has(this.phase) || this.completionPending; }
  /** A quiet-terminated hold extension is running (not part of isActive()). */
  get quietWaiting() { return QUIET.has(this.phase); }
  /** The controller owns control: recorded programme or a quiet extension. */
  /** The release-and-lift micro-phase is running (v6g; not part of isActive()). */
  get releaseWaiting() { return RELEASE.has(this.phase); }
  isBusy() { return this.isActive() || this.quietWaiting || this.releaseWaiting; }

  reset() {
    this.phase = 'inactive'; this.phaseControls = this.totalControls = this.retreatControls = 0;
    this.objectBodyName = null; this.objectPointsLocal = null; this.standOffWorld = null;
    this.holdHandClamp = null;   // v6h
    this.holdSource = null;   // v6j
    this.releaseLift = null; this.releasePlan = null; this.releaseControls = 0; this.releaseClearControls = 0; this.releaseSampledControl = -1;
    this.releaseReview = null; this.releaseCurrent = null; this.releaseMaxHumanForceN = null;
    this.holdPlan = this.finalHoldPlan = this.worldFrames = this.referencePlan = null;
    this.goal = this.outcome = this.completionReason = null;
    this.entered = this.pendingAdvance = this.completionPending = false;
    this.quietHoldControls = this.quietSettlingControls = 0; this.quietSampledControl = -1;
    this.quietSummary = null; this.quietReviews = {}; this.quietMonitor?.reset();
    this.settlingPolicy = null; this.settlingDecision = null;
  }

  // quietFinalHold=false (WS-G, intermediate segment exits): the final hold runs its full fixed
  // length instead of the quiet-terminated minimum. Evidence: H083 — the shortened exit moved the
  // next pickup's approach start and it was refused on sweep clearance; the placed box needs the
  // full settling before the next approach is planned from this stance.
  // standOffWorld (v6e, 2026-09-21, default null = v5 behaviour): optional horizontal [dx, dy, 0] by which the initial hold reference and the
  // retreat reference are translated away from a placed object that stands taller than the largebox (object_exit_standoff.js). The final
  // settling hold is planned from the already-translated retreat frames, so the offset is applied once. Recorded in the outcome only when set.
  // releaseLift (v6g, 2026-09-21, default null = v6f behaviour): optional release-and-lift micro-phase (object_exit_release.js) that runs BEFORE the
  // first hold: the terminal hold pose with both hand chains lifted above the placed object's top face and moved out of its side faces, hand contact
  // flags cleared; ended by the caller's contact measurement (human<->object normal force quiet for a window) or by its control cap (recorded).
  // holdHandClamp (v6h, 2026-09-21, default null = the v5 hold): the INITIAL hold reference is the terminal frame with both hand chains clamped above the placed
  // object's top face (object_exit_release.js planExitHandClampFrame), so the hold never pulls the hands down along a taller object's side face (B5 section 14).
  // holdSource (v6j, 2026-09-21, default null = the v5 terminal-frame hold): 'retreat_row0' => the INITIAL hold reference is the retreat clip's row-0 frame aligned to the live
  // root by turnReferenceTransform (+ the same stand-off), i.e. exactly the frame the retreat would start from; the retreat then follows frames 1.. of that SAME alignment
  // (no re-alignment at the retreat start), so the hold -> retreat seam is one clip. B5 section 16.
  start(proprio, { terminalFrame, objectBodyName, objectPointsLocal, quietFinalHold = true, quietHolds = true, standOffWorld = null, releaseLift = null, holdHandClamp = null, holdSource = null } = {}) {
    if (!['inactive', 'complete'].includes(this.phase) || this.completionPending) throw new Error('Reset or finish the previous box exit first');
    proprioception(proprio); reference(terminalFrame, 'Executed terminal frame');
    if (standOffWorld !== null) {
      vector(standOffWorld, 3, 'Exit stand-off'); if (standOffWorld[2] !== 0) throw new Error('Exit stand-off must be horizontal (z = 0)');
    }
    if (releaseLift !== null) validateExitRelease(releaseLift);
    if (holdHandClamp !== null) validateExitHandClamp(holdHandClamp);
    if (holdHandClamp !== null && releaseLift !== null) throw new Error('Exit hold hand clamp and release-and-lift are exclusive');
    if (holdSource !== null && holdSource !== 'retreat_row0') throw new Error('Exit hold source must be retreat_row0 or null');
    if (holdSource !== null && (holdHandClamp !== null || releaseLift !== null)) throw new Error('Exit hold source is exclusive with the hand clamp and release-and-lift');
    if (typeof objectBodyName !== 'string' || !objectBodyName) throw new Error('Placed object body name required');
    if (proprio.objectBodyName && proprio.objectBodyName !== objectBodyName) throw new Error('Placed object body does not match proprioception');
    if (!Array.isArray(objectPointsLocal) || objectPointsLocal.length !== 256) throw new Error('The actual 256 object points are required');
    for (const point of objectPointsLocal) vector(point, 3, 'Object point');
    this.reset(); this.objectBodyName = objectBodyName;
    this.standOffWorld = standOffWorld === null ? null : Object.freeze(Array.from(standOffWorld));
    this.quietHolds = quietHolds !== false; this.quietFinalHold = this.quietHolds && quietFinalHold !== false;
    this.objectPointsLocal = objectPointsLocal.map(point => Array.from(point));
    this.releaseLift = releaseLift === null ? null : Object.freeze({ ...releaseLift });
    this.holdHandClamp = holdHandClamp === null ? null : Object.freeze({ ...holdHandClamp });
    this.holdSource = holdSource;
    this.phase = this.releaseLift ? EXIT_RELEASE_PHASE : 'teacher_exit_hold'; this.entered = true;
    this.outcome = { initialRootPositionWorld: Array.from(proprio.rootPosWorld),
      initialObjectPositionWorld: Array.from(proprio.objPosWorld),
      minRootHeightM: Infinity, minUpright: Infinity, maxObjectPlanarDriftM: 0,
      ...(this.standOffWorld ? { standOffWorld: Array.from(this.standOffWorld) } : {}) };
    this._observe(proprio);
    if (this.phase !== 'unsupported') {
      // v6h hand clamp: the initial hold reference is the terminal frame with the hand chains clamped above the object's top face (the v6g release frame geometry),
      // planned exactly like the v5 hold (fixed root, live object, recomputed interaction graph, same stand-off). null => the terminal frame (v5 hold).
      const clamp = this.holdHandClamp ? planExitHandClampFrame(terminalFrame, { objectPosWorld: proprio.objPosWorld, ...this.holdHandClamp }) : null;
      if (this.holdSource === 'retreat_row0') {
        // v6j: align the retreat clip to the live root NOW (the retreat's own alignment code + the same stand-off), hold its frame 0 for the whole initial hold and let the
        // retreat continue from frame 1 of this SAME alignment (step() sees worldFrames already set). The hold frame goes through the standard hold planner (fixed root,
        // live object, recomputed interaction graph, zero velocities); the stand-off is already inside the transform, so _fixed() gets none.
        const aligned = turnReferenceTransform(this.retreatSkill.frames[0], proprio.rootPosWorld, proprio.rootQuatXyzwWorld);
        const transform = this.standOffWorld ? { ...aligned, translation: aligned.translation.map((v, i) => v + this.standOffWorld[i]) } : aligned;
        this.referencePlan = { transform, holdSource: 'retreat_row0', ...(this.standOffWorld ? { standOffWorld: Array.from(this.standOffWorld) } : {}) };
        this.worldFrames = this.retreatSkill.frames.map(frame => transformTeacherReference(frame, transform));
        this.goal = Array.from(this.worldFrames[this.sourceFrames - 1].slice(0, 3));
        this.holdPlan = { ...this._fixed(this.worldFrames[0], proprio, null), holdSource: 'retreat_row0' };
        this.outcome.holdSource = 'retreat_row0';
      } else {
        this.holdPlan = this._fixed(clamp ? clamp.frame : terminalFrame, proprio, this.standOffWorld);
        if (clamp) { this.holdPlan = { ...this.holdPlan, handClamp: clamp.hands }; this.outcome.holdHandClamp = { hands: clamp.hands, objectTopZ: clamp.objectTopZ }; }
      }
    }
    if (this.phase !== 'unsupported' && this.releaseLift) {
      // v6g: the release frame is the SAME terminal pose with the hand chains displaced (pure geometry), then planned exactly like the hold
      // (fixed root, live object, recomputed interaction graph, same stand-off) so the release -> hold hand-off is a hand motion only.
      const release = planExitReleaseFrame(terminalFrame, { objectPosWorld: proprio.objPosWorld, hands: this.releaseLift.hands,
        standingHalfExtentM: this.releaseLift.standingHalfExtentM, clearanceMarginM: this.releaseLift.clearanceMarginM, outM: this.releaseLift.outM });
      this.releasePlan = { ...this._fixed(release.frame, proprio, this.standOffWorld), hands: release.hands, objectTopZ: release.objectTopZ };
      this.outcome.releaseLift = { hands: release.hands, objectTopZ: release.objectTopZ, minControls: this.releaseLift.minControls,
        clearWindow: this.releaseLift.clearWindow, maxControls: this.releaseLift.maxControls, humanForceMaximumN: this.releaseLift.humanForceMaximumN };
    }
  }

  // Raw intentions stay queued in the caller. Reset is the explicit interruption.
  requestCancel() { return false; }

  _fixed(frame, proprio, standOff = null) {
    const plan = planTeacherStandingReference(frame, { alignment: 'original',
      rootPosition: proprio.rootPosWorld, rootQuaternion: proprio.rootQuatXyzwWorld,
      objectPosition: proprio.objPosWorld, objectQuaternion: proprio.objQuatXyzwWorld,
      objectPointsLocal: this.objectPointsLocal });
    // v6e stand-off: the whole hold frame (root, object channels, bodies) is translated by the same horizontal vector (common translation, as
    // teacher_goal_warp does); null => the plan is returned untouched (v5).
    return standOff ? { ...plan, frame: transformTeacherReference(plan.frame, { yawRadians: 0, translation: Array.from(standOff) }), standOffWorld: Array.from(standOff) } : plan;
  }

  _observe(proprio) {
    const root = proprio.rootPosWorld, q = proprio.rootQuatXyzwWorld;
    const upright = proprio.uprightScore ?? 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    this.outcome.minRootHeightM = Math.min(this.outcome.minRootHeightM, root[2]);
    this.outcome.minUpright = Math.min(this.outcome.minUpright, upright);
    this.outcome.finalRootPositionWorld = Array.from(root);
    this.outcome.finalObjectPositionWorld = Array.from(proprio.objPosWorld);
    this.outcome.rootPlanarDisplacementM = Math.hypot(root[0] - this.outcome.initialRootPositionWorld[0], root[1] - this.outcome.initialRootPositionWorld[1]);
    this.outcome.maxObjectPlanarDriftM = Math.max(this.outcome.maxObjectPlanarDriftM,
      Math.hypot(proprio.objPosWorld[0] - this.outcome.initialObjectPositionWorld[0], proprio.objPosWorld[1] - this.outcome.initialObjectPositionWorld[1]));
    this.outcome.finalRootGoalErrorM = this.goal ? Math.hypot(root[0] - this.goal[0], root[1] - this.goal[1]) : null;
    if (root[2] < this.options.minRootHeightM || upright < this.options.minUpright) {
      this.phase = 'unsupported'; this.completionReason = 'lost_balance';
      this.pendingAdvance = this.completionPending = this.entered = false;
    }
  }

  /** One live quiet sample per hold control; the sample describes the state
   * after the previous control. Never steps physics. */
  _observeQuiet() {
    const control = this.totalControls + this.quietHoldControls + this.quietSettlingControls;
    if (this.quietSampledControl === control) return;
    this.quietSampledControl = control;
    const sample = this.quietHold.measure();
    this.quietMonitor.observe(sample);
    this.quietSummary = { phase: this.phase, ...this.quietMonitor.summary() };
    if (this.settlingPolicy && (this.phase === 'teacher_exit_settling' || this.phase === 'teacher_exit_quiet_settling')) {
      this.settlingPolicy = observeQuietExitHold(this.settlingPolicy, sample, { controls: this._settlingControlsIssued() });
    }
  }

  /** Settling controls already issued (tracked minimum + quiet extension). */
  _settlingControlsIssued() {
    return (this.phase === 'teacher_exit_settling' ? this.phaseControls : this.options.finalHoldTrackedControls) + this.quietSettlingControls;
  }
  _requestQueued() { return this.quietHold?.requestQueued?.() === true; }
  /** Policy decision for the final settling (B2). Legacy quiet holds (no mode) never reach this. */
  _settlingPolicyEnds() {
    if (!this.settlingPolicy) return null;
    const decision = decideQuietExitHold(this.settlingPolicy, { controls: this._settlingControlsIssued(), requestPending: this._requestQueued() });
    this.settlingDecision = decision;
    if (decision.decision === 'endHold') this.settlingPolicy = latchQuietExitHoldEnd(this.settlingPolicy, decision);
    return decision.decision === 'endHold';
  }

  /** v6g: one contact sample per release control (the caller's synchronous measurement, never steps physics). The phase closes when the
   *  human<->object normal force has been at or below the limit for a full window (never before minControls); advance() closes it at maxControls. */
  _observeRelease() {
    if (this.releaseSampledControl === this.releaseControls) return;
    this.releaseSampledControl = this.releaseControls;
    const sample = this.releaseLift.measure(), force = sample?.humanNormalForceN;
    if (!Number.isFinite(force)) throw new Error('Release-and-lift requires a finite human-object normal force sample');
    this.releaseClearControls = force <= this.releaseLift.humanForceMaximumN ? this.releaseClearControls + 1 : 0;
    this.releaseMaxHumanForceN = Math.max(this.releaseMaxHumanForceN ?? 0, force);
    this.releaseCurrent = { controls: this.releaseControls, clearControls: this.releaseClearControls, humanNormalForceN: force, maxHumanForceN: this.releaseMaxHumanForceN };
    if (this.releaseControls >= this.releaseLift.minControls && this.releaseClearControls >= this.releaseLift.clearWindow) this._closeRelease(true);
  }
  _closeRelease(cleared) {
    this.releaseReview = { controls: this.releaseControls, clearControls: this.releaseClearControls, cleared, maxControls: this.releaseLift.maxControls,
      maxHumanForceN: this.releaseMaxHumanForceN ?? null, finalHumanForceN: this.releaseCurrent?.humanNormalForceN ?? null };
    this.outcome.releaseLift = { ...this.outcome.releaseLift, review: this.releaseReview };
    this.phase = 'teacher_exit_hold'; this.entered = true;
  }
  _closeHold(initial) {
    if (this.quietHold) {
      const o = this.options, tracked = initial ? o.initialHoldTrackedControls : o.finalHoldTrackedControls;
      const quiet = initial ? this.quietHoldControls : this.quietSettlingControls;
      this.quietReviews[initial ? 'initialHold' : 'finalHold'] = { trackedControls: tracked, quietControls: quiet,
        totalControls: tracked + quiet, maxControls: initial ? o.initialHoldControls : o.finalHoldControls,
        terminatedQuiet: this.quietMonitor.satisfied, ...this.quietMonitor.summary(),
        ...(!initial && this.settlingPolicy ? { policy: { mode: this.quietPolicyMode, requestQueuedAtStart: this.settlingPolicy.requestQueuedAtStart,
          decision: this.settlingDecision ? { ...this.settlingDecision } : null } } : {}) };
    }
    if (initial) { this.phase = 'teacher_exit_retreat'; this.entered = true; }
    else { this.phase = 'complete'; this.completionReason = 'finished'; this.completionPending = true; this.entered = false; }
  }

  step(proprio) {
    if (HOLDING.has(this.phase) || this.completionPending) {
      proprioception(proprio);
      if (proprio.objectBodyName && proprio.objectBodyName !== this.objectBodyName) throw new Error('Placed object changed during box exit');
      this._observe(proprio);
      if (this.quietHold && HOLDING.has(this.phase) && this.phase !== 'teacher_exit_retreat' && !RELEASE.has(this.phase)) this._observeQuiet();   // v6g: the release never feeds the hold's quiet window
      // A quiet extension ends before this control is issued, once the window
      // of consecutive quiet controls has been observed (same rule as the
      // carry sequence's settling). The maximum is enforced in advance().
      if (this.phase === 'teacher_exit_quiet_settling' && this.settlingPolicy) {
        if (this._settlingPolicyEnds()) this._closeHold(false);
      } else if (QUIET.has(this.phase) && this.quietMonitor.satisfied) this._closeHold(this.phase === 'teacher_exit_quiet_hold');
    }
    let referenceFrames = null;
    if (RELEASE.has(this.phase)) {
      this._observeRelease();   // v6g: may close the release (-> hold) before this control's reference is chosen
      if (RELEASE.has(this.phase)) referenceFrames = [this.releasePlan.frame, this.releasePlan.frame];
    }
    if (this.phase === 'teacher_exit_hold' || this.phase === 'teacher_exit_quiet_hold') referenceFrames = [this.holdPlan.frame, this.holdPlan.frame];
    if (this.phase === 'teacher_exit_retreat') {
      if (!this.worldFrames) {
        const aligned = turnReferenceTransform(this.retreatSkill.frames[0], proprio.rootPosWorld, proprio.rootQuatXyzwWorld);
        // v6e stand-off: the retreat is placed at the live root PLUS the same horizontal offset as the hold (null => `aligned` unchanged, v5).
        const transform = this.standOffWorld ? { ...aligned, translation: aligned.translation.map((v, i) => v + this.standOffWorld[i]) } : aligned;
        this.referencePlan = { transform, ...(this.standOffWorld ? { standOffWorld: Array.from(this.standOffWorld) } : {}) };
        this.worldFrames = this.retreatSkill.frames.map(frame => transformTeacherReference(frame, transform));
        this.goal = Array.from(this.worldFrames[this.sourceFrames - 1].slice(0, 3));
        this._observe(proprio);
      }
      referenceFrames = [this.worldFrames[this.phaseControls + 1], this.worldFrames[this.phaseControls + 16]];
    }
    if (this.phase === 'teacher_exit_settling' || this.phase === 'teacher_exit_quiet_settling') {
      this.finalHoldPlan ||= this._fixed(this.worldFrames[this.sourceFrames - 1], proprio);
      referenceFrames = [this.finalHoldPlan.frame, this.finalHoldPlan.frame];
    }
    const justCompleted = this.completionPending;
    const result = { phase: this.phase, mode: referenceFrames ? 'teacher' : 'none',
      supported: this.phase !== 'unsupported', referenceFrames, locomotionOnly: this.locomotionOnly,
      justEnteredTeacher: this.entered && Boolean(referenceFrames), justCompleted,
      referenceIndex: this.referenceIndex, sourceFrames: this.sourceFrames, recordClock: this.recordClock,
      requestedRootGoalWorld: this.requestedRootGoalWorld, completionReason: this.completionReason,
      outcome: this.outcome ? structuredClone(this.outcome) : null };
    this.entered = this.completionPending = false;
    this.pendingAdvance = Boolean(referenceFrames);
    return result;
  }

  advance() {
    if (!this.pendingAdvance || !HOLDING.has(this.phase)) return;
    this.pendingAdvance = false;
    if (QUIET.has(this.phase)) {
      // Quiet extension: same hold reference, ended once quiet for a full
      // window (never before the tracked minimum, never past the maximum).
      const initial = this.phase === 'teacher_exit_quiet_hold';
      const count = initial ? ++this.quietHoldControls : ++this.quietSettlingControls;
      if (count >= this._phaseDuration()) {
        if (!initial && this.settlingPolicy) this._settlingPolicyEnds();   // B2 telemetry: records 'max_controls_reached'
        this._closeHold(initial);
      }
      return;
    }
    if (RELEASE.has(this.phase)) {
      // v6g: release controls are counted apart from the recorded programme; the cap closes the phase without clearance (the preview veto stays).
      if (++this.releaseControls >= this.releaseLift.maxControls) this._closeRelease(false);
      return;
    }
    this.phaseControls++; this.totalControls++;
    if (this.phase === 'teacher_exit_retreat') this.retreatControls++;
    if (this.phaseControls < this._phaseDuration()) return;
    this.phaseControls = 0;
    if (this.phase === 'teacher_exit_hold') {
      if (this.quietHold && !this.quietMonitor.satisfied && this._phaseDuration('teacher_exit_quiet_hold') > 0) {
        this.phase = 'teacher_exit_quiet_hold'; this.entered = false; return;
      }
      this._closeHold(true);
    } else if (this.phase === 'teacher_exit_retreat') {
      this.phase = 'teacher_exit_settling'; this.entered = true; this.quietMonitor?.reset();
      if (this.quietPolicyMode && this.quietFinalHold !== false) {
        const o = this.options;
        this.settlingPolicy = createQuietExitHoldState({ phase: 'teacher_exit_settling', mode: this.quietPolicyMode,
          minControls: o.finalHoldTrackedControls, maxControls: o.finalHoldControls, window: this.quietHold.window,
          requestQueuedAtStart: this._requestQueued() });
        this.settlingDecision = null;
      }
    } else {
      if (this.settlingPolicy) {
        // B2: the tracked minimum has run; the policy (not the bare quiet monitor) decides whether the
        // settling ends now or continues as the quiet extension up to the original maximum.
        if (!this._settlingPolicyEnds() && this._phaseDuration('teacher_exit_quiet_settling') > 0) {
          this.phase = 'teacher_exit_quiet_settling'; this.entered = false; return;
        }
      } else if (this.quietHold && !this.quietMonitor.satisfied && this._phaseDuration('teacher_exit_quiet_settling') > 0) {
        this.phase = 'teacher_exit_quiet_settling'; this.entered = false; return;
      }
      this._closeHold(false);
    }
  }
}
