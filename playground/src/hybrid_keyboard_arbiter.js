// hybrid_keyboard_arbiter.js — decides, once per control and before any
// reference is built, whether the recorded supervisor (RestrictedLocomotionController
// + teacher) or the student translator (GoalTranslator + student policy) owns the
// held keyboard command. It never touches physics, references, observations or the
// policies; main.js applies the decision. Box tasks, floor clicks and any active
// skill always stay with the recorded supervisor.
import { studentKeyboardAdmission, HYBRID_GATE_DEFAULTS } from './keyboard_clearance_gate.js';
import { HYBRID_ENVELOPE_DEFAULTS, commandFromKeys, rampCap, clearanceLimitedCap, translationScale,
  StartupBackwardDetector, RunawayBackwardDetector, commandCap, isReversal, StopController, SpeedAverage } from './keyboard_speed_envelope.js';

const MOVING_PHASES = new Set(['teacher_step', 'teacher_turn']);
const READY_PHASES = new Set(['teacher_standing', 'teacher_settling', 'inactive']);
const STATUS = Object.freeze({
  student: 'Student policy is walking (hybrid). Release the keys to stop.',
  student_turn: 'Student policy is turning (hybrid). Release the keys to stop.',
  student_warmup: 'Student policy taking over…',
  stop_release: 'Slowing down, then handing to the measured standing…',
  stop_clearance: 'Approaching a box: slowing down before switching to measured steps…',
  stop_reversal: 'Slowing down before reversing direction…',
  stop_overspeed: 'Too fast: slowing down before continuing…',
  startup_backward_drift: 'The student drifted backward on start-up; finishing a measured step first.',
  runaway_backward: 'The student ran backward against the command; measured steps take over.',
  overspeed_lock: 'Too fast twice; measured steps take over for this key press.',
  clearance: 'Near a box: measured steps and turns only.',
  startup_fallback_lock: 'Measured step in progress before the student may resume.',
});

export class HybridKeyboardArbiter {
  constructor({ gate = {}, envelope = {}, maxStartupFallbacks = 2, warmupControls = 30 } = {}) {
    this.gateOptions = Object.freeze({ ...HYBRID_GATE_DEFAULTS, ...gate });
    this.envelopeOptions = Object.freeze({ ...HYBRID_ENVELOPE_DEFAULTS, ...envelope });
    if (!Number.isInteger(maxStartupFallbacks) || maxStartupFallbacks < 0) throw new Error('A whole start-up fallback limit is required');
    this.maxStartupFallbacks = maxStartupFallbacks;
    if (!Number.isInteger(warmupControls) || warmupControls < 0) throw new Error('A whole student warm-up length is required');
    // Student IDLE (stand) controls after each hand-over before the held keys are applied, so the
    // student's observation history holds its own actions rather than the teacher's.
    this.warmupControls = warmupControls;
    this.startup = new StartupBackwardDetector(this.envelopeOptions);
    this.runaway = new RunawayBackwardDetector(this.envelopeOptions);
    this.stop = new StopController(this.envelopeOptions);
    this.speed = new SpeedAverage(this.envelopeOptions);
    this.reset();
  }

  reset() {
    this.state = 'recorded'; this.studentStart = null; this.afterStop = 'recorded'; this.warmupUntil = null;
    this.startupFallbacks = 0; this.fallbackLock = false; this.sawRecordedMotion = false; this.overspeedStops = 0;
    this.keyFingerprint = ''; this.last = null; this.transitions = []; this.studentEntries = 0;
    this.startup.reset(); this.runaway.reset(); this.stop.reset(); this.speed.reset();
  }

  _transition(state, control, reason) {
    if (state !== this.state) { this.transitions.push({ control, from: this.state, to: state, reason }); if (this.transitions.length > 64) this.transitions.shift(); }
    this.state = state;
  }

  _admission(command, root, yaw, rects, controlsSinceStart) {
    const ramp = commandCap(command, rampCap(controlsSinceStart, this.envelopeOptions), this.envelopeOptions);
    const first = studentKeyboardAdmission({ root, yaw, command, rects, capSpeedMps: ramp, options: this.gateOptions });
    if (first.kind !== 'directional') return { ...first, rampCapMps: ramp, capSpeedMps: ramp };
    // Ramp down toward the box: the cap follows the free travel, never below the minimum student speed.
    const cap = clearanceLimitedCap(ramp, first.freeTravelM, this.gateOptions.horizonS, this.envelopeOptions);
    if (cap === ramp) return { ...first, rampCapMps: ramp, capSpeedMps: ramp };
    const limited = studentKeyboardAdmission({ root, yaw, command, rects, capSpeedMps: cap, options: this.gateOptions });
    return { ...limited, rampCapMps: ramp, capSpeedMps: cap };
  }

  /** input: { control, held, root: { pos, yaw, velWorld }, rects, restricted: { phase, initialStandingPending, intentType, suspended } } */
  decide(input) {
    const { control, held, root, rects, restricted } = input;
    if (!Number.isInteger(control) || !root || !Number.isFinite(root.yaw) || !(Array.isArray(root.pos) || ArrayBuffer.isView(root.pos))) throw new Error('A control index and live root are required');
    const command = commandFromKeys(held);
    const fingerprint = ['forward', 'backward', 'left', 'right', 'turnLeft', 'turnRight'].map(k => Number(Boolean(held?.[k]))).join('');
    const keysChanged = fingerprint !== this.keyFingerprint;
    if (keysChanged) { this.keyFingerprint = fingerprint; this.startupFallbacks = 0; this.fallbackLock = false; this.sawRecordedMotion = false; this.overspeedStops = 0; this.runaway.reset(); }
    const c = Math.cos(root.yaw), s = Math.sin(root.yaw), vx = root.velWorld[0], vy = root.velWorld[1];
    const vForward = c * vx + s * vy, vLateral = -s * vx + c * vy, speed = Math.hypot(vx, vy);
    this.speed.observe(speed);
    const restrictedMoving = MOVING_PHASES.has(restricted.phase);
    if (restrictedMoving) this.sawRecordedMotion = true;
    const recordedOnly = Boolean(restricted.suspended) || restricted.intentType === 'floor' || Boolean(restricted.skillActive);
    const handoffReady = !recordedOnly && !restricted.initialStandingPending && READY_PHASES.has(restricted.phase ?? 'inactive')
      && speed <= this.envelopeOptions.stopSpeedMps;
    const pos = [root.pos[0], root.pos[1]];
    let decision;
    const recorded = (reason, gate = null, reanchor = false) => {
      this._transition('recorded', control, reason);
      return { owner: 'recorded', shaping: null, reason, status: STATUS[reason] ?? null, gate, reanchor, enteredStudent: false };
    };
    const student = (gate) => {
      const entered = this.state !== 'student';
      if (entered) { this._transition('student', control, 'student_admitted'); this.studentStart = control; this.warmupUntil = control + this.warmupControls; this.startup.reset(); this.runaway.reset(); this.studentEntries++; }
      if (control < this.warmupUntil) return { owner: 'student', warmup: true, shaping: null, reason: 'student_warmup', status: STATUS.student_warmup,
        gate, reanchor: false, enteredStudent: entered, warmupControls: this.warmupUntil - control };
      // The fixed student turns 3-6x faster than encoded for Q/E alone, but Q/E+W chords turned at the encoded
      // rate in the baseline and stopped turning when scaled; scale pure turns only.
      const pureTurn = Boolean(command.turn) && !command.forward && !command.lateral;
      const shaping = { translationScale: translationScale(command, gate.capSpeedMps), yawScale: pureTurn ? this.envelopeOptions.yawScale : 1, capSpeedMps: gate.capSpeedMps };
      return { owner: 'student', warmup: false, shaping, reason: 'student', status: command.turn && !command.forward && !command.lateral ? STATUS.student_turn : STATUS.student,
        gate, reanchor: false, enteredStudent: entered };
    };
    // Leaving a stop: resume the student (reversal / overspeed, or a release re-pressed) when admissible, else hand to the recorded supervisor.
    const finishStop = () => {
      const stopReason = this.stop.reason;
      const resume = command.anyMotion && (this.afterStop === 'student' || stopReason === 'stop_release');
      const gate = command.anyMotion ? this._admission(command, pos, root.yaw, rects, 0) : null;
      this.stop.reset();
      // A clearance stop hands the still-held key to the recorded supervisor even if the slower
      // cap would now pass the gate; the student is re-admitted only once that record has finished.
      return resume && gate.admissible ? student(gate) : recorded(!command.anyMotion ? 'stopped' : stopReason === 'stop_overspeed' ? 'overspeed_lock' : gate.admissible ? 'recorded_after_stop' : 'clearance', gate, true);
    };
    const stopping = (reason, afterStop) => {
      if (this.state !== 'stopping') { this._transition('stopping', control, reason); this.stop.begin(control, reason); }
      this.afterStop = afterStop;
      // Already slow enough (e.g. a release at walking speed): hand over on this very control, without a student IDLE control.
      if (this.stop.shouldExit({ control, speedMps: speed })) return finishStop();
      return { owner: 'stop', shaping: null, reason, status: STATUS[reason] ?? STATUS.stop_release, gate: null, reanchor: false, enteredStudent: false,
        stopControls: this.stop.elapsed(control) };
    };
    if (recordedOnly && this.state !== 'recorded') {
      // A floor click or suspension always returns ownership to the recorded supervisor.
      this.stop.reset();
      decision = recorded('recorded_only', null, true);
    } else if (this.state === 'recorded') {
      if (!command.anyMotion) decision = recorded(command.anyKey ? 'opposed_keys' : 'no_keys');
      else if (!handoffReady) decision = recorded(restrictedMoving ? 'recorded_motion_in_progress' : 'recorded_not_ready');
      else if (this.fallbackLock && !(this.sawRecordedMotion && this.startupFallbacks < this.maxStartupFallbacks)) decision = recorded('startup_fallback_lock');
      else {
        const gate = this._admission(command, pos, root.yaw, rects, 0);
        decision = gate.admissible ? student(gate) : recorded('clearance', gate);
      }
    } else if (this.state === 'student') {
      const since = Math.max(0, control - Math.max(this.studentStart, this.warmupUntil ?? this.studentStart));
      if (!command.anyMotion) decision = stopping('stop_release', 'recorded');
      else {
        const gate = this._admission(command, pos, root.yaw, rects, since);
        if (!gate.admissible) decision = stopping('stop_clearance', 'recorded');
        else if (control < this.warmupUntil) decision = student(gate);
        else if (this.startup.observe({ controlsSinceStart: since, forwardHeld: command.forward > 0, backwardHeld: command.forward < 0,
            lateralHeld: command.lateral !== 0, turnHeld: command.turn !== 0, vForwardMps: vForward })) {
          this.startupFallbacks++; this.fallbackLock = true; this.sawRecordedMotion = false;
          decision = recorded('startup_backward_drift', gate, true);
        } else if (this.runaway.observe({ forwardHeld: command.forward > 0, vForwardMps: vForward })) {
          this.startupFallbacks++; this.fallbackLock = true; this.sawRecordedMotion = false;
          decision = recorded('runaway_backward', gate, true);
        } else if (keysChanged && isReversal({ vForwardMps: vForward, vLateralMps: vLateral, command, o: this.envelopeOptions })) decision = stopping('stop_reversal', 'student');
        else if (this.speed.overspeed) {
          this.overspeedStops++;
          if (this.overspeedStops > this.envelopeOptions.overspeedStopsPerPress) { this.fallbackLock = true; this.sawRecordedMotion = false; this.startupFallbacks = this.maxStartupFallbacks; decision = stopping('stop_overspeed', 'recorded'); }
          else decision = stopping('stop_overspeed', 'student');
        } else decision = student(gate);
      }
    } else {
      // stopping: student IDLE until slow or timed out, then hand over.
      if (!command.anyMotion) this.afterStop = 'recorded';
      decision = this.stop.shouldExit({ control, speedMps: speed }) ? finishStop() : stopping(this.stop.reason, this.afterStop);
    }
    decision.command = { forward: command.forward, lateral: command.lateral, turn: command.turn };
    decision.speedMps = speed; decision.vForwardMps = vForward; decision.control = control;
    this.last = decision;
    return decision;
  }

  snapshot() {
    const d = this.last;
    return { state: this.state, owner: d?.owner ?? null, reason: d?.reason ?? null, control: d?.control ?? null,
      studentStart: this.studentStart, warmupUntil: this.warmupUntil, warmup: d?.warmup ?? false, studentEntries: this.studentEntries, overspeedStops: this.overspeedStops, startupFallbacks: this.startupFallbacks, fallbackLock: this.fallbackLock,
      stopControls: this.stop.active ? d?.stopControls ?? null : null, stopReason: this.stop.reason,
      shaping: d?.shaping ? { ...d.shaping } : null,
      gate: d?.gate ? { kind: d.gate.kind, admissible: d.gate.admissible, capSpeedMps: d.gate.capSpeedMps, rampCapMps: d.gate.rampCapMps,
        freeTravelM: d.gate.freeTravelM ?? null, requiredTravelM: d.gate.requiredTravelM ?? null,
        requiredClearanceM: d.gate.requiredClearanceM ?? null, minClearanceM: d.gate.minClearanceM ?? null, limitingIndex: d.gate.limitingIndex ?? null } : null,
      speedMps: d?.speedMps ?? null, speedEmaMps: this.speed.value, vForwardMps: d?.vForwardMps ?? null,
      transitions: this.transitions.slice(-16), options: { gate: this.gateOptions, envelope: this.envelopeOptions, maxStartupFallbacks: this.maxStartupFallbacks, warmupControls: this.warmupControls } };
  }
}

/** Read optional URL overrides for the hybrid knobs (all default to the module constants). */
export function readHybridKeyboardOptions(params) {
  const num = (name, fallback) => { const raw = params.get(name); const v = raw === null ? NaN : Number(raw); return Number.isFinite(v) ? v : fallback; };
  return {
    gate: { clearanceM: num('hybridClearanceM', HYBRID_GATE_DEFAULTS.clearanceM), footprintM: num('hybridFootprintM', HYBRID_GATE_DEFAULTS.footprintM),
      horizonS: num('hybridHorizonS', HYBRID_GATE_DEFAULTS.horizonS), turnDriftMps: num('hybridTurnDriftMps', HYBRID_GATE_DEFAULTS.turnDriftMps) },
    envelope: { startSpeedMps: num('hybridStartSpeedMps', HYBRID_ENVELOPE_DEFAULTS.startSpeedMps), fullSpeedMps: num('hybridFullSpeedMps', HYBRID_ENVELOPE_DEFAULTS.fullSpeedMps),
      rampControls: num('hybridRampControls', HYBRID_ENVELOPE_DEFAULTS.rampControls), minStudentSpeedMps: num('hybridMinStudentSpeedMps', HYBRID_ENVELOPE_DEFAULTS.minStudentSpeedMps),
      yawScale: num('hybridYawScale', HYBRID_ENVELOPE_DEFAULTS.yawScale), stopMaxControls: num('hybridStopMaxControls', HYBRID_ENVELOPE_DEFAULTS.stopMaxControls),
      stopSpeedMps: num('hybridStopSpeedMps', HYBRID_ENVELOPE_DEFAULTS.stopSpeedMps),
      releaseStopMaxControls: num('hybridReleaseStopMaxControls', HYBRID_ENVELOPE_DEFAULTS.releaseStopMaxControls),
      releaseHandoverSpeedMps: num('hybridReleaseHandoverSpeedMps', HYBRID_ENVELOPE_DEFAULTS.releaseHandoverSpeedMps),
      overspeedMps: num('hybridOverspeedMps', HYBRID_ENVELOPE_DEFAULTS.overspeedMps),
      runawayBackwardMps: num('hybridRunawayBackwardMps', HYBRID_ENVELOPE_DEFAULTS.runawayBackwardMps),
      runawayConsecutive: num('hybridRunawayConsecutive', HYBRID_ENVELOPE_DEFAULTS.runawayConsecutive),
      backwardCapMps: num('hybridBackwardCapMps', HYBRID_ENVELOPE_DEFAULTS.backwardCapMps),
      overspeedStopsPerPress: num('hybridOverspeedStopsPerPress', HYBRID_ENVELOPE_DEFAULTS.overspeedStopsPerPress),
      startupBackwardMps: num('hybridStartupBackwardMps', HYBRID_ENVELOPE_DEFAULTS.startupBackwardMps),
      startupConsecutive: num('hybridStartupConsecutive', HYBRID_ENVELOPE_DEFAULTS.startupConsecutive),
      startupWindowControls: num('hybridStartupWindowControls', HYBRID_ENVELOPE_DEFAULTS.startupWindowControls),
      startupMinControls: num('hybridStartupMinControls', HYBRID_ENVELOPE_DEFAULTS.startupMinControls),
      reversalMinSpeedMps: num('hybridReversalMinSpeedMps', HYBRID_ENVELOPE_DEFAULTS.reversalMinSpeedMps) },
    warmupControls: num('hybridWarmupControls', 30), maxStartupFallbacks: num('hybridMaxStartupFallbacks', 2),
  };
}

/** Held-key snapshot in the RestrictedLocomotionController's names from a UserState. */
export function heldKeysFromUser(user) {
  return { forward: Boolean(user.w), backward: Boolean(user.s), left: Boolean(user.a), right: Boolean(user.d), turnLeft: Boolean(user.q), turnRight: Boolean(user.e) };
}

/** A UserState view for GoalTranslator: same object semantics, with the hybrid shaping attached
 * (or every key released for the IDLE stop). Prototype delegation keeps humanGoalWorld/objGoalWorld/activeObjName live. */
export function studentUserView(user, decision) {
  const view = Object.create(user);
  if (decision.owner === 'stop' || decision.warmup) { for (const key of ['w', 's', 'a', 'd', 'q', 'e']) view[key] = false; }
  view.keyboardShaping = decision.owner === 'student' && !decision.warmup ? decision.shaping : null;
  return view;
}

/** Opt-in (hybridFallbackStep=1): when the runaway-backward fallback fires while a W+turn chord is
 * held, hand the recorded supervisor the chord WITHOUT its turn keys, so it runs a forward STEP
 * record (173 controls) instead of the pure 273-control TURN that Q/E precedence would select.
 * Returns the key snapshot to latch, or null when no re-latch is needed. Pure function. */
export function fallbackKeyView(user, decision, enabled) {
  if (!enabled || !decision || decision.owner !== 'recorded' || decision.reason !== 'runaway_backward') return null;
  if (!user?.w || !(user.q || user.e)) return null;
  return { ...user, q: false, e: false };
}
