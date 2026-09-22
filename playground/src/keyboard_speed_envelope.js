// keyboard_speed_envelope.js — command shaping for the fixed student policy in
// the hybrid keyboard mode. Only the 13-D LOCO query built by GoalTranslator is
// scaled; the policy, its observations and the physics are untouched.
//
// Encoded key magnitudes mirror goal_translator.js: W +1.0, S -BACK_FACTOR (0.5),
// A/D ±KEYBOARD_STRAFE_FACTOR (0.25), Q/E ±YAW_RATE (0.5 rad/s) before scaling.
export const HYBRID_ENVELOPE_DEFAULTS = Object.freeze({
  startSpeedMps: .6, fullSpeedMps: 1, rampControls: 60, minStudentSpeedMps: .4, yawScale: .3,
  // Start-up drift detector: armed only for a pure W command (chord pivots swing the body-frame forward
  // velocity negative for a few controls) and only after startupMinControls following the warm-up.
  startupWindowControls: 90, startupBackwardMps: -.15, startupConsecutive: 3, startupMinControls: 20, startupPureForwardOnly: true,
  // Runaway against the command: any time the student owns a command with a forward component, a sustained
  // body-frame backward velocity hands the keys to the recorded supervisor (K60 falls O21/O22/O23 developed
  // from a near standstill as −0.4 → −1.6 m/s over ~40 controls while Q/E+W was held).
  runawayBackwardMps: -.3, runawayConsecutive: 5,
  // The student's backward gait overshoots (S encoded 0.5 m/s realised 1.5–2.6 m/s, fall O02): cap it lower.
  backwardCapMps: .3, overspeedStopsPerPress: 1,
  reversalMinSpeedMps: .4, stopMaxControls: 30, stopSpeedMps: .35,
  // Release: hand to the measured (teacher) standing as soon as the planar speed is at most
  // releaseHandoverSpeedMps (a student IDLE phase accrued 11-23 deg of heading drift in the K60
  // diagnostics, the teacher standing holds heading), but never from a faster run: the teacher lost
  // balance taking over a 1.7 m/s backward walk, while a bounded student IDLE first was safe.
  releaseStopMaxControls: 60, releaseHandoverSpeedMps: 1.25,
  // Realised planar speed (EMA) above which the student is interrupted by a bounded stop.
  overspeedMps: 1.6, speedEmaAlpha: .2,
});
export const ENCODED_KEY = Object.freeze({ forward: 1, backward: .5, lateral: .25 });

function options(o) {
  const v = { ...HYBRID_ENVELOPE_DEFAULTS, ...o };
  if (!(v.startSpeedMps > 0 && v.fullSpeedMps >= v.startSpeedMps && v.rampControls >= 1 && v.minStudentSpeedMps > 0
      && v.minStudentSpeedMps <= v.startSpeedMps && v.yawScale > 0 && v.yawScale <= 1 && v.startupWindowControls >= 1
      && v.startupBackwardMps < 0 && v.startupConsecutive >= 1 && v.startupMinControls >= 0 && typeof v.startupPureForwardOnly === 'boolean' && v.reversalMinSpeedMps > 0 && v.stopMaxControls >= 1 && v.stopSpeedMps > 0
      && v.releaseStopMaxControls >= 0 && v.releaseHandoverSpeedMps > 0 && v.overspeedMps > 0 && v.speedEmaAlpha > 0 && v.speedEmaAlpha <= 1
      && v.runawayBackwardMps < 0 && v.runawayConsecutive >= 1 && v.backwardCapMps > 0 && v.overspeedStopsPerPress >= 0)) {
    throw new Error('Valid hybrid envelope options are required');
  }
  return v;
}

/** Held-key snapshot → cancelled command (opposed keys cancel) plus encoded magnitudes. */
export function commandFromKeys(held) {
  const b = name => Boolean(held?.[name]);
  const forward = Number(b('forward')) - Number(b('backward')), lateral = Number(b('left')) - Number(b('right')), turn = Number(b('turnLeft')) - Number(b('turnRight'));
  const encodedForward = forward > 0 ? ENCODED_KEY.forward : forward < 0 ? -ENCODED_KEY.backward : 0;
  const encodedLateral = lateral * ENCODED_KEY.lateral;
  return { forward, lateral, turn, encodedForward, encodedLateral, encodedNormMps: Math.hypot(encodedForward, encodedLateral),
    anyMotion: Boolean(forward || lateral || turn), anyKey: ['forward', 'backward', 'left', 'right', 'turnLeft', 'turnRight'].some(b) };
}

/** Commanded speed cap: startSpeed → fullSpeed over rampControls from a standing start. */
export function rampCap(controlsSinceStart, o = {}) {
  const v = options(o);
  const t = Math.max(0, Math.min(1, controlsSinceStart / v.rampControls));
  return v.startSpeedMps + (v.fullSpeedMps - v.startSpeedMps) * t;
}

/** Cap after clearance: the incoming cap, further limited by the free travel per horizon, floored at the minimum
 * student speed but never raised above the incoming cap (the backward cap can be below that floor). */
export function clearanceLimitedCap(rampCapMps, freeTravelM, horizonS, o = {}) {
  const v = options(o);
  if (!Number.isFinite(freeTravelM)) return rampCapMps;
  return Math.min(rampCapMps, Math.max(v.minStudentSpeedMps, freeTravelM / horizonS));
}

/** Uniform scale applied to the encoded forward/lateral query so its norm never exceeds the cap. */
export function translationScale(command, capSpeedMps) {
  if (!Number.isFinite(capSpeedMps) || capSpeedMps < 0) throw new Error('A finite speed cap is required');
  return command.encodedNormMps > capSpeedMps && command.encodedNormMps > 0 ? capSpeedMps / command.encodedNormMps : 1;
}

/** "W from standing → hesitation → backward acceleration" start-up mode detector. */
export class StartupBackwardDetector {
  constructor(o = {}) { this.options = options(o); this.reset(); }
  reset() { this.consecutive = 0; this.triggered = false; this.triggeredAt = null; }
  /** Returns true on the control where the detector fires. */
  observe({ controlsSinceStart, forwardHeld, backwardHeld, lateralHeld = false, turnHeld = false, vForwardMps }) {
    if (this.triggered) return false;
    const inWindow = controlsSinceStart >= this.options.startupMinControls && controlsSinceStart <= this.options.startupWindowControls;
    const pure = !this.options.startupPureForwardOnly || (!lateralHeld && !turnHeld);
    if (!inWindow || !forwardHeld || backwardHeld || !pure || !Number.isFinite(vForwardMps)) { this.consecutive = 0; return false; }
    this.consecutive = vForwardMps < this.options.startupBackwardMps ? this.consecutive + 1 : 0;
    if (this.consecutive >= this.options.startupConsecutive) { this.triggered = true; this.triggeredAt = controlsSinceStart; return true; }
    return false;
  }
}

/** Sustained body-frame backward velocity while a forward component is commanded (any chord with W, any time). */
export class RunawayBackwardDetector {
  constructor(o = {}) { this.options = options(o); this.reset(); }
  reset() { this.consecutive = 0; }
  observe({ forwardHeld, vForwardMps }) {
    if (!forwardHeld || !Number.isFinite(vForwardMps)) { this.consecutive = 0; return false; }
    this.consecutive = vForwardMps < this.options.runawayBackwardMps ? this.consecutive + 1 : 0;
    if (this.consecutive >= this.options.runawayConsecutive) { this.consecutive = 0; return true; }
    return false;
  }
}

/** Command cap: the ramp cap, or the backward cap for S-led commands (student backward gait overshoots). */
export function commandCap(command, rampCapMps, o = {}) {
  const v = options(o);
  return command.forward < 0 ? Math.min(rampCapMps, v.backwardCapMps) : rampCapMps;
}

/** A new translation command opposing the current body-frame velocity while moving fast. */
export function isReversal({ vForwardMps, vLateralMps, command, o = {} }) {
  const v = options(o);
  if (!command.forward && !command.lateral) return false;
  const speed = Math.hypot(vForwardMps, vLateralMps);
  if (speed < v.reversalMinSpeedMps) return false;
  return vForwardMps * command.encodedForward + vLateralMps * command.encodedLateral < 0;
}

/** Release / hand-off stop: student IDLE for a bounded number of controls or until slow. */
export class StopController {
  constructor(o = {}) { this.options = options(o); this.reset(); }
  reset() { this.startControl = null; this.reason = null; }
  begin(control, reason) { this.startControl = control; this.reason = reason; }
  get active() { return this.startControl !== null; }
  elapsed(control) { return this.active ? control - this.startControl : null; }
  /** Release stops hand over at releaseHandoverSpeedMps within releaseStopMaxControls; reversal / clearance / overspeed stops wait for stopSpeedMps within stopMaxControls. */
  maxControls() { return this.reason === 'stop_release' ? this.options.releaseStopMaxControls : this.options.stopMaxControls; }
  exitSpeed() { return this.reason === 'stop_release' ? this.options.releaseHandoverSpeedMps : this.options.stopSpeedMps; }
  shouldExit({ control, speedMps }) {
    if (!this.active) return true;
    return control - this.startControl >= this.maxControls() || speedMps <= this.exitSpeed();
  }
}

/** Exponential moving average of the realised planar speed for the overspeed guard. */
export class SpeedAverage {
  constructor(o = {}) { this.options = options(o); this.reset(); }
  reset() { this.value = null; }
  observe(speedMps) { this.value = this.value === null ? speedMps : this.value + this.options.speedEmaAlpha * (speedMps - this.value); return this.value; }
  get overspeed() { return this.value !== null && this.value > this.options.overspeedMps; }
}
