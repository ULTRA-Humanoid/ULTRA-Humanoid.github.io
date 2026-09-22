// Phase B B5 v6g (2026-09-21, coordinator GO after pnat2 section 12.5): RELEASE-AND-LIFT exit micro-phase. PURE GEOMETRY + option validation,
// no controller state. Finding (B5 section 12.5, 4/4 v6f cells): after the suitcase is set down STANDING (0.425 m tall, top face ~0.435 m) the
// largebox-calibrated exit hold keeps both hands beside its side faces (hand centres 0.38-0.39 m high, ~5 cm below the top) and the restricted-
// control preview refuses `preview_contact` on the previewed retraction toward the hold pose: the releasing hand brushes the side face. A horizontal
// stand-off (v6e/v6f) does not open that contact. Rule (B5 section 11 item 1): BEFORE teacher_exit_hold, hold the SAME terminal pose with each hand
// chain displaced UP (so the hand collision box's lowest point clears the object's top face by a margin) and OUT (horizontally away from the object
// centre through the hand), contact flags of the displaced bodies cleared, until the caller's contact measurement reports the human<->object normal
// force quiet for a window (never before minControls, never past maxControls). Object-agnostic: only the live object pose/extent and the terminal
// frame are read. The largebox declares no rule (object_profiles.js exitRelease null) => the v5 exit path is untouched (PERF trace identity).
import { TEACHER_HUMAN_BODY_NAMES } from './teacher_obs.js';
import { standingHalfExtentM } from './object_exit_standoff.js';
import { QUIET_ENDING_LIMITS } from './quiet_ending.js';

export const EXIT_RELEASE_PHASE = 'teacher_exit_release';
/** g1 rubber-hand collision box half sizes (public/g1_scene.xml, both hands): the hand's live vertical half extent is derived from them. */
export const HAND_BOX_HALF_SIZES_M = Object.freeze([0.026, 0.062, 0.074]);
export const EXIT_RELEASE_DEFAULTS = Object.freeze({
  clearanceMarginM: 0.03,                                  // hand box bottom -> object top face after the lift
  outM: 0.05,                                              // horizontal displacement away from the object centre (hand half thickness 0.026 + margin)
  minControls: 10, clearWindow: 10, maxControls: 60,       // at most one initial-hold length (teacher_box_exit_controller initialHoldControls 60)
  humanForceMaximumN: QUIET_ENDING_LIMITS.humanForceMaximumN,   // 0.1 N: the v5 quiet-ending human<->object limit, single source
});
/** Hand chain displaced with each hand: the distal wrist links move with the hand, the elbow by half (kinematic coherence of the reference). */
export const HAND_CHAINS = Object.freeze({
  left_rubber_hand: Object.freeze({ full: Object.freeze(['left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'left_rubber_hand']), half: Object.freeze(['left_elbow_link']) }),
  right_rubber_hand: Object.freeze({ full: Object.freeze(['right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand']), half: Object.freeze(['right_elbow_link']) }),
});
const BODY_INDEX = new Map(TEACHER_HUMAN_BODY_NAMES.map((name, i) => [name, i]));
const finite3 = v => (Array.isArray(v) || ArrayBuffer.isView(v)) && v.length === 3 && Array.from(v).every(Number.isFinite);
const unitXyzw = q => (Array.isArray(q) || ArrayBuffer.isView(q)) && q.length === 4 && Array.from(q).every(Number.isFinite) && Math.abs(Math.hypot(...q) - 1) <= 1e-2;
const positive = (v, label) => { if (!Number.isFinite(v) || v <= 0) throw new Error(`Release-and-lift ${label} must be a positive finite number`); return v; };
const wholeCount = (v, label) => { if (!Number.isInteger(v) || v < 1) throw new Error(`Release-and-lift ${label} must be a positive whole number of controls`); return v; };

/** World-vertical half extent of the hand collision box for the hand's xyzw quaternion: sum_k |R_zk| * half_k (third rotation-matrix row). */
export function handVerticalHalfExtentM(quatXyzw) {
  if (!unitXyzw(quatXyzw)) throw new Error('Release-and-lift requires a finite unit xyzw hand quaternion');
  const [x, y, z, w] = quatXyzw;
  const rz = [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  return rz.reduce((s, c, k) => s + Math.abs(c) * HAND_BOX_HALF_SIZES_M[k], 0);
}

/** Validate the option object spread into TeacherBoxExitController.start() as `releaseLift` (fail closed on every field). */
export function validateExitRelease(option) {
  if (!option || typeof option !== 'object') throw new Error('Release-and-lift option must be an object');
  if (!Array.isArray(option.hands) || option.hands.length < 1 || option.hands.length > 2 || new Set(option.hands).size !== option.hands.length
      || !option.hands.every(h => Object.prototype.hasOwnProperty.call(HAND_CHAINS, h))) throw new Error('Release-and-lift hands must name distinct rubber hands');
  positive(option.standingHalfExtentM, 'standingHalfExtentM'); positive(option.clearanceMarginM, 'clearanceMarginM'); positive(option.outM, 'outM');
  positive(option.humanForceMaximumN, 'humanForceMaximumN');
  const min = wholeCount(option.minControls, 'minControls'), win = wholeCount(option.clearWindow, 'clearWindow'), max = wholeCount(option.maxControls, 'maxControls');
  if (min > max || win > max) throw new Error('Release-and-lift minControls and clearWindow must not exceed maxControls');
  if (typeof option.measure !== 'function') throw new Error('Release-and-lift requires a synchronous contact measure() function');
  return option;
}

/** Build the start() option from a profile: null when the profile declares no rule (largebox/plasticbox/smallbox); otherwise the frozen option
 *  with the live standing half extent (same geometry as the exit stand-off) and the caller's measure(). Throws on unusable poses (fail closed). */
export function exitReleaseOption(profile, proprio, { measure } = {}) {
  if (!profile || typeof profile !== 'object') throw new Error('Release-and-lift requires an object profile');
  if (profile.exitRelease === null || profile.exitRelease === undefined) return null;
  const standing = standingHalfExtentM(profile, proprio?.objQuatXyzwWorld);
  if (!finite3(proprio?.objPosWorld)) throw new Error('Release-and-lift requires a finite object position');
  return validateExitRelease(Object.freeze({ hands: Array.from(profile.exitRelease.hands), standingHalfExtentM: standing, ...EXIT_RELEASE_DEFAULTS, measure }));
}

/** The release frame: the terminal frame with each declared hand chain displaced up/out and its contact flags cleared. Pure; returns a report per hand. */
export function planExitReleaseFrame(terminalFrame, { objectPosWorld, hands, standingHalfExtentM: standing, clearanceMarginM, outM }) {
  if (!terminalFrame || terminalFrame.length !== 747 || !Array.from(terminalFrame).every(Number.isFinite)) throw new Error('Release-and-lift requires a finite 747-channel terminal frame');
  if (!finite3(objectPosWorld)) throw new Error('Release-and-lift requires a finite object position');
  if (!Array.isArray(hands) || hands.length < 1) throw new Error('Release-and-lift requires at least one hand');
  positive(standing, 'standingHalfExtentM'); positive(clearanceMarginM, 'clearanceMarginM'); positive(outM, 'outM');
  const frame = Float32Array.from(terminalFrame), topZ = objectPosWorld[2] + standing, report = {};
  for (const hand of hands) {
    const chain = HAND_CHAINS[hand]; if (!chain) throw new Error(`Release-and-lift: unknown hand ${hand}`);
    const hi = BODY_INDEX.get(hand), pos = Array.from(terminalFrame.slice(84 + 3 * hi, 87 + 3 * hi)), quat = Array.from(terminalFrame.slice(201 + 4 * hi, 205 + 4 * hi));
    const vertical = handVerticalHalfExtentM(quat);
    const liftM = Math.max(0, topZ + clearanceMarginM - (pos[2] - vertical));
    const dx = pos[0] - objectPosWorld[0], dy = pos[1] - objectPosWorld[1], d = Math.hypot(dx, dy);
    if (d < 1e-3) throw new Error('Release-and-lift out direction undefined: the hand is above the object centre');
    const delta = [dx / d * outM, dy / d * outM, liftM];
    const move = (name, scale) => {
      const i = BODY_INDEX.get(name); if (i === undefined) throw new Error(`Release-and-lift: unknown body ${name}`);
      for (let k = 0; k < 3; k++) frame[84 + 3 * i + k] = terminalFrame[84 + 3 * i + k] + scale * delta[k];
      frame[591 + i] = 0;
    };
    chain.full.forEach(name => move(name, 1)); chain.half.forEach(name => move(name, 0.5));
    report[hand] = Object.freeze({ terminalWorld: pos, targetWorld: Array.from(frame.slice(84 + 3 * hi, 87 + 3 * hi)), liftM, outM, handVerticalHalfExtentM: vertical,
      objectTopZ: topZ, clearanceAboveTopM: (pos[2] + liftM - vertical) - topZ, terminalContactFlag: terminalFrame[591 + hi] });
  }
  return Object.freeze({ frame, hands: Object.freeze(report), objectTopZ: topZ });
}

// v6h (2026-09-21, B5 section 14): EXIT-HOLD HAND CLAMP. The v6g release micro-phase was vacuous (B5 13.8): the hands start the exit at ~0.70 m and the
// terminal hold pose PULLS THEM DOWN to the grasp height beside the object (0.38 m), crossing a standing suitcase's 0.435 m top face at ~58 % of the descent.
// The clamp uses the SAME frame geometry as the release frame (hand chains at top + margin + hand half extent, 5 cm out of the centre) as the INITIAL HOLD
// reference itself: no micro-phase, no contact criterion. The retreat clip is untouched (its hands stay >= 0.14 m above the top, replay 14.1).
export function validateExitHandClamp(option) {
  if (!option || typeof option !== 'object') throw new Error('Exit hand clamp option must be an object');
  if (!Array.isArray(option.hands) || option.hands.length < 1 || option.hands.length > 2 || new Set(option.hands).size !== option.hands.length
      || !option.hands.every(h => Object.prototype.hasOwnProperty.call(HAND_CHAINS, h))) throw new Error('Exit hand clamp hands must name distinct rubber hands');
  positive(option.standingHalfExtentM, 'standingHalfExtentM'); positive(option.clearanceMarginM, 'clearanceMarginM'); positive(option.outM, 'outM');
  for (const k of Object.keys(option)) if (!['hands', 'standingHalfExtentM', 'clearanceMarginM', 'outM'].includes(k)) throw new Error('Exit hand clamp option has an unknown field: ' + k);
  return option;
}
/** Build the hold hand-clamp option from a profile (null when the profile declares no `exitHandClamp`); the live standing half extent as for the stand-off. */
export function exitHandClampOption(profile, proprio) {
  if (!profile || typeof profile !== 'object') throw new Error('Exit hand clamp requires an object profile');
  if (profile.exitHandClamp === null || profile.exitHandClamp === undefined) return null;
  const standing = standingHalfExtentM(profile, proprio?.objQuatXyzwWorld);
  if (!finite3(proprio?.objPosWorld)) throw new Error('Exit hand clamp requires a finite object position');
  // v6i: a profile may declare its own DATA-DERIVED clearance margin (object_profiles.js SUITCASE_EXIT_HAND_CLAMP_MARGIN_M); absent => the v6h default. Fail closed on a non-positive value.
  const declared = profile.exitHandClamp.clearanceMarginM;
  if (declared !== undefined && !(Number.isFinite(declared) && declared > 0)) throw new Error('Exit hand clamp clearanceMarginM must be a positive finite number when declared');
  return validateExitHandClamp(Object.freeze({ hands: Array.from(profile.exitHandClamp.hands), standingHalfExtentM: standing, clearanceMarginM: declared ?? EXIT_RELEASE_DEFAULTS.clearanceMarginM, outM: EXIT_RELEASE_DEFAULTS.outM }));
}
/** The clamped hold frame = the release frame geometry (same function; the name states the v6h use). */
export const planExitHandClampFrame = planExitReleaseFrame;
