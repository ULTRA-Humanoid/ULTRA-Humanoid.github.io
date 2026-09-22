// Phase B · B2 — quiet exit-hold policy. DORMANT: not imported by main.js or by any
// controller. Pure functions only: no DOM, no physics, no controller calls.
//
// Scope. After a placed-box task the recorded exit runs three tracked phases
// (teacher_exit_hold 60 → teacher_exit_retreat 199 → teacher_exit_settling 180;
// TeacherBoxExitController). v5 already carries an optional quiet extension for the two
// holds (QuietHoldMonitor + quietHoldComplete in quiet_ending.js) but it is default-OFF
// because the v2/v3 release profile lost T20 H001 when the task-1 ending was shortened
// (LARGEBOX_V3_DIAGNOSIS_20260920.md §A). This module is the pure decision rule for a
// SAFER re-enablement that changes nothing before teacher_exit_hold:
//
//   • the same QUIET_ENDING_LIMITS object v5 defines (never a copy);
//   • a documented minimum (`minControls`) below which a hold never ends;
//   • a hold ends early only after `window` CONSECUTIVE quiet samples;
//   • hysteresis: once a disturbance (non-quiet sample) is observed at/after the
//     minimum, the required consecutive run becomes `window + rearmControls`;
//   • an `endHold` decision is latched (it never flips back to `continueHold`);
//   • a non-finite / missing sample is never quiet (it resets the run) — such a
//     stream can only end at `maxControls`, exactly like v5's fixed hold;
//   • mode `quiet_on_request` (the recommended B2 variant): the settling ends early
//     ONLY when a user request is pending AND the request arrived after the settling
//     started. A request already queued when the settling starts (every frozen T20
//     second click: issued at the retreat start, see PHASEB_B2_B6_DESIGN_20260921.md)
//     runs the full fixed settling, so every frozen P100/T20 control stream is
//     unchanged by construction.
//
// Frozen-input evidence that shaped the defaults (benchmarks/phaseb-smoothing/
// predict_quiet_exit_windows.py over the frozen c5 T20 and v5 P100 cells):
//   • teacher_exit_hold is NEVER quiet (0/163 windows): the hold tracks the crouched
//     set-down terminal frame, so `upright >= .95` cannot be met. A quiet initial
//     hold is therefore a structural no-op; the default keeps it FIXED at 60 (= v5).
//   • teacher_exit_settling is quiet at exactly sample 30 in 161/161 windows: with the
//     v5 minimum of 30 the quiet extension is a deterministic −150 controls per
//     ending — the exact T20 H001 mechanism. Hence `quiet_on_request`.
import { QUIET_ENDING_LIMITS, QUIET_ENDING_DEFAULTS, classifyQuietSample } from './quiet_ending.js';

export const QUIET_EXIT_HOLD_MODES = Object.freeze(['fixed', 'quiet', 'quiet_on_request']);

export const QUIET_EXIT_HOLD_DEFAULTS = Object.freeze({
  mode: 'quiet_on_request',
  window: QUIET_ENDING_DEFAULTS.window,            // 30 consecutive quiet samples
  rearmControls: 15,                               // hysteresis after a disturbance
  // v5 exit programme (TeacherBoxExitController defaults): hold 60, retreat 199, settling 180.
  initialHold: Object.freeze({ minControls: 60, maxControls: 60 }),   // fixed = v5 (never quiet)
  finalSettling: Object.freeze({ minControls: 30, maxControls: 180 }), // v5 quiet minimum, v5 maximum
  retreatControls: 199,
  limits: QUIET_ENDING_LIMITS,                     // the largebox default limits object itself
});

const whole = (value, minimum = 0) => Number.isInteger(value) && value >= minimum;

function validateBounds({ minControls, maxControls, window, rearmControls }) {
  if (!whole(minControls, 1) || !whole(maxControls, 1) || minControls > maxControls)
    throw new Error('Quiet exit hold needs whole minControls <= maxControls (both >= 1)');
  if (!whole(window, 1) || window > 1000) throw new Error('Quiet window must be 1–1000 controls');
  if (!whole(rearmControls, 0)) throw new Error('rearmControls must be a whole nonnegative count');
}

/** Immutable per-hold state. `requestQueuedAtStart` is read ONCE when the hold starts
 * (mode quiet_on_request): true → this hold runs to its maximum, exactly like v5. */
export function createQuietExitHoldState({ phase = 'teacher_exit_settling', mode = QUIET_EXIT_HOLD_DEFAULTS.mode,
  minControls, maxControls, window = QUIET_EXIT_HOLD_DEFAULTS.window,
  rearmControls = QUIET_EXIT_HOLD_DEFAULTS.rearmControls, limits = QUIET_ENDING_LIMITS,
  requestQueuedAtStart = false } = {}) {
  if (!QUIET_EXIT_HOLD_MODES.includes(mode)) throw new Error(`Unknown quiet exit hold mode: ${mode}`);
  if (!['teacher_exit_hold', 'teacher_exit_settling'].includes(phase)) throw new Error('Policy applies to teacher_exit_hold or teacher_exit_settling only');
  const bounds = phase === 'teacher_exit_hold' ? QUIET_EXIT_HOLD_DEFAULTS.initialHold : QUIET_EXIT_HOLD_DEFAULTS.finalSettling;
  const resolved = { minControls: minControls ?? bounds.minControls, maxControls: maxControls ?? bounds.maxControls, window, rearmControls };
  validateBounds(resolved);
  if (!limits || typeof limits !== 'object') throw new Error('Quiet limits object required');
  return Object.freeze({ phase, mode, ...resolved, limits, requestQueuedAtStart: requestQueuedAtStart === true,
    samples: 0, quietSamples: 0, consecutiveQuiet: 0, disturbedAfterMinimum: false,
    lastReasons: Object.freeze([]), ended: false, endReason: null, endControls: null });
}

/** Observe one measured sample (the state after the previous hold control). Pure: returns a
 * new state. `controls` = hold controls already issued when the sample was taken. */
export function observeQuietExitHold(state, sample, { controls } = {}) {
  if (!state || state.ended) return state;
  if (!whole(controls, 0)) throw new Error('controls (hold controls already issued) must be a whole count');
  const classified = classifyQuietSample(sample, state.limits);   // NaN/missing → quiet:false
  const consecutiveQuiet = classified.quiet ? state.consecutiveQuiet + 1 : 0;
  const disturbedAfterMinimum = state.disturbedAfterMinimum || (!classified.quiet && controls >= state.minControls);
  return Object.freeze({ ...state, samples: state.samples + 1, quietSamples: state.quietSamples + (classified.quiet ? 1 : 0),
    consecutiveQuiet, disturbedAfterMinimum, lastReasons: Object.freeze([...classified.reasons]) });
}

/** The required consecutive-quiet run: `window`, or `window + rearmControls` after a
 * disturbance seen at/after the minimum (hysteresis). */
export function requiredQuietRun(state) {
  return state.window + (state.disturbedAfterMinimum ? state.rearmControls : 0);
}

/** Decide before issuing hold control `controls + 1`. Pure. Returns
 * { decision: 'continueHold' | 'endHold', reason, ... }. A prior endHold is latched. */
export function decideQuietExitHold(state, { controls, requestPending = false } = {}) {
  if (!whole(controls, 0)) throw new Error('controls must be a whole count');
  const base = { controls, consecutiveQuiet: state.consecutiveQuiet, requiredRun: requiredQuietRun(state),
    minControls: state.minControls, maxControls: state.maxControls, mode: state.mode, lastReasons: state.lastReasons };
  if (state.ended) return Object.freeze({ ...base, decision: 'endHold', reason: state.endReason, latched: true });
  if (controls >= state.maxControls) return Object.freeze({ ...base, decision: 'endHold', reason: 'max_controls_reached' });
  if (controls < state.minControls) return Object.freeze({ ...base, decision: 'continueHold', reason: 'below_minimum' });
  if (state.mode === 'fixed') return Object.freeze({ ...base, decision: 'continueHold', reason: 'fixed_mode' });
  if (state.mode === 'quiet_on_request') {
    if (state.requestQueuedAtStart) return Object.freeze({ ...base, decision: 'continueHold', reason: 'request_queued_before_hold_start' });
    if (requestPending !== true) return Object.freeze({ ...base, decision: 'continueHold', reason: 'no_request_pending' });
  }
  if (state.consecutiveQuiet >= base.requiredRun) {
    return Object.freeze({ ...base, decision: 'endHold',
      reason: state.mode === 'quiet_on_request' ? 'quiet_window_satisfied_on_request' : 'quiet_window_satisfied' });
  }
  return Object.freeze({ ...base, decision: 'continueHold', reason: state.lastReasons.length ? `not_quiet:${state.lastReasons.join(',')}` : 'quiet_run_short' });
}

/** Latch an endHold decision into the state (the caller records why and when). */
export function latchQuietExitHoldEnd(state, decision) {
  if (decision?.decision !== 'endHold') return state;
  if (state.ended) return state;
  return Object.freeze({ ...state, ended: true, endReason: decision.reason, endControls: decision.controls });
}

/**
 * Run a whole synthetic sample stream. Sample i describes the state after hold control i
 * (i = 0 … samples.length−1; the sample taken at hold start is index 0). The hold ends at
 * the first control k (k ≥ 1) whose decision is endHold; otherwise at maxControls.
 * `requestPendingFrom` (control index, or null) models a user click arriving during the
 * hold; `requestQueuedAtStart` models a click already queued when the hold began.
 */
export function evaluateQuietExitHoldStream({ samples, requestPendingFrom = null, requestQueuedAtStart = false, ...options } = {}) {
  if (!Array.isArray(samples)) throw new Error('samples array required');
  let state = createQuietExitHoldState({ ...options, requestQueuedAtStart });
  const decisions = [];
  for (let controls = 0; controls <= state.maxControls; controls++) {
    if (controls < samples.length) state = observeQuietExitHold(state, samples[controls], { controls });
    const requestPending = requestPendingFrom !== null && controls >= requestPendingFrom;
    const decision = decideQuietExitHold(state, { controls, requestPending });
    decisions.push(decision);
    if (decision.decision === 'endHold') {
      state = latchQuietExitHoldEnd(state, decision);
      return Object.freeze({ endControls: controls, reason: decision.reason, shortenedBy: state.maxControls - controls,
        state, decisions: Object.freeze(decisions) });
    }
  }
  // unreachable: controls === maxControls always ends
  throw new Error('Quiet exit hold stream did not terminate');
}

/**
 * Options resolver for the enablement path (documented in PHASEB_B2_B6_DESIGN_20260921.md,
 * site S2). Returns the `quietHold` option object TeacherBoxExitController already accepts,
 * or null for mode `fixed` (= v5 default construction). The initial hold minimum equals its
 * maximum (60), so the teacher_exit_hold phase, its label and its recordClock are v5's.
 */
export function resolveQuietExitHoldOptions({ mode = QUIET_EXIT_HOLD_DEFAULTS.mode, measure, requestQueued = null,
  window = QUIET_EXIT_HOLD_DEFAULTS.window, finalSettlingMinControls = QUIET_EXIT_HOLD_DEFAULTS.finalSettling.minControls } = {}) {
  if (!QUIET_EXIT_HOLD_MODES.includes(mode)) throw new Error(`Unknown quiet exit hold mode: ${mode}`);
  if (mode === 'fixed') return null;
  if (typeof measure !== 'function') throw new Error('Quiet exit holds require a synchronous measure() function');
  if (mode === 'quiet_on_request' && typeof requestQueued !== 'function') throw new Error('quiet_on_request requires a requestQueued() reader');
  validateBounds({ minControls: finalSettlingMinControls, maxControls: QUIET_EXIT_HOLD_DEFAULTS.finalSettling.maxControls, window, rearmControls: 0 });
  return Object.freeze({ mode, window, measure, requestQueued,
    initialHoldMinControls: QUIET_EXIT_HOLD_DEFAULTS.initialHold.maxControls,   // 60 = fixed hold, v5-identical
    finalHoldMinControls: finalSettlingMinControls });
}

/** Schedule arithmetic used by the design note: how much earlier the next approach may start. */
export function exitScheduleShift({ settlingEndControls, settlingMaxControls = QUIET_EXIT_HOLD_DEFAULTS.finalSettling.maxControls } = {}) {
  if (!whole(settlingEndControls, 1) || !whole(settlingMaxControls, 1) || settlingEndControls > settlingMaxControls)
    throw new Error('settlingEndControls must be a whole count within the maximum');
  return settlingMaxControls - settlingEndControls;
}
