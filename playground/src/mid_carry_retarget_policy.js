// Phase B · B6 — mid-carry goal retarget admission policy. DORMANT: not imported by
// main.js or by any controller. Pure functions only: no DOM, no physics, no inference.
//
// What v5 already has (all default-OFF behind `midCarryRetarget=1`, main.js:733):
//   picker click during an active carry → planarPickerRetargetGoal (keeps the requested
//   set-down Z) → requestLoadedCarryRetarget → previewLoadedCarryRetarget (zero-control
//   old/new policy inference on the spliced reference; StreamingCarryGoalUpdateOwner
//   admission; spliceLoadedCarryGoal publish:false) → applyLoadedCarryRetarget (identity /
//   exact-state re-check, publish:true). Every safety bound lives in those modules; this
//   policy REPEATS them as a single ordered, testable admission so the enablement path can
//   refuse cheaply BEFORE spending the zero-control preview, and so that a refusal has one
//   documented fallback (`refusalDisposition`) instead of dropping the user's click.
//
// Fixed constants (never tuned here):
//   actionDeltaMaxAbs 0.0089015 — main.js MID_CARRY_RETARGET_ACTION_MAX_DELTA, "fixed before
//     merged-path physics from the exact-state paired policy probe (job34816825). It is not
//     adjusted from the later physical outcome."
//   minRampControls 16 — loaded_reference_retarget.js: "at least sixteen unexecuted loaded
//     ramp controls before descent".
//   toleranceM 0.10 — CARRY_PLACEMENT_TOLERANCE_M (teacher_carry_sequence.js); a retarget moves
//     the goal, never the tolerance.
//   planarZToleranceM 1e-5, geometryRemainderMaxAbs 2e-7 — loaded_reference_retarget.js.
//   duplicateToleranceM 1e-9 — StreamingCarryGoalUpdateOwner default.
//   grasp: both hand normal forces > 1 N and object height > .5 m — main.js preview/apply.
import { CARRY_PLACEMENT_TOLERANCE_M } from './teacher_carry_sequence.js';
import { CARRY_TIME_COSTS } from './carry_time_budget.js';

export const MID_CARRY_RETARGET_POLICY_LIMITS = Object.freeze({
  actionDeltaMaxAbs: 0.0089015,
  minRampControls: 16,
  planarZToleranceM: 1e-5,
  duplicateToleranceM: 1e-9,
  geometryRemainderMaxAbs: 2e-7,
  minHandNormalForceN: 1,
  minLoadedObjectHeightM: .5,
  maxCorrectionDefaultM: .25,
  toleranceM: CARRY_PLACEMENT_TOLERANCE_M,
  // Controls that must remain after the current source ends: post-placement settling,
  // the recorded exit and the panel's stable-standing window (carry_time_budget.js).
  reserveControls: CARRY_TIME_COSTS.settlingControls + CARRY_TIME_COSTS.exitControls + CARRY_TIME_COSTS.finalStandingControls,
});

/** Contract the arbiter (stream A, skill_arbiter.js) needs for a floor click during an
 * active carry. Data only; stream A's files are not touched here. */
export const MID_CARRY_RETARGET_ARBITER_CONTRACT = Object.freeze({
  // Aligned 2026-09-21 with stream A's IMPLEMENTED input (phaseb-unified-v9a main.js:2722-2728, skill_arbiter.js v2):
  // activeTask = taskInFlight ? { kind, sequencePhase, segmentIndex, objectBodyName } : null. The arbiter only routes
  // (non-null -> blocker 'task_active' -> carry = v5's synchronous queue). childPhase / segmentCount are NOT arbiter inputs:
  // admitMidCarryRetargetRequest reads them from the live carry controller in main.js.
  requestField: 'activeTask',
  requestFields: Object.freeze(['kind', 'sequencePhase', 'segmentIndex', 'objectBodyName']),
  kindValues: Object.freeze(['carry', 'pickup', 'loading', 'push_pending']),   // A's main.js: activeBoxTask ?? queuedBoxTask.kind ?? loading | push_pending
  retargetKind: 'carry',                             // only this kind can ever reach the retarget route; every other kind -> queue_after_release
  arbiterBlocker: 'task_active',
  skill: 'carry',
  routes: Object.freeze(['mid_carry_retarget', 'queue_after_release']),
  reasons: Object.freeze({ retarget: 'mid_carry_goal_update', queue: 'carry_in_progress_queue_after_release' }),
  blockerWhileLoaded: 'object_in_hand',              // never `push` / `pickup` while a carry owns the object
  neverChanges: Object.freeze(['toleranceM', 'actionDeltaMaxAbs', 'safetyVetoes']),
});

const finite3 = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const whole = (value, minimum = 0) => Number.isInteger(value) && value >= minimum;
const refuse = (reason, extra = {}) => Object.freeze({ accept: false, stage: 'request', reason, ...extra });

/**
 * Stage 1 — request admission (pure, no preview cost). Ordered fail-closed branches; the
 * first failing branch is the reason. `accept:true` means "spend the zero-control preview",
 * never "apply".
 */
export function admitMidCarryRetargetRequest({ enabled = false, owners = {}, carry = {}, grasp = {},
  currentGoal, requestedGoal, budget = {}, toleranceM = CARRY_PLACEMENT_TOLERANCE_M, limits = MID_CARRY_RETARGET_POLICY_LIMITS } = {}) {
  if (toleranceM !== CARRY_PLACEMENT_TOLERANCE_M) throw new Error('The 10 cm placement tolerance is fixed');
  if (enabled !== true) return refuse('retarget_disabled');
  if (!finite3(currentGoal) || !finite3(requestedGoal)) return refuse('invalid_goal');
  if (owners.retargetInFlight === true) return refuse('retarget_in_progress');
  if (owners.pendingPreview === true) return refuse('preview_pending');
  if (owners.privateOwner === true) return refuse('private_owner_active');
  if (owners.queuedUserTask === true) return refuse('user_command_queued');
  if (carry.sequencePhase !== 'teacher' || carry.childPhase !== 'teacher') return refuse('not_in_loaded_transport',
    { sequencePhase: carry.sequencePhase ?? null, childPhase: carry.childPhase ?? null });
  if (!whole(carry.segmentIndex) || !whole(carry.segmentCount, 1) || carry.segmentIndex !== carry.segmentCount - 1)
    return refuse('not_final_segment', { segmentIndex: carry.segmentIndex ?? null, segmentCount: carry.segmentCount ?? null });
  if (Math.abs(requestedGoal[2] - currentGoal[2]) > limits.planarZToleranceM) return refuse('non_planar_goal');
  const displacement = [requestedGoal[0] - currentGoal[0], requestedGoal[1] - currentGoal[1], 0];
  const displacementM = Math.hypot(displacement[0], displacement[1]);
  if (displacementM <= limits.duplicateToleranceM) return refuse('duplicate_goal', { displacementM });
  const { referenceIndex, warpStartFrame, warpEndFrame, sourceFrames } = carry;
  if (![referenceIndex, warpStartFrame, warpEndFrame, sourceFrames].every(v => whole(v)) || sourceFrames < 1
      || warpStartFrame >= warpEndFrame) return refuse('invalid_reference_clock');
  if (referenceIndex < warpStartFrame) return refuse('before_loaded_ramp', { referenceIndex, warpStartFrame });
  const endFrame = Math.min(warpEndFrame, sourceFrames - 1);
  const remainingRampControls = endFrame - referenceIndex;
  if (remainingRampControls < limits.minRampControls) return refuse('ramp_too_short', { remainingRampControls, minRampControls: limits.minRampControls });
  const maxCorrectionM = Number.isFinite(carry.maxCorrectionM) ? carry.maxCorrectionM : limits.maxCorrectionDefaultM;
  if (!(maxCorrectionM > 0) || maxCorrectionM > limits.maxCorrectionDefaultM) return refuse('invalid_correction_bound', { maxCorrectionM });
  const prior = finite3(carry.priorCorrectionWorld) ? carry.priorCorrectionWorld : [0, 0, 0];
  const cumulative = [prior[0] + displacement[0], prior[1] + displacement[1], 0];
  const cumulativeCorrectionM = Math.hypot(cumulative[0], cumulative[1]);
  if (cumulativeCorrectionM > maxCorrectionM + 1e-10) return refuse('exceeds_correction_bound', { cumulativeCorrectionM, maxCorrectionM, displacementM });
  const graspRetained = grasp.leftHandN > limits.minHandNormalForceN && grasp.rightHandN > limits.minHandNormalForceN
    && grasp.objectHeightM > limits.minLoadedObjectHeightM;
  if (graspRetained !== true) return refuse('grasp_not_retained', { leftHandN: grasp.leftHandN ?? null, rightHandN: grasp.rightHandN ?? null, objectHeightM: grasp.objectHeightM ?? null });
  if (!whole(budget.remainingControls)) return refuse('request_clock_unavailable');
  const requiredControls = (sourceFrames - referenceIndex) + limits.reserveControls;
  if (budget.remainingControls < requiredControls) return refuse('insufficient_budget', { remainingControls: budget.remainingControls, requiredControls });
  return Object.freeze({ accept: true, stage: 'request', reason: 'preview_required',
    displacementWorld: Object.freeze(displacement), displacementM, cumulativeCorrectionWorld: Object.freeze(cumulative),
    cumulativeCorrectionM, maxCorrectionM, remainingRampControls, requiredControls, remainingControls: budget.remainingControls,
    toleranceM: CARRY_PLACEMENT_TOLERANCE_M, actionDeltaMaxAbs: limits.actionDeltaMaxAbs });
}

/**
 * Stage 2 — preview admission. Consumes the zero-control preview record (main.js
 * previewLoadedCarryRetarget review + spliceLoadedCarryGoal record). The action-delta
 * bound is FIXED; passing any other value throws. `physicalGoalUpdateQualified` stays
 * false here: physical qualification is a panel outcome (B6 focused gate), not a policy result.
 */
export function admitMidCarryRetargetPreview({ request, preview = {}, actionDeltaBound = MID_CARRY_RETARGET_POLICY_LIMITS.actionDeltaMaxAbs,
  limits = MID_CARRY_RETARGET_POLICY_LIMITS } = {}) {
  if (actionDeltaBound !== MID_CARRY_RETARGET_POLICY_LIMITS.actionDeltaMaxAbs) throw new Error('The mid-carry action-delta bound is fixed (job34816825)');
  const out = (accept, reason, extra = {}) => Object.freeze({ accept, stage: 'preview', reason, physicalGoalUpdateQualified: false,
    actionDeltaMaxAbs: preview.actionDeltaMaxAbs ?? null, actionDeltaBound, ...extra });
  if (request?.accept !== true || request.stage !== 'request') return out(false, 'request_not_admitted');
  if (preview.stateUnchanged !== true) return out(false, 'state_changed_during_preview');
  for (const key of ['positionBoundaryMaxAbs', 'rotationBoundaryMaxAbs', 'velocityBoundaryMaxAbs']) {
    if (preview[key] !== 0) return out(false, 'splice_boundary_discontinuity', { channel: key, value: preview[key] ?? null });
  }
  if (!Number.isFinite(preview.relativeGeometryRemainderMaxAbs) || preview.relativeGeometryRemainderMaxAbs > limits.geometryRemainderMaxAbs)
    return out(false, 'geometry_remainder_exceeded', { relativeGeometryRemainderMaxAbs: preview.relativeGeometryRemainderMaxAbs ?? null });
  if (!Number.isFinite(preview.actionDeltaMaxAbs)) return out(false, 'action_delta_unavailable');
  if (preview.actionDeltaMaxAbs > actionDeltaBound) return out(false, 'action_delta_exceeds_bound');
  return out(true, 'zero_control_preview_passed');
}

/** What the enablement path does with a refused click: never drop the user's intent.
 *  - 'reject'           : malformed input, nothing to route;
 *  - 'ignore'           : the carry already targets this goal;
 *  - 'ordinary_request' : route the click through v5's ordinary carry request path
 *                         (main.js onObjectGoal → startCarryToGoal → queued after the
 *                         current task, exactly as with midCarryRetarget off). */
export function refusalDisposition(reason) {
  if (reason === 'invalid_goal') return 'reject';
  if (reason === 'duplicate_goal') return 'ignore';
  return 'ordinary_request';
}
