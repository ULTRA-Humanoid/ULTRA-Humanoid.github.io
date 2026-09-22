// Deterministic skill arbiter for the single object + floor-goal interface.
// Pure: no DOM, no physics, no controller calls. Given the selected object,
// the requested floor destination and the measured robot/object pose, it
// chooses which EXISTING lane (`push` / `carry` / `pickup`) the request is
// routed to, or `refuse`. Every decision returns a transparent `reason`, the
// feature values it was computed from, and the list of push blockers.
//
// Every numeric threshold below is either read from an existing lane contract
// (file:line cited) or labelled `arbiterPolicy` (a routing preference the push
// lane itself does not require). None of them changes physics, controller
// options or tolerances; they only decide which existing entry point is called.

export const SKILL_ARBITER_VERSION = 'phasec1-arbiter-v5';   // v2: activeTask -> blocker task_active (B2 queued-carry semantics)
// v3 (v15a): push admission also requires (i) the box face within PUSH_FACE_ALIGNMENT_MAX_DEG of square-on to the push direction
// (blocker off_face) and (ii) the request distance within the lane's DELIVERED reach PUSH_MAX_DISTANCE_M (blocker above_delivered_reach).
// v4 (Phase C): an off-cone request can push only when the caller proves a direct, collision-free route to the exact source-aligned
// push stance. This widens visibility without relaxing v15's distance, face, floor, live-entry or safety gates.
// v5 (Phase C.1): v4's off-cone widening is withdrawn after job 35278460 proved that planar line clearance does not certify
// source-boundary compatibility: the right-lateral case took a 2.152 m cross-side approach and lost balance before useful contact.
// Exact-target geometry stays in telemetry, but outside-cone entry fails closed as entry_state_unvalidated until pose/velocity/history
// compatibility has physical evidence. The ordinary straight-ahead click lane remains reachable and smoke-validated.

/** Object capability table. Body names are the MJCF names the picker reports
 * (`active_` + point-cloud key). Only the large box has any routed lane in v5:
 * `onObjectGoal` refuses other selections with UNSUPPORTED_OBJECT_MESSAGE
 * (main.js, the `user.activeObjName !== 'active_largebox_080_080_080'` check),
 * `startGroundPushToGoal` is hard-bound to the large box, and BOX_TASKS
 * (pickup/carry) load large-box teacher references. */
export const OBJECT_CAPABILITIES = Object.freeze({
  active_largebox_080_080_080: Object.freeze(['pickup', 'carry', 'push']),
  active_plasticbox_080_080_080: Object.freeze([]),
  active_smallbox_080_080_080: Object.freeze([]),
  active_suitcase_080_080_080: Object.freeze([]),
});
/** Exact v5 refusal text (main.js onObjectGoal). */
export const UNSUPPORTED_OBJECT_MESSAGE = 'Carry controls currently use the large box.';

/** Push lane contract, read from the lane's own source; never adjusted here.
 * - sourceDisplacementM: planar travel of the hand002 source object between
 *   reference_frames747[0] and [sourceFrames-1] (indices 71,72); equals
 *   normal-api-smoke.json `sourceDisplacement` (release-validation-20260919/
 *   largebox-carry-push-locomotion-v5).
 * - maxCorrectionM: main.js GroundPushGoalSequenceController options
 *   `maxCorrection:.05`, prepareGroundPushGoalWarpEntry `maxCorrection:.05`,
 *   and ground_push_mapping.js correctionLimit (`> .05` throws).
 * - maxSegments: 1 (ground_push_sequence.js constructor throws otherwise).
 * - displacementIntervalM: teacher_carry_sequence.js planCarrySegments with
 *   maxSegments 1 supports distance in [sourceTravelM - maxCorrection,
 *   sourceTravelM + maxCorrection]; outside -> reason 'unsupported_distance'.
 * - approachClearanceM: largebox_push_live_entry_runtime.js
 *   prepareOutcomeBasedNoResetEntry planBoxApproach(..., {clearance:.55}).
 * - budgets: normal-api-smoke.json `budgets` (measured envelope, informational). */
export const PUSH_LANE = Object.freeze({
  sourceIdentity: 'sub7_largebox_002_081_081_076_080_080_080',
  objectBody: 'active_largebox_080_080_080',
  sourceDisplacementM: 0.953262055755702,
  maxCorrectionM: 0.05,
  maxSegments: 1,
  displacementIntervalM: Object.freeze([0.953262055755702 - 0.05, 0.953262055755702 + 0.05]),
  approachClearanceM: 0.55,
  budgets: Object.freeze({ approachControls: 1800, pushAndExitControls: 2200 }),
  // v15a (ii) DELIVERED reach. The lane executes the complete 225-frame source and stops where the source object
  // stopped; it never re-plans. Measured deliveries for the natural request (public/task-assets/native-summary.json,
  // exact-source reproduction, and release-validation-20260919/largebox-carry-push-locomotion-v14/normal-api-smoke.json):
  //   requested 0.953262 m -> delivered 0.874298 m (exact replay, signed 0.873845, cross-track 0.028) and 0.871782 m (smoke,
  //   residual 0.088631 to the goal = 1.1 cm inside the 0.10 m placement tolerance).
  // Two admissible reading of "delivered vs requested":
  //   (a) constant delivery ~0.874 regardless of the +-5 cm endpoint warp -> a request d misses once d - 0.874 > 0.10 - cross(0.03),
  //       i.e. d > 0.944 (the coordinator's arithmetic: delivered + (0.10 - 0.03));
  //   (b) constant shortfall ~0.079 (the warp shifts the whole endpoint) -> residual hypot(0.079, cross) <= 0.10 for any d in the band.
  // The only VALIDATED point is the natural request itself (0.953262, smoke PASS with 1.1 cm margin), which lies between (a) and (b).
  // PROVISIONAL bound = that validated point plus 1 mm of click/measurement slack (the smoke's own request measures 0.95326207 m,
  //   1.2e-8 above the natural displacement): requests above it are not admitted until a delivery sweep exists.
  // (H026 asked 0.961486 -> delivered 0.803 along / -0.117 across, placement_missed 0.175 m.)
  deliveredDisplacementM: Object.freeze({ exactReplay: 0.8742976968559365, smoke: 0.8717822959044412, crossTrackM: 0.028124770065732857 }),
});
export const PUSH_MAX_DISTANCE_M = PUSH_LANE.sourceDisplacementM + 0.001;   // 0.954262 m, provisional (see PUSH_LANE.deliveredDisplacementM)

/** v15a (i) FACE ALIGNMENT, provisional. The lane warps the source box-centre -> goal (teacher_carry_controller.js _referencePlan:
 * planCarryToGoal with snapObjectYaw off), so the box FACE is never aligned to the source. In the source the face is 32.6 deg
 * (mod 90) off the push direction at frame 0 and self-squares to ~5 deg under the hands; a face near the 45 deg diagonal rotates
 * the other way and presents a corner (v14 H026: face 43.3 deg -> 90 deg spin, right-hand/knee propulsion, 0.175 m miss, knee-contact
 * strict violation). Measured points: 0.0 deg PASS (smoke), 32.6 deg PASS (source clip), 43.3 deg FAIL (H026). Nothing in between is
 * measured, so the bound is the source clip's own geometry class rounded up (35 deg = 32.6 + margin, 10 deg short of the diagonal).
 * Fail-closed: a missing/non-finite object yaw blocks push (object_yaw_missing). */
export const PUSH_FACE_ALIGNMENT_MAX_DEG = 35;
export const PUSH_FACE_EVIDENCE_DEG = Object.freeze({ smokePass: 0.0, sourceClipPass: 32.6, h026Fail: 43.3, diagonal: 45 });

/** Exact frame-0 geometry of hand002_predecessor69_phase_reference.json. It is
 * used only to preflight the same aligned root target that
 * prepareGroundPushGoalWarpEntry later sends to the no-reset approach lane. */
export const PUSH_APPROACH_GEOMETRY = Object.freeze({
  sourceRootPosition: Object.freeze([-0.07943997532129288, -0.11404800415039062, 0.7061561942100525]),
  sourceObjectPosition: Object.freeze([0.20422419905662537, -0.4284019470214844, 0.15027539432048798]),
  sourceObjectEndPosition: Object.freeze([1.1474868059158325, -0.2906913757324219, 0.1502753049135208]),
});

/** Compute the frame-0 stance produced by planCarryToGoal's rigid alignment.
 * The endpoint correction starts after frame 0 and cannot move this stance. */
export function alignedPushApproachTarget(objectPosWorld, goalWorld) {
  if (!finiteVector(objectPosWorld, 3) || !finiteVector(goalWorld, 3)) return null;
  const targetDx = goalWorld[0] - objectPosWorld[0], targetDy = goalWorld[1] - objectPosWorld[1];
  if (Math.hypot(targetDx, targetDy) <= 1e-9) return null;
  const source = PUSH_APPROACH_GEOMETRY;
  const sourceDx = source.sourceObjectEndPosition[0] - source.sourceObjectPosition[0];
  const sourceDy = source.sourceObjectEndPosition[1] - source.sourceObjectPosition[1];
  const yaw = Math.atan2(targetDy, targetDx) - Math.atan2(sourceDy, sourceDx);
  const relX = source.sourceRootPosition[0] - source.sourceObjectPosition[0];
  const relY = source.sourceRootPosition[1] - source.sourceObjectPosition[1];
  const c = Math.cos(yaw), s = Math.sin(yaw);
  return Object.freeze([objectPosWorld[0] + c * relX - s * relY,
    objectPosWorld[1] + s * relX + c * relY, source.sourceRootPosition[2]]);
}
/** Fold a face angle onto [0, 45] deg: a square footprint repeats every 90 deg and a face is as good as its mirror. */
export function faceOffsetDeg(objectYawRad, pushDirectionRad) {
  if (!Number.isFinite(objectYawRad) || !Number.isFinite(pushDirectionRad)) return null;
  const deg = ((objectYawRad - pushDirectionRad) * 180 / Math.PI) % 90;
  const folded = ((deg % 90) + 90) % 90;              // [0, 90)
  return Math.min(folded, 90 - folded);                // [0, 45]
}

/** Arbiter thresholds with provenance. */
export const ARBITER_THRESHOLDS = Object.freeze({
  // teacher_carry_sequence.js:14 CARRY_PLACEMENT_TOLERANCE_M = 0.10; main.js
  // already treats a carry goal within this radius as "already placed".
  pickupDistanceM: 0.10,
  // Same 10 cm criterion reused as the vertical band for "goal is on the
  // floor": objectGoalOnFloor (goal_geometry.js) returns z = object resting
  // centre height for a floor click, the API smoke uses z = 0. Accept
  // -band <= goalZ <= objectZ + band.
  floorBandM: 0.10,
  // arbiterPolicy: half-angle of the "straight ahead" cone between
  // bearing(robot->object) and bearing(object->goal). The lane itself is
  // direction-agnostic (teacher_goal_warp.js planCarryToGoal yaw-aligns the
  // source to the object->goal bearing); the only measured push evidence
  // (normal-api-smoke.json) is at 0 rad bearing error. 45 deg keeps the
  // robot on the far side of the box so the no-reset approach is short.
  pushConeHalfAngleRad: Math.PI / 4,
  // v15a: see PUSH_FACE_ALIGNMENT_MAX_DEG / PUSH_MAX_DISTANCE_M above (both provisional, evidence cited there).
  pushFaceAlignmentMaxDeg: PUSH_FACE_ALIGNMENT_MAX_DEG,
  pushMaxDistanceM: PUSH_MAX_DISTANCE_M,
});

const SKILLS = Object.freeze(['push', 'carry', 'pickup', 'refuse']);
const wrap = value => Math.atan2(Math.sin(value), Math.cos(value));
const finiteVector = (value, length) => value != null && typeof value.length === 'number'
  && value.length === length && Array.from(value).every(v => typeof v === 'number' && Number.isFinite(v));
const finiteNumber = value => typeof value === 'number' && Number.isFinite(value);
const round = (value, digits = 6) => (finiteNumber(value) ? Number(value.toFixed(digits)) : null);

/** FNV-1a over a canonical request string. The arbiter uses NO randomness;
 * the seed is exposed so any future tie-break can be reproduced from the
 * same click sequence (goal, object, requestId). */
export function requestSeed({ objectBodyName, goalWorld, requestId } = {}) {
  const canonical = JSON.stringify([String(objectBodyName ?? ''),
    finiteVector(goalWorld, 3) ? Array.from(goalWorld).map(v => round(v, 4)) : null,
    requestId ?? null]);
  let hash = 0x811c9dc5;
  for (let i = 0; i < canonical.length; i++) {
    hash ^= canonical.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function decision(skill, reason, features, blockers, extra = {}) {
  if (!SKILLS.includes(skill)) throw new Error(`Unknown skill ${skill}`);
  return Object.freeze({ skill, reason, features: Object.freeze(features),
    pushBlockers: Object.freeze(Array.from(blockers)), version: SKILL_ARBITER_VERSION,
    thresholds: ARBITER_THRESHOLDS, pushLane: PUSH_LANE, ...extra });
}

/**
 * Choose the lane for a floor-goal request.
 * @param {object} request
 *  objectBodyName  selected MJCF body name (required)
 *  capability      optional explicit capability list; defaults to OBJECT_CAPABILITIES
 *  objectPosWorld  [x,y,z] measured object position (required)
 *  objectYawRad    measured object yaw (v15a: REQUIRED for push; face alignment vs the push direction, blocker off_face /
 *                  object_yaw_missing)
 *  goalWorld       [x,y,z] requested floor destination (required)
 *  rootPosWorld    [x,y,z] robot pelvis position (required for push eligibility)
 *  rootYawRad      robot yaw (feature only)
 *  approach        { clearLine:boolean, liveEntryAvailable:boolean, distanceM, directBlocked,
 *                    clearanceBlocked, reason, alignedTarget:{directClear:boolean,...} }.
 *                    alignedTarget is a planBoxApproach preflight to
 *                    alignedPushApproachTarget using every compiled obstacle.
 *  budget          optional { maxApproachDistanceM } -> blocker 'approach_too_far'
 *  activeTask      null/undefined when no box task is active; otherwise ANY value (well-formed
 *                  { kind, sequencePhase, segmentIndex, objectBodyName } or malformed) -> blocker
 *                  'task_active': the click is routed to carry, i.e. v5's synchronous queued-carry
 *                  path (startBoxTask queues while busy), never to the push lane's task_busy refusal
 *                  and never to in-place pickup (phaseb-motion B2 `quiet_on_request` predicate).
 *                  Design-only follow-up (B6): activeTask.objectBodyName === selected + held object
 *                  -> blocker 'object_in_hand' (not implemented).
 *  requestId       caller's serial, folded into the exposed seed
 * Fail-closed: non-finite / missing required inputs -> 'refuse' (reason
 * 'invalid_input'); missing robot pose or approach facts -> push ineligible; a present but
 * malformed activeTask still blocks push/pickup (recorded as features.activeTask.malformed).
 */
export function chooseSkill(request = {}) {
  const { objectBodyName, objectPosWorld, objectYawRad, goalWorld, rootPosWorld, rootYawRad,
    approach = null, budget = null, requestId = null, activeTask = null } = request ?? {};
  const taskActive = activeTask !== null && activeTask !== undefined;
  const activeTaskFeature = !taskActive ? null
    : (typeof activeTask === 'object' && !Array.isArray(activeTask))
      ? { kind: typeof activeTask.kind === 'string' ? activeTask.kind : null,
          sequencePhase: typeof activeTask.sequencePhase === 'string' ? activeTask.sequencePhase : null,
          segmentIndex: Number.isInteger(activeTask.segmentIndex) ? activeTask.segmentIndex : null,
          objectBodyName: typeof activeTask.objectBodyName === 'string' ? activeTask.objectBodyName : null, malformed: false }
      : { kind: null, sequencePhase: null, segmentIndex: null, objectBodyName: null, malformed: true };
  const seed = requestSeed({ objectBodyName, goalWorld, requestId });
  const capability = Array.isArray(request?.capability) ? Array.from(request.capability)
    : (typeof objectBodyName === 'string' ? OBJECT_CAPABILITIES[objectBodyName] : undefined);
  const base = { objectBodyName: typeof objectBodyName === 'string' ? objectBodyName : null,
    capability: capability ? Array.from(capability) : null, seed, requestId, activeTask: activeTaskFeature };
  if (!capability || !capability.length) {
    return decision('refuse', 'unsupported_object', base, ['unsupported_object'],
      { message: UNSUPPORTED_OBJECT_MESSAGE });
  }
  if (!finiteVector(objectPosWorld, 3) || !finiteVector(goalWorld, 3)) {
    return decision('refuse', 'invalid_input', { ...base,
      objectPosWorldFinite: finiteVector(objectPosWorld, 3), goalWorldFinite: finiteVector(goalWorld, 3) },
      ['invalid_input'], { message: 'Destination and object position must contain finite XYZ.' });
  }
  const dx = goalWorld[0] - objectPosWorld[0], dy = goalWorld[1] - objectPosWorld[1];
  const distanceM = Math.hypot(dx, dy);
  const bearingObjectToGoalRad = distanceM > 1e-9 ? Math.atan2(dy, dx) : null;
  const goalZ = goalWorld[2], objectZ = objectPosWorld[2];
  const onFloor = goalZ >= -ARBITER_THRESHOLDS.floorBandM && goalZ <= objectZ + ARBITER_THRESHOLDS.floorBandM;
  const robotPoseKnown = finiteVector(rootPosWorld, 3);
  const bearingRobotToObjectRad = robotPoseKnown
    ? Math.atan2(objectPosWorld[1] - rootPosWorld[1], objectPosWorld[0] - rootPosWorld[0]) : null;
  const robotToObjectDistanceM = robotPoseKnown
    ? Math.hypot(objectPosWorld[0] - rootPosWorld[0], objectPosWorld[1] - rootPosWorld[1]) : null;
  const coneErrorRad = robotPoseKnown && bearingObjectToGoalRad !== null
    ? wrap(bearingRobotToObjectRad - bearingObjectToGoalRad) : null;
  const robotFacingErrorRad = robotPoseKnown && finiteNumber(rootYawRad) ? wrap(bearingRobotToObjectRad - rootYawRad) : null;
  const faceOffset = bearingObjectToGoalRad !== null ? faceOffsetDeg(objectYawRad, bearingObjectToGoalRad) : null;
  const features = { ...base,
    distanceM: round(distanceM), bearingObjectToGoalRad: round(bearingObjectToGoalRad),
    goalZ: round(goalZ), objectZ: round(objectZ), onFloor,
    objectYawRad: finiteNumber(objectYawRad) ? round(objectYawRad) : null,
    rootYawRad: finiteNumber(rootYawRad) ? round(rootYawRad) : null,
    bearingRobotToObjectRad: round(bearingRobotToObjectRad), robotToObjectDistanceM: round(robotToObjectDistanceM),
    coneErrorRad: round(coneErrorRad), robotFacingErrorRad: round(robotFacingErrorRad),
    pushDisplacementIntervalM: PUSH_LANE.displacementIntervalM,
    faceOffsetDeg: round(faceOffset, 3), pushFaceAlignmentMaxDeg: PUSH_FACE_ALIGNMENT_MAX_DEG, pushMaxDistanceM: PUSH_MAX_DISTANCE_M,
    approach: approach ? { clearLine: approach.clearLine === true, liveEntryAvailable: approach.liveEntryAvailable === true,
      distanceM: round(approach.distanceM), directBlocked: approach.directBlocked ?? null,
      clearanceBlocked: approach.clearanceBlocked ?? null, reason: approach.reason ?? null,
      alignedTarget: approach.alignedTarget ? {
        directClear: approach.alignedTarget.directClear === true,
        supported: approach.alignedTarget.supported === true,
        directBlocked: approach.alignedTarget.directBlocked ?? null,
        clearanceBlocked: approach.alignedTarget.clearanceBlocked ?? null,
        pathLengthM: round(approach.alignedTarget.pathLengthM),
        targetWorld: finiteVector(approach.alignedTarget.targetWorld, 3)
          ? Array.from(approach.alignedTarget.targetWorld).map(v => round(v)) : null,
        reason: approach.alignedTarget.reason ?? null,
      } : null } : null,
    budget: budget ? { maxApproachDistanceM: round(budget.maxApproachDistanceM) } : null };

  // 1. Goal at (within placement tolerance of) the object -> pickup in place.
  // Same inclusive epsilon main.js uses for the placement check (`<= CARRY_PLACEMENT_TOLERANCE_M + 1e-12`).
  //    Unless a task is active: then the click must take v5's queued-carry path (B2), never an in-place pickup.
  if (distanceM <= ARBITER_THRESHOLDS.pickupDistanceM + 1e-12 && !taskActive) {
    if (capability.includes('pickup')) return decision('pickup', 'goal_within_placement_tolerance', features, ['distance_below_pickup_tolerance']);
    if (capability.includes('carry')) return decision('carry', 'goal_within_placement_tolerance_no_pickup_lane', features, ['distance_below_pickup_tolerance']);
    return decision('refuse', 'unsupported_object', features, ['unsupported_object'], { message: UNSUPPORTED_OBJECT_MESSAGE });
  }

  // 2. Push only when EVERY push precondition holds. An active task blocks first (B2).
  const blockers = [];
  if (taskActive) blockers.push('task_active');
  if (!capability.includes('push')) blockers.push('no_push_capability');
  const [minPush, maxPush] = PUSH_LANE.displacementIntervalM;
  if (distanceM < minPush) blockers.push('below_push_range');
  else if (distanceM > maxPush) blockers.push('above_push_range');
  else if (distanceM > PUSH_MAX_DISTANCE_M + 1e-12) blockers.push('above_delivered_reach');   // v15a (ii)
  if (!onFloor) blockers.push('goal_not_on_floor');
  if (faceOffset === null) blockers.push('object_yaw_missing');                                 // v15a (i), fail-closed
  else if (faceOffset > PUSH_FACE_ALIGNMENT_MAX_DEG) blockers.push('off_face');                 // v15a (i)
  const insideLegacyCone = robotPoseKnown && coneErrorRad !== null
    && Math.abs(coneErrorRad) <= ARBITER_THRESHOLDS.pushConeHalfAngleRad;
  const alignedTargetDirectClear = approach?.alignedTarget?.directClear === true;
  const pushVisibilityRule = insideLegacyCone ? 'legacy_cone' : null;
  features.pushVisibilityRule = pushVisibilityRule;
  if (!robotPoseKnown) blockers.push('robot_pose_missing');
  else if (!pushVisibilityRule) {
    // 35278460_2: direct-clear target geometry was true, yet the long approach reached the recorded source with an
    // unqualified physical boundary and the push lost balance before meaningful contact. Do not infer dynamics from XY clearance.
    if (alignedTargetDirectClear) blockers.push('entry_state_unvalidated');
    blockers.push('off_cone');
  }
  if (!approach) blockers.push('approach_unknown');
  else {
    if (approach.clearLine !== true) blockers.push('approach_blocked');
    if (approach.liveEntryAvailable !== true) blockers.push('live_entry_unavailable');
    if (budget && finiteNumber(budget.maxApproachDistanceM)
        && !(finiteNumber(approach.distanceM) && approach.distanceM <= budget.maxApproachDistanceM)) blockers.push('approach_too_far');
  }
  if (!blockers.length) return decision('push', 'push_preconditions_satisfied:validated_forward_entry', features, []);

  // 3. Otherwise carry (the ordinary planner keeps its own range/clearance checks).
  if (capability.includes('carry')) return decision('carry', `push_blocked:${blockers.join(',')}`, features, blockers);
  return decision('refuse', 'unsupported_object', features, blockers, { message: UNSUPPORTED_OBJECT_MESSAGE });
}

/** v14a R2: describe the push lane's admission result for the router. Pure and fail-closed: anything but
 * `{supported:true}` is a refusal; the status text is what #task-status shows (the router prefixes `[push] `). */
export const PUSH_REFUSAL_STATUS_PREFIX = 'Push unavailable: ';
export function describePushRefusal(result) {
  if (result !== null && typeof result === 'object' && result.supported === true) {
    return Object.freeze({ refused: false, reason: null, statusText: null, requestId: result.requestId ?? null });
  }
  const reason = (result !== null && typeof result === 'object' && typeof result.reason === 'string' && result.reason)
    ? result.reason : (result !== null && typeof result === 'object' && result.message ? 'task_load_error' : 'push_refused');
  return Object.freeze({ refused: true, reason, statusText: PUSH_REFUSAL_STATUS_PREFIX + reason,
    requestId: (result !== null && typeof result === 'object') ? (result.requestId ?? null) : null });
}

/** Time one arbiter call with performance.now() (Date.now fallback). */
export function measureChooseSkill(request) {
  const clock = globalThis.performance && typeof globalThis.performance.now === 'function'
    ? () => globalThis.performance.now() : () => Date.now();
  const startedMs = clock();
  const result = chooseSkill(request);
  return { decision: result, decisionMs: clock() - startedMs };
}
