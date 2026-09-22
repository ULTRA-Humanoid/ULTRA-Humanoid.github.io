// Full-reference geometry checks for an explicitly planned mixed carry.
import { planCarryToGoal } from './teacher_goal_warp.js';
import { checkCarrySegmentClearance } from './carry_request_clearance.js';
import { projectObjectCollisionMeshes } from './carry_destination_footprint.js';
import { planBoxApproach } from './box_approach.js';
import { turnReferenceTransform } from './teacher_turn_controller.js';
import { transformTeacherReference } from './teacher_reference.js';
import { evaluatePickupPoseReachability, pickupPoseReachabilityKey, PICKUP_POSE_REACHABILITY_LIMITS } from './pickup_pose_reachability.js';

const yaw = q => Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
export const PICKUP_POSE_SEARCH_REQUIREMENTS = Object.freeze(['all', 'first', 'none']);

function readPickupPoseSearch(search) {
  if (!search) return null;
  const { walkSkills, turnSkills = [], sweeps, require = 'all', cache = null, referenceCache = null, checkCache = null, limits = PICKUP_POSE_REACHABILITY_LIMITS } = search;
  if (!Array.isArray(walkSkills) || !walkSkills.length || !(sweeps instanceof Map) || walkSkills.some(skill => !sweeps.get(skill))
      || !Array.isArray(turnSkills) || !PICKUP_POSE_SEARCH_REQUIREMENTS.includes(require) || (cache !== null && !(cache instanceof Map))
      || (referenceCache !== null && !(referenceCache instanceof Map)) || (checkCache !== null && !(checkCache instanceof Map)))
    throw new Error('Pickup pose search requires the bounded loaded walking library, its sweeps and an explicit requirement');
  return { walkSkills, turnSkills, sweeps, require, cache, referenceCache, checkCache, limits };
}

// Compact per-segment record for plan review and controller state; no source rows.
function compactReachability(result) {
  if (!result) return null;
  const a = result.approach, f = result.facingProgramme;
  return { reachable: result.reachable, reason: result.reason, uncertain: result.uncertain ?? false, pickupYawRad: result.pickupYawRad ?? null,
    route: result.route ? { supported: result.route.supported, routed: result.route.routed, pathLengthM: result.route.pathLengthM,
      waypoints: result.route.waypoints.map(p => [...p]) } : null,
    approach: a ? { end: a.end, arrivalKind: a.arrivalKind ?? null, refusalReason: a.refusalReason ?? null, finalRootXY: [...a.finalRootXY],
      finalYawRad: a.finalYawRad, predictedArrivalYawRad: a.predictedArrivalYawRad ?? null, facingErrorRad: a.facingErrorRad, distanceToGoalM: a.distanceToGoalM, handoffZone: a.handoffZone ?? false,
      studentRecoveryEligible: a.studentRecoveryEligible, rootYawAssumed: a.rootYawAssumed,
      clips: a.clips.map(c => ({ kind: c.kind, sweepName: c.sweepName, sourceFrames: c.sourceFrames, headingRad: c.headingRad, admitted: c.admitted,
        obstacleName: c.obstacleName ?? null, alternative: c.alternative ?? false })) } : null,
    facingProgramme: f ? { available: f.available, reason: f.reason, evaluated: f.evaluated, selected: f.selected ? { offsetM: [...f.selected.offsetM],
      terminalGoalWorld: [...f.selected.terminalGoalWorld], sweepName: f.selected.sweepName, sourceFrames: f.selected.sourceFrames,
      entryPose: [...f.selected.entryPose], entryRouteExact: f.selected.entryRouteExact, entryRouteLengthM: f.selected.entryRouteLengthM } : null } : null };
}

/** pickupPoseSearch (optional): { walkSkills, sweeps, require:'all'|'first'|'none', cache?, referenceCache? }.
 * With require 'all' every segment whose start root is known (actual root for the
 * first, predicted post-exit root for later ones) must have a reachable pickup
 * pose, 'first' only the first segment, 'none' only reports. Reachability runs
 * before the expensive path check so an unreachable plan costs only geometry. */
export function checkMixedCarryClearance({ plan, liveData, destinationGeometry, pathChecker,
  rootPositionWorld = null, rootQuaternionXyzwWorld = null, exitSkill = null, pickupPoseSearch = null }) {
  if (plan?.supported !== true || !plan.segments?.length) throw new Error('A supported mixed carry plan is required');
  const search = readPickupPoseSearch(pickupPoseSearch);
  const checks = [];
  let firstReferenceRoot = null, maximumPlannedEndpointResidualM = 0;
  const predictedApproaches = [], pickupPoseReachability = [];
  let predictedRoot = rootPositionWorld ? Array.from(rootPositionWorld) : null;
  let predictedYaw = rootQuaternionXyzwWorld?.length === 4 && Array.from(rootQuaternionXyzwWorld).every(Number.isFinite)
    ? yaw(rootQuaternionXyzwWorld) : null;
  const obstacles = exitSkill || search ? destinationGeometry.read(liveData) : null;
  if (exitSkill && (!predictedRoot || exitSkill.locomotionOnly !== true
      || ![199, 249].includes(exitSkill.sourceFrames)
      || exitSkill.frames?.length < exitSkill.sourceFrames + 16)) {
    throw new Error('All-transition ranking requires the actual complete exit reference and current root position');
  }
  // pickupPoseReachable is the ranking verdict: the first segment from the
  // actual root. Later segments start from a nominal post-exit root and a
  // nominal placement; their verdicts are reported for the executing loop but
  // (validated on the frozen panel) are not reliable enough to re-rank a plan.
  const reachabilitySummary = () => ({
    pickupPoseReachable: search ? pickupPoseReachability[0]?.reachable ?? null : null,
    firstPickupReachable: search ? pickupPoseReachability[0]?.reachable ?? null : null,
    allPickupPosesPredictedReachable: search ? pickupPoseReachability.some(r => r.reachable !== null) ? pickupPoseReachability.every(r => r.reachable !== false) : null : null,
    pickupPoseReachability: search ? pickupPoseReachability : null, pickupPoseSearchRequirement: search?.require ?? null });
  for (const [segmentIndex, segment] of plan.segments.entries()) {
    const { skill, carryOptions } = segment;
    const referenceKey = search?.referenceCache ? [segment.candidateId, ...segment.plannedStartObjectPositionWorld, ...segment.goalWorld,
      carryOptions.warpStartFrame, carryOptions.warpEndFrame, carryOptions.maxCorrection].join(',') : null;
    let reference = referenceKey ? search.referenceCache.get(referenceKey) : undefined;
    if (!reference) {
      reference = planCarryToGoal(skill.frames, skill.sourceFrames,
        segment.plannedStartObjectPositionWorld, segment.goalWorld, {
          startFrame: carryOptions.warpStartFrame, endFrame: carryOptions.warpEndFrame,
          maxCorrection: carryOptions.maxCorrection,
        });
      if (referenceKey) search.referenceCache.set(referenceKey, reference);
    }
    firstReferenceRoot ??= Array.from(reference.frames[0].slice(0, 3));
    let distanceM = null, shortIntersegmentCandidate = false;
    if (predictedRoot) {
      const approachGoal = reference.frames[0];
      distanceM = Math.hypot(approachGoal[0] - predictedRoot[0], approachGoal[1] - predictedRoot[1]);
      shortIntersegmentCandidate = segmentIndex > 0 && distanceM <= .5;
    }
    if (exitSkill) {
      const approachGoal = Array.from(reference.frames[0].slice(0, 3));
      const route = planBoxApproach(predictedRoot, approachGoal, obstacles,
        { clearance: .55, respectTransitClearance: !shortIntersegmentCandidate });
      predictedApproaches.push({ segmentIndex, candidateId: segment.candidateId,
        startRootPositionWorld: [...predictedRoot], approachGoalWorld: approachGoal,
        rootFromActualState: segmentIndex === 0, rootFromNominalExitReference: segmentIndex > 0,
        directDistanceM: distanceM, route,
        shortDirectStudentGeometry: shortIntersegmentCandidate && route.supported && !route.directBlocked });
      if (!route.supported) return { supported: false, reason: 'predicted_approach_route_unavailable',
        requestedGoalWorld: [...plan.requestedGoalWorld], segmentIndex, checks, predictedApproaches, ...reachabilitySummary() };
    }
    if (search) {
      // The first segment starts from the actual root; later ones from the
      // nominal post-exit root, which exists only with an exit reference.
      const rootKnown = predictedRoot !== null && (segmentIndex === 0 || exitSkill !== null);
      let result = null;
      if (rootKnown) {
        const options = { pickupPose: reference.frames[0], objectPositionWorld: segment.plannedStartObjectPositionWorld,
          rootPositionWorld: predictedRoot, rootYawRad: predictedYaw, obstacles, walkSkills: search.walkSkills,
          turnSkills: search.turnSkills, sweeps: search.sweeps, respectTransitClearance: !shortIntersegmentCandidate, limits: search.limits };
        const key = search.cache ? pickupPoseReachabilityKey({ candidateId: segment.candidateId, ...options }) : null;
        result = key ? search.cache.get(key) : undefined;
        if (!result) { result = evaluatePickupPoseReachability(options); if (key) search.cache.set(key, result); }
      }
      pickupPoseReachability.push({ segmentIndex, candidateId: segment.candidateId,
        rootSource: !rootKnown ? 'unknown' : segmentIndex === 0 ? 'actual' : 'predicted_after_exit',
        rootPositionWorld: predictedRoot ? [predictedRoot[0], predictedRoot[1]] : null, rootYawRad: predictedYaw,
        pickupPose: [reference.frames[0][0], reference.frames[0][1], yaw(reference.frames[0].slice(3, 7))],
        ...(compactReachability(result) ?? { reachable: null, reason: 'root_unknown', route: null, approach: null, facingProgramme: null }) });
      const blocking = result && !result.reachable && (search.require === 'all' || (search.require === 'first' && segmentIndex === 0));
      if (blocking) return { supported: false, reason: 'pickup_pose_unreachable', segmentIndex,
        requestedGoalWorld: [...plan.requestedGoalWorld], checks, predictedApproaches, ...reachabilitySummary() };
    }
    maximumPlannedEndpointResidualM = Math.max(maximumPlannedEndpointResidualM,
      Math.hypot(reference.referenceGoalWorld[0] - segment.goalWorld[0], reference.referenceGoalWorld[1] - segment.goalWorld[1]));
    // The same segment reference recurs across ranking passes; its footprint and
    // path verdicts depend only on the reference and the fixed live scene.
    let check = referenceKey && search.checkCache ? search.checkCache.get(referenceKey) : undefined;
    if (!check) {
      check = checkCarrySegmentClearance({ referenceFrames: reference.frames,
        sourceFrames: skill.sourceFrames, objectBodyName: skill.objectBodyName,
        requestedGoalWorld: segment.goalWorld, liveData, destinationGeometry, pathChecker });
      if (referenceKey && search.checkCache) search.checkCache.set(referenceKey, check);
    }
    checks.push({ segmentIndex, candidateId: segment.candidateId, ...check });
    if (!check.supported) return { supported: false, reason: check.reason, segmentIndex,
      requestedGoalWorld: [...plan.requestedGoalWorld], checks, ...reachabilitySummary() };
    if (exitSkill) {
      // Predict the existing complete retreat's endpoint from the complete
      // carry terminal. Actual execution still aligns after real settling.
      const terminal = reference.frames[skill.sourceFrames - 1];
      const transform = turnReferenceTransform(exitSkill.frames[0], terminal.slice(0, 3), terminal.slice(3, 7));
      const exitTerminal = transformTeacherReference(exitSkill.frames[exitSkill.sourceFrames - 1], transform);
      predictedRoot = Array.from(exitTerminal.slice(0, 3));
      predictedYaw = yaw(exitTerminal.slice(3, 7));
      const object = destinationGeometry.objects.find(o => o.name === skill.objectBodyName);
      const placed = projectObjectCollisionMeshes(object, terminal.slice(71, 74), terminal.slice(74, 78));
      const index = obstacles.findIndex(o => o.name === object.name);
      if (index < 0) throw new Error('The carried box is required in approach geometry');
      obstacles[index] = placed;
    }
  }
  // The checker establishes its fixed reserve, not a measured minimum margin.
  // Straight initial approach distance is a ranking estimate, not route proof.
  return { supported: true, reason: null, requestedGoalWorld: [...plan.requestedGoalWorld], checks,
    predictedApproachCostM: exitSkill ? predictedApproaches.reduce((sum, a) => sum + a.route.pathLengthM, 0)
      : rootPositionWorld ? Math.hypot(firstReferenceRoot[0] - rootPositionWorld[0], firstReferenceRoot[1] - rootPositionWorld[1]) : null,
    predictedApproaches, approachPrediction: exitSkill ? 'all nominal carry-exit-approach transitions' : 'initial straight distance only',
    predictedEndpointResidualM: maximumPlannedEndpointResidualM, ...reachabilitySummary() };
}
