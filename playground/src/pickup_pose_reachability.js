// Plan-time pickup-pose reachability. Pure geometry, no physics, no arrival
// credit. Two complementary predictions for one carry reference's first-frame
// root ("pickup pose"), from an actual or predicted root:
//  1. approach dry-run: the recorded approach that main.js executes
//     (BoxApproachPlanner route -> TeacherWaypointController step/turn
//     selection and heading rule -> runtime hull admission with the same
//     reserves -> trySmallerStep -> standing fallback within the parent's
//     0.25 m handoff radius) replayed on nominal source terminals. Its end
//     state predicts the arrival heading (hence the parent's facing check) or
//     a refusal before arrival (hence the first-pickup student recovery).
//  2. facing programme: pickup_facing_approach generalized to every carry
//     source: some walking source, ending at the pickup pose (+ entry-region
//     offset) with the required heading, is admissible and its entry has a
//     0.55 m transit route. Reported for owners that can walk to an entry
//     pose first; it does not describe the ordinary approach's heading.
import { planBoxApproach } from './box_approach.js';
import { planNavigationPath, segmentEntersRectangle } from './navigation_planner.js';
import { checkAnchoredSweepGeometry } from './restricted_motion_geometry.js';
import { planRecordedSteps } from './teacher_waypoint_controller.js';
import { PICKUP_ENTRY_REGION_OFFSETS_M, PICKUP_ENTRY_REGION_POSITION_MARGIN_M } from './pickup_facing_entry_region.js';

export const PICKUP_POSE_REACHABILITY_LIMITS = Object.freeze({
  transitClearanceM: .55, stepTrackingReserveM: .1, turnTrackingReserveM: .15, verticalTrackingReserveM: .1,
  maxFacingErrorRad: 1.3,
  // Nominal source terminals ignore tracking. Over the 81 recorded combined
  // approaches the actual arrival yaw sat a systematic ~15 deg clockwise of the
  // dry-run heading (student tracking of forward records), with ~10 deg spread.
  arrivalYawBiasRad: -15 * Math.PI / 180,
  facingMarginRad: 10 * Math.PI / 180,
  parentArrivalRadiusM: .25, routeArrivalRadiusM: .1, waypointRadiusM: .28,
  maxHeadingOffsetRad: Math.PI / 3, maxSteps: 8, maxTurns: 4,
  // combined first-pickup student recovery window (staged_student_approach_controller).
  studentRecoveryDistanceM: .5,
  entryRegionMarginM: PICKUP_ENTRY_REGION_POSITION_MARGIN_M,
});
const finite = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);
const yaw = q => Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const wrap = x => Math.atan2(Math.sin(x), Math.cos(x));
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const insideBounds = (p, r) => p[0] > r.minX && p[0] < r.maxX && p[1] > r.minY && p[1] < r.maxY;
const round = (v, digits = 6) => Number.isFinite(v) ? v.toFixed(digits) : String(v);
const rotate = (x, y, angle) => [Math.cos(angle) * x - Math.sin(angle) * y, Math.sin(angle) * x + Math.cos(angle) * y];

/** Same source description as TeacherWaypointController.describeSkill. */
function describe(skill, sweep) {
  const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
  const sourceYawRad = yaw(first.slice(3, 7));
  return { skill, sweep, travelM: distance(first, last), sourceYawRad, yawChangeRad: wrap(yaw(last.slice(3, 7)) - sourceYawRad),
    travelDirectionRad: Math.atan2(last[1] - first[1], last[0] - first[0]), deltaXY: [last[0] - first[0], last[1] - first[1]] };
}

/** Root anchor (first-frame root XY/yaw) of a sweep whose terminal root is placed
 * at terminalXY/terminalYaw. Equals turnReferenceTransform(last, terminal) applied
 * to frames[0], because bindMotionSweep pins initial/terminal root poses to the
 * source's first/last frames. */
export function sweepAnchorFromTerminal(sweep, terminalXY, terminalYawRad) {
  const initial = sweep.initialRootPose, terminal = sweep.terminalRootPose;
  const yaw0 = yaw(initial.slice(3, 7)), yaw1 = yaw(terminal.slice(3, 7)), deltaYaw = wrap(yaw1 - yaw0);
  const [lx, ly] = rotate(terminal[0] - initial[0], terminal[1] - initial[1], -yaw0);
  const anchorYawRad = wrap(terminalYawRad - deltaYaw), [wx, wy] = rotate(lx, ly, anchorYawRad);
  return { anchorXY: [terminalXY[0] - wx, terminalXY[1] - wy], anchorYawRad, travelM: Math.hypot(lx, ly), deltaYawRad: deltaYaw };
}

/** Stable key for memoizing one evaluation inside a planning request. */
export function pickupPoseReachabilityKey({ candidateId, pickupPose, rootPositionWorld, rootYawRad, obstacles, respectTransitClearance }) {
  const pose = [pickupPose[0], pickupPose[1], yaw(pickupPose.slice(3, 7))].map(v => round(v)).join(',');
  const root = [rootPositionWorld[0], rootPositionWorld[1], rootYawRad ?? NaN].map(v => round(v)).join(',');
  const bounds = obstacles.map(o => [o.name ?? '', o.minX, o.maxX, o.minY, o.maxY, o.minZ, o.maxZ].map(v => typeof v === 'number' ? round(v, 4) : v).join(',')).join(';');
  return `${candidateId}|${pose}|${root}|${respectTransitClearance ? 't' : 'd'}|${bounds}`;
}

/** Geometric replay of the recorded approach from rootXY/rootYawRad through
 * waypoints to goalXY. Nominal source terminals stand in for tracked motion.
 * End states: arrival (parent may sample facing), refused (owned approach
 * refusal at distanceToGoalM > 0.25), needs_heading, unsupported_step_distance,
 * step_limit, turn_limit, standing_refused. */
export function simulateRecordedApproach({ rootXY, rootYawRad, goalXY, waypoints = [], obstacles, walkSkills, turnSkills = [], sweeps,
  limits = PICKUP_POSE_REACHABILITY_LIMITS, maxClips = 16 } = {}) {
  const steps = walkSkills.map(skill => describe(skill, sweeps.get(skill)));
  const turns = turnSkills.filter(skill => sweeps.get(skill)).map(skill => describe(skill, sweeps.get(skill)));
  const neutralSweep = steps[0].sweep;
  const targets = [...waypoints.map(p => [p[0], p[1]]), [goalXY[0], goalXY[1]]];
  let pos = [rootXY[0], rootXY[1]], heading = rootYawRad, index = 0, stepCount = 0, turnCount = 0;
  const clips = [];
  const finish = (end, extra = {}) => Object.freeze({ end, finalRootXY: Object.freeze([...pos]), finalYawRad: heading,
    distanceToGoalM: distance(pos, goalXY), clips: Object.freeze(clips), ...extra });
  const standingOk = () => checkAnchoredSweepGeometry({ sweep: neutralSweep, stance: true, anchorXY: pos, anchorYawRad: heading,
    obstacles, trackingReserve: limits.stepTrackingReserveM }).supported;
  const refused = reason => {
    // RestrictedLocomotionController retains a standing fallback; the recorded
    // approach hands off to the parent only inside its handoff radius.
    if (!standingOk()) return finish('standing_refused', { refusalReason: reason });
    if (distance(pos, goalXY) <= limits.parentArrivalRadiusM) return finish('arrival', { arrivalKind: 'parent_handoff_region', refusalReason: reason });
    return finish('refused', { refusalReason: reason });
  };
  for (let guard = 0; guard < maxClips; guard++) {
    while (index < targets.length && distance(pos, targets[index]) <= (index === targets.length - 1 ? limits.routeArrivalRadiusM : limits.waypointRadiusM)) index++;
    if (index === targets.length) return finish('arrival', { arrivalKind: 'floor_goal' });
    if (stepCount >= limits.maxSteps) return finish('step_limit');
    const target = targets[index], radius = index === targets.length - 1 ? limits.routeArrivalRadiusM : limits.waypointRadiusM;
    const remaining = distance(pos, target);
    const plan = planRecordedSteps(remaining, steps.map(s => s.travelM), radius, Math.min(3, limits.maxSteps - stepCount));
    if (!plan) return finish('unsupported_step_distance');
    const direction = Math.atan2(target[1] - pos[1], target[0] - pos[0]);
    const headingFor = s => wrap(direction + s.sourceYawRad - s.travelDirectionRad);
    let descriptor = steps[plan.indices[0]], desired = headingFor(descriptor), offset = wrap(desired - heading);
    if (Math.abs(offset) > limits.maxHeadingOffsetRad) {
      const turn = turns.map(t => ({ t, residual: Math.abs(wrap(offset - t.yawChangeRad)) }))
        .filter(c => c.residual < Math.abs(offset) - 1e-6).sort((a, b) => a.residual - b.residual)[0]?.t;
      if (!turn) return finish('needs_heading');
      if (turnCount >= limits.maxTurns) return finish('turn_limit');
      const geometry = checkAnchoredSweepGeometry({ sweep: turn.sweep, anchorXY: pos, anchorYawRad: heading, obstacles, trackingReserve: limits.turnTrackingReserveM });
      clips.push({ kind: 'turn', sweepName: turn.sweep.name, sourceFrames: turn.skill.sourceFrames, rootXY: [...pos], headingRad: heading, admitted: geometry.supported, obstacleName: geometry.obstacleName ?? null });
      if (!geometry.supported) return refused(geometry.reason);
      turnCount++;
      const rotation = heading - turn.sourceYawRad, [dx, dy] = rotate(turn.deltaXY[0], turn.deltaXY[1], rotation);
      pos = [pos[0] + dx, pos[1] + dy]; heading = wrap(heading + turn.yawChangeRad);
      continue;
    }
    // Admission at the live root with the step's aligned heading; a refusal
    // previews strictly shorter complete records at the same state.
    let admitted = null;
    const candidates = [descriptor, ...steps.filter(s => s.travelM < descriptor.travelM - 1e-6 && s.travelM <= remaining + radius
      && Math.abs(remaining - s.travelM) < remaining).sort((a, b) => b.travelM - a.travelM)];
    for (const candidate of candidates) {
      const candidateHeading = headingFor(candidate);
      if (candidate !== descriptor && Math.abs(wrap(candidateHeading - heading)) > limits.maxHeadingOffsetRad) continue;
      const geometry = checkAnchoredSweepGeometry({ sweep: candidate.sweep, anchorXY: pos, anchorYawRad: candidateHeading, obstacles, trackingReserve: limits.stepTrackingReserveM });
      clips.push({ kind: 'step', sweepName: candidate.sweep.name, sourceFrames: candidate.skill.sourceFrames, rootXY: [...pos], headingRad: candidateHeading,
        targetXY: [...target], admitted: geometry.supported, obstacleName: geometry.obstacleName ?? null, alternative: candidate !== descriptor });
      if (geometry.supported) { admitted = { candidate, heading: candidateHeading }; break; }
    }
    if (!admitted) return refused('reference_sweep_clearance');
    stepCount++;
    const rotation = admitted.heading - admitted.candidate.sourceYawRad, [dx, dy] = rotate(admitted.candidate.deltaXY[0], admitted.candidate.deltaXY[1], rotation);
    pos = [pos[0] + dx, pos[1] + dy]; heading = wrap(admitted.heading + admitted.candidate.yawChangeRad);
  }
  return finish('clip_limit');
}

/** Facing programme: some walking source ending at pickupXY(+offset) with the
 * pickup heading is admissible (whole + detailed hull, terminal stance) and its
 * entry has a 0.55 m transit route (exact entry preferred, recorded). */
export function planFacingProgramme({ pickupXY, pickupYawRad, objectPositionWorld, rootXY, obstacles, walkSkills, sweeps,
  offsets = PICKUP_ENTRY_REGION_OFFSETS_M, limits = PICKUP_POSE_REACHABILITY_LIMITS } = {}) {
  const dx = pickupXY[0] - objectPositionWorld[0], dy = pickupXY[1] - objectPositionWorld[1], length = Math.hypot(dx, dy);
  if (!(length > 1e-6)) return Object.freeze({ available: false, reason: 'pickup_pose_on_object', evaluated: 0, selected: null });
  const ux = dx / length, uy = dy / length, neutralSweep = sweeps.get(walkSkills[0]);
  let evaluated = 0, reason = 'no_supported_final_source', fallback = null;
  for (const [radial, tangent] of offsets) {
    if (Math.hypot(radial, tangent) > limits.parentArrivalRadiusM - limits.entryRegionMarginM + 1e-12) continue;
    const terminal = [pickupXY[0] + radial * ux - tangent * uy, pickupXY[1] + radial * uy + tangent * ux];
    if (obstacles.some(r => insideBounds(terminal, r))) continue;
    for (const skill of walkSkills) {
      const sweep = sweeps.get(skill); evaluated++;
      const { anchorXY, anchorYawRad } = sweepAnchorFromTerminal(sweep, terminal, pickupYawRad);
      if (obstacles.some(r => insideBounds(anchorXY, r))) continue;
      const whole = checkAnchoredSweepGeometry({ sweep, anchorXY, anchorYawRad, obstacles, trackingReserve: limits.stepTrackingReserveM });
      if (!whole.supported) continue;
      const detailed = checkAnchoredSweepGeometry({ sweep, anchorXY, anchorYawRad, obstacles, trackingReserve: limits.stepTrackingReserveM,
        heightAware: true, verticalTrackingReserve: limits.verticalTrackingReserveM });
      if (!detailed.supported) continue;
      const stance = checkAnchoredSweepGeometry({ sweep: neutralSweep, stance: true, anchorXY: terminal, anchorYawRad: pickupYawRad, obstacles, trackingReserve: limits.stepTrackingReserveM });
      if (!stance.supported) continue;
      const navigation = planNavigationPath(rootXY, anchorXY, obstacles, limits.transitClearanceM);
      const entryRouteExact = navigation.reachableGoal !== null && navigation.path.length > 0 && distance(navigation.reachableGoal, anchorXY) <= 1e-7;
      const selected = Object.freeze({ offsetM: Object.freeze([radial, tangent]), terminalGoalWorld: Object.freeze([...terminal]),
        sweepName: sweep.name, sourceFrames: skill.sourceFrames, entryPose: Object.freeze([...anchorXY, anchorYawRad]), entryRouteExact,
        entryRouteLengthM: navigation.path.reduce((acc, p) => ({ sum: acc.sum + distance(acc.prev, p), prev: p }), { sum: 0, prev: rootXY }).sum });
      if (entryRouteExact) return Object.freeze({ available: true, reason: null, evaluated, selected });
      fallback ??= selected; reason = 'staging_route_projected';
    }
  }
  return Object.freeze({ available: false, reason, evaluated, selected: fallback });
}

/**
 * pickupPose: full747 first frame (or any >=7 vector: root XYZ + XYZW quaternion).
 * objectPositionWorld: the box the reference was anchored on. rootPositionWorld /
 * rootYawRad: actual or predicted root (yaw required for the dry-run heading
 * rule; null falls back to the direction of the first route leg).
 * obstacles: live mesh bounds ({minX..maxZ,name}) for every scene box.
 * walkSkills/turnSkills/sweeps: the bounded loaded library (walkSkills[0] is the
 * neutral stance source). Never a physical guarantee.
 */
export function evaluatePickupPoseReachability({ pickupPose, objectPositionWorld, rootPositionWorld, rootYawRad = null,
  obstacles, walkSkills, turnSkills = [], sweeps, offsets = PICKUP_ENTRY_REGION_OFFSETS_M, respectTransitClearance = true,
  limits = PICKUP_POSE_REACHABILITY_LIMITS } = {}) {
  const no = (reason, extra = {}) => Object.freeze({ reachable: false, reason, approach: null, facingProgramme: null, ...extra });
  if (!(pickupPose?.length >= 7) || !Array.from(pickupPose.slice(0, 7)).every(Number.isFinite)
      || !finite(objectPositionWorld?.slice(0, 2), 2) || !finite(rootPositionWorld?.slice(0, 2), 2)
      || (rootYawRad !== null && !Number.isFinite(rootYawRad))) return no('invalid_pose_geometry');
  if (!Array.isArray(walkSkills) || !walkSkills.length || !(sweeps instanceof Map) || walkSkills.some(skill => !sweeps.get(skill))
      || !Array.isArray(turnSkills)) return no('bounded_loaded_source_library_required');
  if (!Array.isArray(obstacles) || !obstacles.every(r => r && ['minX', 'maxX', 'minY', 'maxY', 'minZ', 'maxZ'].every(k => Number.isFinite(r[k]))
      && r.minX < r.maxX && r.minY < r.maxY && r.minZ < r.maxZ)) return no('complete_obstacle_geometry_required');
  const pickupXY = [pickupPose[0], pickupPose[1]], pickupYawRad = yaw(pickupPose.slice(3, 7));
  const rootXY = [rootPositionWorld[0], rootPositionWorld[1]];
  let route;
  try { route = planBoxApproach(rootXY, pickupXY, obstacles, { clearance: limits.transitClearanceM, respectTransitClearance }); }
  catch { return no('invalid_route_geometry'); }
  const routeSummary = { supported: route.supported, reason: route.reason ?? null, routed: route.routed, directBlocked: route.directBlocked,
    pathLengthM: route.pathLengthM, waypoints: route.path.map(p => [...p]) };
  const facingProgramme = planFacingProgramme({ pickupXY, pickupYawRad, objectPositionWorld, rootXY, obstacles, walkSkills, sweeps, offsets, limits });
  if (!route.supported) return no('approach_route_unavailable', { route: routeSummary, facingProgramme, pickupYawRad });
  const firstLeg = route.path.length ? route.path[0] : pickupXY;
  const startYaw = rootYawRad ?? Math.atan2(firstLeg[1] - rootXY[1], firstLeg[0] - rootXY[0]);
  const approach = simulateRecordedApproach({ rootXY, rootYawRad: startYaw, goalXY: pickupXY, waypoints: route.path, obstacles, walkSkills, turnSkills, sweeps, limits });
  const predictedArrivalYawRad = approach.end === 'arrival' ? wrap(approach.finalYawRad + limits.arrivalYawBiasRad) : null;
  const facingErrorRad = approach.end === 'arrival' ? wrap(pickupYawRad - predictedArrivalYawRad) : null;
  const directUnblocked = !obstacles.some(r => segmentEntersRectangle(approach.finalRootXY, pickupXY, r));
  const studentRecoveryEligible = approach.end === 'refused' && approach.distanceToGoalM <= limits.studentRecoveryDistanceM && directUnblocked;
  // A final step refused inside the parent's handoff radius is the ambiguous
  // zone: a few centimetres decide between a heading-preserving handoff and an
  // owned refusal that the combined recoveries reorient toward the box. Both
  // observed outcomes exist; report it as reachable but uncertain.
  const handoffZone = approach.end === 'arrival' && approach.arrivalKind === 'parent_handoff_region';
  let reachable, reason, uncertain = false;
  if (approach.end === 'arrival') {
    const facingOk = Math.abs(facingErrorRad) <= limits.maxFacingErrorRad - limits.facingMarginRad;
    reachable = facingOk || handoffZone; uncertain = !facingOk && handoffZone;
    reason = reachable ? null : 'predicted_arrival_facing_outside_parent_limit';
  } else if (approach.end === 'refused') {
    reachable = studentRecoveryEligible; uncertain = reachable;
    reason = reachable ? null : 'predicted_approach_refusal_beyond_recovery_window';
  } else { reachable = false; reason = `predicted_approach_${approach.end}`; }
  return Object.freeze({ reachable, reason, uncertain, pickupYawRad, route: routeSummary,
    approach: Object.freeze({ ...approach, predictedArrivalYawRad, facingErrorRad, studentRecoveryEligible, handoffZone, rootYawAssumed: rootYawRad === null }), facingProgramme });
}
