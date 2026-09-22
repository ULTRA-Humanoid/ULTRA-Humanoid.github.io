// Evaluate the exact intended carry segments before beginning a box request.
// Actual entry and tracking remain separate; this does not move the user's goal.
import { planCarrySegments } from './teacher_carry_sequence.js';
import { planCarryToGoal } from './teacher_goal_warp.js';
import { checkCarryDestinationFootprint } from './carry_destination_footprint.js';
import { CARRY_REFERENCE_TRACKING_RESERVE_M } from './carry_reference_clearance.js';

/** Check an already planned segment, including an entry-time replan. */
export function checkCarrySegmentClearance({ referenceFrames, sourceFrames, objectBodyName,
  requestedGoalWorld, liveData, destinationGeometry, pathChecker }) {
  const object = destinationGeometry.objects.find(value => value.name === objectBodyName);
  if (!object) throw new Error('The carried object is absent from compiled collision geometry');
  const destination = checkCarryDestinationFootprint({ object, requestedGoalWorld,
    finalReferenceFrame: referenceFrames[sourceFrames - 1], obstacles: destinationGeometry.read(liveData),
    trackingReserve: CARRY_REFERENCE_TRACKING_RESERVE_M });
  const path = destination.supported ? pathChecker.check(liveData, {
    referenceFrames, sourceFrames, objectBodyName }) : null;
  return { supported: destination.supported && path.supported,
    reason: destination.supported ? path.reason : destination.reason, destination, path };
}

export function checkCarryRequestClearance({ skill, objectPositionWorld, requestedGoalWorld,
  liveData, destinationGeometry, pathChecker }) {
  const plan = planCarrySegments(objectPositionWorld, requestedGoalWorld, skill);
  if (!plan.supported) return { supported: false, reason: plan.reason, plan, checks: [] };
  const [startFrame, endFrame] = skill.carryInterval ?? [];
  if (!Number.isInteger(startFrame) || !Number.isInteger(endFrame)) {
    throw new Error('The complete carry interval is required for request preflight');
  }
  const checks = [];
  let from = Array.from(objectPositionWorld);
  for (const [segmentIndex, goal] of plan.goals.entries()) {
    const reference = planCarryToGoal(skill.frames, skill.sourceFrames, from, goal,
      { startFrame, endFrame, maxCorrection: .25 });
    const check = checkCarrySegmentClearance({ referenceFrames: reference.frames,
      sourceFrames: skill.sourceFrames, objectBodyName: skill.objectBodyName,
      requestedGoalWorld: goal, liveData, destinationGeometry, pathChecker });
    checks.push({ segmentIndex, ...check });
    if (!check.supported) {
      return { supported: false, reason: check.reason,
        requestedGoalWorld: Array.from(requestedGoalWorld), segmentIndex, checks };
    }
    from = Array.from(reference.referenceGoalWorld);
  }
  return { supported: true, reason: null, requestedGoalWorld: Array.from(requestedGoalWorld), checks };
}
