// Private footprint trial: preflight uses the same branch/planner as execution.
// The existing compiled path/destination checks and reserves stay unchanged.
import { planCarrySegments } from './teacher_carry_sequence.js';
import { checkCarrySegmentClearance } from './carry_request_clearance.js';
import { planAlignedGroundPush } from './ground_push_mapping.js';
import { groundPushBranch } from './teacher_footprint_ground_push_controller.js';

export function checkGroundPushRequestClearance({skill,objectPositionWorld,requestedGoalWorld,liveData,destinationGeometry,pathChecker}) {
  const branch=groundPushBranch(skill);
  const plan=planCarrySegments(objectPositionWorld,requestedGoalWorld,skill,{maxSegments:1,maxCorrection:.05});
  if(!plan.supported)return{supported:false,reason:plan.reason,plan,checks:[]};
  if(plan.goals.length!==1)throw new Error('One complete ground push is required');
  const body=destinationGeometry.objects.find(object=>object.name===skill.objectBodyName);
  if(!body)throw new Error('The pushed object is absent from compiled collision geometry');
  const wxyz=liveData.xquat.slice(body.bodyId*4,body.bodyId*4+4),quat=[wxyz[1],wxyz[2],wxyz[3],wxyz[0]];
  const [startFrame,endFrame]=skill.pushInterval;
  const reference=planAlignedGroundPush(skill.frames,skill.sourceFrames,objectPositionWorld,quat,requestedGoalWorld,
    {objectHeadingOffsetRadians:branch.objectHeadingOffsetRadians,startFrame,endFrame,maxCorrection:.05});
  const check=checkCarrySegmentClearance({referenceFrames:reference.frames,sourceFrames:skill.sourceFrames,
    objectBodyName:skill.objectBodyName,requestedGoalWorld,liveData,destinationGeometry,pathChecker});
  return{supported:check.supported,reason:check.reason,requestedGoalWorld:Array.from(requestedGoalWorld),
    checks:[{segmentIndex:0,...check}],pushReferenceAlignment:{transform:reference.transform,
      actualObjectQuaternion:quat,objectHeadingOffsetRadians:branch.objectHeadingOffsetRadians,
      referenceGoalWorld:Array.from(reference.referenceGoalWorld)}};
}
