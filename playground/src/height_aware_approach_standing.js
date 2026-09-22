// Private-experiment admission only. The caller owns clocks, history, physics,
// invalidation after movement, and every complete all-box action preview.
import {checkRestrictedReferenceSweep} from './restricted_motion_geometry.js';
import {planTeacherStandingReference} from './teacher_standing_reference.js';

export const HEIGHT_AWARE_APPROACH_STANDING_LIMITS=Object.freeze({trackingReserveM:.1,verticalTrackingReserveM:.1,
  maxPlanarSpeedMps:.05,minUpright:.95,minRootHeightM:.7});
const finite=(v,n)=>v?.length===n&&Array.from(v).every(Number.isFinite);
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const count=v=>Number.isSafeInteger(v)&&v>=0;
const quaternion=q=>finite(q,4)&&Math.abs(Math.hypot(...q)-1)<=1e-5;
const no=reason=>({supported:false,reason,frame:null,requiresPreview:false});

/** Reconsider ONLY the current canonical standing target, after a real full
 * approach source ended inside its unchanged parent radius. No target is
 * replaced and no prior settling/arrival/facing credit is created. Identity of
 * the TeacherRecordedApproachController is checked by the caller; its actual
 * inner pending-standing plan and saved completed-source lease are checked here.
 */
export function checkHeightAwareApproachStanding({owner,parent,savedTerminal,request,sweep,wholeGeometry,
  obstacles,live,episode,requestId,physicalControl}={}) {
  if(![episode,requestId,physicalControl].every(count))return no('invalid_identity');
  const inner=owner?.controller,limits=HEIGHT_AWARE_APPROACH_STANDING_LIMITS;
  if(!inner||owner.cancelRequested!==false||['inactive','complete','unsupported'].includes(owner.phase)
      ||parent?.phase!=='approach'||parent.finishRequested!==false||parent.referenceIndex!==0
      ||!finite(parent.approachGoalWorld,3)||!finite(parent.requestedGoalWorld,3)
      ||!same(owner.requestedGoalWorld,parent.approachGoalWorld)
      ||!same(inner.requestedGoalWorld,parent.approachGoalWorld)
      ||!Number.isFinite(parent.arrivalRadius)||parent.arrivalRadius<=0)return no('approach_not_owned');
  const saved=savedTerminal;
  if(!saved||saved.owner!==owner||saved.parent!==parent||saved.episode!==episode||saved.requestId!==requestId
      ||!same(saved.approachGoalWorld,parent.approachGoalWorld)||!same(saved.originalGoalWorld,parent.requestedGoalWorld)
      ||!count(saved.executedAtControl)||saved.executedAtControl<1||saved.executedAtControl>physicalControl
      ||!Number.isFinite(saved.endpointDistanceM)||saved.endpointDistanceM<0||saved.endpointDistanceM>parent.arrivalRadius)
    return no('completed_terminal_not_owned');
  if(saved.skill?.locomotionOnly!==true||!Number.isSafeInteger(saved.skill.sourceFrames)||saved.skill.sourceFrames<2
      ||!Array.isArray(saved.skill.frames)||saved.skill.frames.length<saved.skill.sourceFrames+16||!finite(saved.terminal,747)
      ||!quaternion(saved.terminal.slice(3,7))||!finite(saved.plan?.frame,747))return no('invalid_completed_terminal');
  const plan=inner._standing,skill=inner.neutralSkill,frame=request?.alignedReferenceFrames?.[0];
  if(!['teacher_settling','teacher_standing'].includes(request?.phase)||inner.phase!==request.phase
      ||request.sourceFrames!==1||request.alignedReferenceFrames?.length!==1
      ||request.skill!==skill||inner.standingSkill!==skill||skill?.locomotionOnly!==true
      ||!Number.isSafeInteger(skill.sourceFrames)||skill.sourceFrames<2||sweep?.sourceFrames!==skill.sourceFrames
      ||!Array.isArray(skill.frames)||skill.frames.length<skill.sourceFrames+16||!finite(skill.frames[skill.sourceFrames-1],747)
      ||!plan||request.referencePlan!==plan||plan.frame!==frame||!finite(frame,747)
      ||plan.alignment!=='live-root'||!same(request.requestedGoalWorld,parent.approachGoalWorld))
    return no('current_canonical_standing_required');
  if(wholeGeometry?.supported!==false||wholeGeometry.reason!=='reference_sweep_clearance'
      ||wholeGeometry.trackingReserve!==limits.trackingReserveM||wholeGeometry.heightAware===true
      ||!Array.isArray(wholeGeometry.worldHull)||!wholeGeometry.worldHull.every(p=>finite(p,2)))
    return no('original_whole_sweep_must_refuse');
  if(!finite(live?.rootPosWorld,3)||!quaternion(live.rootQuatXyzwWorld)||!finite(live.rootVelWorld,3)
      ||!finite(live.objPosWorld,3)||!quaternion(live.objQuatXyzwWorld)
      ||typeof parent.skill?.objectBodyName!=='string'||live.objectBodyName!==parent.skill.objectBodyName)
    return no('invalid_live_state_or_object');
  const q=live.rootQuatXyzwWorld,upright=1-2*(q[0]**2+q[1]**2),speed=Math.hypot(...live.rootVelWorld.slice(0,2));
  const distance=Math.hypot(live.rootPosWorld[0]-parent.approachGoalWorld[0],live.rootPosWorld[1]-parent.approachGoalWorld[1]);
  if(live.rootPosWorld[2]<limits.minRootHeightM||upright<limits.minUpright||speed>limits.maxPlanarSpeedMps)
    return no('entry_not_quiet_standing');
  if(distance>parent.arrivalRadius)return no('outside_parent_region');
  // _neutral can anchor at the current root or the actual completed floor goal.
  // Rebuild only to validate all747 original targets; return the original object.
  const goal=plan.requestedGoalWorld;
  if(goal!==null&&(!finite(goal,3)||!same(goal,parent.approachGoalWorld)
      ||!Number.isFinite(inner.options?.arrivalRadius)
      ||distance>inner.options.arrivalRadius+1e-6))return no('unsupported_standing_alignment');
  let expected;
  try {expected=planTeacherStandingReference(skill.frames[skill.sourceFrames-1],{alignment:'live-root',
    rootPosition:goal===null?live.rootPosWorld:[goal[0],goal[1],live.rootPosWorld[2]],
    rootQuaternion:live.rootQuatXyzwWorld,objectPosition:live.objPosWorld,objectQuaternion:live.objQuatXyzwWorld,
    objectPointsLocal:skill.objectPointsLocal}).frame;}catch{return no('invalid_canonical_source');}
  if(!same(frame,expected))return no('canonical_frame_changed');
  if(!Array.isArray(obstacles)||obstacles.length===0)return no('missing_obstacle_geometry');
  // Recheck the supplied refusal against the same current frame and bounds so
  // stale/mismatched whole-geometry evidence cannot admit a different request.
  const options={sweep,sourceFrames:1,alignedReferenceFrames:request.alignedReferenceFrames,obstacles,
    trackingReserve:limits.trackingReserveM};
  const originalGeometry=checkRestrictedReferenceSweep(options);
  if(originalGeometry.supported!==false||originalGeometry.reason!=='reference_sweep_clearance'
      ||originalGeometry.obstacleIndex!==wholeGeometry.obstacleIndex
      ||!same(originalGeometry.worldHull?.flat(),wholeGeometry.worldHull?.flat()))
    return no('original_geometry_changed');
  const geometry=checkRestrictedReferenceSweep({...options,heightAware:true,verticalTrackingReserve:limits.verticalTrackingReserveM});
  return geometry.supported?{supported:true,reason:null,frame,requiresPreview:true,geometry,originalGeometry,
    distanceM:distance,upright,planarSpeedMps:speed,executedAtControl:saved.executedAtControl}
    :{...no(geometry.reason),geometry,originalGeometry};
}
