// Pure reference/route proposal. No execution, arrival, or settling credit.
import {turnReferenceTransform} from './teacher_turn_controller.js';
import {transformTeacherReference} from './teacher_reference.js';
import {planTeacherStandingReference} from './teacher_standing_reference.js';
import {bindMotionSweep,checkRestrictedReferenceSweep} from './restricted_motion_geometry.js';
import {planNavigationPath} from './navigation_planner.js';

export const PICKUP_FACING_APPROACH_LIMITS=Object.freeze({transitClearanceM:.55,trackingReserveM:.1,
  verticalTrackingReserveM:.1,maximumSources:4,minimumLookahead:16,settlingControls:60,quietSamples:12,
  maxIncomingHeadingOffsetRad:Math.PI/3});
const finite=(v,n)=>v?.length===n&&Array.from(v).every(Number.isFinite);
const quat=q=>finite(q,4)&&Math.abs(Math.hypot(...q)-1)<=1e-5;
const count=v=>Number.isSafeInteger(v)&&v>=0;
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const frozen=v=>Object.freeze(Array.from(v));
const yaw=q=>Math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2));
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const pointGap=(p,r)=>Math.hypot(Math.max(r.minX-p[0],0,p[0]-r.maxX),Math.max(r.minY-p[1],0,p[1]-r.maxY));
const no=(reason,candidates=[])=>({supported:false,reason,selected:null,candidates});

/** Select a complete final walking record whose original terminal XY/yaw fits
 * the parent's existing pickup pose. stepSkills[0] remains the current neutral
 * source. All sources and sweeps come from the caller's already loaded library.
 * The selected source is a mandatory future obligation, not an ordinary
 * skippable waypoint. Entry is a reference proposal, never measured arrival.
 */
export function planPickupFacingApproach({parent,stepSkills,sweeps,obstacles,live,episode,requestId,physicalControl,terminalGoalWorld=null}={}) {
  const limits=PICKUP_FACING_APPROACH_LIMITS;
  if(![episode,requestId,physicalControl].every(count))return no('invalid_identity');
  const first=parent?.referencePlan?.first;
  if(parent?.phase!=='approach'||parent.finishRequested!==false||parent.referenceIndex!==0
      ||!finite(parent.approachGoalWorld,3)||!finite(parent.requestedGoalWorld,3)||!finite(first,747)
      ||!quat(first.slice(3,7))||!same(first.slice(0,2),parent.approachGoalWorld.slice(0,2))
      ||!Number.isFinite(parent.arrivalRadius)||parent.arrivalRadius<=0
      ||!Number.isFinite(parent.maxFacingError)||parent.maxFacingError<=0||parent.maxFacingError>Math.PI)
    return no('unstarted_parent_pickup_pose_required');
  if(!finite(live?.rootPosWorld,3)||!quat(live.rootQuatXyzwWorld)||!finite(live.rootVelWorld,3)
      ||!finite(live.objPosWorld,3)||!quat(live.objQuatXyzwWorld)||typeof parent.skill?.objectBodyName!=='string'
      ||live.objectBodyName!==parent.skill.objectBodyName)return no('invalid_live_state_or_object');
  if(!Array.isArray(stepSkills)||stepSkills.length<1||stepSkills.length>limits.maximumSources
      ||new Set(stepSkills).size!==stepSkills.length||!(sweeps instanceof Map))return no('bounded_loaded_source_library_required');
  if(!Array.isArray(obstacles)||!obstacles.length||!obstacles.every(r=>r&&['minX','maxX','minY','maxY','minZ','maxZ'].every(k=>Number.isFinite(r[k]))
      &&r.minX<r.maxX&&r.minY<r.maxY&&r.minZ<r.maxZ))return no('complete_obstacle_geometry_required');
  const goal=frozen(parent.approachGoalWorld);
  const terminalGoal=terminalGoalWorld??goal;
  if(!finite(terminalGoal,3)||terminalGoal[2]!==goal[2]
      ||terminalGoalWorld!==null&&distance(terminalGoal,goal)>parent.arrivalRadius-.06+1e-12)
    return no('terminal_goal_outside_existing_pickup_region');
  const pickupPose=frozen([terminalGoal[0],terminalGoal[1],...first.slice(2,7)]),targetYaw=yaw(pickupPose.slice(3,7));
  const neutral=stepSkills[0],neutralSweep=sweeps.get(neutral),candidates=[];
  // Reject an incomplete supplied library before producing a misleading subset.
  for(const skill of stepSkills){
    if(skill?.locomotionOnly!==true||!Number.isSafeInteger(skill.sourceFrames)||skill.sourceFrames<2
        ||!Array.isArray(skill.frames)||skill.frames.length<skill.sourceFrames+limits.minimumLookahead
        ||!skill.frames.every(frame=>finite(frame,747))||!quat(skill.frames[0].slice(3,7))
        ||!quat(skill.frames[skill.sourceFrames-1].slice(3,7)))return no('incomplete_source_library');
    const sweep=sweeps.get(skill);
    try {bindMotionSweep(skill,{sweeps:[sweep]},sweep?.name);}catch{return no('unbound_source_geometry');}
    if(!Array.isArray(sweep.collisionParts)||!sweep.collisionParts.length)return no('missing_source_part_geometry');
  }
  for(const [index,skill] of stepSkills.entries()){
    const sweep=sweeps.get(skill),last=skill.frames[skill.sourceFrames-1];
    const transform=turnReferenceTransform(last,pickupPose.slice(0,3),pickupPose.slice(3,7));
    const frames=skill.frames.map(frame=>transformTeacherReference(frame,transform));
    const entry=frozen(frames[0].slice(0,7)),terminal=frozen(frames[skill.sourceFrames-1].slice(0,7));
    const geometryOptions={sweep,sourceFrames:skill.sourceFrames,alignedReferenceFrames:frames,obstacles,trackingReserve:limits.trackingReserveM};
    const wholeGeometry=checkRestrictedReferenceSweep(geometryOptions);
    const detailedGeometry=checkRestrictedReferenceSweep({...geometryOptions,heightAware:true,verticalTrackingReserve:limits.verticalTrackingReserveM});
    let standingPlan;
    try {standingPlan=planTeacherStandingReference(neutral.frames[neutral.sourceFrames-1],{alignment:'live-root',
      rootPosition:terminal.slice(0,3),rootQuaternion:terminal.slice(3,7),objectPosition:live.objPosWorld,
      objectQuaternion:live.objQuatXyzwWorld,objectPointsLocal:neutral.objectPointsLocal});}
    catch{return no('invalid_neutral_standing_source');}
    const terminalStandingGeometry=checkRestrictedReferenceSweep({sweep:neutralSweep,sourceFrames:1,
      alignedReferenceFrames:[standingPlan.frame],obstacles,trackingReserve:limits.trackingReserveM});
    const navigation=planNavigationPath(live.rootPosWorld,entry,obstacles,limits.transitClearanceM);
    const exactStage=navigation.reachableGoal!==null&&distance(navigation.reachableGoal,entry)<=1e-7;
    let routeLengthM=0,previous=live.rootPosWorld;
    for(const point of navigation.path){routeLengthM+=distance(previous,point);previous=point;}
    const terminalPositionErrorM=distance(terminal,goal),terminalFacingErrorRad=wrap(yaw(terminal.slice(3,7))-targetYaw);
    const reason=!wholeGeometry.supported?'final_source_whole_geometry_refused':!detailedGeometry.supported?'final_source_detailed_geometry_refused'
      :!terminalStandingGeometry.supported?'terminal_canonical_geometry_refused':!exactStage?'projected_staging_goal'
      :!navigation.path.length?'staging_route_unavailable':terminalPositionErrorM>parent.arrivalRadius?'terminal_outside_parent_radius'
      :Math.abs(terminalFacingErrorRad)>parent.maxFacingError?'terminal_facing_outside_parent_limit':null;
    candidates.push(Object.freeze({supported:reason===null,reason,sourceIndex:index,sourceSkill:skill,sourceFrames:skill.sourceFrames,
      sourceTravelM:distance(skill.frames[0],last),referenceFrames:Object.freeze(frames),
      transform:Object.freeze({yawRadians:transform.yawRadians,translation:frozen(transform.translation)}),
      requiredEntryPose:entry,requiredEntryYawRad:yaw(entry.slice(3,7)),referenceTerminalPose:terminal,
      expectedTerminalPositionErrorM:terminalPositionErrorM,expectedTerminalFacingErrorRad:terminalFacingErrorRad,
      entryPointClearanceBeyondTransitM:Math.min(...obstacles.map(r=>pointGap(entry,r)))-limits.transitClearanceM,
      wholeGeometry,detailedGeometry,terminalStandingGeometry,
      transit:Object.freeze({clearanceM:limits.transitClearanceM,exactEntry:exactStage,
        reachableGoal:navigation.reachableGoal?frozen(navigation.reachableGoal):null,
        waypoints:Object.freeze(navigation.path.map(frozen)),lengthM:routeLengthM}),
      obligation:Object.freeze({mandatory:true,maySkipOnWaypointArrival:false,sourceFrames:skill.sourceFrames,
        providedLookaheadFrames:frames.length-skill.sourceFrames,expectedActualReferenceStartIndex:0,
        settleControlsAfterFullSource:limits.settlingControls,quietSamples:limits.quietSamples,
        parentArrivalRadiusM:parent.arrivalRadius,parentMaxFacingErrorRad:parent.maxFacingError,
        maxIncomingHeadingOffsetRad:limits.maxIncomingHeadingOffsetRad})}));
  }
  const supported=candidates.filter(c=>c.supported).sort((a,b)=>b.entryPointClearanceBeyondTransitM-a.entryPointClearanceBeyondTransitM
    ||a.sourceFrames-b.sourceFrames||a.transit.lengthM-b.transit.lengthM||a.sourceIndex-b.sourceIndex);
  if(!supported.length)return no('no_supported_pickup_facing_final_source',candidates);
  return Object.freeze({supported:true,reason:null,parent,episode,requestId,physicalControl,
    neutralSkill:neutral,neutralSweep,
    originalGoalWorld:frozen(parent.requestedGoalWorld),approachGoalWorld:goal,terminalGoalWorld:frozen(terminalGoal),pickupPose,pickupYawRad:targetYaw,
    sourceParentFrame:first,objectBodyName:parent.skill.objectBodyName,
    selectionRule:'maximum_entry_point_clearance_beyond_unchanged_transit_reserve_then_fewer_controls_then_shorter_route',
    selected:supported[0],candidates:Object.freeze(candidates)});
}
