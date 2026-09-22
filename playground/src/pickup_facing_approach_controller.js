/** Optional selective approach owner. Source clocks stay in existing controllers. */
import {TeacherRecordedApproachController} from './teacher_recorded_approach_controller.js';
import {TeacherWaypointController} from './teacher_waypoint_controller.js';
import {prepareTeacherFacingTurn} from './teacher_turn_controller.js';
import {planTeacherStandingReference} from './teacher_standing_reference.js';
import {checkRestrictedReferenceSweep} from './restricted_motion_geometry.js';
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const yaw=q=>Math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2)),wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
const distance=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
export class PickupFacingApproachController{
 constructor({plan,stepSkills,turnSkills,approveReference,readObstacles}){
  if(!plan?.supported||typeof approveReference!=='function'||typeof readObstacles!=='function')throw new Error('Qualified geometric plan and actual geometry readers required');
  Object.assign(this,{plan,parent:plan.parent,stepSkills,turnSkills,approveReference,readObstacles});
  this.phase='inactive';this.stage='inactive';this.delegate=null;this.cancelRequested=false;this.controls=0;
  this.turnCount=0;this.events=[];this.pendingAdvance=false;this.completed=false;this.invalidated=false;
 }
 get skill(){return this.delegate?.skill??this.stepSkills[0];}
 get sourceFrames(){return this.delegate?.sourceFrames??0;}
 get referenceIndex(){return this.delegate?.referenceIndex??0;}
 get locomotionOnly(){return true;}
 get requestedGoalWorld(){return Array.from(this.plan.approachGoalWorld);}
 isOwnedBy(c){return !this.invalidated&&c.owner===this&&c.parent===this.parent&&c.episode===this.plan.episode&&c.requestId===this.plan.requestId
  &&same(this.parent.requestedGoalWorld,this.plan.originalGoalWorld)&&same(this.parent.approachGoalWorld,this.plan.approachGoalWorld)
  &&this.parent.referenceIndex===0&&this.parent.referencePlan.first===this.plan.sourceParentFrame;}
 reset(){
  // Main resets physics/history separately and then replaces this owner. Revoke
  // it immediately, including an already queried candidate in the same episode.
  // Retain actual completed counts/events for review; reset is not execution.
  this.invalidated=true;this.pendingAdvance=false;this.cancelRequested=true;
  this.delegate?.reset?.();this.phase='unsupported';this.stage='reset';
  this.completionReason='reset';this.completed=true;
 }
 #record(kind,live,details={}){this.events.push({kind,controls:this.controls,rootPositionWorld:Array.from(live.rootPosWorld),
  rootYawRad:yaw(live.rootQuatXyzwWorld),rootPlanarSpeedMps:Math.hypot(...live.rootVelWorld.slice(0,2)),...details});}
 #recorded(live,goal,waypoints=[]){const owner=new TeacherRecordedApproachController(this.stepSkills,{turnSkills:this.turnSkills,
  approveReference:this.approveReference,handoffRadius:.1});owner.start(live,{finalGoalWorld:goal,waypoints});return owner;}
 start(live){if(this.phase!=='inactive')throw new Error('Finite approach starts once');
  const stage=this.plan.selected.requiredEntryPose;
  this.delegate=this.#recorded(live,[stage[0],stage[1],0],this.plan.selected.transit.waypoints.slice(0,-1).map(p=>[...p,0]));
  this.stage='transit';this.phase='approach';this.#record('transit_started',live,{requiredEntryPose:Array.from(stage)});}
 requestCancel(){this.cancelRequested=true;this.delegate?.requestCancel();}
 #stop(reason,live){this.phase=reason==='cancelled'?'complete':'unsupported';this.completionReason=reason;this.pendingAdvance=false;
  this.#record('ended',live,{reason});return this.#terminalResult();}
 #terminalResult(){const result={phase:this.phase,mode:this.phase==='complete'?'student':'none',supported:this.phase==='complete',
  justCompleted:!this.completed&&this.phase==='complete',completionReason:this.completionReason,referenceFrames:null,
  outcome:{controls:this.controls,stages:this.events,completionReason:this.completionReason},requestedGoalWorld:this.requestedGoalWorld};
  if(result.justCompleted)this.completed=true;
  if(this.completionReason==='finished')result.handoffReferenceFrames=this.handoffFrames.map(f=>Float32Array.from(f));
  return result;}
 #quiet(live,next){this.delegate=this.#recorded(live,[live.rootPosWorld[0],live.rootPosWorld[1],0]);
  this.stage=next;this.#record(next+'_started',live);}
 #prepareFinal(live){
  const q=live.rootQuatXyzwWorld,speed=Math.hypot(...live.rootVelWorld.slice(0,2)),up=1-2*(q[0]**2+q[1]**2);
  if(speed>.05||live.rootPosWorld[2]<.7||up<.95)return this.#stop('final_entry_not_quiet',live);
  const selected=this.plan.selected,goal=this.plan.terminalGoalWorld??this.plan.approachGoalWorld;
  // The original transit needs its own actual arrival, before any facing turn.
  if(this.turnCount===0&&distance(live.rootPosWorld,selected.requiredEntryPose)>.1)return this.#stop('staging_not_reached',live);
  const executor=new TeacherWaypointController([selected.sourceSkill],{turnSkills:[],arrivalRadius:.25,
    maxSteps:1,maxTurns:0,settlingPolicy:'teacher'});
  const d=executor.steps[0],direction=Math.atan2(goal[1]-live.rootPosWorld[1],goal[0]-live.rootPosWorld[0]);
  const desired=wrap(direction+d.sourceYawRad-d.travelDirectionRad),error=wrap(yaw(q)-desired);
  if(Math.abs(error)>Math.PI/3){
    if(this.turnCount>=4)return this.#stop('turn_limit',live);
    const turn=prepareTeacherFacingTurn(this.turnSkills,live,error,{approveReference:this.approveReference});
    if(!turn.supported)return this.#stop(turn.reason,live);
    this.turnCount++;this.delegate=turn.controller;this.stage='facing_turn';
    this.#record('full_facing_turn_started',live,{sourceControls:this.delegate.sourceFrames,desiredHeadingRad:desired});return null;
  }
  executor.start(live,{finalGoalWorld:goal});
  if(executor.phase!=='teacher_step'||executor.skill!==selected.sourceSkill||executor.referenceIndex!==0)
    return this.#stop('mandatory_final_source_not_planned',live);
  const terminal=executor.worldFrames[selected.sourceFrames-1],positionError=distance(terminal,this.plan.approachGoalWorld),facing=wrap(yaw(terminal.slice(3,7))-this.plan.pickupYawRad);
  if(positionError>this.parent.arrivalRadius||Math.abs(facing)>this.parent.maxFacingError)return this.#stop('final_reference_misses_parent_pose',live);
  const approval=this.approveReference(live,{phase:'teacher_step',skill:executor.skill,sourceFrames:executor.sourceFrames,
    alignedReferenceFrames:executor.worldFrames,referencePlan:executor.referencePlan,requestedGoalWorld:this.requestedGoalWorld});
  if(approval?.supported!==true)return this.#stop(approval?.reason??'final_reference_geometry',live);
  const neutral=this.stepSkills[0],stance=planTeacherStandingReference(neutral.frames[neutral.sourceFrames-1],{alignment:'live-root',
    rootPosition:terminal.slice(0,3),rootQuaternion:terminal.slice(3,7),objectPosition:live.objPosWorld,
    objectQuaternion:live.objQuatXyzwWorld,objectPointsLocal:neutral.objectPointsLocal});
  const terminalGeometry=checkRestrictedReferenceSweep({sweep:this.plan.neutralSweep,sourceFrames:1,alignedReferenceFrames:[stance.frame],
    obstacles:this.readObstacles(),trackingReserve:.1});
  if(!terminalGeometry.supported)return this.#stop(terminalGeometry.reason,live);
  this.delegate=executor;this.stage='final_record';this.#record('mandatory_final_record_started',live,{sourceControls:selected.sourceFrames,
    referenceEndpoint:Array.from(terminal.slice(0,7)),referencePositionErrorM:positionError,referenceFacingErrorRad:facing});return null;
 }
 step(live){
  if(['complete','unsupported'].includes(this.phase))return this.#terminalResult();
  for(let transitions=0;transitions<6;transitions++){
    if(this.stage==='prepare_final'){
      if(this.cancelRequested)return this.#stop('cancelled',live);
      const result=this.#prepareFinal(live);if(result)return result;
    }
    const result=this.delegate.step(live);
    if(result.mode==='none')return this.#stop(result.completionReason??'reference_sweep_clearance',live);
    if(this.stage==='final_record'&&result.stepFinished){
      if(this.delegate.referenceIndex!==this.plan.selected.sourceFrames)throw new Error('Final source ended before all original controls');
      this.#record('mandatory_final_record_completed',live,{sourceControls:this.delegate.referenceIndex,outcome:result.outcome});
      this.#quiet(live,'final_quiet');continue;
    }
    if(result.justCompleted){
      this.#record(this.stage+'_completed',live,{completionReason:result.completionReason,outcome:result.outcome});
      if(!['finished','cancelled'].includes(result.completionReason))return this.#stop(result.completionReason,live);
      if(this.stage==='facing_turn'){this.#quiet(live,'turn_quiet');continue;}
      if(this.cancelRequested||result.completionReason==='cancelled')return this.#stop('cancelled',live);
      if(this.stage==='final_quiet'){
        const error=distance(live.rootPosWorld,this.plan.approachGoalWorld),facing=wrap(yaw(live.rootQuatXyzwWorld)-this.plan.pickupYawRad);
        if(error>this.parent.arrivalRadius||Math.abs(facing)>this.parent.maxFacingError)return this.#stop('actual_final_pose_outside_parent',live);
        this.handoffFrames=result.handoffReferenceFrames;this.completionReason='finished';this.phase='complete';
        this.#record('actual_parent_handoff',live,{distanceM:error,facingErrorRad:facing});return this.#terminalResult();
      }
      this.stage='prepare_final';continue;
    }
    if(result.mode!=='teacher')throw new Error('No student settling or unowned action in explicit final approach');
    this.phase=result.phase;this.pendingAdvance=true;return{...result,justCompleted:false,requestedGoalWorld:this.requestedGoalWorld,
      pickupFacingStage:this.stage,outcome:{...result.outcome,outerControls:this.controls,stages:this.events}};
  }
  throw new Error('Finite approach exceeded bounded metadata transitions');
 }
 advance(){if(!this.pendingAdvance)throw new Error('Exactly one actual action before advance');this.pendingAdvance=false;this.delegate.advance();this.controls++;}
 review(){return{controls:this.controls,phase:this.phase,stage:this.stage,turns:this.turnCount,originalGoalWorld:Array.from(this.plan.originalGoalWorld),
  pregraspGoalWorld:this.requestedGoalWorld,selectedSourceControls:this.plan.selected.sourceFrames,events:this.events,completionReason:this.completionReason??null};}
}
