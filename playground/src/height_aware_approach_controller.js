// Detailed source geometry for one explicitly supported small approach skill.
// No margin is reduced. Every newly admitted teacher action still needs a
// complete all-box physical preview. The ordinary owner keeps its source clock.
import {checkRestrictedReferenceSweep} from './restricted_motion_geometry.js';
export const HEIGHT_AWARE_APPROACH_LIMITS=Object.freeze({trackingReserveM:.1,verticalTrackingReserveM:.1,
  maxPregraspDistanceM:.5,minRootHeightM:.7,minUpright:.95,maxPlanarSpeedMps:.05});
const finite=(v,n)=>v?.length===n&&Array.from(v).every(Number.isFinite);
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const count=v=>Number.isSafeInteger(v)&&v>=0;
const no=reason=>({supported:false,reason,controller:null});
export function checkHeightAwareApproachEntry({owner,parent,eligibleSkill,request,sweep,wholeGeometry,obstacles,live,episode,requestId,physicalControl}){
  if(![episode,requestId,physicalControl].every(count))return no('invalid_identity');
  if(!parent||parent.phase!=='approach'||parent.finishRequested||parent.referenceIndex!==0||parent.segmentIndex<1
      ||!owner||owner.cancelRequested||!same(owner.requestedGoalWorld,parent.approachGoalWorld))return no('later_approach_not_owned');
  const exit=parent.segmentExitResults?.at(-1),clock=exit?.recordClock;
  if(exit?.completionReason!=='finished'||exit.segmentIndex!==parent.segmentIndex-1
      ||parent.exitedSegmentIndex!==parent.segmentIndex-1||!count(clock?.totalControls)||clock.totalControls<1
      ||clock.totalControls!==clock.expectedTotalControls)return no('previous_exit_incomplete');
  if(!eligibleSkill||request?.skill!==eligibleSkill||eligibleSkill.locomotionOnly!==true||request.phase!=='teacher_step'
      ||request.sourceFrames!==eligibleSkill.sourceFrames||request.sourceFrames!==sweep?.sourceFrames
      ||!count(request.sourceFrames)||request.sourceFrames<2
      ||request.alignedReferenceFrames?.length<request.sourceFrames+16)return no('complete_explicit_approach_skill_required');
  if(wholeGeometry?.supported!==false||wholeGeometry.reason!=='reference_sweep_clearance'
      ||wholeGeometry.trackingReserve!==HEIGHT_AWARE_APPROACH_LIMITS.trackingReserveM)return no('original_whole_sweep_must_refuse');
  if(!finite(live?.rootPosWorld,3)||!finite(live.rootQuatXyzwWorld,4)||!finite(live.rootVelWorld,3)
      ||!finite(parent.approachGoalWorld,3)||!finite(parent.requestedGoalWorld,3)
      ||Math.abs(Math.hypot(...live.rootQuatXyzwWorld)-1)>1e-5||live.objectBodyName!==parent.skill?.objectBodyName)return no('invalid_live_state_or_object');
  const limits=HEIGHT_AWARE_APPROACH_LIMITS,q=live.rootQuatXyzwWorld,upright=1-2*(q[0]**2+q[1]**2);
  const speed=Math.hypot(...live.rootVelWorld.slice(0,2)),distance=Math.hypot(...[0,1].map(i=>parent.approachGoalWorld[i]-live.rootPosWorld[i]));
  if(live.rootPosWorld[2]<limits.minRootHeightM||upright<limits.minUpright||speed>limits.maxPlanarSpeedMps)return no('entry_not_quiet_standing');
  if(distance>limits.maxPregraspDistanceM||distance<=parent.arrivalRadius)return no('pregrasp_distance_outside_scope');
  const geometry=checkRestrictedReferenceSweep({sweep,sourceFrames:request.sourceFrames,alignedReferenceFrames:request.alignedReferenceFrames,
    obstacles,trackingReserve:limits.trackingReserveM,heightAware:true,verticalTrackingReserve:limits.verticalTrackingReserveM});
  return geometry.supported?{supported:true,reason:null,geometry,distanceM:distance,upright,planarSpeedMps:speed}
    :{...no(geometry.reason),geometry};
}
export class HeightAwareApproachController{
  #owner;#parent;#skill;#last;#records=[];#ended=null;
  static tryStart(options){const entry=checkHeightAwareApproachEntry(options);return entry.supported
    ?{...entry,controller:new HeightAwareApproachController(options,entry)}:entry;}
  constructor(options,entry=checkHeightAwareApproachEntry(options)){
    if(!entry.supported)throw new Error('A complete detailed-geometry admission is required');
    this.#owner=options.owner;this.#parent=options.parent;this.#skill=options.request.skill;
    this.episode=options.episode;this.requestId=options.requestId;this.segmentIndex=options.parent.segmentIndex;
    this.startControl=this.#last=options.physicalControl;this.sourceFrames=options.request.sourceFrames;
    this.originalDestinationWorld=Object.freeze(Array.from(options.parent.requestedGoalWorld));
    this.geometry=structuredClone(entry.geometry);this.wholeGeometry=structuredClone(options.wholeGeometry);
  }
  get active(){return this.#ended===null;}
  get controls(){return this.#last-this.startControl;}
  get ended(){return structuredClone(this.#ended);}
  isOwnedBy({owner,parent,episode,requestId}){return owner===this.#owner&&parent===this.#parent&&episode===this.episode
    &&requestId===this.requestId&&parent?.segmentIndex===this.segmentIndex&&same(parent.requestedGoalWorld,this.originalDestinationWorld);}
  #finish(reason){this.#ended??={reason,atControl:this.#last,controls:this.controls,sourceFrames:this.sourceFrames,
    completeSourceExecuted:this.controls===this.sourceFrames};return{active:false,...this.ended};}
  // Cancellation of a queued task does not bypass previews or truncate the
  // active recorded source. The original owner finishes its current motion.
  observe(step,context){
    if(!this.active)return{active:false,...this.ended};
    if(!this.isOwnedBy(context))return this.#finish(context.episode!==this.episode?'episode_changed':'owner_changed');
    if(context.physicalControl!==this.#last)throw new Error('Height-aware approach requires its committed physical clock');
    if(this.controls===this.sourceFrames)return this.#finish('complete_teacher_source_executed');
    if(context.owner.skill!==this.#skill||step?.phase!=='teacher_step'||step?.mode!=='teacher'
        ||context.owner.referenceIndex!==this.controls)throw new Error('New teacher source must execute continuously with previews');
    return{active:true,requiresAllBoxPreview:true,sourceIndex:this.controls};
  }
  commit({record,...context}){
    if(!this.active||!this.isOwnedBy(context)||context.physicalControl!==this.#last+1||this.controls>=this.sourceFrames
        ||record?.preview?.supported!==true||record.preview.completedSubsteps!==17
        ||record.preview.unwantedContactCount!==0||record.preview.allowedContactCount!==0)
      throw new Error('Height-aware teacher commit requires an actual complete all-box contact-free preview');
    this.#records.push({...structuredClone(record),sourceIndexBefore:this.controls,sourceIndexAfter:this.controls+1,
      preControl:this.#last,physicalControl:context.physicalControl});this.#last=context.physicalControl;
  }
  refuse({record,...context}){if(!this.isOwnedBy(context)||context.physicalControl!==this.#last||record?.preview?.supported!==false)
    throw new Error('A refusal must belong to the current unexecuted teacher candidate');
    this.refusedPreview={...structuredClone(record),physicalControl:this.#last,sourceIndex:this.controls,executed:false};
    return this.#finish('teacher_preview_refused');}
  cancel(reason){return this.#finish(reason);}
  review(){return{episode:this.episode,requestId:this.requestId,segmentIndex:this.segmentIndex,startControl:this.startControl,
    controls:this.controls,sourceFrames:this.sourceFrames,originalDestinationWorld:Array.from(this.originalDestinationWorld),
    wholeGeometry:structuredClone(this.wholeGeometry),geometry:structuredClone(this.geometry),records:structuredClone(this.#records),
    ended:this.ended,refusedPreview:structuredClone(this.refusedPreview??null)};}
}
