// A finite student action window inside an already approved complete LOCO turn.
// The recorded approach retains its source clock and completes the teacher suffix.
import {encodeStageGoal,packLegacyStageStudentInput} from './stage_goal.js';
export const REFERENCE_STUDENT_TURN_LIMITS=Object.freeze({horizonControls:180,minRootHeightM:.7,minUpright:.95,maxPlanarSpeedMps:.05});
const finite=(v,n)=>v?.length===n&&Array.from(v).every(Number.isFinite);
const count=v=>Number.isSafeInteger(v)&&v>=0;
const equal=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
// Geometry/plan snapshots can contain typed-array anchors, which cannot be
// frozen with indexed elements. Retain their serialized numeric values.
const copy=v=>v==null?v:JSON.parse(JSON.stringify(v));
const freeze=v=>{if(v&&typeof v==='object'){Object.values(v).forEach(freeze);Object.freeze(v);}return v;};
const no=reason=>({supported:false,reason,controller:null});
export function checkReferenceStudentTurnEntry({owner,parent,request,geometry,live,episode,requestId,physicalControl,alreadyUsed=false}){
  if(![episode,requestId,physicalControl].every(count))return no('invalid_identity');
  if(alreadyUsed)return no('student_turn_already_used_for_segment');
  if(!parent||parent.phase!=='approach'||parent.finishRequested||parent.referenceIndex!==0||parent.segmentIndex<1)
    return no('later_carry_not_unstarted');
  const previous=parent.segmentExitResults?.at(-1),clock=previous?.recordClock;
  if(previous?.completionReason!=='finished'||previous.segmentIndex!==parent.segmentIndex-1
      ||parent.exitedSegmentIndex!==parent.segmentIndex-1||!count(clock?.totalControls)||clock.totalControls<1
      ||clock.totalControls!==clock.expectedTotalControls)return no('previous_exit_incomplete');
  if(!owner||owner.cancelRequested||owner.motionControls!==0||owner.referenceIndex!==0
      ||owner.skill!==request?.skill||!equal(owner.requestedGoalWorld,parent.approachGoalWorld))return no('first_recorded_turn_not_owned');
  if(request.phase!=='teacher_turn'||request.skill?.locomotionOnly!==true||request.sourceFrames!==request.skill.sourceFrames
      ||request.sourceFrames<=REFERENCE_STUDENT_TURN_LIMITS.horizonControls
      ||request.alignedReferenceFrames?.length<request.sourceFrames+16
      ||!request.alignedReferenceFrames.every(r=>finite(r,747)))return no('complete_turn_with_teacher_suffix_required');
  if(geometry?.supported!==true)return no('full_recorded_geometry_not_approved');
  if(!finite(live?.rootPosWorld,3)||!finite(live.rootQuatXyzwWorld,4)||!finite(live.rootVelWorld,3)
      ||!finite(live.objPosWorld,3)||!finite(parent.requestedGoalWorld,3)||Math.abs(Math.hypot(...live.rootQuatXyzwWorld)-1)>1e-5
      ||live.objectBodyName!==parent.skill?.objectBodyName)return no('invalid_live_state_or_object');
  const upright=1-2*(live.rootQuatXyzwWorld[0]**2+live.rootQuatXyzwWorld[1]**2),speed=Math.hypot(...live.rootVelWorld.slice(0,2));
  const limits=REFERENCE_STUDENT_TURN_LIMITS;
  if(live.rootPosWorld[2]<limits.minRootHeightM||upright<limits.minUpright||speed>limits.maxPlanarSpeedMps)return no('entry_not_quiet_standing');
  return{supported:true,reason:null,segmentIndex:parent.segmentIndex,objectBodyName:parent.skill.objectBodyName,upright,planarSpeedMps:speed};
}
export class ReferenceStudentTurnController{
  #owner;#parent;#skill;#last;#records=[];#pending=null;#ended=null;#refused=null;
  static tryStart(options){const entry=checkReferenceStudentTurnEntry(options);return entry.supported
    ?{...entry,controller:new ReferenceStudentTurnController(options,entry)}:entry;}
  constructor(options,entry=checkReferenceStudentTurnEntry(options)){
    if(!entry.supported)throw new Error('An admitted complete recorded turn is required');
    const {owner,parent,request,geometry,episode,requestId,physicalControl}=options;
    this.#owner=owner;this.#parent=parent;this.#skill=request.skill;this.#last=physicalControl;
    this.episode=episode;this.requestId=requestId;this.segmentIndex=entry.segmentIndex;this.startControl=physicalControl;
    this.horizonControls=REFERENCE_STUDENT_TURN_LIMITS.horizonControls;this.sourceFrames=request.sourceFrames;
    this.plan=freeze({stage:'approach',mode:'LOCO',sourceName:request.skill.name,sourceFrames:request.sourceFrames,
      targetSourceIndex:this.horizonControls,alignedReferenceFrames:request.alignedReferenceFrames.map(r=>Array.from(r)),
      originalDestinationWorld:Array.from(parent.requestedGoalWorld),approachDestinationWorld:Array.from(parent.approachGoalWorld),
      geometry:copy(geometry),referencePlan:copy(request.referencePlan??null)});
  }
  get active(){return this.#ended===null;}
  get controls(){return this.#last-this.startControl;}
  get ended(){return copy(this.#ended);}
  isForParentSegment(parent){return parent===this.#parent&&parent.segmentIndex===this.segmentIndex;}
  isOwnedBy({owner,parent,episode,requestId}){return owner===this.#owner&&parent===this.#parent&&episode===this.episode
    &&requestId===this.requestId&&parent?.segmentIndex===this.segmentIndex&&owner?.skill===this.#skill
    &&equal(parent?.requestedGoalWorld,this.plan.originalDestinationWorld);}
  #clock(context){if(!this.isOwnedBy(context)||context.physicalControl!==this.#last)throw new Error('Student turn requires its actual owner and committed clock');}
  #finish(reason,{requiresTeacherResume=false}={}){this.#ended??={reason,atControl:this.#last,controls:this.controls,
    requiresTeacherResume,resumeSourceIndex:this.controls,remainingTeacherSourceControls:this.sourceFrames-this.controls,arrived:false};
    this.#pending=null;return{active:false,...this.ended};}
  // Call after the real owner's step(), before inference. The wrapper exposes
  // the last sampled source until its next step, so verify the freshly sampled index.
  observe(step,context){
    if(!this.active)return{active:false,...this.ended};
    if(!this.isOwnedBy(context))return this.#finish(context.episode!==this.episode?'episode_changed':'owner_changed');
    this.#clock(context);
    if(context.owner.cancelRequested||context.parent.finishRequested)return this.#finish('cancelled',{requiresTeacherResume:true});
    if(context.owner.referenceIndex!==this.controls)throw new Error('Recorded turn source did not advance with physical controls');
    if(this.controls===this.horizonControls)return this.#finish('student_turn_window_complete',{requiresTeacherResume:true});
    if(step?.phase!=='teacher_turn'||step?.mode!=='teacher')return this.#finish('recorded_turn_ended');
    return{active:true,remainingControls:this.horizonControls-this.controls,arrived:false};
  }
  sample(live,context){
    this.#clock(context);if(!this.active||this.controls>=this.horizonControls)throw new Error('Observe turn expiry before another inference');
    const target=this.plan.alignedReferenceFrames[this.plan.targetSourceIndex];
    const encoded=encodeStageGoal({stage:'approach',mode:'LOCO',humanGoalWorld:target.slice(0,3),humanGoalRotationWorld:target.slice(3,7),
      objectGoalWorld:target.slice(71,74),finalDestinationWorld:this.plan.originalDestinationWorld,remainingControls:this.horizonControls-this.controls},
      {rootPositionWorld:live.rootPosWorld,rootQuaternionWorld:live.rootQuatXyzwWorld,objectPositionWorld:live.objPosWorld});
    this.#pending={sourceIndex:this.controls,physicalControl:this.#last,command:Array.from(encoded.command)};
    return{...encoded,expired:false,arrived:false};
  }
  buildObservation(encoded,bodyObservation){return packLegacyStageStudentInput(encoded,bodyObservation,null);}
  commit({record,...context}){
    if(!this.active||!this.isOwnedBy(context)||context.physicalControl!==this.#last+1||!this.#pending
        ||record?.preview?.supported!==true||record.preview.completedSubsteps!==17
        ||record.preview.unwantedContactCount!==0||record.preview.allowedContactCount!==0||!finite(record.rawAction,29))
      throw new Error('Student turn commit requires one actually executed previewed control');
    this.#records.push({...copy(record),sourceIndexBefore:this.controls,sourceIndexAfter:this.controls+1,
      command:this.#pending.command,preControl:this.#last,physicalControl:context.physicalControl});
    this.#last=context.physicalControl;this.#pending=null;
  }
  requestPreviewFallback({record,...context}){
    this.#clock(context);if(!this.active||!this.#pending||record?.preview?.supported!==false)throw new Error('A current refused turn candidate is required');
    this.#refused={...copy(record),...this.#pending,executed:false};return this.#finish('student_turn_preview_refused',{requiresTeacherResume:true});
  }
  cancel(reason,context){if(this.isOwnedBy(context))this.#clock(context);return this.#finish(reason,{requiresTeacherResume:context.episode===this.episode});}
  review(){return{episode:this.episode,requestId:this.requestId,segmentIndex:this.segmentIndex,startControl:this.startControl,
    controls:this.controls,horizonControls:this.horizonControls,sourceFrames:this.sourceFrames,plan:copy(this.plan),
    records:copy(this.#records),refusedPreview:copy(this.#refused),ended:this.ended};}
}
