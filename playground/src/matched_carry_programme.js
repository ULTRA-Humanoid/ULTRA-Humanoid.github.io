/** Fixed private original3m programme; physical clocks remain actual episode clocks. */
const invariant=(condition,message='Matched carry invariant failed')=>{if(!condition)throw new Error(message);};
const equal=(actual,expected,message)=>invariant(actual===expected,message??`Expected ${expected}; received ${actual}`);
import {BoundedStageGoalWindow,packLegacyStageStudentInput} from './stage_goal.js';
import {transformTeacherReference} from './teacher_reference.js';
import {measurePlacementRelease} from './placement_feedback.js';
import {isTaskRelativeCarryRequest} from './matched_carry_request.js';
import {OBJECT_PROFILES} from './object_profiles.js';
export const ORIGINAL3M_GOAL=Object.freeze([4.520183086395264,-0.018612559884786606,0.13593138754367828]);
export const FIRST_INTERMEDIATE_GOAL=Object.freeze([3.6500000953674316,-4.440892098500626e-16,0.13593138754367828]);
export const WHOLE_BOX_ROLES=Object.freeze({firstStudent:90,secondStudent:60,groundingStudent:106,releaseTeacher:44,
  settling:180,exit:439,standing:120,afterFirst90:949});
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
const same=(a,b)=>a?.length===b?.length&&Array.from(a).every((v,i)=>v===b[i]);
const vector=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
const dist=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
const upright=q=>1-2*(q[0]**2+q[1]**2);
const endingPhase=n=>n<180?'settling':n<240?'teacher_exit_hold':n<439?'teacher_exit_retreat':n<619?'teacher_exit_settling':'teacher_standing';
export function isFixedOriginal3mGoal(goal){return same(goal,ORIGINAL3M_GOAL);}
export function makeOriginal3mRequest({episode,requestId,physicalControl,originalGoalWorld}){
  invariant(Number.isSafeInteger(episode)&&episode>=0&&Number.isSafeInteger(requestId)&&requestId>0&&Number.isSafeInteger(physicalControl)&&physicalControl>=0);
  invariant(isFixedOriginal3mGoal(originalGoalWorld),'Only the declared fixed original3m request is supported');
  return freeze({episode,requestId,issuedAtPhysicalControl:physicalControl,originalGoalWorld:Array.from(originalGoalWorld),
    firstIntermediateGoalWorld:Array.from(FIRST_INTERMEDIATE_GOAL),scope:'one fixed original3m request from Reset'});
}
function loaded(live,{target=null,foot=false}={}){
  if(!vector(live?.rootPositionWorld,3)||!vector(live.rootQuaternionWorld,4)||!vector(live.objectPositionWorld,3)
    ||Math.abs(Math.hypot(...live.rootQuaternionWorld)-1)>1e-5||live.rootPositionWorld[2]<.65||upright(live.rootQuaternionWorld)<.8||live.objectPositionWorld[2]<=.5
    ||!vector(live.handNormalForceN,2)||live.handNormalForceN.some(x=>x<=1))return false;
  if(target&&(dist(live.rootPositionWorld,target.slice(0,3))>.15||dist(live.objectPositionWorld,target.slice(71,74))>.15))return false;
  return!foot||(Number.isSafeInteger(live.loadedFootGroundContacts)&&live.loadedFootGroundContacts>0
    &&Number.isFinite(live.loadedFootNormalForceN)&&live.loadedFootNormalForceN>5);
}
export class MatchedCarryProgrammeOwner{
  constructor({request,rawSkill,prefixParent}){
    invariant(Object.isFrozen(request)&&(isFixedOriginal3mGoal(request.originalGoalWorld)||isTaskRelativeCarryRequest(request)),
      'A legacy fixed request or a current task-relative factory request is required');
    equal(rawSkill.sourceFrames,366);equal(rawSkill.frames.length,386);
    equal(rawSkill.objectBodyName,OBJECT_PROFILES.largebox.bodyName);invariant(rawSkill.frames.every(r=>vector(r,747)));
    this.request=request;this.raw=freeze({name:rawSkill.name,objectBodyName:rawSkill.objectBodyName,sourceFrames:366,
      objectPointsLocal:structuredClone(rawSkill.objectPointsLocal),frames:rawSkill.frames.map(r=>Array.from(r))});
    this.prefixParent=prefixParent;this.parent=null;this.role='prefix';this.counts={student_second_loaded:0,student_grounding106:0,teacher_release44:0,postplacement:0};
    this.lastPhysicalControl=null;this.entryPhysicalControl=null;this.window=null;this.pending=null;this.serial=0;this.released=0;
    this.transitions=[];this.ended=null;this.anyActualViolation=false;this.cancelRequested=false;
    this.finishRequested=false;this.permittedLatestRequestId=request.requestId;this.finishRequests=[];this.finishRequestCount=0;this.returnedToMain=false;
  }
  requestCurrent(c){return c.episode===this.request.episode&&c.requestId===this.request.requestId&&c.latestRequestId===this.permittedLatestRequestId
    &&same(c.originalGoalWorld,this.request.originalGoalWorld)&&!this.cancelRequested;}
  current(c){return this.requestCurrent(c)&&c.parent===this.parent&&c.physicalControl===this.lastPhysicalControl
    &&this.parent?.worldFrames===this.bank&&this.parent.skill===this.skill&&same(this.parent.requestedGoalWorld,this.request.originalGoalWorld)
    &&this.parent.segmentIndex===0&&this.parent.referenceIndex===this.sourceIndex&&this.parent.phase===this.role;}
  get active(){return this.role!=='prefix'&&!this.ended;}get sourceIndex(){return this.role==='student_second_loaded'?179+this.counts.student_second_loaded:
    this.role==='student_grounding106'?239:this.role==='teacher_release44'?322+this.counts.teacher_release44:366;}
  cancel(reason='cancelled'){this.cancelRequested=true;this.pending=null;
    if(!this.ended)this.ended={completionReason:reason,goalReached:false,physicalControl:this.lastPhysicalControl};return this.ended;}
  requestPrefixFinish(c){
    invariant(this.role==='prefix'&&!this.ended&&!this.cancelRequested&&c.episode===this.request.episode
      &&c.requestId===this.request.requestId&&c.parent===this.prefixParent&&this.prefixParent.phase==='teacher'
      &&same(c.originalGoalWorld,this.request.originalGoalWorld),'Only the current loaded prefix may register finite finish');
    invariant(Number.isSafeInteger(c.physicalControl)&&c.physicalControl>=this.request.issuedAtPhysicalControl
      &&c.physicalControl>=(this.lastPrefixCommandPhysicalControl??this.request.issuedAtPhysicalControl),'Prefix command clock cannot rewind');
    invariant(Number.isSafeInteger(c.latestRequestId)&&c.latestRequestId>=this.permittedLatestRequestId,'Latest prefix request cannot rewind');
    if(c.latestRequestId!==this.permittedLatestRequestId)equal(c.queuedRequestId,c.latestRequestId,'A newer prefix request must be explicitly queued');
    this.finishRequested=true;this.permittedLatestRequestId=c.latestRequestId;this.lastPrefixCommandPhysicalControl=c.physicalControl;
    this.finishRequests.push({physicalControl:c.physicalControl,queuedRequestId:c.queuedRequestId??null,latestRequestId:c.latestRequestId,
      role:'prefix',behavior:'complete existing pickup, student windows, placement and full ending; preserve original goal'});
    this.finishRequestCount++;if(this.finishRequests.length>32)this.finishRequests.shift();
    // Do not cancel the legacy parent: its finish flag would prevent first90.
    return this.review();
  }
  requestFinish(c){
    invariant(this.active&&c.episode===this.request.episode&&c.requestId===this.request.requestId&&c.parent===this.parent
      &&c.physicalControl===this.lastPhysicalControl&&same(c.originalGoalWorld,this.request.originalGoalWorld));
    invariant(Number.isSafeInteger(c.latestRequestId)&&c.latestRequestId>=this.request.requestId);
    if(c.latestRequestId!==this.permittedLatestRequestId)equal(c.queuedRequestId,c.latestRequestId,'Only an explicitly queued request can change latest ownership');
    this.finishRequested=true;this.permittedLatestRequestId=c.latestRequestId;
    this.finishRequests.push({physicalControl:c.physicalControl,queuedRequestId:c.queuedRequestId??null,latestRequestId:c.latestRequestId,
      behavior:'complete existing placement and full ending; preserve original goal'});
    this.finishRequestCount++;if(this.finishRequests.length>32)this.finishRequests.shift();
    return this.review();
  }
  canReturnToMain(c){return !this.returnedToMain&&this.ended?.goalReached===true&&this.requestCurrent(c)&&c.parent===this.parent
    &&c.physicalControl===this.lastPhysicalControl&&this.ended.physicalControl===c.physicalControl
    &&this.parent.phase==='complete'&&this.parent.referenceIndex===366&&this.parent.segmentIndex===0&&this.parent.skill===this.skill
    &&this.parent.worldFrames===this.bank&&same(this.parent.requestedGoalWorld,this.request.originalGoalWorld)
    &&this.counts.student_second_loaded===60&&this.counts.student_grounding106===106&&this.counts.teacher_release44===44&&this.counts.postplacement===739;}
  returnToMain(c,transfer){invariant(this.canReturnToMain(c),'Only current completed full programme may return control');
    equal(typeof transfer,'function');const value=transfer();invariant(!value||typeof value.then!=='function','Ownership transfer must be synchronous');
    this.returnedToMain=true;this.returnedAtPhysicalControl=c.physicalControl;return this.review();}
  beginAfter90(c,{student,live,prefixSafety,readCommandContext=null}){
    equal(this.role,'prefix');
    invariant(prefixSafety&&prefixSafety.evaluatedSubsteps===c.physicalControl*17&&!prefixSafety.missingEvaluation,'Complete actual prefix substep audit required');
    this.prefixSafety=structuredClone(prefixSafety);this.anyActualViolation=Boolean(prefixSafety.anyActualViolation);
    invariant(!this.anyActualViolation,'Actual initial approach/pickup/student90 prefix must have no disallowed violation');invariant(this.requestCurrent(c)&&c.parent===this.prefixParent,'Current original request and prefix owner required');
    equal(this.prefixParent.skill.sourceFrames,456);invariant(same(this.prefixParent.skill.studentTransportInterval,[240,330]));
    equal(this.prefixParent.referenceIndex,330);equal(student.controls,90);equal(student.ended?.atControl,c.physicalControl);
    equal(student.ended.reason,'moving_window_complete');equal(student.ended.fallback,false);
    const a=this.raw.frames[179],z=this.raw.frames[365],box=live.objectPositionWorld,goal=this.request.originalGoalWorld;
    const yaw=Math.atan2(goal[1]-box[1],goal[0]-box[0])-Math.atan2(z[72]-a[72],z[71]-a[71]);
    const co=Math.cos(yaw),si=Math.sin(yaw);this.transform=freeze({yawRadians:yaw,translation:[box[0]-co*a[71]+si*a[72],box[1]-si*a[71]-co*a[72],0]});
    this.bank=freeze(this.raw.frames.map(row=>Array.from(transformTeacherReference(row,this.transform))));
    this.skill=freeze({name:this.raw.name,objectBodyName:this.raw.objectBodyName,sourceFrames:366,objectPointsLocal:this.raw.objectPointsLocal,locomotionOnly:false,
      studentTransportInterval:[179,239],privateSecondLoadedProfile:'original087_179_239_local_second_window'});
    invariant(loaded(live,{target:this.bank[179]}),'Actual second60 loaded reference entry required');
    this.lastPhysicalControl=this.entryPhysicalControl=c.physicalControl;
    this.parent={phase:'student_second_loaded',segmentIndex:0,referenceIndex:179,skill:this.skill,sourceFrames:366,worldFrames:this.bank,
      requestedGoalWorld:this.request.originalGoalWorld,plan:{goals:[this.request.originalGoalWorld]},referencePlan:{transform:this.transform},
      segmentResults:[],segmentExitResults:[],outcome:{},reset:()=>this.cancel('episode_reset'),isActive:()=>!this.ended,requestCancel:()=>this.requestFinish(readCommandContext?readCommandContext():{...c,parent:this.parent,physicalControl:this.lastPhysicalControl})};
    this.enter('student_second_loaded',60,239);return this.parent;
  }
  enter(role,horizon=null,target=null){this.role=role;this.parent.phase=role;this.parent.referenceIndex=this.sourceIndex;
    this.transitions.push({role,physicalControl:this.lastPhysicalControl,sourceIndex:this.sourceIndex});this.pending=null;
    if(horizon){const f=this.bank[target];this.window=new BoundedStageGoalWindow({stage:role==='student_second_loaded'?'transport':'place',mode:'HOI_FULL',
      humanGoalWorld:f.slice(0,3),humanGoalRotationWorld:f.slice(3,7),objectGoalWorld:f.slice(71,74),finalDestinationWorld:this.request.originalGoalWorld},
      {episode:this.request.episode,physicalControl:this.lastPhysicalControl,horizonControls:horizon});}else this.window=null;
  }
  sample(c,live){invariant(this.active&&this.current(c),'Current real programme clock and owner required');
    const sample={candidateId:++this.serial,phase:this.role==='postplacement'?endingPhase(this.counts.postplacement):this.role,sourceIndex:this.sourceIndex};
    if(this.window){sample.encoded=this.window.sample(live,c);sample.mode='student';}
    else if(this.role==='teacher_release44'){sample.mode='teacher';sample.referenceFrames=[this.bank[this.sourceIndex+1],this.bank[this.sourceIndex+16]];}
    else sample.mode=sample.phase==='settling'?'student':'teacher';
    this.pending={candidateId:sample.candidateId,phase:sample.phase,sourceIndex:sample.sourceIndex};return sample;
  }
  packStudent(sample,body,points){equal(sample.candidateId,this.pending?.candidateId);invariant(sample.encoded);
    return packLegacyStageStudentInput(sample.encoded,body,points);}
  canActuate(c,sample){return this.active&&this.current(c)&&this.pending?.candidateId===sample.candidateId&&this.pending.phase===sample.phase;}
  commit(c,{sample,physicsSubsteps,preview,live,actualViolation=false}){
    invariant(this.active&&this.current({...c,physicalControl:this.lastPhysicalControl})&&c.physicalControl===this.lastPhysicalControl+1
      &&this.pending?.candidateId===sample.candidateId&&physicsSubsteps===17,'Only next real owned programme control can commit');
    if(this.role!=='postplacement'||sample.phase!=='settling')invariant(preview?.supported&&preview.requestedSubsteps===17&&preview.completedSubsteps===17&&preview.unwantedContactCount===0);
    this.anyActualViolation||=actualViolation;this.lastPhysicalControl=c.physicalControl;this.pending=null;this.counts[this.role]++;
    this.parent.referenceIndex=this.sourceIndex;
    if(this.role==='student_second_loaded'&&this.counts.student_second_loaded===60){
      if(!loaded(live,{foot:true}))return this.cancel('grounding_entry_not_loaded_and_balanced');this.enter('student_grounding106',106,345);
    }else if(this.role==='student_grounding106'&&this.counts.student_grounding106===106){
      const m=measurePlacementRelease(live,this.bank[322]);this.groundingEndpoint={physicalControl:this.lastPhysicalControl,measured:m};
      if(!m.valid||!m.balanced||m.released||m.rootErrorM>.15||m.boxErrorM>.15||m.boxSpeedMps>.2
        ||!vector(live.handNormalForceN,2)||live.handNormalForceN.some(x=>x<=1)||!Number.isSafeInteger(live.loadedFootGroundContacts)||live.loadedFootGroundContacts<1
        ||!Number.isFinite(live.loadedFootNormalForceN)||live.loadedFootNormalForceN<=5)
        return this.cancel('matched_release_entry_incompatible');
      this.released=0;this.enter('teacher_release44');
    }else if(this.role==='teacher_release44'){
      const m=measurePlacementRelease(live,this.bank[Math.min(365,322+this.counts.teacher_release44)]);this.released=m.released?this.released+1:0;
      if(!m.valid||!m.balanced)return this.cancel('actual_release_balance_or_measurement_failure');
      if(this.counts.teacher_release44===44){this.releaseEndpoint={physicalControl:this.lastPhysicalControl,measured:m,releasedControls:this.released};
        if(this.released<6||!m.handoff)return this.cancel('teacher_release_incomplete');this.enter('postplacement');}
    }else if(this.role==='postplacement'&&this.counts.postplacement===739){
      const m=measurePlacementRelease(live,this.bank[365]),error=dist(live.objectPositionWorld.slice(0,2),this.request.originalGoalWorld.slice(0,2));
      const complete=m.released&&error<=.1&&!this.anyActualViolation;
      this.ended={completionReason:complete?'finished':'ending_incomplete',goalReached:complete,physicalControl:this.lastPhysicalControl,
        remainingDistanceM:error,measuredRelease:m,originalGoalWorld:this.request.originalGoalWorld};
      this.parent.phase='complete';this.parent.completionReason=this.ended.completionReason;this.parent.outcome=this.ended;
    }
    return this.review();
  }
  review(){return{request:this.request,role:this.role,counts:{...this.counts},entryPhysicalControl:this.entryPhysicalControl,lastPhysicalControl:this.lastPhysicalControl,
    sourceIndex:this.sourceIndex,transitions:structuredClone(this.transitions),ended:this.ended?structuredClone(this.ended):null,
    groundingEndpoint:this.groundingEndpoint??null,releaseEndpoint:this.releaseEndpoint??null,anyActualViolation:this.anyActualViolation,
    oldPrefixSourcePaused:this.prefixParent.referenceIndex,oldPrefixSourceControls:456,unexecutedOldTeacherRows:[330,455],
    prefixSafety:this.prefixSafety??null,firstStudentControls:this.role==='prefix'?null:90,sourcePhaseSelectionDoesNotRewindPhysics:true,
    finishRequested:this.finishRequested,finishRequests:structuredClone(this.finishRequests),finishRequestCount:this.finishRequestCount,permittedLatestRequestId:this.permittedLatestRequestId,
    returnedToMain:this.returnedToMain,returnedAtPhysicalControl:this.returnedAtPhysicalControl??null};}
}
