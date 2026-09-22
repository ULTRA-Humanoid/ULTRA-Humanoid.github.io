/** Private reusable recovery lineage and original-first facing admission; no physics. */
import {TeacherApproachRecoveryController} from './teacher_approach_recovery_controller.js';
import {prepareTeacherFacingTurn,TeacherTurnController} from './teacher_turn_controller.js';
import {TerminalFacing273Probe} from './terminal_facing_turn_owner.js';

const equal=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const finite=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
const clock=n=>Number.isSafeInteger(n)&&n>=0;
const segment=parent=>parent?.segmentIndex??0;
const child=parent=>parent?.child??parent;

export class RecoveredFacingTurnAdmission {
 constructor({enabled=false,substepsPerControl=17}={}){
  if(typeof enabled!=='boolean'||!Number.isSafeInteger(substepsPerControl)||substepsPerControl<1)throw Error('Explicit mode and control substeps required');
  this.enabled=enabled;this.substepsPerControl=substepsPerControl;this.generation=0;this.reset();
 }
 reset(){this.generation++;this.byRecovery=new WeakMap();this.byParent=new WeakMap();this.events=[];this.latest=null;}
 event(kind,ticket,extra={}){
  this.events.push({event:kind,...this.summary(ticket),...extra});if(this.events.length>64)this.events.shift();
 }
 summary(t){return t?{episode:t.episode,requestId:t.requestId,segmentIndex:t.segmentIndex,
  startControl:t.startControl,sourceExecutedAtControl:t.sourceExecutedAtControl,sourceFrames:t.sourceFrames,
  actualRecoveryControls:t.controls,actualRecoverySubsteps:t.actualSubsteps,partialRecoverySubsteps:t.partial,
  completedAtControl:t.completedAtControl??null,admittedAtControl:t.admittedAtControl??null,
  status:t.status,invalidReason:t.invalidReason??null,originalGoalWorld:[...t.originalGoal]}:null;}
 review(){return {enabled:this.enabled,latest:this.summary(this.latest),events:structuredClone(this.events)};}
 key(requestId,segmentIndex){return `${requestId}:${segmentIndex}`;}
 parentIdentity(t,c){
  return t.generation===this.generation&&c.parent===t.parent&&c.episode===t.episode&&c.requestId===t.requestId
   &&segment(c.parent)===t.segmentIndex&&!c.parent.finishRequested&&!c.parent.cancelRequested
   &&c.parent.referenceIndex===0&&equal(c.parent.requestedGoalWorld,t.originalGoal)
   &&child(c.parent)===t.carryChild&&c.parent.skill===t.carrySkill
   &&c.parent.skill?.frames===t.carryFrames&&c.parent.skill?.sourceFrames===t.carrySourceFrames
   &&equal(child(c.parent).requestedGoalWorld,t.segmentGoal)
   // The original carry refreshes its object-bound plan after approach settles.
   // Freeze that live plan only when the special turn is actually admitted.
   &&(t.status!=='admitted'||(c.parent.referencePlan===t.admittedReferencePlan
     &&equal(c.parent.referencePlan?.first,t.admittedFirst)));
 }
 invalidate(t,reason){if(t.status!=='invalid'){t.status='invalid';t.invalidReason=reason;this.event('recovery_evidence_invalid',t);}return false;}
 registerRecovery(c){
  if(!this.enabled)return false;
  const {recovery,saved,parent,owner,episode,requestId,physicalControl}=c;
  if(!(recovery instanceof TeacherApproachRecoveryController)||!saved||saved.episode!==episode
    ||saved.parent!==parent||saved.owner!==owner||parent!==c.activeParent||owner?.cancelRequested
    ||parent?.finishRequested||parent?.cancelRequested||parent?.referenceIndex!==0
    ||![episode,requestId,physicalControl,saved.executedAtControl,segment(parent)].every(clock)
    ||saved.executedAtControl>physicalControl||!finite(parent.requestedGoalWorld,3)
    ||!finite(child(parent).requestedGoalWorld,3)||!Array.isArray(parent.skill?.frames)
    ||!clock(parent.skill.sourceFrames)||parent.skill.sourceFrames<1
    ||!finite(parent.referencePlan?.first,747)||!finite(saved.plan?.frame,747)
    ||!finite(saved.terminal,747)||!equal(saved.plan.frame,recovery.plan?.frame)
    ||saved.skill!==recovery.skill||!clock(saved.skill?.sourceFrames)||saved.skill.sourceFrames<1
    ||recovery.controls!==0||recovery.referenceIndex!==0||recovery.phase!=='teacher_settling'
    ||recovery.pending||recovery.issued||recovery.cancelled||recovery.completed
    ||!equal(recovery.goalWorld,parent.approachGoalWorld))return false;
  let records=this.byParent.get(parent);if(!records){records=new Map();this.byParent.set(parent,records);}
  const key=this.key(requestId,segment(parent)),previous=records.get(key);
  // A special attempt remains spent even if a later recovery is created.
  if(previous?.attempted)return false;
  if(previous)this.invalidate(previous,'new_recovery_replaced_old');
  const t={generation:this.generation,parent,requestId,episode,segmentIndex:segment(parent),recovery,
   originalGoal:Array.from(parent.requestedGoalWorld),carryChild:child(parent),carrySkill:parent.skill,
   carryFrames:parent.skill.frames,carrySourceFrames:parent.skill.sourceFrames,
   segmentGoal:Array.from(child(parent).requestedGoalWorld),
   terminal:Array.from(saved.plan.frame),terminalObject:recovery.plan.frame,sourceSkill:saved.skill,
   approachGoal:Array.from(recovery.goalWorld),sourceFrames:saved.skill.sourceFrames,
   sourceExecutedAtControl:saved.executedAtControl,startControl:physicalControl,lastControl:physicalControl,
   controls:0,actualSubsteps:0,partial:0,status:'recovering',attempted:false};
  records.set(key,t);this.byRecovery.set(recovery,t);this.latest=t;this.event('recovery_registered',t);return true;
 }
 recoveryIdentity(t,c){
  return this.parentIdentity(t,c)&&c.controller===t.recovery&&c.owner===t.recovery
   &&t.recovery.skill===t.sourceSkill&&t.recovery.plan.frame===t.terminalObject
   &&equal(t.recovery.plan.frame,t.terminal)&&equal(t.recovery.goalWorld,t.approachGoal)
   &&!t.recovery.cancelled;
 }
 observeActualSubstep(c){
  const t=this.byRecovery.get(c.controller);if(!t||t.status!=='recovering')return false;
  if(!this.recoveryIdentity(t,c)||c.physicalControl!==t.lastControl+1||c.substep!==t.partial+1
    ||c.substep>this.substepsPerControl||t.recovery.controls!==t.controls
    ||t.recovery.referenceIndex!==t.controls||!t.recovery.issued||t.recovery.pending)
   return this.invalidate(t,'recovery_substep_ownership_or_clock');
  t.partial++;t.actualSubsteps++;return true;
 }
 commitRecovery(c){
  const t=this.byRecovery.get(c.controller);if(!t||t.status!=='recovering')return false;
  const p=c.preview;
  if(!this.recoveryIdentity(t,c)||c.physicalControl!==t.lastControl+1||t.partial!==this.substepsPerControl
    ||t.recovery.controls!==t.controls+1||t.recovery.referenceIndex!==t.controls+1
    ||!t.recovery.pending||t.recovery.issued||p?.supported!==true||p.completedSubsteps!==this.substepsPerControl
    ||p.unwantedContactCount!==0||p.endpointUnwantedContactCount!==0||p.allowedContactCount!==0)
   return this.invalidate(t,'recovery_commit_or_preview');
  t.controls++;t.lastControl=c.physicalControl;t.partial=0;return true;
 }
 completeRecovery(c){
  const t=this.byRecovery.get(c.controller);if(!t||t.status!=='recovering')return false;
  const r=t.recovery,s=c.step;
  if(!this.recoveryIdentity(t,c)||c.physicalControl!==t.lastControl||t.controls<1||t.partial!==0
    ||t.actualSubsteps!==t.controls*this.substepsPerControl||s?.justCompleted!==true||s.completionReason!=='finished'
    ||r.phase!=='complete'||r.completionReason!=='finished'||!r.completed||r.pending||r.issued
    ||r.controls!==t.controls||r.referenceIndex!==t.controls||s.outcome?.controls!==t.controls
    ||r.records.length!==t.controls||r.records.some((row,i)=>row.control!==i+1)
    ||s.handoffReferenceFrames?.length!==2||!s.handoffReferenceFrames.every(row=>equal(row,t.terminal)))
   return this.invalidate(t,'recovery_completion_not_physically_owned');
  t.status='recovered';t.completedAtControl=c.physicalControl;this.event('recovery_finished',t);return true;
 }
 eligible(c,request,geometry,leftSkill){
  if(!this.enabled||!c.parent)return null;
  const t=this.byParent.get(c.parent)?.get(this.key(c.requestId,segment(c.parent)));
  if(!t||t.status!=='recovered'||t.attempted||!this.parentIdentity(t,c)||c.owner!==c.parent
    ||c.activeParent!==c.parent||!clock(c.physicalControl)||c.physicalControl<t.completedAtControl
    ||c.parent.completionReason!=='needs_facing'||c.completionReason!=='needs_facing'
    ||!finite(c.parent.referencePlan?.first,747)
    ||request?.skill!==leftSkill||request?.phase!=='teacher_turn'||request?.sourceFrames!==273
    ||geometry?.supported!==false||geometry.reason!=='reference_sweep_clearance'||geometry.trackingReserve!==.15)return null;
  return t;
 }
 consume(t,c){t.admittedReferencePlan=c.parent.referencePlan;t.admittedFirst=Array.from(c.parent.referencePlan.first);
  t.attempted=true;t.status='admitted';t.admittedAtControl=c.physicalControl;this.event('recovered_turn_preview_admitted',t);}
 isAdmissionCurrent(t,c){return t?.status==='admitted'&&t.attempted&&this.parentIdentity(t,c);}
}

/** Complete the ordinary selection first; never pre-empt another accepted candidate. */
export function prepareRecoveredFacingTurn(skills,proprio,facingErrorRad,{approveReference=null,admission,context}={}){
 let refusedLeft=null;
 const ordinary=prepareTeacherFacingTurn(skills,proprio,facingErrorRad,{approveReference:approveReference===null?null:(live,request)=>{
  const geometry=approveReference(live,request);
  if(request.skill===skills[0]&&request.sourceFrames===273&&geometry?.supported===false
      &&geometry.reason==='reference_sweep_clearance'&&geometry.trackingReserve===.15)
   refusedLeft={request,geometry};
  return geometry;
 }});
 if(ordinary.supported||!refusedLeft||!(admission instanceof RecoveredFacingTurnAdmission))return ordinary;
 const {request,geometry}=refusedLeft,t=admission.eligible(context,request,geometry,skills[0]);
 if(!t)return ordinary;
 admission.consume(t,context);
 const controller=new TeacherTurnController(request.skill);controller.start(proprio);
 if(controller.worldFrames.length!==request.alignedReferenceFrames.length||controller.worldFrames.some((row,i)=>!equal(row,request.alignedReferenceFrames[i]))){
  admission.invalidate(t,'original_refused_source_changed');return ordinary;
 }
 // Keep the very source bank whose ordinary geometry was recorded as refused.
 controller.worldFrames=request.alignedReferenceFrames;
 return {supported:true,reason:null,controller,recoveredFacingPending:{request,geometry:structuredClone(geometry),
  parent:context.parent,episode:context.episode,requestId:context.requestId,physicalControl:context.physicalControl,
  recoveryAdmission:admission,recoveryTicket:t}};
}

export class RecoveredFacingTurnProbe extends TerminalFacing273Probe{
 constructor(options){super(options);this.recoveryAdmission=options.recoveryAdmission;this.recoveryTicket=options.recoveryTicket;}
 invalidReason(context){return super.invalidReason(context)
  ||(!this.recoveryAdmission.isAdmissionCurrent(this.recoveryTicket,context)?'recovery_parent_or_segment_changed':null);}
 review(){return {...super.review(),recoveryEvidence:this.recoveryAdmission.summary(this.recoveryTicket)};}
}
