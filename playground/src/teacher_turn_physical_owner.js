/** Finite private teacher-turn ownership. Does not certify source geometry. */
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const finite=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
export class FacingTurnPhysicalProbe {
 constructor({owner,parent,request,geometry,episode,requestId,physicalControl}){
  if(!owner||!parent||owner.phase!=='teacher_turn'||owner.referenceIndex!==0||owner.sourceFrames!==273
    ||owner.worldFrames!==request?.alignedReferenceFrames||request.skill!==owner.skill||request.sourceFrames!==273
    ||request.phase!=='teacher_turn'||geometry?.supported!==false||geometry.reason!=='reference_sweep_clearance'
    ||geometry.trackingReserve!==.15||parent.referenceIndex!==0||parent.finishRequested
    ||!finite(parent.requestedGoalWorld,3)||!finite(parent.referencePlan?.first,747)
    ||![episode,requestId,physicalControl].every(v=>Number.isSafeInteger(v)&&v>=0))throw new Error('Exact refused complete273 turn and original parent required');
  this.owner=owner;this.parent=parent;this.episode=episode;this.requestId=requestId;this.startControl=physicalControl;
  this.originalGoal=Object.freeze(Array.from(parent.requestedGoalWorld));this.pickupPose=Object.freeze(Array.from(parent.referencePlan.first.slice(0,7)));
  this.bank=owner.worldFrames;this.sourceControls=273;this.controls=0;this.ended=null;this.records=[];
  this.originalGeometry=structuredClone(geometry);
 }
 get active(){return this.ended===null;}
 invalidReason(c){
  if(!this.active)return 'probe_ended';
  if(c.episode!==this.episode)return 'episode_changed';
  if(c.owner!==this.owner||c.parent!==this.parent||c.requestId!==this.requestId)return 'owner_changed';
  if(this.owner.finishRequested||this.parent.finishRequested)return 'cancelled';
  if(!same(this.parent.requestedGoalWorld,this.originalGoal)||!same(this.parent.referencePlan?.first?.slice(0,7),this.pickupPose))return 'goal_changed';
  if(c.physicalControl!==this.startControl+this.controls||this.owner.referenceIndex!==this.controls
    ||this.owner.phase!=='teacher_turn'||this.owner.worldFrames!==this.bank||this.parent.referenceIndex!==0)return 'source_clock_changed';
  return null;
 }
 begin(c,record){const reason=this.invalidReason(c);if(reason)throw new Error(reason);
  if(this.records.at(-1)?.executed===false)throw new Error('Unresolved original candidate');
  if(!finite(record.teacherObservation,4052)||!finite(record.previousAction,29)||!finite(record.previousDofPos,29)||!finite(record.previousDofVel,29))throw new Error('Original input/history required');
  const row={...record,sourceIndex:this.controls,teacherTargetIndices:[this.controls+1,this.controls+16],
    preControl:c.physicalControl,executed:false};this.records.push(row);return row;}
 finish(reason){this.ended??={reason,controls:this.controls,physicalControl:this.startControl+this.controls,completeSource:this.controls===273};return this.ended;}
 reject(row,reason){if(row!==this.records.at(-1)||row.executed)throw new Error('Only pending candidate may be refused');row.discardedReason=reason;return this.finish(reason);}
 commit(c,row){
  if(!this.active||row!==this.records.at(-1)||row.executed||c.owner!==this.owner||c.parent!==this.parent
    ||c.episode!==this.episode||c.requestId!==this.requestId||c.physicalControl!==this.startControl+this.controls+1
    ||this.owner.referenceIndex!==this.controls+1||!row.preview?.supported||row.preview.completedSubsteps!==17
    ||row.preview.allowedContactCount!==0||row.preview.unwantedContactCount!==0)throw new Error('Original17-substep preview and exactly one physical advance required');
  row.executed=true;row.physicalControl=c.physicalControl;row.physicsSubsteps=17;this.controls++;
  if(this.controls===273)this.finish('complete_teacher_turn_executed');
 }
 review(){return{episode:this.episode,requestId:this.requestId,startControl:this.startControl,controls:this.controls,
  originalGoalWorld:[...this.originalGoal],pickupPose:[...this.pickupPose],sourceControls:this.sourceControls,
  sourceGeometrySupported:false,privateReferenceAdmissionOverride:true,originalGeometry:this.originalGeometry,
  fullReference:this.bank.map(row=>Array.from(row)),records:this.records,ended:this.ended};}
}
