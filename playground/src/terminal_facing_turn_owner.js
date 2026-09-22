/** One already selected full teacher turn, with current-builder and pending-step checks. */
import {FacingTurnPhysicalProbe} from './teacher_turn_physical_owner.js';
const equal=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
export class TerminalFacing273Probe extends FacingTurnPhysicalProbe{
 constructor(options){
  super(options);this.builder=null;this.frameObjects=[...this.bank];
  this.frozenFrames=this.bank.map(row=>Object.freeze(Array.from(row)));
 }
 bindBuilder(builder){
  if(!builder||builder.jacp==null)throw Error('A live turn teacher builder is required');
  if(this.builder&&this.builder!==builder)throw Error('The turn teacher builder changed');
  this.builder=builder;
 }
 invalidReason(context){
  const reason=super.invalidReason(context);if(reason)return reason;
  if(this.builder&&(context.teacherBuilder!==this.builder||this.builder.jacp==null))return 'teacher_builder_changed';
  if(this.bank.length!==this.frameObjects.length)return 'reference_bank_changed';
  for(const index of new Set([0,this.controls+1,this.controls+16,this.sourceControls-1])){
   if(this.bank[index]!==this.frameObjects[index]||!equal(this.bank[index],this.frozenFrames[index]))return 'reference_values_changed';
  }
  return null;
 }
 begin(context,record){
  this.bindBuilder(context.teacherBuilder);
  const row=super.begin(context,record);row.actualSubsteps=0;return row;
 }
 observeActualSubstep(row){
  if(row!==this.records.at(-1)||row.executed||row.actualSubsteps>=17)throw Error('One current actual substep is required');
  row.actualSubsteps++;
 }
 commit(context,row){
  if(row.actualSubsteps!==17||context.teacherBuilder!==this.builder||this.builder?.jacp==null
      ||row.preview?.endpointUnwantedContactCount!==0)throw Error('All17 actual steps and the live original builder are required');
  return super.commit(context,row);
 }
 review(){return {...super.review(),teacherBuilderBound:Boolean(this.builder),
  actualSubsteps:this.records.reduce((n,r)=>n+(r.actualSubsteps??0),0),
  partialUncommittedActualSubsteps:this.records.filter(r=>!r.executed).reduce((n,r)=>n+(r.actualSubsteps??0),0)};}
}
