/** Private ownership of existing restricted standing after a finite student
 * window. Never advances the parent or physics itself. */
import assert from './recovery_assert.js';

export const FIRST_PICKUP_STANDING_LIMIT=60;
const count=v=>Number.isSafeInteger(v)&&v>=0;
const equal=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const finite=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);

export class FirstPickupStandingRecovery {
  #parent;#student;#child;#skill;#raw;#readContext;#identity;#goal;#approachGoal;
  #lastControl;#initialApproachSteps;#lastSample=null;#pending=null;#ended=null;
  #records=[];#samples=[];
  constructor({parent,student,readContext}){
    assert.equal(typeof readContext,'function');
    const c=readContext(),ended=student?.ended;
    assert.ok(student?.firstPickupOrigin&&student.isOwnedBy(c),'Owned first-pickup student required');
    assert.equal(ended?.reason,'stage_goal_window_elapsed');assert.equal(ended.arrived,false);assert.equal(ended.fallback,true);
    assert.equal(student.controls,student.horizonControls);assert.equal(student.controls,180);
    assert.equal(c.physicalControl,ended.atControl);assert.equal(c.episode,student.episode);
    assert.equal(c.student,student);assert.equal(c.parent,parent);assert.equal(c.activeParent,parent);
    assert.equal(parent.phase,'approach');assert.equal(parent.referenceIndex,0);assert.equal(parent.segmentIndex,0);
    assert.equal(parent.finishRequested,false);assert.ok(count(parent.child.approachSteps));
    assert.ok(count(c.commandRevision)&&typeof c.paused==='boolean');
    assert.equal(c.latestRequestId,c.requestId);assert.equal(c.queuedRequestId,null);assert.ok(!c.suspended);
    assert.equal(c.request?.disposition,'started');assert.equal(c.request?.episodeVersion,c.episode);
    assert.equal(c.request?.requestId,c.requestId);assert.equal(c.request?.task,'carry');
    assert.ok(finite(parent.requestedGoalWorld,3)&&equal(c.originalGoalWorld,parent.requestedGoalWorld));
    assert.ok(equal(c.request.goalWorld,parent.requestedGoalWorld));
    assert.equal(c.selectedObjectBodyName,parent.rawSkill.objectBodyName);
    assert.equal(parent.skill.sourceFrames,parent.rawSkill.sourceFrames+90);
    this.#parent=parent;this.#student=student;this.#child=parent.child;this.#skill=parent.skill;this.#raw=parent.rawSkill;
    this.#readContext=readContext;this.#identity={episode:c.episode,requestId:c.requestId,commandRevision:c.commandRevision,paused:c.paused};
    this.#goal=Array.from(parent.requestedGoalWorld);this.#approachGoal=Array.from(parent.approachGoalWorld);
    this.startControl=this.#lastControl=c.physicalControl;
    this.#initialApproachSteps=parent.child.approachSteps;
    assert.ok(this.isOwnedBy(c));
  }
  get active(){return this.#ended===null;}
  get controls(){return this.#lastControl-this.startControl;}
  get ended(){return this.#ended?structuredClone(this.#ended):null;}
  isOwnedBy(context){
    const c=this.#readContext(),p=this.#parent,id=this.#identity;
    return context?.parent===p&&context.student===this.#student&&context.episode===id.episode
      &&context.requestId===id.requestId&&c.parent===p&&c.activeParent===p&&c.student===this.#student
      &&c.episode===id.episode&&c.requestId===id.requestId&&c.latestRequestId===id.requestId&&c.queuedRequestId===null
      &&c.commandRevision===id.commandRevision&&c.paused===id.paused&&!c.suspended
      &&c.request?.disposition==='started'&&c.request.requestId===id.requestId&&c.request.episodeVersion===id.episode
      &&c.request.task==='carry'&&equal(c.request.goalWorld,this.#goal)&&equal(c.originalGoalWorld,this.#goal)
      &&c.selectedObjectBodyName===this.#raw.objectBodyName&&p.child===this.#child&&p.skill===this.#skill&&p.rawSkill===this.#raw
      &&p.segmentIndex===0&&!p.finishRequested&&p.completionReason!=='cancelled'
      &&equal(p.requestedGoalWorld,this.#goal);
  }
  #clock(context){
    assert.ok(this.isOwnedBy(context),'Standing recovery owner changed');
    assert.ok(count(context.physicalControl));assert.equal(context.physicalControl,this.#lastControl);
    assert.equal(this.#readContext().physicalControl,this.#lastControl);
  }
  #finish(reason,{arrived=false,fallback=false}={}){
    this.#ended??={reason,atControl:this.#lastControl,controls:this.controls,arrived,fallback};
    return this.ended;
  }
  observeParent(step,context,live){
    if(!this.active)return this.ended;
    this.#clock(context);assert.equal(this.#pending,null,'A pending physical action must close before another sample');
    const child=this.#parent.child;
    assert.equal(child.approachSteps,this.#initialApproachSteps+this.controls,'Parent approach clock must advance once per actual standing action');
    if(this.#lastSample!==context.physicalControl){
      assert.ok(finite(live.rootPosWorld,3)&&finite(live.rootVelWorld,3));
      this.#samples.push({physicalControl:context.physicalControl,actualStandingControls:this.controls,
        parentPhase:this.#parent.phase,parentApproachSteps:child.approachSteps,parentLastApproachSample:child._lastApproachSample,
        parentSettledSamples:child.settled,rootPositionWorld:Array.from(live.rootPosWorld),
        planarSpeedMps:Math.hypot(...live.rootVelWorld.slice(0,2)),
        distanceToAdmissionPickupM:Math.hypot(live.rootPosWorld[0]-this.#approachGoal[0],live.rootPosWorld[1]-this.#approachGoal[1]),
        stepMode:step?.mode,completionReason:step?.completionReason??null});
      this.#lastSample=context.physicalControl;
    }
    if(this.#parent.phase==='teacher'&&step?.mode==='teacher'&&step.justEnteredTeacher)
      return this.#finish('ordinary_parent_teacher_handoff',{arrived:true});
    if(this.#parent.phase!=='approach'||step?.phase!=='approach')
      return this.#finish('ordinary_parent_transition');
    if(this.controls>=FIRST_PICKUP_STANDING_LIMIT)return this.#finish('standing_window_elapsed',{fallback:true});
    return {active:true,remainingControls:FIRST_PICKUP_STANDING_LIMIT-this.controls};
  }
  begin(context,{referenceFrames,observation,rootPositionWorld}){
    this.#clock(context);assert.ok(this.active&&this.controls<FIRST_PICKUP_STANDING_LIMIT);
    assert.equal(this.#parent.phase,'approach');assert.equal(this.#parent.referenceIndex,0);
    assert.equal(this.#lastSample,this.#lastControl);assert.equal(this.#pending,null);
    assert.ok(referenceFrames?.length===2&&referenceFrames.every(row=>finite(row,747)));
    assert.ok(finite(observation,4052)&&finite(rootPositionWorld,3));
    const row={preControl:this.#lastControl,physicalControl:null,actualSubsteps:0,actualCommitted:false,
      parentApproachStepsBefore:this.#parent.child.approachSteps,rootPositionWorld:Array.from(rootPositionWorld),
      teacherReferenceFrames:referenceFrames.map(row=>Array.from(row)),teacherObservation:Array.from(observation),
      rawAction:null,preview:null};
    this.#pending=row;return row;
  }
  observeActualSubstep(row,context){
    this.#clock(context);assert.ok(this.active);assert.equal(row,this.#pending);
    assert.ok(row.actualSubsteps<17);row.actualSubsteps++;
  }
  commit(row,context){
    assert.ok(this.active&&this.isOwnedBy(context));assert.equal(row,this.#pending);
    assert.equal(context.physicalControl,this.#lastControl+1);
    assert.equal(this.#readContext().physicalControl,this.#lastControl,'Public control commits after both controller clocks');
    assert.equal(row.actualSubsteps,17);assert.ok(finite(row.rawAction,29));
    assert.equal(row.preview?.supported,true);assert.equal(row.preview.completedSubsteps,17);
    assert.equal(row.preview.unwantedContactCount,0);assert.equal(row.preview.allowedContactCount,0);
    assert.equal(this.#parent.child.approachSteps,this.#initialApproachSteps+this.controls+1);
    assert.equal(this.#parent.phase,'approach');assert.equal(this.#parent.referenceIndex,0);
    row.physicalControl=context.physicalControl;row.actualCommitted=true;
    row.parentApproachStepsAfter=this.#parent.child.approachSteps;
    this.#records.push(structuredClone(row));this.#pending=null;this.#lastControl=context.physicalControl;
  }
  abort(reason,row=null){
    assert.ok(typeof reason==='string'&&reason);if(row)assert.equal(row,this.#pending);
    if(this.#pending)this.#pending.discardedReason=reason;
    return this.#finish(reason);
  }

  progress(){return {startControl:this.startControl,controls:this.controls,maximumStandingControls:FIRST_PICKUP_STANDING_LIMIT,
    identity:{...this.#identity},originalGoalWorld:[...this.#goal],initialApproachSteps:this.#initialApproachSteps,
    ended:this.ended,sampleCount:this.#samples.length,lastSample:this.#samples.length?structuredClone(this.#samples.at(-1)):null,
    recordCount:this.#records.length,pending:this.#pending?{preControl:this.#pending.preControl,
      physicalControl:this.#pending.physicalControl,actualSubsteps:this.#pending.actualSubsteps,
      actualCommitted:this.#pending.actualCommitted,discardedReason:this.#pending.discardedReason??null}:null};}
  review(){return {startControl:this.startControl,controls:this.controls,maximumStandingControls:FIRST_PICKUP_STANDING_LIMIT,
    identity:{...this.#identity},originalGoalWorld:[...this.#goal],initialApproachSteps:this.#initialApproachSteps,
    ended:this.ended,samples:structuredClone(this.#samples),records:structuredClone(this.#records),
    pending:this.#pending?structuredClone(this.#pending):null};}
}
