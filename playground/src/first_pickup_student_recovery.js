/** Private first-source admission/preparation; never advances or writes physics. */
import {withInitialStance} from './teacher_skill.js';
import {STAGED_STUDENT_APPROACH_LIMITS as LIMITS} from './staged_student_approach_controller.js';

const finite=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
const same=(a,b)=>a?.length===b?.length&&Array.from(a??[]).every((v,i)=>v===b[i]);
const count=n=>Number.isSafeInteger(n)&&n>=0;
const deny=reason=>({supported:false,reason});
const velocityRanges=[[7,13],[42,71],[78,84],[357,591]];

/** Direct current solver measurement, independent of any saved-data observer.
 * Contact activation uses the existing preview's .1 N / nonpositive-distance
 * definition. An unloaded box must have measured floor support. */
export function readFirstPickupContacts({mujoco,model,data,objectId,rootId,feet,objectIds}) {
  if(!Number.isInteger(objectId)||objectId<=0||!Number.isInteger(rootId)||rootId<=0
      ||!Array.isArray(feet)||feet.some(x=>!Number.isInteger(x)||x<=0)
      ||!Array.isArray(objectIds)||!objectIds.includes(objectId)) throw Error('Known scene object, humanoid and feet required');
  const humans=new Set();
  for(let id=1;id<model.nbody;id++){
    let ancestor=id;
    while(ancestor>0&&ancestor!==rootId)ancestor=model.body_parentid[ancestor];
    if(ancestor===rootId)humans.add(id);
  }
  const boxes=new Set(objectIds),footIds=new Set(feet),force=new mujoco.DoubleBuffer(6);
  const measurement={evaluatedContacts:0,missingEvidence:false,selectedFloorNormalN:0,
    loadedFootNormalN:0,activeHumanBoxContacts:0,activeBoxPairContacts:0,contacts:[]};
  const contacts=data.contact;
  try {
    if(!count(data.ncon))throw Error('Missing actual contact count');
    for(let index=0;index<data.ncon;index++){
      const c=contacts.get(index);
      try {
        force.GetView().fill(NaN);mujoco.mj_contactForce(model,data,index,force);
        const f=Array.from(force.GetView()),a=model.geom_bodyid[c.geom1],b=model.geom_bodyid[c.geom2];
        if(!finite(f,6)||!Number.isFinite(c.dist)||!count(a)||!count(b))throw Error('Nonfinite actual contact evidence');
        const normalN=Math.max(0,f[0]),active=c.dist<=0||normalN>.1;
        measurement.evaluatedContacts++;
        if((a===objectId&&b===0)||(b===objectId&&a===0))measurement.selectedFloorNormalN+=normalN;
        if((a===0&&footIds.has(b))||(b===0&&footIds.has(a)))measurement.loadedFootNormalN+=normalN;
        const humanBox=(boxes.has(a)&&humans.has(b))||(boxes.has(b)&&humans.has(a));
        const boxPair=boxes.has(a)&&boxes.has(b)&&a!==b;
        if(active&&humanBox)measurement.activeHumanBoxContacts++;
        if(active&&boxPair)measurement.activeBoxPairContacts++;
        measurement.contacts.push({index,geom1:c.geom1,geom2:c.geom2,body1:a,body2:b,distanceM:c.dist,force6:f,active});
      } finally {c.delete();}
    }
  } finally {contacts.delete();force.delete();}
  return measurement;
}

export function checkFirstPickupStudentRecovery({parent,owner,step,live,route,contact,episode,
    physicalControl,requestId,request,activeParent,pendingParent,originalGoalWorld,latestRequestId,
    queuedRequestId,selectedObjectBodyName}) {
  if(!count(episode)||!count(physicalControl)||!count(requestId)||requestId===0)return deny('invalid_physical_request_clock');
  if(!parent||parent!==activeParent||parent!==pendingParent||!owner||owner===parent||owner.cancelRequested
      ||step?.completionReason!=='reference_sweep_clearance'
      ||!(step.justCompleted===true||(step.mode==='none'&&step.supported===false&&step.justCompleted===false)))
    return deny('not_owned_recorded_approach_refusal');
  if(request?.requestId!==requestId||request.episodeVersion!==episode||request.task!=='carry'
      ||!['started'].includes(request.disposition)||requestId!==latestRequestId||queuedRequestId!==null
      ||!finite(originalGoalWorld,3)||!same(request.goalWorld,originalGoalWorld)
      ||!same(parent.requestedGoalWorld,originalGoalWorld))return deny('original_request_not_current');
  const child=parent.child,raw=parent.rawSkill,skill=parent.skill;
  if(parent.phase!=='approach'||child?.phase!=='approach'||parent.finishRequested||child.finishRequested
      ||parent.segmentIndex!==0||parent.referenceIndex!==0||child.worldFrames!==null
      ||parent.exitedSegmentIndex!==null||parent.segmentExitResults?.length!==0||parent.segmentResults?.length!==0
      ||parent.segmentPreparations?.length!==0||!count(child.approachSteps)||!count(child.settled)
      ||!Number.isInteger(child._lastApproachSample)||child._lastApproachSample>child.approachSteps)
    return deny('first_source_already_started_or_prepared');
  const oldStance=skill?.sourceFrames-raw?.sourceFrames;
  if(!Number.isInteger(oldStance)||oldStance<1||oldStance>90||!Array.isArray(skill?.frames)
      ||!Array.isArray(raw?.frames)||skill.frames.length!==raw.frames.length+oldStance
      ||raw.frames.some((row,i)=>skill.frames[oldStance+i]!==row)
      ||!count(child.warpStartFrame)||!count(child.warpEndFrame)||child.warpStartFrame<oldStance
      ||child.warpStartFrame>=child.warpEndFrame||child.warpEndFrame>=skill.sourceFrames)
    return deny('complete_prepared_source_or_warp_missing');
  const quiet=Float32Array.from(raw.frames[0]??[]);
  for(const [a,b]of velocityRanges)quiet.fill(0,a,b);
  const plan=child.referencePlan,first=plan?.first,goal=parent.approachGoalWorld;
  if(!finite(quiet,747)||skill.frames.slice(0,oldStance).some(row=>!same(row,quiet))
      ||!Array.isArray(plan?.frames)||plan.frames.length!==skill.frames.length||plan.frames[0]!==first
      ||plan.frames.some(row=>!finite(row,747))||!finite(first,747)
      ||velocityRanges.some(([a,b])=>Array.from(first.slice(a,b)).some(v=>v!==0)))
    return deny('existing_quiet_preparation_or_warped_plan_missing');
  if(!finite(live?.rootPosWorld,3)||!finite(live?.rootQuatXyzwWorld,4)||!finite(live?.rootVelWorld,3)
      ||!finite(live?.objPosWorld,3)||!finite(goal,3)
      ||live.objectBodyName!==raw.objectBodyName||selectedObjectBodyName!==raw.objectBodyName
      ||Math.abs(Math.hypot(...live.rootQuatXyzwWorld)-1)>1e-5
      ||Math.abs(Math.hypot(...first.slice(3,7))-1)>1e-5)return deny('invalid_live_state_or_selected_object');
  const q=live.rootQuatXyzwWorld,upright=1-2*(q[0]**2+q[1]**2);
  const planarSpeedMps=Math.hypot(...live.rootVelWorld.slice(0,2));
  const distanceM=Math.hypot(goal[0]-live.rootPosWorld[0],goal[1]-live.rootPosWorld[1]);
  if(live.rootPosWorld[2]<LIMITS.minRootHeightM||upright<LIMITS.minUpright||planarSpeedMps>LIMITS.maxPlanarSpeedMps)
    return deny('reached_state_not_stably_standing');
  if(contact?.missingEvidence!==false||!count(contact.evaluatedContacts)||contact.evaluatedContacts===0
      ||!Number.isFinite(contact.selectedFloorNormalN)||contact.selectedFloorNormalN<=.1
      ||!Number.isFinite(contact.loadedFootNormalN)||contact.loadedFootNormalN<=.1
      ||contact.activeHumanBoxContacts!==0||contact.activeBoxPairContacts!==0)
    return deny('selected_box_not_measured_unloaded_on_floor');
  if(route?.supported!==true||route.directBlocked!==false)return deny('direct_approach_not_supported');
  if(distanceM>LIMITS.maxDistanceM)return deny('student_approach_distance');
  return {supported:true,reason:null,objectBodyName:raw.objectBodyName,segmentIndex:0,requestId,episode,
    physicalControl,distanceM,planarSpeedMps,upright,oldStanceFrames:oldStance,initialStanceFrames:90,
    addedStanceFrames:90-oldStance,rawSourceFrames:raw.sourceFrames,preparedSourceFrames:raw.sourceFrames+90,
    priorApproachSteps:child.approachSteps,priorLastApproachSample:child._lastApproachSample,priorSettled:child.settled,
    plan:{stage:'approach',mode:'LOCO',humanGoalWorld:[goal[0],goal[1],first[2]],
      humanGoalRotationWorld:Array.from(first.slice(3,7)),objectGoalWorld:Array.from(live.objPosWorld),
      finalDestinationWorld:Array.from(originalGoalWorld)}};
}

/** Same child, old warped plan suffix, source and sampled approach clock. Only
 * new quiet reference rows and their warp-index offset are added. The parent's
 * ordinary live arrival/facing/clearance checks still determine teacher entry. */
function prepare(parent,entry) {
  const child=parent.child,oldSkill=child.skill,oldPlan=child.referencePlan,added=entry.addedStanceFrames;
  const skill=withInitialStance(oldSkill,added);
  const frames=[...Array.from({length:added},()=>Float32Array.from(oldPlan.first)),...oldPlan.frames];
  const plan={...oldPlan,frames,first:frames[0]};
  const preparation={segmentIndex:0,initialStanceFrames:90,addedStanceFrames:added,
    originalPreparedSourceFrames:oldSkill.sourceFrames,preparedSourceFrames:skill.sourceFrames,
    originalWarpStartFrame:child.warpStartFrame,originalWarpEndFrame:child.warpEndFrame,
    warpStartFrame:child.warpStartFrame+added,warpEndFrame:child.warpEndFrame+added,
    previousApproachSteps:child.approachSteps,previousLastApproachSample:child._lastApproachSample,
    previousSettled:child.settled,reason:'owned_first_pickup_student_recovery'};
  child.skill=skill;child.referencePlan=plan;
  child.warpStartFrame+=added;child.warpEndFrame+=added;
  parent.segmentPreparations.push(preparation);
  return {preparation,oldSkill,oldPlan};
}

export class FirstPickupStudentRecoveryRegistry {
  #attempted=new WeakSet();
  #leases=[];
  tryClaim(options) {
    if(options.parent&&this.#attempted.has(options.parent))return deny('first_pickup_recovery_already_used');
    const entry=checkFirstPickupStudentRecovery(options);
    if(!entry.supported)return entry;
    if(typeof options.readContext!=='function')return deny('live_command_context_required');
    const {parent,episode,requestId,physicalControl}=options,child=parent.child,raw=parent.rawSkill;
    const readContext=options.readContext,before=readContext();
    const commandRevision=before.commandRevision;
    if(!count(commandRevision)||before.physicalControl!==physicalControl||typeof before.paused!=='boolean')
      return deny('current_command_or_manual_step_context_missing');
    const admissionPaused=before.paused;
    const fixedGoal=Array.from(parent.requestedGoalWorld),fixedApproach=Array.from(parent.approachGoalWorld);
    const lease={metadata:structuredClone({...entry,contact:options.contact,route:options.route}),
      isOwnedBy(context) {
        const current=readContext();
        return context.parent===parent&&context.episode===episode&&context.requestId===requestId
          &&current.parent===parent&&current.activeParent===parent&&current.episode===episode
          &&current.requestId===requestId&&current.latestRequestId===requestId&&current.queuedRequestId===null
          &&current.commandRevision===commandRevision&&current.selectedObjectBodyName===raw.objectBodyName
          &&current.paused===admissionPaused&&!current.suspended
          &&parent.child===child&&parent.rawSkill===raw&&parent.segmentIndex===0&&!parent.finishRequested
          &&same(parent.requestedGoalWorld,fixedGoal)&&same(current.originalGoalWorld,fixedGoal)
          &&current.request?.requestId===requestId&&current.request?.episodeVersion===episode
          &&current.request?.task==='carry'&&current.request?.disposition==='started'&&same(current.request?.goalWorld,fixedGoal)
          &&(parent.phase==='approach'?parent.referenceIndex===0&&same(parent.approachGoalWorld,fixedApproach):parent.phase==='teacher');
      },
      initialControl:physicalControl,
    };
    if(!lease.isOwnedBy(options))return deny('refusal_owner_changed_before_preparation');
    const prepared=prepare(parent,entry);
    lease.metadata.preparation=structuredClone(prepared.preparation);
    const preparedSkill=child.skill;
    const owns=lease.isOwnedBy;
    lease.isOwnedBy=context=>child.skill===preparedSkill&&owns(context);
    this.#attempted.add(parent);this.#leases.push(lease);
    return {supported:true,reason:null,lease,entry:lease.metadata};
  }
  get admissionCount() { return this.#leases.length; }
  review() { return this.#leases.map(l=>structuredClone(l.metadata)); }
}

/** Capture the ordinary parent's live alignment at its actual inference
 * boundary. This is not a committed teacher action or a full-source outcome. */
export function captureFirstPickupTeacherHandoff({controller,parent,step,live,episode,requestId,physicalControl}) {
  const ended=controller?.ended,preparation=parent?.segmentPreparations?.find(r=>r.segmentIndex===0);
  const directStudentHandoff=ended?.arrived===true&&ended.atControl===physicalControl;
  const recordedFallbackHandoff=ended?.fallback===true&&count(ended.atControl)&&physicalControl>=ended.atControl;
  if(!controller?.firstPickupOrigin||(!directStudentHandoff&&!recordedFallbackHandoff)
      ||!controller.isOwnedBy({parent,episode,requestId})||parent.phase!=='teacher'
      ||step?.justEnteredTeacher!==true||step.mode!=='teacher'||parent.referenceIndex!==0
      ||!preparation||!same(parent.requestedGoalWorld,controller.plan.finalDestinationWorld)
      ||live?.objectBodyName!==parent.skill?.objectBodyName||!finite(live.rootPosWorld,3)
      ||!finite(live.rootQuatXyzwWorld,4)||!finite(live.rootVelWorld,3)||!finite(live.objPosWorld,3)
      ||!finite(live.objQuatXyzwWorld,4))
    throw Error('An owned ordinary first teacher handoff is required');
  const raw=parent.rawSkill,skill=parent.skill,child=parent.child,plan=parent.referencePlan;
  const quiet=Float32Array.from(raw.frames[0]);
  for(const [a,b]of velocityRanges)quiet.fill(0,a,b);
  if(skill.sourceFrames!==raw.sourceFrames+90||skill.frames.length!==raw.frames.length+90
      ||raw.frames.some((row,i)=>skill.frames[90+i]!==row)
      ||skill.frames.slice(0,90).some(row=>!same(row,quiet))
      ||child.warpStartFrame!==preparation.warpStartFrame||child.warpEndFrame!==preparation.warpEndFrame
      ||parent.worldFrames!==plan.frames||plan.frames.length!==skill.frames.length
      ||plan.frames.some(row=>!finite(row,747))||!same(plan.requestedGoalWorld,child.requestedGoalWorld)
      ||step.referenceFrames?.[0]!==plan.frames[1]||step.referenceFrames?.[1]!==plan.frames[16])
    throw Error('Teacher handoff changed complete source, quiet preparation, warp or lookahead');
  return {episode,requestId,physicalControl,originalGoalWorld:Array.from(parent.requestedGoalWorld),live:structuredClone(live),
    parentApproachSteps:child.approachSteps,referenceIndex:0,rawSourceFrames:raw.sourceFrames,
    preparedSourceFrames:skill.sourceFrames,initialStanceFrames:90,addedStanceFrames:preparation.addedStanceFrames,
    rawSuffixUnchanged:true,quietPreparationUnchanged:true,warpStartFrame:child.warpStartFrame,warpEndFrame:child.warpEndFrame,
    teacherReferenceFrames:step.referenceFrames.map(r=>Array.from(r)),
    actualLiveReferencePlan:structuredClone({...plan,frames:plan.frames.map(r=>Array.from(r)),first:Array.from(plan.first)}),
    alignment:'ordinary_parent_live_recomputation_at_quiet_arrival',
    arrivalOwner:directStudentHandoff?'finite_student_window':'recorded_fallback_after_student_window',
    firstTeacherControlCommitted:false,actualTeacherSubsteps:0};
}
