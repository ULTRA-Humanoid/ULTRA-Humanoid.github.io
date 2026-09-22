/** Per-control private continuation in the actual main simulator; no snapshots. */
const invariant=(condition,message='Matched carry runtime invariant failed')=>{if(!condition)throw new Error(message);};
const equal=(actual,expected,message)=>invariant(actual===expected,message??`Expected ${expected}; received ${actual}`);
const equalState=(a,b)=>invariant(a?.length===b?.length&&Array.from(a).every((v,i)=>v===b[i]),'Recorded preview and actual state differ');
import {TeacherObsBuilder} from './teacher_obs.js';
import {GroundingContactPreview,measureGroundingRelease,StageFeedbackMeasurement,summarizePlacementSafety} from './placement_feedback.js';
import {STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES} from './staged_student_transport_controller.js';
import {computeObjPointsHeadingFrame,buildObs} from './obs_builder.js';
import {createObjectSelection} from './object_selection.js';
import {GoalTranslator} from './goal_translator.js';
import {UserState} from './state.js';
import {loadTeacherSkill} from './teacher_skill.js';
import {TeacherBoxExitController} from './teacher_box_exit_controller.js';
import {RestrictedLocomotionController} from './restricted_locomotion_controller.js';
import {applyPDTorques,ACTION_SCALE} from './pd_control.js';
export class MatchedCarryRuntime{
  constructor(env){Object.assign(this,env);const {model,mujoco,addresses}=this;
    invariant(env.physicsMonitor&&typeof env.physicsMonitor.observe==='function'&&typeof env.physicsMonitor.snapshot==='function',
      'An actual streaming physics monitor is required independently of optional recorders');
    this.captureDiagnostics=env.captureDiagnostics===true;this.diagnosticCapacity=env.diagnosticCapacity??32;
    invariant(Number.isSafeInteger(this.diagnosticCapacity)&&this.diagnosticCapacity>=1&&this.diagnosticCapacity<=256,'Diagnostic capacity must be1…256');
    const nameAt=o=>{let s='';while(model.names[o])s+=String.fromCharCode(model.names[o++]);return s;};
    this.names=Array.from({length:model.nbody},(_,i)=>nameAt(model.name_bodyadr[i]));this.ids=Object.fromEntries(this.names.map((n,i)=>[n,i]));
    this.objectId=this.ids[env.owner.raw.objectBodyName];this.rootId=this.ids.pelvis;
    this.objects=new Set(this.names.flatMap((n,i)=>n.startsWith('active_')?[i]:[]));this.humans=new Set();
    for(let i=1;i<model.nbody;i++){let p=i;while(p>0&&p!==this.rootId)p=model.body_parentid[p];if(p===this.rootId)this.humans.add(i);}
    this.force=new mujoco.DoubleBuffer(6);this.preview=new GroundingContactPreview(mujoco,model,{addresses,minRootHeightM:.45,minUpright:.5});
    this.measurement=new StageFeedbackMeasurement(mujoco,model,{rootBodyId:this.rootId,objectBodyId:this.objectId,
      leftHandBodyId:this.ids.left_rubber_hand,rightHandBodyId:this.ids.right_rubber_hand});
    this.points=createObjectSelection(env.owner.raw.objectBodyName,this.objectId,env.pointCloudDb).pointsLocal;
    this.teacher=null;this.teacherPhase=null;this.ending=null;this.records=[];this.dense=this.captureDiagnostics?[]:null;this.disposed=false;
    this.observerErrorCount=0;this.lastObserverError=null;
  }
  observeOptional(callback,value){if(!callback)return;try{callback(value);}catch(error){
    this.observerErrorCount=(this.observerErrorCount??0)+1;this.lastObserverError={message:String(error?.message??error),physicalControl:this.context().physicalControl};}}
  pos(id){return Array.from(this.data.xpos.slice(3*id,3*id+3));}
  quat(id){const q=this.data.xquat.slice(4*id,4*id+4);return[q[1],q[2],q[3],q[0]];}
  live(){return measureGroundingRelease({...this,measurement:this.measurement,objectId:this.objectId,
    distalBodyIds:STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.map(n=>this.ids[n]),forceBuffer:this.force});}
  proprio(){const l=this.live();return{rootPosWorld:l.rootPositionWorld,rootQuatXyzwWorld:l.rootQuaternionWorld,
    rootVelWorld:Array.from(this.data.qvel.slice(0,3)),pelvisZ:l.rootPositionWorld[2],rootHeight:l.rootPositionWorld[2],
    uprightScore:1-2*(l.rootQuaternionWorld[0]**2+l.rootQuaternionWorld[1]**2),objPosWorld:l.objectPositionWorld,
    objQuatXyzwWorld:this.quat(this.objectId),objectBodyName:this.owner.skill.objectBodyName,
    footContactL:Math.min(...['left_ankle_pitch_link','left_ankle_roll_link'].map(n=>this.data.xpos[this.ids[n]*3+2]))<.1?1:0,
    footContactR:Math.min(...['right_ankle_pitch_link','right_ankle_roll_link'].map(n=>this.data.xpos[this.ids[n]*3+2]))<.1?1:0};}
  history(){return{bodyInitialized:this.body.hasInitialized,bodyHistory:Array.from(this.body.historyBuf),lastAction:Array.from(this.action),targetQ:Array.from(this.target),
    lastTorque:Array.from(this.torque),previousDofPos:Array.from(this.previousDofPos),previousDofVel:Array.from(this.previousDofVel),
    activeTeacher:this.teacher?{lastDofPos:Array.from(this.teacher.lastDofPos),lastDofVel:Array.from(this.teacher.lastDofVel)}:null};}
  async prepareEnding(){if(this.ending)return;const translator=new GoalTranslator({clickPositionSource:'stable_receding'});
    await translator.load('public/clip_db.bin','public/clip_db.json');translator.reset();translator.rng=()=>.5;
    const retreat=await loadTeacherSkill('public/teacher_box_exit_reference.json'),neutral=await loadTeacherSkill('public/teacher_walk_medium_reference.json');
    const user=new UserState();user.activeObjName=this.owner.skill.objectBodyName;user.deterministic=true;
    this.ending={translator,user,exit:new TeacherBoxExitController(retreat),standing:new RestrictedLocomotionController([neutral],{settleOnStart:true})};
  }
  endingSample(sample,proprio,bodyObs){const n=this.owner.counts.postplacement,e=this.ending;
    if(n<180){const tr=e.translator.step({user:e.user,proprio});return{observation:buildObs(e.user,proprio.rootPosWorld,
      proprio.rootQuatXyzwWorld,proprio.objPosWorld,proprio.objQuatXyzwWorld,this.points,bodyObs,tr),actorSkill:null};}
    if(n===180){invariant(proprio.objPosWorld[2]<=.25&&proprio.rootPosWorld[2]>=.45&&proprio.uprightScore>=.5,
      'Existing measured post-settling placement and balance required before exit');
      e.exit.start(proprio,{terminalFrame:this.owner.endingReferenceFrame??this.owner.bank[365],objectBodyName:this.owner.skill.objectBodyName,objectPointsLocal:this.owner.skill.objectPointsLocal});}
    if(n===619){const done=e.exit.step(proprio);invariant(done.justCompleted);equal(e.exit.totalControls,439);e.standing.reanchor();}
    const actor=n<619?e.exit:e.standing,next=actor.step(proprio);invariant(next.supported,'Existing exit/standing reference must be supported');
    equal(next.phase,sample.phase);return{referenceFrames:next.referenceFrames,actorSkill:actor.skill,actor};
  }
  readContacts(){const events=[],table=this.data.contact;
    try{for(let i=0;i<this.data.ncon;i++){const c=table.get(i);try{
      const a=this.model.geom_bodyid[c.geom1],b=this.model.geom_bodyid[c.geom2],object=this.objects.has(a)?a:this.objects.has(b)?b:-1;
      const other=object===a?b:a;if(object<0||(!this.humans.has(other)&&!this.objects.has(other)))continue;
      this.force.GetView().fill(0);this.mujoco.mj_contactForce(this.model,this.data,i,this.force);const normal=Math.max(0,this.force.GetView()[0]);
      invariant(Number.isFinite(normal)&&Number.isFinite(c.dist));if(normal>.1||c.dist<=0)events.push({object:this.names[object],body:this.names[other],
        normal_force_n:normal,distance_m:c.dist,leg:/_(hip|knee|ankle)_/.test(this.names[other]),object_pair:this.objects.has(other),
        allowedLoadedSupport:object===this.objectId&&STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.includes(this.names[other])});
    }finally{c.delete();}}}finally{table.delete();}return events;
  }
  physical(){return{qpos:Array.from(this.data.qpos),qvel:Array.from(this.data.qvel),ctrl:Array.from(this.data.ctrl),root:this.pos(this.rootId),rootQuaternion:this.quat(this.rootId),
    object:this.pos(this.objectId),objectQuaternion:this.quat(this.objectId),allObjects:Object.fromEntries([...this.objects].map(id=>[this.names[id],{position:this.pos(id),quaternion:this.quat(id)}])),simTime:this.data.time};}
  async step(){const owner=this.owner,entryContext=this.context();invariant(owner.active&&owner.current(entryContext));
    if(owner.role==='postplacement'){await this.prepareEnding();if(!owner.current(this.context())){owner.cancel('owner_changed_during_ending_load');return false;}}
    const live=this.live(),sample=owner.sample(this.context(),live),before=this.captureDiagnostics?this.history():{
      bodyInitialized:this.body.hasInitialized,bodyHistory:Float32Array.from(this.body.historyBuf),lastAction:Float32Array.from(this.action)};let beforeTeacher=null,teacherAtQuery=null,speculativeBody=null,speculativeTeacher=null,actualSubsteps=0;
    const same=(a,b)=>a?.length===b?.length&&Array.from(a).every((v,i)=>v===b[i]);
    const currentTeacher=()=>!teacherAtQuery||(this.teacher===teacherAtQuery&&this.getCurrentTeacherBuilder()===teacherAtQuery
      &&teacherAtQuery.jacp!=null);
    const current=()=>!this.disposed&&currentTeacher()&&owner.canActuate(this.context(),sample);
    const rollback=()=>{const c=this.context();if(actualSubsteps===0&&c.episode===entryContext.episode&&c.physicalControl===entryContext.physicalControl){
      if(same(this.body.historyBuf,speculativeBody)&&same(this.action,before.lastAction)){this.body.historyBuf.set(before.bodyHistory);this.body.hasInitialized=before.bodyInitialized;}
      if(teacherAtQuery&&beforeTeacher&&!this.disposed&&this.teacher===teacherAtQuery&&this.getCurrentTeacherBuilder()===teacherAtQuery
        &&teacherAtQuery.jacp!=null&&same(teacherAtQuery.lastDofPos,speculativeTeacher?.lastDofPos)
        &&same(teacherAtQuery.lastDofVel,speculativeTeacher?.lastDofVel))teacherAtQuery.reset(beforeTeacher);}};
    const record={physicalControlBefore:entryContext.physicalControl,phase:sample.phase,sourceIndex:sample.sourceIndex,executed:false,
      ...(this.captureDiagnostics?{historyBefore:before}:{})};this.records.push(record);
    if(!this.captureDiagnostics&&this.records.length>this.diagnosticCapacity)this.records.shift();
    let obs,mu,checked=null,ending=null;
    try{
      const bodyObs=this.body.build(this.data,this.action);speculativeBody=Float32Array.from(this.body.historyBuf);if(this.captureDiagnostics)record.bodyObservation=Array.from(bodyObs);
      if(owner.window){const points=computeObjPointsHeadingFrame(this.points,live.objectPositionWorld,this.quat(this.objectId),live.rootPositionWorld,live.rootQuaternionWorld);
        obs=owner.packStudent(sample,bodyObs,points);
      }else{
        ending=owner.role==='postplacement'?this.endingSample(sample,this.proprio(),bodyObs):null;
        if(ending?.observation)obs=ending.observation;
        else{if(this.teacherPhase!==sample.phase){this.teacher?.dispose();this.teacher=new TeacherObsBuilder(this.mujoco,this.model,ending?.actorSkill??owner.skill);
            this.teacher.reset({lastDofPos:this.previousDofPos,lastDofVel:this.previousDofVel});this.teacherPhase=sample.phase;this.setTeacherBuilder(this.teacher);}
          beforeTeacher={lastDofPos:Array.from(this.teacher.lastDofPos),lastDofVel:Array.from(this.teacher.lastDofVel)};
          obs=this.teacher.build(this.data,ending?.referenceFrames??sample.referenceFrames,this.action,this.torque);teacherAtQuery=this.teacher;
          speculativeTeacher={lastDofPos:Array.from(this.teacher.lastDofPos),lastDofVel:Array.from(this.teacher.lastDofVel)};}
      }
      invariant(obs?.length===(sample.mode==='teacher'?4052:1422)&&Array.from(obs).every(Number.isFinite),
        'Finite complete teacher4052 or student1422 input required before inference');
      record.observationDimension=obs.length;if(this.captureDiagnostics)record.observation=Array.from(obs);
      this.observeOptional(this.onActorInput,{context:entryContext,sample,observation:obs});
      if(!current()){rollback();owner.cancel('owner_changed_before_query');return false;}
      mu=sample.mode==='teacher'?await this.teacherPolicy.infer(obs):await this.policy.infer(obs,this.zeroNoise);
      this.observeOptional(this.onActorResult,{context:entryContext,sample,observation:obs,rawAction:mu});
      if(this.captureDiagnostics)record.rawAction=Array.from(mu);if(!current()){rollback();owner.cancel('owner_changed_during_inference');return false;}
      invariant(mu.length===29&&Array.from(mu).every(Number.isFinite));const applied=Float32Array.from(mu,x=>Math.max(-1,Math.min(1,x))),target=Float32Array.from(applied,x=>ACTION_SCALE*x);
      if(sample.phase!=='settling'){
        const predicted=[];checked=this.preview.evaluate(this.data,target,{allowContact:item=>owner.role!=='postplacement'
          &&item.objectName===owner.skill.objectBodyName&&STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.includes(item.bodyName),
          captureState:this.captureDiagnostics,onSubstep:this.captureDiagnostics?s=>predicted.push(s):null});
        if(this.captureDiagnostics)checked={...checked,substeps:predicted};record.preview=checked;
        if(!current()){rollback();owner.cancel('owner_changed_during_preview');return false;}
        if(!checked.supported){rollback();owner.cancel('private_programme_preview_refused');record.reason=checked.reason;return false;}
        equal(checked.requestedSubsteps,17);equal(checked.completedSubsteps,17);equal(checked.unwantedContactCount,0);
      }
      this.onBeforePhysical({sample,obs,mu,checked});
      if(!current()){rollback();owner.cancel('owner_changed_before_physics');return false;}
      this.action.set(applied);this.target.set(target);
      this.previousDofPos.set(Float32Array.from(this.addresses.qposAddr,id=>this.data.qpos[id]));
      this.previousDofVel.set(Float32Array.from(this.addresses.qvelAddr,id=>this.data.qvel[id]));
      const dense=[];let monitoredActualViolation=false;
      for(let substep=1;substep<=17;substep++){
        applyPDTorques(this.model,this.data,this.target,this.addresses);this.mujoco.mj_step(this.model,this.data);actualSubsteps++;record.actualSubsteps=actualSubsteps;
        // Account the real substep before any additional measurement can throw
        // or a diagnostic callback can Reset/change the current owner.
        const monitorClear=this.physicsMonitor.observe(this.data,{episode:entryContext.episode,physicalControl:entryContext.physicalControl+1,
          substep,phase:sample.phase,allowLoadedSupport:owner.role!=='postplacement'});
        invariant(typeof monitorClear==='boolean','Actual streaming monitor must return its measured result');
        monitoredActualViolation||=!monitorClear;
        const q=this.data.xquat,offset=this.rootId*4,s={episodeControl:entryContext.physicalControl+1,substep,phase:sample.phase,
          sourceIndex:sample.sourceIndex,rootZ:this.data.xpos[this.rootId*3+2],upright:1-2*(q[offset+1]**2+q[offset+2]**2),contacts:this.readContacts()};
        const state=this.captureDiagnostics?{...this.physical(),...s}:s;
        if(this.captureDiagnostics&&checked)for(const key of ['qpos','qvel','ctrl'])equalState(state[key],checked.substeps[substep-1][key]);
        dense.push(s);if(this.captureDiagnostics)this.dense.push(state);this.observeOptional(this.onActualSubstep,state);
        if(!current()){owner.cancel('owner_changed_during_substep_observer');return false;}
        if(this.onRecordedState)this.observeOptional(this.onRecordedState,this.captureDiagnostics?state:{...this.physical(),...s});
        if(!current()){owner.cancel('owner_changed_during_state_observer');return false;}
      }
      for(let j=0;j<29;j++)this.torque[j]=this.data.ctrl[this.addresses.actuatorOrder[j]];
      ending?.actor?.advance();this.onPhysicalCommitted();
      const measured=this.live(),phaseForSafety=['student_second_loaded','student_grounding106'].includes(sample.phase)?'teacher_release44':sample.phase;
      const safety=summarizePlacementSafety(dense.map(s=>({...s,phase:phaseForSafety})));
      owner.commit(this.context(),{sample,physicsSubsteps:17,preview:checked,live:measured,
        actualViolation:monitoredActualViolation||!safety.fullEndingHasNoActualViolation});
      Object.assign(record,{executed:true,physicalControlAfter:this.context().physicalControl,measured,safety,monitoredActualViolation,ownerAfter:owner.review(),
        ...(this.captureDiagnostics?{historyAfter:this.history(),physical:this.physical()}:{})});
      this.onAfterControl(record);return true;
    }catch(error){rollback();owner.cancel('private_programme_error');record.error={message:error.message,...(this.captureDiagnostics?{stack:error.stack}:{})};throw error;}
    finally{this.observeOptional(this.onControlDiagnostic,record);}
  }
  dispose(){if(this.disposed)return;this.disposed=true;this.teacher?.dispose();this.preview.dispose();this.measurement.dispose();this.force.delete();}
}
