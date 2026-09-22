// Predict one already-inferred control action in independent MuJoCo data.
// The caller owns inference, smoothing, policy history and committing the action.
import {applyPDTorques,resolveJointAddresses,SIM_DT,SIM_DECIMATION} from './pd_control.js';

function namesOf(model) {
  return Array.from({length:model.nbody},(_,id)=>{
    let name='',offset=model.name_bodyadr[id];
    while(model.names[offset])name+=String.fromCharCode(model.names[offset++]);
    return name;
  });
}

function finiteValues(values,length) {
  return values!=null&&values.length===length&&Array.from(values).every(Number.isFinite);
}

/** One-action prediction, valid only while model, input state and targets agree.
 * This does not infer later actions or establish future recoverability.
 */
export class ControlPreview {
  constructor(mujoco,model,{
    addresses=resolveJointAddresses(model),
    minRootHeightM=.70,minUpright=.90,contactForceThresholdN=.1,
  }={}) {
    if(!Number.isFinite(minRootHeightM)||!Number.isFinite(minUpright)||minUpright< -1||minUpright>1
      ||!Number.isFinite(contactForceThresholdN)||contactForceThresholdN<0) {
      throw new Error('Valid preview balance and contact thresholds are required');
    }
    if(!Number.isFinite(model.opt.timestep)||Math.abs(model.opt.timestep-SIM_DT)>1e-12)throw new Error('Control preview requires the active control timestep');
    this.mujoco=mujoco;this.model=model;this.addresses=addresses;
    this.minRootHeightM=minRootHeightM;this.minUpright=minUpright;
    this.contactForceThresholdN=contactForceThresholdN;
    this.names=namesOf(model);this.rootId=this.names.indexOf('pelvis');
    if(this.rootId<1)throw new Error('Control preview requires the humanoid pelvis');
    this.rootQpos=model.jnt_qposadr[model.body_jntadr[this.rootId]];
    this.objects=new Set(this.names.flatMap((name,id)=>name.startsWith('active_')?[id]:[]));
    this.humans=new Set();
    for(let id=1;id<model.nbody;id++){
      let ancestor=id;
      while(ancestor>0&&ancestor!==this.rootId)ancestor=model.body_parentid[ancestor];
      if(ancestor===this.rootId)this.humans.add(id);
    }
    this.specification=mujoco.mjtState.mjSTATE_INTEGRATION.value;
    this.stateSize=mujoco.mj_stateSize(model,this.specification);
    this._data=new mujoco.MjData(model);
    this._stateBuffer=new mujoco.DoubleBuffer(this.stateSize);
    this._forceBuffer=new mujoco.DoubleBuffer(6);
    this._stateValues=new Array(this.stateSize);
    this._warmstart=new Float64Array(model.nv);
    this._targetQ=new Float64Array(addresses.qposAddr.length);
    this.disposed=false;
  }

  /** Warmstart of the last evaluated endpoint (copy), for continuing an isolated lookahead from `endpoint.integration`. */
  lastWarmstart() { return Float64Array.from(this._data.qacc_warmstart); }

  dispose() {
    if(this.disposed)return;
    this._forceBuffer.delete();this._stateBuffer.delete();this._data.delete();
    this.disposed=true;
  }

  _balance(data,result) {
    const q=data.qpos,k=this.rootQpos;
    const height=q[k+2],upright=1-2*(q[k+4]**2+q[k+5]**2);
    result.minRootHeightM=Math.min(result.minRootHeightM,height);
    result.minUpright=Math.min(result.minUpright,upright);
    if(height<this.minRootHeightM||upright<this.minUpright){
      result.supported=false;result.reason??='preview_balance';
    }
  }

  _contacts(data,result,{substep,endpoint,allowContact}) {
    const contacts=data.contact;
    try {
      for(let i=0;i<data.ncon;i++){
        const contact=contacts.get(i);
        try {
          const a=this.model.geom_bodyid[contact.geom1],b=this.model.geom_bodyid[contact.geom2];
          const objectId=this.objects.has(a)?a:this.objects.has(b)?b:-1;
          const bodyId=objectId===a?b:a;
          if(objectId<0||!this.humans.has(bodyId))continue;
          let normalForceN=null;
          if(!endpoint){
            this._forceBuffer.GetView().fill(0);
            this.mujoco.mj_contactForce(this.model,data,i,this._forceBuffer);
            normalForceN=Math.max(0,this._forceBuffer.GetView()[0]);
          }
          const distanceM=contact.dist;
          if(!Number.isFinite(distanceM)||(normalForceN!==null&&!Number.isFinite(normalForceN))){
            result.supported=false;result.reason??='preview_nonfinite_contact';continue;
          }
          if(distanceM>0&&(normalForceN===null||normalForceN<=this.contactForceThresholdN))continue;
          const bodyName=this.names[bodyId];
          const item={substep,endpoint,bodyId,bodyName,objectId,objectName:this.names[objectId],
            hand:bodyName==='left_rubber_hand'||bodyName==='right_rubber_hand',
            leg:/_(hip|knee|ankle)_/.test(bodyName),normalForceN,distanceM};
          if(allowContact(item)===true){result.allowedContactCount++;continue;}
          result.unwantedContactCount++;
          if(endpoint)result.endpointUnwantedContactCount++;
          result.peakUnwantedForceN=Math.max(result.peakUnwantedForceN,normalForceN??0);
          if(result.unwantedContacts.length<32)result.unwantedContacts.push(item);
          result.supported=false;result.reason??='preview_contact';
        } finally {contact.delete();}
      }
    } finally {contacts.delete();}
  }

  /** Evaluate already-smoothed joint targets without changing liveData.
   * allowContact receives body/object pair metadata, signed distance and force.
   * Only an explicit true allows that contact; default denies every robot/box
   * contact. Endpoint force is null after position-only reconstruction.
   * captureState/onSubstep are optional diagnostics; snapshots are detached.
   */
  evaluate(liveData,targetQ,{allowContact=()=>false,captureState=false,onSubstep=null,onSubstepData=null}={}) {
    const result={supported:false,reason:null,completedSubsteps:0,requestedSubsteps:SIM_DECIMATION,
      minRootHeightM:Infinity,minUpright:Infinity,unwantedContactCount:0,endpointUnwantedContactCount:0,
      allowedContactCount:0,peakUnwantedForceN:0,unwantedContacts:[],stateSize:this.stateSize};
    if(this.disposed)return{...result,reason:'preview_disposed'};
    if(typeof allowContact!=='function'||(onSubstep!==null&&typeof onSubstep!=='function')
      ||(onSubstepData!==null&&typeof onSubstepData!=='function'))return{...result,reason:'preview_invalid_callback'};
    if(!finiteValues(targetQ,this._targetQ.length))return{...result,reason:'preview_invalid_target'};
    if(!Number.isFinite(this.model.opt.timestep)||Math.abs(this.model.opt.timestep-SIM_DT)>1e-12)return{...result,reason:'preview_timestep'};
    if(!finiteValues(liveData.qpos,this.model.nq)||!finiteValues(liveData.qvel,this.model.nv)
      ||!finiteValues(liveData.qacc_warmstart,this.model.nv))return{...result,reason:'preview_nonfinite_state'};
    const start=performance.now();
    try {
      this.mujoco.mj_getState(this.model,liveData,this._stateBuffer,this.specification);
      const view=this._stateBuffer.GetView();
      for(let i=0;i<this.stateSize;i++){
        if(!Number.isFinite(view[i]))return{...result,reason:'preview_nonfinite_state'};
        this._stateValues[i]=view[i];
      }
      this._warmstart.set(liveData.qacc_warmstart);this._targetQ.set(targetQ);
      this.mujoco.mj_setState(this.model,this._data,this._stateValues,this.specification);
      this.mujoco.mj_forward(this.model,this._data);
      this._data.qacc_warmstart.set(this._warmstart);
      result.copyMs=performance.now()-start;result.supported=true;
      this._balance(this._data,result);
      let finitePhysics=true;
      for(let substep=1;substep<=SIM_DECIMATION;substep++){
        applyPDTorques(this.model,this._data,this._targetQ,this.addresses);
        this.mujoco.mj_step(this.model,this._data);
        result.completedSubsteps=substep;
        if(!finiteValues(this._data.qpos,this.model.nq)||!finiteValues(this._data.qvel,this.model.nv)){
          result.supported=false;result.reason??='preview_nonfinite_state';finitePhysics=false;break;
        }
        this._balance(this._data,result);
        this._contacts(this._data,result,{substep,endpoint:false,allowContact});
        // Optional read-only proximity/geometry diagnostics on the PRIVATE scratch data (kinematics refreshed after the step).
        if(onSubstepData){this.mujoco.mj_kinematics(this.model,this._data);onSubstepData(this._data,substep);}
        if(onSubstep)onSubstep({substep,time:this._data.time,qpos:Array.from(this._data.qpos),
          qvel:Array.from(this._data.qvel),ctrl:Array.from(this._data.ctrl)});
      }
      if(finitePhysics&&result.completedSubsteps===SIM_DECIMATION){
        this.mujoco.mj_fwdPosition(this.model,this._data);
        this._contacts(this._data,result,{substep:SIM_DECIMATION,endpoint:true,allowContact});
      }
      if(captureState){
        this.mujoco.mj_getState(this.model,this._data,this._stateBuffer,this.specification);
        result.endpoint={time:this._data.time,qpos:Array.from(this._data.qpos),qvel:Array.from(this._data.qvel),
          ctrl:Array.from(this._data.ctrl),integration:Array.from(this._stateBuffer.GetView())};
      }
    } catch(error) {
      result.supported=false;result.reason??='preview_error';result.error=String(error?.message??error);
    }
    result.elapsedMs=performance.now()-start;
    return result;
  }
}
