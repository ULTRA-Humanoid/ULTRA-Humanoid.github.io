/** Current simulator contact and release measurements for matched carry roles.
 * Reads solver results without stepping or forwarding. Browser-compatible;
 * numerical conventions match the retained successful teacher/student trial.
 */
import {ObjectMeshBounds} from './box_approach.js';
import {ControlPreview} from './control_preview.js';
const requireCondition=(condition,message='Invalid placement measurement')=>{
  if(!condition)throw new Error(message);
};
const requireEqual=(actual,expected,message='Unexpected placement measurement value')=>{
  requireCondition(Object.is(actual,expected),message);
};
export const PLACEMENT_RELEASE_LIMITS=Object.freeze({groundForceN:.1,maxDistalForceN:.1,
  maxBoxSpeedMps:.2,maxBottomClearanceM:.02,maxHandoffPoseErrorM:.15});
const finite=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
const dist=(a,b)=>Math.hypot(...Array.from(a,(v,i)=>v-b[i]));
const upright=q=>1-2*(q[0]**2+q[1]**2);
const pose=l=>finite(l?.rootPositionWorld,3)&&finite(l?.rootQuaternionWorld,4)
 &&Math.abs(Math.hypot(...l.rootQuaternionWorld)-1)<1e-5&&finite(l?.objectPositionWorld,3);

export class StageFeedbackMeasurement {
  constructor(mujoco,model,{rootBodyId,objectBodyId,leftHandBodyId,rightHandBodyId}){
    this.mujoco=mujoco;this.model=model;this.root=rootBodyId;this.object=objectBodyId;
    this.hands=[leftHandBodyId,rightHandBodyId];
    for(const id of [this.root,this.object,...this.hands])requireCondition(Number.isInteger(id)&&id>0&&id<model.nbody,'Known body ID required');
    const planes=Array.from({length:model.ngeom},(_,id)=>id).filter(id=>model.geom_bodyid[id]===0&&model.geom_type[id]===0
      &&(model.geom_contype[id]!==0||model.geom_conaffinity[id]!==0));
    requireEqual(planes.length,1,'This bounded collector requires one compiled world floor plane');
    this.floor=planes[0];this.bounds=new ObjectMeshBounds(model,[this.object]);
    this.jacp=new mujoco.DoubleBuffer(3*model.nv);this.jacr=new mujoco.DoubleBuffer(3*model.nv);
    this.force=new mujoco.DoubleBuffer(6);
  }

  bodyVelocity(data,id){
    this.jacp.GetView().fill(0);this.jacr.GetView().fill(0);
    this.mujoco.mj_jacBody(this.model,data,this.jacp,this.jacr,id);
    const linear=[0,0,0],angular=[0,0,0],p=this.jacp.GetView(),r=this.jacr.GetView();
    for(let axis=0;axis<3;axis++)for(let dof=0;dof<this.model.nv;dof++){
      linear[axis]+=p[axis*this.model.nv+dof]*data.qvel[dof];
      angular[axis]+=r[axis*this.model.nv+dof]*data.qvel[dof];
    }
    return{linear,angular};
  }

  read(data){
    requireCondition(this.force,'Measurement buffers have been disposed');
    const m=this.floor*9;
    requireCondition(Math.abs(data.geom_xmat[m+2])<1e-9&&Math.abs(data.geom_xmat[m+5])<1e-9
      &&Math.abs(data.geom_xmat[m+8]-1)<1e-9,'This collector requires a horizontal upward floor plane');
    const position=id=>Array.from(data.xpos.slice(id*3,id*3+3));
    const q=data.xquat.slice(this.root*4,this.root*4+4);
    const root=this.bodyVelocity(data,this.root),object=this.bodyVelocity(data,this.object);
    const loads=[0,0],handContact=[false,false];let objectFloorContact=false;
    const contacts=data.contact;
    try{for(let i=0;i<data.ncon;i++){
      const c=contacts.get(i);try{
        const a=this.model.geom_bodyid[c.geom1],b=this.model.geom_bodyid[c.geom2];
        if(a!==this.object&&b!==this.object)continue;
        const other=a===this.object?b:a,hand=this.hands.indexOf(other);
        const floor=c.geom1===this.floor||c.geom2===this.floor;
        if(hand<0&&!floor)continue;
        this.force.GetView().fill(0);this.mujoco.mj_contactForce(this.model,data,i,this.force);
        const force=Math.max(0,this.force.GetView()[0]);requireCondition(Number.isFinite(force));
        if(hand>=0){loads[hand]+=force;if(force>0)handContact[hand]=true;}
        if(floor&&force>0)objectFloorContact=true;
      }finally{c.delete();}
    }}finally{contacts.delete();}
    const live={rootPositionWorld:position(this.root),rootQuaternionWorld:[q[1],q[2],q[3],q[0]],
      objectPositionWorld:position(this.object),rootLinearVelocityWorld:root.linear,
      objectLinearVelocityWorld:object.linear,objectAngularVelocityWorld:object.angular,
      handContact,handNormalForceN:loads,objectFloorContact,floorHeightWorld:data.geom_xpos[this.floor*3+2],
      objectBottomWorldZ:this.bounds.read(data)[0].minZ};
    return{live,measurement:{simTime:data.time,sample:'current pre-query state; solver forces retained from preceding physical step',
      velocity:'body-origin Jacobian times current qvel',contact:'positive current solver normal force, summed per hand',
      bottom:'minimum world Z of all compiled selected-object collision mesh vertices',
      floor:'compiled unique horizontal world plane',rootBodyId:this.root,objectBodyId:this.object,
      handBodyIds:[...this.hands],floorGeomId:this.floor,simulatorForwardOrStepCalled:false}};
  }

  dispose(){this.jacp?.delete();this.jacr?.delete();this.force?.delete();this.jacp=this.jacr=this.force=null;}
}

export function classifyGroundingBoxPair({aObject,bObject,distanceM,normalForceN,endpoint=false}){
  if(!aObject||!bObject)return{unwanted:false,invalid:false};
  const invalid=!Number.isFinite(distanceM)||(!endpoint&&(!Number.isFinite(normalForceN)||normalForceN<0));
  return{unwanted:invalid||distanceM<=0||(!endpoint&&normalForceN>.1),invalid};
}
export class GroundingContactPreview extends ControlPreview{
  _contacts(data,result,options){
    super._contacts(data,result,options);
    const contacts=data.contact;result.boxBoxUnwantedCount??=0;
    try{for(let i=0;i<data.ncon;i++){const contact=contacts.get(i);try{
      const a=this.model.geom_bodyid[contact.geom1],b=this.model.geom_bodyid[contact.geom2];
      if(a===b||!this.objects.has(a)||!this.objects.has(b))continue;
      let normalForceN=null;
      if(!options.endpoint){this._forceBuffer.GetView().fill(0);this.mujoco.mj_contactForce(this.model,data,i,this._forceBuffer);
        normalForceN=Math.max(0,this._forceBuffer.GetView()[0]);}
      const classified=classifyGroundingBoxPair({aObject:true,bObject:true,distanceM:contact.dist,normalForceN,endpoint:options.endpoint});
      if(!classified.unwanted)continue;
      result.supported=false;result.reason??=classified.invalid?'preview_nonfinite_box_box_contact':'preview_box_box_contact';
      result.boxBoxUnwantedCount++;result.unwantedContactCount++;if(options.endpoint)result.endpointUnwantedContactCount++;
      result.peakUnwantedForceN=Math.max(result.peakUnwantedForceN,normalForceN??0);
      if(result.unwantedContacts.length<32)result.unwantedContacts.push({substep:options.substep,endpoint:options.endpoint,
        objectId:a,objectName:this.names[a],bodyId:b,bodyName:this.names[b],boxBox:true,leg:false,hand:false,
        normalForceN,distanceM:contact.dist});
    }finally{contact.delete();}}}finally{contacts.delete();}
  }
}

/** Actual solver-force release measurement: both hands plus all four allowed
 * distal wrist links, and the compiled floor plane. No force values are inferred.
 */
export function measureGroundingRelease({mujoco,model,data,measurement,diagnostic,objectId,distalBodyIds,forceBuffer}){
  if(distalBodyIds.length!==6||new Set(distalBodyIds).size!==6)throw new Error('Exact six existing distal support bodies required');
  const measured=measurement.read(data),distalSupportNormalForceN=new Array(6).fill(0);let objectFloorNormalForceN=0;
  const contacts=data.contact;
  try{for(let i=0;i<data.ncon;i++){const contact=contacts.get(i);try{
    const a=model.geom_bodyid[contact.geom1],b=model.geom_bodyid[contact.geom2];if(a!==objectId&&b!==objectId)continue;
    const other=a===objectId?b:a,index=distalBodyIds.indexOf(other);
    const floor=contact.geom1===measurement.floor||contact.geom2===measurement.floor;if(index<0&&!floor)continue;
    forceBuffer.GetView().fill(0);mujoco.mj_contactForce(model,data,i,forceBuffer);const force=forceBuffer.GetView()[0];
    if(!Number.isFinite(force))throw new Error('Actual finite contact force required');const normal=Math.max(0,force);
    if(index>=0)distalSupportNormalForceN[index]+=normal;if(floor)objectFloorNormalForceN+=normal;
  }finally{contact.delete();}}}finally{contacts.delete();}
  const loads=diagnostic.read(data,objectId);
  return{...measured.live,objectLinearVelocityWorldMps:measured.live.objectLinearVelocityWorld,
    distalSupportNormalForceN,objectFloorNormalForceN,
    loadedFootGroundContacts:loads.loadedFootGroundContacts,loadedFootNormalForceN:loads.loadedFootNormalForceN};
}

export function measurePlacementRelease(live,target){
 const p=PLACEMENT_RELEASE_LIMITS;
 if(!finite(target,747)||!pose(live)||!finite(live.objectLinearVelocityWorldMps,3)||!finite(live.distalSupportNormalForceN,6)
   ||live.distalSupportNormalForceN.some(v=>v<0)||!Number.isFinite(live.objectFloorNormalForceN)||live.objectFloorNormalForceN<0
   ||!Number.isFinite(live.objectBottomWorldZ)||!Number.isFinite(live.floorHeightWorld))return{valid:false,released:false,handoff:false};
 const balanced=live.rootPositionWorld[2]>=.45&&upright(live.rootQuaternionWorld)>=.5;
 const floorSupported=live.objectFloorNormalForceN>p.groundForceN
   &&live.objectBottomWorldZ-live.floorHeightWorld<=p.maxBottomClearanceM;
 const distalForceN=live.distalSupportNormalForceN.reduce((a,b)=>a+b,0);
 const boxSpeedMps=Math.hypot(...live.objectLinearVelocityWorldMps);
 const released=balanced&&floorSupported&&distalForceN<=p.maxDistalForceN&&boxSpeedMps<=p.maxBoxSpeedMps;
 const rootErrorM=dist(live.rootPositionWorld,target.slice(0,3)),boxErrorM=dist(live.objectPositionWorld,target.slice(71,74));
 return{valid:true,balanced,floorSupported,distalForceN,boxSpeedMps,released,rootErrorM,boxErrorM,
  handoff:released&&rootErrorM<=p.maxHandoffPoseErrorM&&boxErrorM<=p.maxHandoffPoseErrorM};
}

export function summarizePlacementSafety(denseSamples){
  const contact=e=>!Number.isFinite(e.normal_force_n)||!Number.isFinite(e.distance_m)||e.normal_force_n>.1||e.distance_m<=0;
  const unwanted=(s,e)=>contact(e)&&(e.object_pair||!['teacher_release21','teacher_release44','teacher_descent108'].includes(s.phase)||!e.allowedLoadedSupport);
  const balanceViolationSubsteps=denseSamples.filter(s=>!Number.isFinite(s.rootZ)||!Number.isFinite(s.upright)||s.rootZ<.45||s.upright<.5).length;
  const legContactSubsteps=denseSamples.filter(s=>s.contacts.some(e=>e.leg&&contact(e))).length;
  const boxBoxContactSubsteps=denseSamples.filter(s=>s.contacts.some(e=>e.object_pair&&contact(e))).length;
  const unintendedHumanBoxContactSubsteps=denseSamples.filter(s=>s.contacts.some(e=>!e.object_pair&&unwanted(s,e))).length;
  return{balanceViolationSubsteps,legContactSubsteps,boxBoxContactSubsteps,unintendedHumanBoxContactSubsteps,
    fullEndingHasNoActualViolation:balanceViolationSubsteps+legContactSubsteps+boxBoxContactSubsteps+unintendedHumanBoxContactSubsteps===0,
    actualContactViolationRule:'force>.1N OR distance<=0; six distal support allowed only during declared teacher release role'};
}
