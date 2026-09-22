/** Constant-memory accounting of actual carry physics, independent of observers.
 * Call once after each real substep. This never steps or forwards the simulator.
 */
import {STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES} from './staged_student_transport_controller.js';
import {SIM_DECIMATION} from './pd_control.js';
import {DEFAULT_OBJECT_BODY} from './object_profiles.js';

const loadedPhases=new Set(['teacher','student_transport','student_second_loaded',
  'student_grounding106','teacher_release21','teacher_release44','teacher_descent108']);
const integer=n=>Number.isSafeInteger(n)&&n>=0;
const namesOf=model=>Array.from({length:model.nbody},(_,id)=>{
  let name='',offset=model.name_bodyadr[id];while(model.names[offset])name+=String.fromCharCode(model.names[offset++]);
  return name;
});

export class StreamingCarryPhysicsMonitor{
  constructor(mujoco,model,{objectBodyName=DEFAULT_OBJECT_BODY}={}){
    this.mujoco=mujoco;this.model=model;this.names=namesOf(model);
    this.timestep=model.opt.timestep;
    if(!Number.isFinite(this.timestep)||this.timestep<=0)throw new Error('Actual physics timestep required');
    this.rootId=this.names.indexOf('pelvis');this.objectId=this.names.indexOf(objectBodyName);
    if(this.rootId<1||this.objectId<1)throw new Error('Known humanoid and selected carry object required');
    this.objects=new Set(this.names.flatMap((n,id)=>n.startsWith('active_')?[id]:[]));
    if(!this.objects.has(this.objectId))throw new Error('Selected carry body must be a scene object');
    this.humans=new Set();
    for(let id=1;id<model.nbody;id++){
      let parent=id;while(parent>0&&parent!==this.rootId)parent=model.body_parentid[parent];
      if(parent===this.rootId)this.humans.add(id);
    }
    this.support=new Set(STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES);
    this.force=new mujoco.DoubleBuffer(6);this.disposed=false;this.episode=null;
  }
  reset({episode,simulationTime}){
    if(!integer(episode)||!Number.isFinite(simulationTime)||this.disposed)throw new Error('Valid Reset clock required');
    this.episode=episode;this.lastTime=simulationTime;this.evaluatedSubsteps=0;this.missingEvaluation=false;
    this.counts={balanceViolationSubsteps:0,legContactSubsteps:0,boxBoxContactSubsteps:0,
      unintendedHumanBoxContactSubsteps:0,invalidMeasurementSubsteps:0};
    this.firstViolation=null;this.minRootHeightM=Infinity;this.minUpright=Infinity;
  }
  observe(data,{episode,physicalControl,substep,phase,allowLoadedSupport=false}){
    if(this.disposed||this.episode===null)throw new Error('Active carry physics monitor required');
    // A stale callback cannot damage a newer episode's accounting.
    if(episode!==this.episode)throw new Error('Carry physics sample belongs to another episode');
    const expectedControl=Math.floor(this.evaluatedSubsteps/SIM_DECIMATION)+1,expectedSubstep=this.evaluatedSubsteps%SIM_DECIMATION+1;
    if(physicalControl!==expectedControl||substep!==expectedSubstep||!Number.isFinite(data.time)
      ||data.time<=this.lastTime||Math.abs(data.time-this.lastTime-this.timestep)>1e-9||typeof phase!=='string'){
      this.missingEvaluation=true;throw new Error('Each actual carry substep must be measured once in physical order');
    }
    const rootZ=data.xpos[this.rootId*3+2],q=this.rootId*4;
    const rootPosition=Array.from(data.xpos.slice(this.rootId*3,this.rootId*3+3));
    const quaternion=Array.from(data.xquat.slice(q,q+4));
    const upright=1-2*(data.xquat[q+1]**2+data.xquat[q+2]**2);
    let invalid=!rootPosition.every(Number.isFinite)||!quaternion.every(Number.isFinite)
      ||Math.abs(Math.hypot(...quaternion)-1)>1e-5||!Number.isFinite(upright);
    let balance=invalid||rootZ<.45||upright<.5;
    let leg=false,boxPair=false,unintended=false,firstContact=null;
    if(!invalid){this.minRootHeightM=Math.min(this.minRootHeightM,rootZ);this.minUpright=Math.min(this.minUpright,upright);}
    const supportAllowed=allowLoadedSupport===true&&loadedPhases.has(phase);
    const contacts=data.contact;
    try{
      for(let i=0;i<data.ncon;i++){
        const c=contacts.get(i);
        try{
          const a=this.model.geom_bodyid[c.geom1],b=this.model.geom_bodyid[c.geom2];
          const object=this.objects.has(a)?a:this.objects.has(b)?b:-1,other=object===a?b:a;
          if(object<0||(!this.humans.has(other)&&!this.objects.has(other)))continue;
          this.force.GetView().fill(0);this.mujoco.mj_contactForce(this.model,data,i,this.force);
          const force=this.force.GetView()[0],normalForceN=Math.max(0,force),distanceM=c.dist;
          const invalidContact=!Number.isFinite(force)||!Number.isFinite(normalForceN)||!Number.isFinite(distanceM);
          invalid||=invalidContact;
          if(!invalidContact&&normalForceN<=.1&&distanceM>0)continue;
          const isPair=this.objects.has(other),isLeg=/_(hip|knee|ankle)_/.test(this.names[other]);
          const allowed=!invalidContact&&!isPair&&supportAllowed&&object===this.objectId&&this.support.has(this.names[other]);
          leg||=isLeg;boxPair||=isPair;unintended||=!isPair&&!allowed;
          if(!allowed&&!firstContact)firstContact={object:this.names[object],body:this.names[other],normalForceN,distanceM};
        }finally{c.delete();}
      }
    }catch(error){this.missingEvaluation=true;throw error;}finally{contacts.delete();}
    this.evaluatedSubsteps++;this.lastTime=data.time;
    this.counts.balanceViolationSubsteps+=Number(balance);this.counts.legContactSubsteps+=Number(leg);
    this.counts.boxBoxContactSubsteps+=Number(boxPair);this.counts.unintendedHumanBoxContactSubsteps+=Number(unintended);
    this.counts.invalidMeasurementSubsteps+=Number(invalid);
    if((balance||leg||boxPair||unintended||invalid)&&!this.firstViolation)
      this.firstViolation={episode,physicalControl,substep,phase,rootZ,upright,contact:firstContact,invalidMeasurement:invalid};
    return !(balance||leg||boxPair||unintended||invalid);
  }
  snapshot(){
    if(this.episode===null)throw new Error('Reset carry physics accounting before reading it');
    return{episode:this.episode,evaluatedSubsteps:this.evaluatedSubsteps,
      completedPhysicalControls:Math.floor(this.evaluatedSubsteps/SIM_DECIMATION),partialControlSubsteps:this.evaluatedSubsteps%SIM_DECIMATION,
      missingEvaluation:this.missingEvaluation,anyActualViolation:Object.values(this.counts).some(n=>n>0),
      ...this.counts,minRootHeightM:this.minRootHeightM,minUpright:this.minUpright,
      firstViolation:this.firstViolation?structuredClone(this.firstViolation):null};
  }
  dispose(){if(this.disposed)return;this.force.delete();this.disposed=true;}
}
