/** Shared goal/observation contract for staged box control.
 * This supplies task goals, not body references, joint actions or proof that an
 * existing student can execute a stage.
 */
import {calcHeadingQuatInv,quatRotateOne,quatMulXyzw,quatToRot6d} from './math.js';

export const BOX_GOAL_STAGES=Object.freeze(['approach','grasp','lift','transport','place','release','retreat']);
const MODES=Object.freeze({LOCO:{human:1,object:0,points:0},HOI_FULL:{human:1,object:1,points:1},HOI_OBJ_ONLY:{human:0,object:1,points:1}});
const vector=(value,length,label)=>{
  if(!value||value.length!==length||!Array.from(value).every(Number.isFinite))throw new Error(`${label} requires ${length} finite values`);
  return Array.from(value);
};
const quaternion=(value,label)=>{
  const q=vector(value,4,label);if(Math.abs(Math.hypot(...q)-1)>1e-5)throw new Error(`${label} must be unit xyzw`);return q;
};
const count=(value,label,min=0,max=Number.MAX_SAFE_INTEGER)=>{
  if(!Number.isSafeInteger(value)||value<min||value>max)throw new Error(`${label} requires an integer in [${min},${max}]`);return value;
};
const stageName=stage=>{if(!BOX_GOAL_STAGES.includes(stage))throw new Error('Explicit supported stage name required');return stage;};
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const distance=(a,b)=>Math.hypot(...sub(a,b));
function validatePlan(plan){
  stageName(plan.stage);if(!MODES[plan.mode])throw new Error('Use a current LOCO, HOI_FULL or HOI_OBJ_ONLY mask mode');
  return Object.freeze({stage:plan.stage,mode:plan.mode,
    humanGoalWorld:Object.freeze(vector(plan.humanGoalWorld,3,'Absolute root goal')),
    humanGoalRotationWorld:Object.freeze(quaternion(plan.humanGoalRotationWorld,'Absolute root goal rotation')),
    objectGoalWorld:Object.freeze(vector(plan.objectGoalWorld,3,'Absolute object COM goal')),
    finalDestinationWorld:Object.freeze(vector(plan.finalDestinationWorld,3,'Original final COM destination'))});
}

/** Encode metres in the live yaw-only heading frame, exactly as the current
 * interactive training method. Both root and object goals are absolute world
 * points, but their deltas have different origins. Rotation is full target root
 * orientation relative to current heading; it is not a yaw-only target.
 * A floor click's z=0 must be converted to a physical COM target by its planner
 * before this call. This function never guesses box height or clips distance.
 */
export function encodeStageGoal(plan,live){
  const stage=stageName(plan.stage),mode=MODES[plan.mode];if(!mode)throw new Error('Use a current LOCO, HOI_FULL or HOI_OBJ_ONLY mask mode');
  const human=vector(plan.humanGoalWorld,3,'Absolute root goal'),rotation=quaternion(plan.humanGoalRotationWorld,'Absolute root goal rotation');
  const object=vector(plan.objectGoalWorld,3,'Absolute object COM goal'),final=vector(plan.finalDestinationWorld,3,'Original final COM destination');
  const root=vector(live.rootPositionWorld,3,'Measured root position'),rootQ=quaternion(live.rootQuaternionWorld,'Measured root rotation');
  const currentObject=vector(live.objectPositionWorld,3,'Measured object position');
  const remaining=count(plan.remainingControls,'Remaining actual controls',1,239),inv=calcHeadingQuatInv(rootQ);
  const goalSpec={humanTargetPos:Float32Array.from(mode.human?quatRotateOne(inv,sub(human,root)):[0,0,0]),
    humanTargetRot:Float32Array.from(mode.human?quatToRot6d(quatMulXyzw(inv,rotation)):Array(6).fill(0)),
    objTargetPos:Float32Array.from(mode.object?quatRotateOne(inv,sub(object,currentObject)):[0,0,0]),timeToTarget:Math.fround(remaining/240)};
  const command=new Float32Array(13);command.set(goalSpec.humanTargetPos,0);command.set(goalSpec.humanTargetRot,3);
  command.set(goalSpec.objTargetPos,9);command[12]=goalSpec.timeToTarget;
  if(!command.every(Number.isFinite))throw new Error('Command cannot be represented as finite FP32');
  return{stage,mode:plan.mode,goalSpec,command,mask:{keepHuman:mode.human,keepObj:mode.object,keepObjPoints:mode.points},
    remainingControls:remaining,physicalGoals:{humanGoalWorld:human,humanGoalRotationWorld:rotation,objectGoalWorld:object,finalDestinationWorld:final},
    measured:{humanGoalErrorM:distance(human,root),objectStageGoalErrorM:distance(object,currentObject),
      finalObjectGoalErrorM:distance(final,currentObject)},
    semantics:{positionUnit:'metres',coordinateFrame:'current root heading only',humanPositionOrigin:'current root',
      objectPositionOrigin:'current object',timeNormalizerControls:240,goalWasClipped:false,
      stageVisibleToLegacyStudent:false,physicalExecutionValidated:false}};
}

/** A fixed world subgoal with a finite physical-clock budget. Repeated reads
 * have no side effects. Expiry requests explicit replanning; it never claims
 * arrival, slides the goal with the robot, repeats a pose, or resets a policy.
 * This adapter does not execute the required physical fallback on expiry.
 */
export class BoundedStageGoalWindow{
  constructor(plan,{episode,physicalControl,horizonControls}){
    this.episode=count(episode,'Episode');this.startControl=count(physicalControl,'Physical control');
    this.horizonControls=count(horizonControls,'Goal horizon controls',1,239);
    if('remainingControls'in plan)throw new Error('Window derives remaining controls from physical time');
    this.plan=validatePlan(plan);Object.freeze(this);
  }
  sample(live,{episode,physicalControl}){
    if(count(episode,'Episode')!==this.episode)throw new Error('Goal window belongs to a different physical episode');
    const elapsed=count(physicalControl,'Physical control')-this.startControl;
    if(elapsed<0)throw new Error('Physical time cannot precede stage entry');
    const remaining=this.horizonControls-elapsed;
    if(remaining<=0)return{stage:this.plan.stage,expired:true,reason:'stage_goal_window_elapsed',
      remainingControls:0,command:null,arrived:false,physicalGoals:structuredClone(this.plan)};
    return{...encodeStageGoal({...this.plan,remainingControls:remaining},live),expired:false,arrived:false};
  }
}

/** Preserve the existing 1422 layout; extra privileged fields cannot silently
 * enter a checkpoint trained with this shape. Points are already in the root
 * heading frame, as computed by the unchanged perception path.
 */
export function packLegacyStageStudentInput(encoded,bodyObservation,objectPointsHeading){
  if(encoded.expired||encoded.command?.length!==13)throw new Error('An active encoded goal is required');
  const body=vector(bodyObservation,1012,'Already built body/history observation');
  const out=new Float32Array(1422);out.set(encoded.command);out.set(body,13);
  if(encoded.mask.keepObjPoints){out.set(vector(objectPointsHeading,192,'Measured root-relative object points'),1025);out.fill(1,1217,1409);}
  out.fill(encoded.mask.keepHuman,1409,1418);out.fill(encoded.mask.keepObj,1418,1421);out[1421]=1;
  return out;
}

/** Separate measured features for a future stage-conditioned policy/data
 * collector. They are never concatenated into the old student's 1422 input.
 * Raw SI values are retained; any new learned normalization needs explicit
 * training statistics. There are no fabricated velocities/contact defaults.
 */
export function privilegedStageFeedback(stage,live){
  stageName(stage);const root=vector(live.rootPositionWorld,3,'Measured root position');
  const inv=calcHeadingQuatInv(quaternion(live.rootQuaternionWorld,'Measured root rotation'));
  const rootVelocity=vector(live.rootLinearVelocityWorld,3,'Measured root velocity');
  const objectVelocity=vector(live.objectLinearVelocityWorld,3,'Measured object velocity');
  const angular=vector(live.objectAngularVelocityWorld,3,'Measured object angular velocity');
  const hands=vector(live.handNormalForceN,2,'Measured hand normal forces');
  if(hands.some(v=>v<0)||live.handContact?.length!==2||!live.handContact.every(v=>typeof v==='boolean'))throw new Error('Measured nonnegative hand loads and boolean contacts required');
  if(typeof live.objectFloorContact!=='boolean'||!Number.isFinite(live.floorHeightWorld)||!Number.isFinite(live.objectBottomWorldZ))throw new Error('Measured floor contact and compiled-geometry heights required');
  return{stage,stageOneHot:BOX_GOAL_STAGES.map(s=>Number(s===stage)),rootHeightAboveFloorM:root[2]-live.floorHeightWorld,
    rootLinearVelocityHeadingMps:quatRotateOne(inv,rootVelocity),
    objectRelativeToRootVelocityHeadingMps:quatRotateOne(inv,sub(objectVelocity,rootVelocity)),
    objectAngularVelocityHeadingRadps:quatRotateOne(inv,angular),handContact:[...live.handContact],handNormalForceN:hands,
    objectFloorContact:live.objectFloorContact,objectBottomClearanceM:live.objectBottomWorldZ-live.floorHeightWorld,
    suppliedToLegacyStudent:false};
}
