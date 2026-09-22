/** Task-relative request geometry for the experimental existing-skill programme.
 * This does not admit a physical handoff or change any actor/reference timing.
 */
import {OBJECT_PROFILES} from './object_profiles.js';
const invariant=(condition,message)=>{if(!condition)throw new Error(message);};
const vector=(a,n)=>a?.length===n&&Array.from(a).every(Number.isFinite);
const freeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(freeze);Object.freeze(x);}return x;};
const requests=new WeakSet();
const semanticBearings=new WeakSet();
const same=(a,b)=>a?.length===b?.length&&Array.from(a).every((v,i)=>v===b[i]);
export const MATCHED_CARRY_CONDITIONING=freeze({
  conditioningOffsetTaskXY:[2.1298169021592592,0.018612559987691322],
  conditioningHeightWorld:0.13593138754367828,initialBoxHeightWorld:0.1384663032379996,
  calibratedTaskDistanceM:2.9999998931848904,
  objectBodyName:OBJECT_PROFILES.largebox.bodyName,sourceTimingOrScaleChanged:false,
});
export const MATCHED_CARRY_SEMANTIC_CONDITIONING=freeze({
  conditioningOffsetTaskXY:[2.1298169021570583,0.018612560239556902],
  encodingAllowanceFloat32UlpsPerCoordinate:2,
  directionConvention:'Explicit world bearing retained independently of destination distance; calibration re-expressed in canonical east.',
});
/** Bind explicit directional intent at the command clock. This is not inferred
 * from a rounded click ray and does not admit any physical role transition.
 */
export function makeMatchedCarrySemanticBearing({episode,requestId,commandIssuedAtPhysicalControl,
  objectBodyName,originalGoalWorld,objectPositionWorldAtCommand,directionRadians}){
  invariant(Number.isSafeInteger(episode)&&episode>=0&&Number.isSafeInteger(requestId)&&requestId>0
    &&Number.isSafeInteger(commandIssuedAtPhysicalControl)&&commandIssuedAtPhysicalControl>=0,
  'Semantic direction requires its actual episode, request and command clock');
  invariant(objectBodyName===MATCHED_CARRY_CONDITIONING.objectBodyName,'Semantic direction requires the selected calibrated object');
  invariant(vector(originalGoalWorld,3)&&vector(objectPositionWorldAtCommand,3)&&Number.isFinite(directionRadians),
    'Explicit bearing and measured command-time origin must be finite');
  const turn=2*Math.PI;let yaw=directionRadians%turn;if(yaw<0)yaw+=turn;if(yaw===0)yaw=0;
  const bearing=freeze({episode,requestId,commandIssuedAtPhysicalControl,objectBodyName,
    originalGoalWorld:Array.from(originalGoalWorld),objectPositionWorldAtCommand:Array.from(objectPositionWorldAtCommand),
    directionRadians:yaw,directionWorldXY:[Math.cos(yaw),Math.sin(yaw)],
    source:'explicit_direction_intent',physicalExecutionValidated:false});
  semanticBearings.add(bearing);return bearing;
}
export const isMatchedCarrySemanticBearing=value=>semanticBearings.has(value);
const fp32Ulp=value=>{
  const a=Math.abs(Math.fround(value));
  return a<2**-126?2**-149:2**(Math.floor(Math.log2(a))-23);
};
function rayConsistency(origin,goal,direction){
  if(![...origin,...goal].every(v=>Number.isFinite(Math.fround(v))))return{supported:false,reason:'semantic_coordinates_outside_float32'};
  const [c,s]=direction,dx=goal[0]-origin[0],dy=goal[1]-origin[1];
  const along=c*dx+s*dy,lateral=-s*dx+c*dy;
  const ulps=MATCHED_CARRY_SEMANTIC_CONDITIONING.encodingAllowanceFloat32UlpsPerCoordinate;
  const allowance=ulps*(Math.abs(s)*(fp32Ulp(goal[0])+fp32Ulp(origin[0]))
    +Math.abs(c)*(fp32Ulp(goal[1])+fp32Ulp(origin[1])))
    +32*Number.EPSILON*Math.max(1,...origin.map(Math.abs),...goal.map(Math.abs),Math.hypot(dx,dy));
  return{supported:along>1e-8&&Math.abs(lateral)<=allowance,
    reason:along<=1e-8?'semantic_destination_not_forward':Math.abs(lateral)>allowance?'semantic_ray_inconsistent':null,
    alongM:along,lateralM:lateral,encodingAllowanceM:allowance};
}
class UnsupportedSemanticNormalization extends Error{
  constructor(proposal){super(`Ordinary goal normalization required: ${proposal.reason}`);this.proposal=proposal;}
}
function reviewSemanticBearing(args,bearing){
  invariant(semanticBearings.has(bearing),'Semantic direction must retain its factory-created identity');
  const commandClock=args.commandIssuedAtPhysicalControl??args.physicalControl;
  invariant(bearing.episode===args.episode&&bearing.requestId===args.requestId&&bearing.objectBodyName===args.objectBodyName
    &&bearing.commandIssuedAtPhysicalControl===commandClock&&args.physicalControl>=commandClock
    &&same(bearing.originalGoalWorld,args.originalGoalWorld),'Semantic direction episode, request, object, command clock or original goal changed');
  const atCommand=rayConsistency(bearing.objectPositionWorldAtCommand,args.originalGoalWorld,bearing.directionWorldXY);
  const atExecution=rayConsistency(args.initialObjectPositionWorld,args.originalGoalWorld,bearing.directionWorldXY);
  if(!atCommand.supported||!atExecution.supported){
    const proposal=freeze({supported:false,request:null,reason:!atCommand.supported?'semantic_bearing_inconsistent_at_command':'semantic_bearing_inconsistent_at_execution',
      episode:args.episode,requestId:args.requestId,commandIssuedAtPhysicalControl:commandClock,
      executionStartedAtPhysicalControl:args.physicalControl,objectBodyName:args.objectBodyName,
      originalGoalWorld:Array.from(args.originalGoalWorld),actualExecutionStartObjectPositionWorld:Array.from(args.initialObjectPositionWorld),
      semanticBearing:bearing,atCommand,atExecution,ordinaryFallback:{objectBodyName:args.objectBodyName,
        originalGoalWorld:Array.from(args.originalGoalWorld)},physicalExecutionValidated:false});
    throw new UnsupportedSemanticNormalization(proposal);
  }
  return{atCommand,atExecution};
}
export function makeTaskRelativeCarryRequest({episode,requestId,physicalControl,originalGoalWorld,
  initialObjectPositionWorld,objectBodyName,commandIssuedAtPhysicalControl,semanticBearing}){
  invariant(Number.isSafeInteger(episode)&&episode>=0&&Number.isSafeInteger(requestId)&&requestId>0
    &&Number.isSafeInteger(physicalControl)&&physicalControl>=0,'Actual episode/request/physical clock required');
  invariant(commandIssuedAtPhysicalControl===undefined||(Number.isSafeInteger(commandIssuedAtPhysicalControl)
    &&commandIssuedAtPhysicalControl>=0&&commandIssuedAtPhysicalControl<=physicalControl),'Queued command clock cannot follow execution-start clock');
  invariant(objectBodyName===MATCHED_CARRY_CONDITIONING.objectBodyName,'This programme requires the calibrated selected box');
  invariant(vector(originalGoalWorld,3)&&vector(initialObjectPositionWorld,3),'Measured initial object and original COM goal require finite XYZ');
  const dx=originalGoalWorld[0]-initialObjectPositionWorld[0],dy=originalGoalWorld[1]-initialObjectPositionWorld[1];
  invariant(Math.hypot(dx,dy)>1e-8,'A nonzero box-to-destination direction is required');
  const semanticReview=semanticBearing===undefined?null:reviewSemanticBearing({episode,requestId,physicalControl,
    originalGoalWorld,initialObjectPositionWorld,objectBodyName,commandIssuedAtPhysicalControl},semanticBearing);
  const yaw=semanticReview?semanticBearing.directionRadians:Math.atan2(dy,dx);
  const [c,s]=semanticReview?semanticBearing.directionWorldXY:[Math.cos(yaw),Math.sin(yaw)];
  const [forward,lateral]=(semanticReview?MATCHED_CARRY_SEMANTIC_CONDITIONING:MATCHED_CARRY_CONDITIONING).conditioningOffsetTaskXY;
  const offset=[c*forward-s*lateral,s*forward+c*lateral];
  const intermediate=[initialObjectPositionWorld[0]+offset[0],initialObjectPositionWorld[1]+offset[1],
    MATCHED_CARRY_CONDITIONING.conditioningHeightWorld];
  invariant(vector(intermediate,3),'Conditioning target must remain finite');
  const request=freeze({episode,requestId,issuedAtPhysicalControl:physicalControl,
    ...(commandIssuedAtPhysicalControl===undefined?{}:{commandIssuedAtPhysicalControl}),
    originalGoalWorld:Array.from(originalGoalWorld),firstIntermediateGoalWorld:intermediate,
    initialObjectPositionWorld:Array.from(initialObjectPositionWorld),objectBodyName,
    ...(semanticReview?{semanticBearing}:{}),
    taskRelative:{taskYawRadians:yaw,initialBoxHeightDifferenceM:initialObjectPositionWorld[2]-MATCHED_CARRY_CONDITIONING.initialBoxHeightWorld,
      conditioningOffsetTaskXY:[forward,lateral],prefixConditioningDependsOnDistance:false,
      sourceScale:1,sourceRetime:false,physicalExecutionValidated:false,
      ...(semanticReview?{conditioningDirectionSource:'explicit_semantic_bearing',semanticRayConsistency:semanticReview}:{})},
    scope:'experimental task-relative request; measured existing-source entry, release and complete ending still required'});
  requests.add(request);return request;
}
// Factory identity prevents a deserialized diagnostic proposal from being
// mistaken for an owned request created from the current real request context.
export const isTaskRelativeCarryRequest=request=>requests.has(request);
/** Use this transaction for optional semantic normalization. A materially
 * changed queued origin yields the unchanged ordinary goal without creating
 * a matched request. Stale ownership/Reset and malformed bindings still throw.
 */
export function proposeTaskRelativeCarryRequest(args){
  try{return freeze({supported:true,request:makeTaskRelativeCarryRequest(args)});}
  catch(error){if(error instanceof UnsupportedSemanticNormalization)return error.proposal;throw error;}
}
