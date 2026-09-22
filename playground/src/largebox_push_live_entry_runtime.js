// Task-only hand002 live-handoff diagnostic. It measures, reports and preserves
// the live boundary; it never writes simulator state or calls reset.
export const HAND002_LIVE = Object.freeze({
  sourceIdentity: 'sub7_largebox_002_081_081_076_080_080_080',
  sourcePhaseInclusive: Object.freeze([69, 293]), sourceFrames: 225,
  interactionInterval: Object.freeze([81, 110]),
  objectBody: 'active_largebox_080_080_080',
  geometryIdentity: 'g1_scene.xml#active_largebox_080_080_080',
});
const widths={rootPosition:3,rootQuaternion:4,rootVelocity:6,jointPosition:29,jointVelocity:29,
  objectPosition:3,objectQuaternion:4,objectVelocity:6};
const historyWidths={previousAction:29,appliedTorque:29,previousDofPosition:29,previousDofVelocity:29};
const finite=(x,n)=>Array.isArray(x)&&x.length===n&&x.every(Number.isFinite);
const complete=(x,s)=>Object.entries(s).every(([k,n])=>finite(x?.[k],n));
const l2=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
const rms=(a,b)=>l2(a,b)/Math.sqrt(a.length);
const qdeg=(a,b)=>{const d=Math.abs(a.reduce((s,v,i)=>s+v*b[i],0)/(Math.hypot(...a)*Math.hypot(...b)));
  return 2*Math.acos(Math.max(-1,Math.min(1,d)))*180/Math.PI;};
const no=(reason,details={})=>Object.freeze({supported:false,reason,...details});
const sameExecutionLease=(a,b)=>a?.episodeVersion===b?.episodeVersion&&a?.control===b?.control
  &&a?.objectBinding?.objectBody===b?.objectBinding?.objectBody
  &&a?.objectBinding?.objectBodyId===b?.objectBinding?.objectBodyId
  &&b?.provenance?.initializedFromTargetState===false&&complete(b?.state,widths)&&complete(b?.history,historyWidths);
export const completionLeaseMatchesFreshBoundary=(review,packet)=>{
  const endpoint=review?.endpointPacket,delta=packet?.control-endpoint?.control;
  return review?.phase==='complete'&&review?.completionReason==='finished'
    &&endpoint?.episodeVersion===packet?.episodeVersion&&(delta===0||delta===1)
    &&endpoint?.provenance?.initializedFromTargetState===false
    &&endpoint?.objectBinding?.objectBody===packet?.objectBinding?.objectBody
    &&endpoint?.objectBinding?.objectBodyId===packet?.objectBinding?.objectBodyId
    &&complete(endpoint?.state,widths)&&complete(endpoint?.history,historyWidths);
};

export const LARGEBOX_PUSH_OUTCOME_THRESHOLDS = Object.freeze({
  minimumSignedPushProgressM:.10,minRootHeightM:.45,minUpright:.50,
  maximumFinalObjectCenterHeightM:.25,
});
export function prepareGroundPushGoalWarpEntry({raw,evidence=null,objectPosition,goalWorld,planCarryToGoal}={}) {
  if(typeof planCarryToGoal!=='function')throw new Error('Existing goal-warp planner is required');
  if(!finite(objectPosition,3)||!finite(goalWorld,3))throw new Error('Finite live object and fixed push goal are required');
  const skill=bindHand002Predecessor69(raw,evidence),plan=planCarryToGoal(skill.frames,skill.sourceFrames,
    objectPosition,goalWorld,{startFrame:skill.pushInterval[0],endFrame:skill.pushInterval[1],maxCorrection:.05});
  const first=plan.frames?.[0];if(!first||first.length!==747)throw new Error('Complete goal-warp entry frame is required');
  return Object.freeze({sourceIdentity:skill.sourceIdentity,objectBody:skill.objectBodyName,
    goalWorld:Object.freeze(Array.from(goalWorld)),targetRootPosition:Object.freeze(Array.from(first.slice(0,3))),
    targetRootQuaternionWxyz:Object.freeze([first[6],first[3],first[4],first[5]]),
    referenceObjectPosition:Object.freeze(Array.from(first.slice(71,74))),
    transform:Object.freeze(structuredClone(plan.transform)),correctionWorld:Object.freeze(Array.from(plan.correctionWorld)),
    referenceGoalWorld:Object.freeze(Array.from(plan.referenceGoalWorld)),remainingDistance:plan.remainingDistance,
    state:Object.freeze({rootPosition:Object.freeze(Array.from(first.slice(0,3))),
      rootQuaternion:Object.freeze([first[6],first[3],first[4],first[5]])})});
}
export function prepareOutcomeBasedNoResetEntry({entryPacket,alignedTarget,compiledObstacles,planBoxApproach}={}) {
  if(entryPacket?.provenance?.initializedFromTargetState!==false
      ||entryPacket?.objectBinding?.objectBody!==HAND002_LIVE.objectBody
      ||!Number.isSafeInteger(entryPacket?.objectBinding?.objectBodyId))throw new Error('Current no-reset selected Largebox required');
  if(entryPacket?.coordinateFrame!=='mujoco_world:g1_scene.xml'||entryPacket?.quaternionOrder!=='wxyz')
    throw new Error('Current world frame and WXYZ convention required');
  const start=entryPacket?.state?.rootPosition,target=alignedTarget?.state?.rootPosition,q=alignedTarget?.state?.rootQuaternion;
  if(!finite(start,3)||!finite(target,3)||!finite(q,4))throw new Error('Finite current and target pose required');
  let diagnostic=null;
  try{diagnostic=planBoxApproach(start,target,compiledObstacles,{clearance:.55,respectTransitClearance:false});}
  catch(error){diagnostic={supported:false,reason:'planner_diagnostic_error',message:String(error?.message??error)};}
  const plan=diagnostic?.supported&&Array.isArray(diagnostic.path)&&finite(diagnostic.finalGoal,3)
    ?diagnostic:{supported:false,reason:diagnostic?.reason??'planner_diagnostic_unavailable',path:[],finalGoal:Array.from(target),diagnostic};
  return Object.freeze({targetRootPosition:Object.freeze(Array.from(target)),
    targetRootQuaternionWxyz:Object.freeze(Array.from(q)),plan:Object.freeze(structuredClone(plan)),
    plannerDiagnostic:Object.freeze(structuredClone(diagnostic))});
}
export function shouldRunOutcomeLaneControlPreview(existingRequired,noResetPoseOwner,activeDiagnostic) {
  return Boolean(existingRequired&&!noResetPoseOwner&&!activeDiagnostic);
}
export function noResetActionAdvancedExactlyOnce(ownerControls,inferenceControls) {
  return Number.isSafeInteger(ownerControls)&&Number.isSafeInteger(inferenceControls)&&ownerControls===inferenceControls+1;
}

export function selectRetiredPushForwardPlanObstacles(obstacles,{eligible=false,intent=null,targetBodyId=-1}={}) {
  if(!Array.isArray(obstacles))throw new Error('Obstacle bounds must be an array');
  const keys=intent?.keys,held=keys?['forward','backward','left','right','turnLeft','turnRight'].filter(k=>keys[k]===true):[];
  const exactForward=intent?.type==='keys'&&Number.isSafeInteger(intent.revision)&&intent.revision>0
    &&held.length===1&&held[0]==='forward';
  if(!eligible||!exactForward||!Number.isSafeInteger(targetBodyId)||targetBodyId<=0)
    return Object.freeze({obstacles,admitted:false,ignoredTargetObstacleCount:0});
  const selected=obstacles.filter(obstacle=>obstacle?.bodyId!==targetBodyId);
  return Object.freeze({obstacles:selected,admitted:selected.length!==obstacles.length,
    ignoredTargetObstacleCount:obstacles.length-selected.length});
}
export function shouldCompleteOutcomeLaneAfterTransit(stage,result) {
  return stage==='transit'&&result?.justCompleted===true&&result?.completionReason==='finished';
}
export function evaluateLargeboxPushOutcome({requestRecord,robotTargetContactControls,signedProgress,
  finalObjectPosition,minRoot,minUpright,noReset,selectionStable,exitRetired,
  ordinaryCommandExecuted,qpos,qvel}={}) {
  const t=LARGEBOX_PUSH_OUTCOME_THRESHOLDS;
  const criteria={
    outcomeFinished:requestRecord?.disposition==='outcome'&&requestRecord?.reason==='finished',
    actualRobotTargetContact:Number.isSafeInteger(robotTargetContactControls)&&robotTargetContactControls>0,
    signedHorizontalProgress:Number.isFinite(signedProgress)&&signedProgress>=t.minimumSignedPushProgressM,
    groundedFinalUsingControllerCriterion:finite(finalObjectPosition,3)&&finalObjectPosition[2]<=t.maximumFinalObjectCenterHeightM,
    actualBalance:Number.isFinite(minRoot)&&minRoot>=t.minRootHeightM&&Number.isFinite(minUpright)&&minUpright>=t.minUpright,
    noReset:noReset===true,selectionStable:selectionStable===true,exitRetired:exitRetired===true,
    ordinaryCommandExecuted:ordinaryCommandExecuted===true,
    finite:Array.isArray(qpos)&&qpos.every(Number.isFinite)&&Array.isArray(qvel)&&qvel.every(Number.isFinite),
  };
  return Object.freeze({pass:Object.values(criteria).every(Boolean),criteria:Object.freeze(criteria),thresholds:t});
}

export function bindHand002Predecessor69(raw, evidence) {
  if(raw?.source_frames!==225
    ||raw?.source_phase_inclusive?.[0]!==69||raw?.source_phase_inclusive?.[1]!==293
    ||raw?.interaction_interval_frames?.[0]!==81||raw?.interaction_interval_frames?.[1]!==110
    ||raw?.object_body!==HAND002_LIVE.objectBody
    ||!Array.isArray(raw?.object_points256_local)||raw.object_points256_local.length!==256
    ||!Array.isArray(raw?.reference_frames747)||raw.reference_frames747.length<241
    ||raw.reference_frames747.some(row=>!finite(row,747))) throw new Error('Exact retained hand002 source69-293 asset required');
  return Object.freeze({name:raw.name,sourceIdentity:HAND002_LIVE.sourceIdentity,objectBodyName:raw.object_body,
    objectGeometryIdentity:HAND002_LIVE.geometryIdentity,objectPointsLocal:raw.object_points256_local.map(x=>Array.from(x)),
    sourceFrames:225,frames:raw.reference_frames747.map(x=>Float32Array.from(x)),
    manipulationKind:'ground_push',pushStyle:'left_hand',pushInterval:Object.freeze([81,110]),
    carryInterval:null,studentTransportInterval:null,locomotionOnly:false,teacherCompatible:true,studentCompatible:false,
    sourcePhaseInclusive:Object.freeze([69,293]),qualificationStage:'unqualified',
    sourceReproductionEvidence:evidence?Object.freeze({job:34847876,diagnosticOnly:true}):null});
}

export function measureSource69LiveBoundary(target, packet) {
  const categorical={objectBody:packet?.objectBinding?.objectBody===HAND002_LIVE.objectBody,
    coordinateFrame:packet?.coordinateFrame==='mujoco_world:g1_scene.xml',quaternionOrder:packet?.quaternionOrder==='wxyz',
    noReset:packet?.provenance?.initializedFromTargetState===false};
  if(!Object.values(categorical).every(Boolean))return no('source69_categorical_mismatch',{categorical});
  if(!complete(packet.state,widths)||!complete(packet.history,historyWidths)||!complete(target?.state,widths))
    return no('source69_measurement_incomplete',{categorical,targetHistoryAvailable:complete(target?.history,historyWidths)});
  const state={rootPositionM:l2(target.state.rootPosition,packet.state.rootPosition),
    rootOrientationDeg:qdeg(target.state.rootQuaternion,packet.state.rootQuaternion),
    rootVelocityL2:l2(target.state.rootVelocity,packet.state.rootVelocity),
    jointPositionRmsRad:rms(target.state.jointPosition,packet.state.jointPosition),
    jointVelocityRmsRadps:rms(target.state.jointVelocity,packet.state.jointVelocity),
    objectPositionM:l2(target.state.objectPosition,packet.state.objectPosition),
    objectOrientationDeg:qdeg(target.state.objectQuaternion,packet.state.objectQuaternion),
    objectVelocityL2:l2(target.state.objectVelocity,packet.state.objectVelocity)};
  const targetHistoryAvailable=complete(target.history,historyWidths);
  const history=targetHistoryAvailable?Object.fromEntries(Object.keys(historyWidths).map(k=>[k,
    k.includes('Dof')?rms(target.history[k],packet.history[k]):l2(target.history[k],packet.history[k])])):null;
  return Object.freeze({supported:true,reason:null,categorical:Object.freeze(categorical),state:Object.freeze(state),
    history:history?Object.freeze(history):null,targetHistoryAvailable,exactStateMatch:Object.values(state).every(v=>v<=1e-12),
    toleranceStatus:'no_validated_live_entry_tolerance',compatibleForQualification:false,
    diagnosticAttemptOnly:true});
}

export async function startLargeboxPushLiveDiagnostic(args,deps) {
  const packet=deps.readBoundary(), review=deps.readApproachReview();
  if(review?.completionReason!=='finished'||review?.phase!=='complete')return no('no_reset_approach_not_completed');
  if(packet.episodeVersion!==args.expectedGeneration||packet.control!==args.expectedPhysicalControl
    ||!completionLeaseMatchesFreshBoundary(review,packet))
    return no('live_boundary_token_changed');
  if(deps.busy())return no('controller_busy');
  const selection=deps.readSelection();
  if(selection.objectBody!==HAND002_LIVE.objectBody||selection.objectBodyId!==packet.objectBinding?.objectBodyId)
    return no('selected_largebox_changed');
  const measurement=measureSource69LiveBoundary(args.source69Target,packet);
  if(!measurement.supported)return measurement;
  const [raw,evidence]=await Promise.all([deps.loadAsset(args.assetUrl),deps.loadEvidence(args.evidenceUrl)]);
  const current=deps.readBoundary(),currentSelection=deps.readSelection();
  if(deps.busy())return no('controller_became_busy_during_asset_load');
  if(!sameExecutionLease(packet,current))return no('live_boundary_changed_during_asset_load');
  if(currentSelection.objectBody!==selection.objectBody||currentSelection.objectBodyId!==selection.objectBodyId)
    return no('selection_changed_during_asset_load');
  const skill=bindHand002Predecessor69(raw,evidence);
  const controller=deps.createController(skill,args.goalWorld);
  if(typeof controller.startFromMeasuredEndpoint!=='function')return no('measured_endpoint_start_unavailable',{measurement});
  const live=deps.readLive(controller);controller.startFromMeasuredEndpoint(live);
  if(controller.phase!=='teacher')return no(controller.completionReason??'push_start_refused',{measurement});
  const entry=controller.worldFrames?.[0],goalWarpEntryResidual=entry?Object.freeze({
    rootPositionM:l2(Array.from(entry.slice(0,3)),live.rootPosWorld),
    rootOrientationDeg:qdeg(Array.from(entry.slice(3,7)),live.rootQuatXyzwWorld),
    objectPositionM:l2(Array.from(entry.slice(71,74)),live.objPosWorld),
    referenceRootPosition:Object.freeze(Array.from(entry.slice(0,3))),
    liveRootPosition:Object.freeze(Array.from(live.rootPosWorld)),
  }):null;
  deps.install(controller,{measurement,skill,endpointHistory:structuredClone(current.history)});
  return Object.freeze({supported:true,reason:null,diagnosticOnly:true,liveEntryQualified:false,
    measurement,goalWarpEntryResidual,sourceIdentity:skill.sourceIdentity,sourcePhaseInclusive:skill.sourcePhaseInclusive,
    token:Object.freeze({episodeVersion:packet.episodeVersion,control:packet.control})});
}
