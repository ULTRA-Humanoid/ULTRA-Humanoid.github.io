// Private full-source ground-push planning. No physics, actor calls or success
// claims. The scanned box is not assumed symmetric about its body axes.
import { quatRotateOne } from './math.js';
import { transformTeacherReference } from './teacher_reference.js';
import { commonTranslationWarp } from './teacher_goal_warp.js';

const TAU = 2 * Math.PI;
const wrap = value => Math.atan2(Math.sin(value), Math.cos(value));
const immutable = values => Object.freeze(Array.from(values));
function finite(values, count, label) {
  if (!values || values.length !== count || !Array.from(values).every(Number.isFinite)) throw new Error(`${label} must contain ${count} finite values`);
}
function quaternion(value) {
  finite(value, 4, 'Quaternion');
  if (Math.abs(Math.hypot(...value) - 1) > 1e-4) throw new Error('A unit quaternion is required');
}
function heading(value) {
  quaternion(value); const forward = quatRotateOne(value, [1, 0, 0]);
  return Math.atan2(forward[1], forward[0]);
}
const cross = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function hull(points) {
  const ordered = points.map(p => [p[0],p[1]]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const unique=ordered.filter((p,i)=>!i||p[0]!==ordered[i-1][0]||p[1]!==ordered[i-1][1]);
  if (unique.length<3) throw new Error('A nondegenerate collision footprint is required');
  const half=rows=>{const result=[];for(const p of rows){while(result.length>=2&&cross(result.at(-2),result.at(-1),p)<=0)result.pop();result.push(p);}return result;};
  const low=half(unique),high=half(unique.toReversed());low.pop();high.pop();
  const result=low.concat(high);if(result.length<3)throw new Error('A nondegenerate collision footprint is required');return result;
}
function directedDistance(points, polygon) {
  let maximum=0;
  for(const point of points){
    let inside=true,minimum=Infinity;
    for(let j=0;j<polygon.length;j++){
      const a=polygon[j],b=polygon[(j+1)%polygon.length];inside&&=cross(a,b,point)>=-1e-12;
      const dx=b[0]-a[0],dy=b[1]-a[1];
      const t=Math.max(0,Math.min(1,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(dx*dx+dy*dy)));
      minimum=Math.min(minimum,Math.hypot(point[0]-a[0]-t*dx,point[1]-a[1]-t*dy));
    }
    if(!inside)maximum=Math.max(maximum,minimum);
  }
  return maximum;
}
const rotateXY=(points,yaw)=>{const c=Math.cos(yaw),s=Math.sin(yaw);return points.map(p=>[c*p[0]-s*p[1],s*p[0]+c*p[1]]);};
export function footprintHausdorffXY(a,b) {
  const pa=hull(a),pb=hull(b);return Math.max(directedDistance(pa,pb),directedDistance(pb,pa));
}

/** Enumerate measured local minima of planar collision-footprint mismatch.
 * Each branch is a distinct proposal, never an asserted box symmetry. The
 * caller supplies the actual compiled collision hull in the object's body
 * frame, not the policy's random surface samples or an idealized cuboid.
 */
export function groundPushFootprintAlignments(collisionHullLocal, sourceQuaternion, objectQuaternion) {
  if(!Array.isArray(collisionHullLocal)||collisionHullLocal.length<4)throw new Error('Compiled collision hull vertices are required');
  collisionHullLocal.forEach(p=>finite(p,3,'Collision vertex'));quaternion(sourceQuaternion);quaternion(objectQuaternion);
  const source=hull(collisionHullLocal.map(p=>quatRotateOne(sourceQuaternion,p)));
  const actual=hull(collisionHullLocal.map(p=>quatRotateOne(objectQuaternion,p)));
  const objective=yaw=>{const moved=rotateXY(source,yaw);return Math.max(directedDistance(moved,actual),directedDistance(actual,moved));};
  const count=360,step=TAU/count,scores=Array.from({length:count},(_,i)=>objective(i*step));
  const result=[];
  for(let index=0;index<count;index++){
    if(scores[index]>scores[(index+count-1)%count]||scores[index]>scores[(index+1)%count])continue;
    let lo=(index-1)*step,hi=(index+1)*step;
    // Local refinement preserves every separated coarse-grid minimum.
    for(let iteration=0;iteration<35;iteration++){
      const a=lo+(hi-lo)*.3819660112501051,b=lo+(hi-lo)*.6180339887498949;
      if(objective(a)<objective(b))hi=b;else lo=a;
    }
    const yaw=wrap((lo+hi)/2);
    if(result.some(row=>Math.abs(wrap(row.yawRadians-yaw))<step))continue;
    result.push(Object.freeze({yawRadians:yaw,objectHeadingOffsetRadians:wrap(yaw-heading(objectQuaternion)),
      centeredFootprintErrorM:objective(yaw),geometryOnly:true,physicalSupportEstablished:false}));
  }
  return Object.freeze(result.sort((a,b)=>a.centeredFootprintErrorM-b.centeredFootprintErrorM));
}

function sourceGeometry(referenceFrames,sourceFrames) {
  if(!Array.isArray(referenceFrames)||!Number.isInteger(sourceFrames)||sourceFrames<2||referenceFrames.length<sourceFrames+16)throw new Error('A complete source and terminal lookahead are required');
  const first=referenceFrames[0],last=referenceFrames[sourceFrames-1];finite(first,747,'Full reference');finite(last,747,'Full reference');
  const delta=[last[71]-first[71],last[72]-first[72],0];
  if(Math.hypot(...delta)<1e-4)throw new Error('A nonzero complete push displacement is required');
  return {first,last,delta};
}
function correctionLimit(maxCorrection) {
  if(!Number.isFinite(maxCorrection)||maxCorrection<=0||maxCorrection>.05)throw new Error('Initial push proposals permit at most 5 cm correction');
}

/** Describe one natural full-source direction before a requested goal is
 * mapped. Bounds describe a private experiment, not validated motor skills.
 */
export function describeGroundPushDirection(referenceFrames,sourceFrames,objectPosition,objectQuaternion,alignment,{maxCorrection=.05}={}) {
  finite(objectPosition,3,'Object position');quaternion(objectQuaternion);correctionLimit(maxCorrection);
  const {delta}=sourceGeometry(referenceFrames,sourceFrames);
  if(!Number.isFinite(alignment?.objectHeadingOffsetRadians))throw new Error('An explicit measured alignment branch is required');
  const yaw=heading(objectQuaternion)+alignment.objectHeadingOffsetRadians;
  const rotated=rotateXY([delta],yaw)[0],distance=Math.hypot(...rotated);
  if(distance<=maxCorrection)throw new Error('Push direction must remain positive across its proposed interval');
  const direction=rotated.map(x=>x/distance);
  return Object.freeze({objectPositionWorld:immutable(objectPosition),objectQuaternionXyzw:immutable(objectQuaternion),
    directionWorld:immutable([...direction,0]),naturalDistanceM:distance,naturalFloorGoalWorld:immutable([objectPosition[0]+rotated[0],objectPosition[1]+rotated[1],0]),
    distanceIntervalM:immutable([distance-maxCorrection,distance+maxCorrection]),maxCorrectionM:maxCorrection,
    objectHeadingOffsetRadians:alignment.objectHeadingOffsetRadians,centeredFootprintErrorM:alignment.centeredFootprintErrorM,
    sourceFrames,geometryOnly:true,physicalSupportEstablished:false});
}

/** Return an explicit alternative endpoint on the source's natural ray.
 * The original request remains separate; callers must display/adopt a changed
 * proposal before treating it as the task goal. No silent success relabeling.
 */
export function proposeGroundPushEndpoint(direction,requestedGoalWorld) {
  finite(requestedGoalWorld,3,'Requested goal');
  const origin=direction.objectPositionWorld,u=direction.directionWorld;
  const along=(requestedGoalWorld[0]-origin[0])*u[0]+(requestedGoalWorld[1]-origin[1])*u[1];
  const distance=Math.max(direction.distanceIntervalM[0],Math.min(direction.distanceIntervalM[1],along));
  const proposed=[origin[0]+distance*u[0],origin[1]+distance*u[1],0];
  const residual=Math.hypot(requestedGoalWorld[0]-proposed[0],requestedGoalWorld[1]-proposed[1]);
  return Object.freeze({requestedGoalWorld:immutable(requestedGoalWorld),proposedGoalWorld:immutable(proposed),
    distanceM:distance,requestToProposalXYM:residual,requiresChangedGoal:residual>1e-6,physicalSupportEstablished:false});
}

/** Bind a complete reference to an already chosen immutable physical goal.
 * Only references change. Replanning before entry follows measured object
 * heading/XY on the same branch; any remaining correction must fit 5 cm.
 * During teacher execution the returned bank must remain fixed.
 */
export function planAlignedGroundPush(referenceFrames,sourceFrames,objectPosition,objectQuaternion,goalWorld,
  {objectHeadingOffsetRadians,startFrame,endFrame,maxCorrection=.05,controlHz=60}={}) {
  const {first}=sourceGeometry(referenceFrames,sourceFrames);
  finite(objectPosition,3,'Object position');quaternion(objectQuaternion);finite(goalWorld,3,'Chosen goal');correctionLimit(maxCorrection);
  if(!Number.isFinite(objectHeadingOffsetRadians)||!Number.isInteger(endFrame)||endFrame>=sourceFrames)throw new Error('A fixed orientation branch and complete-source warp interval are required');
  const yaw=heading(objectQuaternion)+objectHeadingOffsetRadians,c=Math.cos(yaw),s=Math.sin(yaw);
  const transform={yawRadians:yaw,translation:[objectPosition[0]-c*first[71]+s*first[72],objectPosition[1]-s*first[71]-c*first[72],0]};
  const aligned=referenceFrames.map(row=>transformTeacherReference(row,transform));
  const last=aligned[sourceFrames-1],correction=[goalWorld[0]-last[71],goalWorld[1]-last[72],0];
  if(Math.hypot(...correction)>maxCorrection+1e-6)throw new Error('Chosen push endpoint exceeds this complete source correction bound');
  const {frames}=commonTranslationWarp(aligned,correction,startFrame,endFrame,controlHz);
  const final=frames[sourceFrames-1];
  return {frames,first:frames[0],transform,requestedGoalWorld:immutable(goalWorld),referenceGoalWorld:immutable(final.slice(71,74)),
    remainingDistance:Math.hypot(goalWorld[0]-final[71],goalWorld[1]-final[72]),correctionWorld:immutable(correction),
    approachGoalWorld:immutable([frames[0][0],frames[0][1],0]),actualObjectQuaternionAtPlan:immutable(objectQuaternion),
    alignmentMode:'measured-collision-footprint-branch',physicalSupportEstablished:false};
}
