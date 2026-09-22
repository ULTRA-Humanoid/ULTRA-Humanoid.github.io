/** Private fixed terminal stance target. Changes reference data only. */
import { transformTeacherReference } from './teacher_reference.js';
import { turnReferenceTransform } from './teacher_turn_controller.js';
import { quatRotateOne } from './math.js';

export const STANCE_VELOCITY_BLOCKS = Object.freeze([[7,13],[42,71],[78,84],[357,591]]);
const f32=Math.fround;
function finite(value,length,label) {
  if (!value || value.length!==length || !Array.from(value).every(Number.isFinite)) throw new Error(`${label} requires ${length} finite values`);
}

export function planTeacherStandingReference(terminal,{alignment,rootPosition,rootQuaternion,objectPosition,objectQuaternion,objectPointsLocal}) {
  finite(terminal,747,'Reference');finite(rootPosition,3,'Root position');finite(rootQuaternion,4,'Root quaternion');
  finite(objectPosition,3,'Object position');finite(objectQuaternion,4,'Object quaternion');
  if (!['original','live-root'].includes(alignment)) throw new Error('Use original or live-root alignment');
  if (!Array.isArray(objectPointsLocal) || objectPointsLocal.length!==256) throw new Error('The actual 256 object points are required');
  for (const point of objectPointsLocal) finite(point,3,'Object point');
  const transform=alignment==='live-root'?turnReferenceTransform(terminal,rootPosition,rootQuaternion):{yawRadians:0,translation:[0,0,0]};
  const stance=alignment==='live-root'?transformTeacherReference(terminal,transform):Float32Array.from(terminal);
  for (const [start,end] of STANCE_VELOCITY_BLOCKS) stance.fill(0,start,end);
  // Both arms target the box actually placed at takeover, including its live
  // orientation. The original user goal is retained only for outcome reporting.
  stance.set(objectPosition,71);stance.set(objectQuaternion,74);
  const forward=quatRotateOne(stance.subarray(3,7),[1,0,0]).map(f32);
  const half=f32(-f32(Math.atan2(forward[1],forward[0]))/2);
  const heading=[0,0,f32(Math.sin(half)),f32(Math.cos(half))];
  const points=objectPointsLocal.map(point=>quatRotateOne(stance.subarray(74,78),point).map((value,axis)=>f32(f32(value)+stance[71+axis])));
  for (let body=0;body<39;body++) {
    const position=stance.subarray(84+3*body,87+3*body);let best=null,bestDistance=Infinity;
    for (const point of points) {
      const delta=point.map((value,axis)=>f32(position[axis]-value));
      const squared=delta.reduce((sum,value)=>sum+value*value,0);
      if (squared<bestDistance) {bestDistance=squared;best=delta;}
    }
    stance.set(quatRotateOne(heading,best).map(f32),630+3*body);
  }
  return {frame:stance,alignment,transform,objectTargetPosition:Array.from(stance.subarray(71,74)),
    objectTargetQuaternion:Array.from(stance.subarray(74,78)),bodyRootTargetPosition:Array.from(stance.subarray(0,3)),
    note:'Fixed terminal body pose and contact targets; fixed actual placed-object pose; zero reference velocities; recomputed target interaction graph; no ongoing reanchor.'};
}

/** Return the full747 fixed stance; use the plan helper to inspect alignment. */
export function buildTeacherStandingReference(terminal,options) {
  return planTeacherStandingReference(terminal,options).frame;
}
