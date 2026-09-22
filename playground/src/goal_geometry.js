import { quatRotateOne } from './math.js';

// Floor clicks specify the support surface. Policy object commands refer to
// the object's frame origin, which normally lies above that surface.
export function objectGoalOnFloor(floorPointWorld, pointsLocal, quatXyzwWorld) {
  let lowestOffset = Infinity;
  for (let i = 0; i < pointsLocal.length; i += 3) {
    const offset = quatRotateOne(quatXyzwWorld, pointsLocal.subarray(i, i + 3));
    lowestOffset = Math.min(lowestOffset, offset[2]);
  }
  if (!Number.isFinite(lowestOffset)) throw new Error('Object surface points are unavailable');
  return new Float32Array([
    floorPointWorld[0], floorPointWorld[1], floorPointWorld[2] - lowestOffset,
  ]);
}
