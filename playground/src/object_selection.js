import { NUM_POINTS } from './obs_builder.js';

// Resolve the complete observation cache before publishing a new selection.
// Shipped records contain {points, bbox, ...}; older exports were bare arrays.
export function createObjectSelection(bodyName, bodyId, pointCloudDb) {
  if (bodyName === null) return { bodyName: null, bodyId: -1, pointsLocal: null };
  if (!Number.isInteger(bodyId) || bodyId < 0) throw new Error(`Object body unavailable: ${bodyName}`);
  const key = bodyName.replace(/^active_/, '');
  const record = pointCloudDb[key];
  const points = record?.points ?? record;
  if (!Array.isArray(points) || points.length < NUM_POINTS) {
    throw new Error(`Object ${key} needs ${NUM_POINTS} surface points`);
  }
  const pointsLocal = new Float32Array(NUM_POINTS * 3);
  for (let i = 0; i < NUM_POINTS; i++) {
    const point = points[i];
    if (!Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) {
      throw new Error(`Object ${key} has invalid surface point ${i}`);
    }
    pointsLocal.set(point, i * 3);
  }
  return { bodyName, bodyId, pointsLocal };
}
