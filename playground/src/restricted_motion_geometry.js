// Collision geometry for complete recorded motions. No physics is changed.
// Source polygons need an additional caller-chosen reserve for actual tracking.
const finite = (values, count) => values?.length === count && Array.from(values).every(Number.isFinite);
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const close = (a, b, tolerance = 1e-5) => Math.abs(a - b) <= tolerance;
function validPolygon(points) {
  if (!Array.isArray(points) || points.length < 3 || !points.every(p => finite(p, 2))) return false;
  let sign = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i], b = points[(i + 1) % points.length], c = points[(i + 2) % points.length];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    if (Math.abs(cross) < 1e-12) continue;
    if (sign && Math.sign(cross) !== sign) return false;
    sign = Math.sign(cross);
  }
  return sign !== 0 && points.every((a, index) => {
    const b = points[(index + 1) % points.length];
    return points.every(p => sign * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-10);
  });
}

/** Bind an explicitly named known motion. Regenerate its geometry when source
 * poses or robot geometry change. This is geometry, not source identification. */
export function bindMotionSweep(skill, asset, key) {
  if (skill?.locomotionOnly !== true || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 2
      || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames
      || !Array.isArray(asset?.sweeps)) throw new Error('A complete locomotion skill and collision sweeps are required');
  const matches = asset.sweeps.filter(sweep => sweep.name === key);
  if (matches.length !== 1 || matches[0].sourceFrames !== skill.sourceFrames) throw new Error('Named collision sweep must match the source frame count');
  const sweep = matches[0];
  if (!validPolygon(sweep.wholeBodyXYHull) || !validPolygon(sweep.terminalBodyXYHull)
      || !finite(sweep.initialRootPose, 7) || !finite(sweep.terminalRootPose, 7)) throw new Error('Finite root anchors and convex collision polygons are required');
  return sweep;
}

function alignedAnchorsMatch(frames, sourcePoses) {
  const first = frames[0], source = sourcePoses[0];
  const delta = yaw(first.slice(3, 7)) - yaw(source.slice(3, 7));
  const c = Math.cos(delta), s = Math.sin(delta), halfS = Math.sin(delta / 2), halfC = Math.cos(delta / 2);
  const tx = first[0] - c * source[0] + s * source[1], ty = first[1] - s * source[0] - c * source[1];
  return frames.every((frame, index) => {
    const expected = sourcePoses[index];
    const [x, y, z, w] = expected.slice(3, 7);
    const quaternion = [halfC * x - halfS * y, halfC * y + halfS * x, halfC * z + halfS * w, halfC * w - halfS * z];
    const sign = quaternion.reduce((sum, value, axis) => sum + value * frame[axis + 3], 0) < 0 ? -1 : 1;
    return close(frame[0], c * expected[0] - s * expected[1] + tx, 1e-4)
      && close(frame[1], s * expected[0] + c * expected[1] + ty, 1e-4) && close(frame[2], expected[2])
      && quaternion.every((value, axis) => close(value * sign, frame[axis + 3]));
  });
}

function separated(polygon, rectangle) {
  const axes = [[1, 0], [0, 1]];
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12) axes.push([a[1] - b[1], b[0] - a[0]]);
  }
  return axes.some(axis => {
    const a = polygon.map(p => p[0] * axis[0] + p[1] * axis[1]);
    const b = rectangle.map(p => p[0] * axis[0] + p[1] * axis[1]);
    // Touching the reserve boundary is unsupported.
    const epsilon = 1e-9 * Math.hypot(...axis);
    return Math.max(...a) < Math.min(...b) - epsilon || Math.max(...b) < Math.min(...a) - epsilon;
  });
}

/** Convex obstacle footprint (rect.hull). Separation must exceed the reserve along
 * some edge normal of either polygon; touching the reserve boundary is unsupported.
 * Edge-normal axes only, like the rectangle rule above (conservative). */
function separatedFromHull(polygon, hull, reserve) {
  const axes = [];
  for (const shape of [polygon, hull]) for (let i = 0; i < shape.length; i++) {
    const a = shape[i], b = shape[(i + 1) % shape.length], length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (length > 1e-12) axes.push([(a[1] - b[1]) / length, (b[0] - a[0]) / length]);
  }
  return axes.some(axis => {
    const a = polygon.map(p => p[0] * axis[0] + p[1] * axis[1]);
    const b = hull.map(p => p[0] * axis[0] + p[1] * axis[1]);
    return Math.max(...a) < Math.min(...b) - reserve - 1e-9 || Math.max(...b) < Math.min(...a) - reserve - 1e-9;
  });
}
const obstacleHull = rect => Array.isArray(rect?.hull) && validPolygon(rect.hull) ? rect.hull : null;
function clearOfObstacle(worldHull, rect, reserve) {
  const hull = obstacleHull(rect);
  if (hull) return separatedFromHull(worldHull, hull, reserve);
  const x0 = rect.minX - reserve, x1 = rect.maxX + reserve, y0 = rect.minY - reserve, y1 = rect.maxY + reserve;
  return separated(worldHull, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
}

/** Signed clearance beyond the reserve (metres) between a world hull and one
 * obstacle, using the same axes as the admission rule: the largest separating
 * distance over the tested axes (positive = admissible; <= 0 = deficit, the
 * smallest overlap over the tested axes). Same rectangle/hull selection as
 * checkRestrictedReferenceSweep, so sign agrees with its supported flag. */
export function restrictedSweepMargin(worldHull, rect, trackingReserve = .1) {
  if (!validPolygon(worldHull) || !rect || !Number.isFinite(trackingReserve)) return NaN;
  const hull = obstacleHull(rect);
  const other = hull ?? [[rect.minX - trackingReserve, rect.minY - trackingReserve], [rect.maxX + trackingReserve, rect.minY - trackingReserve],
    [rect.maxX + trackingReserve, rect.maxY + trackingReserve], [rect.minX - trackingReserve, rect.maxY + trackingReserve]];
  const axes = hull ? [] : [[1, 0], [0, 1]];
  for (const shape of hull ? [worldHull, hull] : [worldHull]) for (let i = 0; i < shape.length; i++) {
    const a = shape[i], b = shape[(i + 1) % shape.length];
    if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12) axes.push([a[1] - b[1], b[0] - a[0]]);
  }
  let best = -Infinity;
  for (const axis of axes) {
    const length = Math.hypot(...axis);
    const a = worldHull.map(p => (p[0] * axis[0] + p[1] * axis[1]) / length);
    const b = other.map(p => (p[0] * axis[0] + p[1] * axis[1]) / length);
    best = Math.max(best, Math.min(...b) - Math.max(...a), Math.min(...a) - Math.max(...b));
  }
  return hull ? best - trackingReserve : best;
}

/** Minimum margin over all obstacles (the binding obstacle). */
export function restrictedSweepClearance(worldHull, obstacles, trackingReserve = .1) {
  let marginM = Infinity, obstacleIndex = null;
  obstacles.forEach((rect, index) => {
    const margin = restrictedSweepMargin(worldHull, rect, trackingReserve);
    if (Number.isFinite(margin) && margin < marginM) { marginM = margin; obstacleIndex = index; }
  });
  return { marginM, obstacleIndex };
}

function monotoneChainHull(points) {
  const sorted = points.map(p => [p[0], p[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique = sorted.filter((p, i) => !i || p[0] !== sorted[i - 1][0] || p[1] !== sorted[i - 1][1]);
  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const half = rows => { const out = []; for (const p of rows) { while (out.length > 1 && cross(out.at(-2), out.at(-1), p) <= 0) out.pop(); out.push(p); } return out; };
  const lower = half(unique), upper = half([...unique].reverse()); lower.pop(); upper.pop();
  return [...lower, ...upper];
}

/** Convex XY obstacles from ObjectCollisionMeshes.read(data) projections: the
 * world AABB fields stay (so every rectangle consumer is unchanged) plus a convex
 * `hull` of all mesh hull vertices, which the sweep rule prefers when present.
 * A yawed box no longer carries the axis-aligned phantom width. */
export function convexObstaclesFromProjections(projections) {
  if (!Array.isArray(projections)) throw new Error('Object collision projections are required');
  return projections.map(object => {
    const points = (object.meshes ?? []).flatMap(mesh => mesh.hull ?? []);
    const hull = points.length >= 3 ? monotoneChainHull(points) : null;
    const rect = { bodyId: object.bodyId, name: object.name, minX: object.minX, maxX: object.maxX,
      minY: object.minY, maxY: object.maxY, minZ: object.minZ, maxZ: object.maxZ };
    return hull && validPolygon(hull) ? { ...rect, hull } : rect;
  });
}

/** Frame-free geometry core: the hull of a named sweep anchored at a root XY and
 * yaw (the source's first-frame root, or the stance root when stance===true),
 * against inflated obstacle bounds. checkRestrictedReferenceSweep validates the
 * aligned frames first and then applies exactly this rule; planners may call it
 * directly with a predicted anchor. A supported result covers the intended
 * source geometry plus the reserve only; it certifies no actual transition. */
export function checkAnchoredSweepGeometry({ sweep, stance = false, anchorXY, anchorYawRad, obstacles, trackingReserve = .1,
  heightAware = false, verticalTrackingReserve = .1, trackingReserveByBody = {}, includePartGeometry = false } = {}) {
  const invalid = reason => ({ supported: false, reason });
  if (!sweep || typeof stance !== 'boolean' || !Number.isFinite(trackingReserve) || trackingReserve < 0 || !Array.isArray(obstacles)
      || !finite(anchorXY, 2) || !Number.isFinite(anchorYawRad)) return invalid('invalid_reference_geometry');
  if (typeof heightAware !== 'boolean' || typeof includePartGeometry !== 'boolean'
      || !Number.isFinite(verticalTrackingReserve) || verticalTrackingReserve < 0
      || !trackingReserveByBody || typeof trackingReserveByBody !== 'object' || Array.isArray(trackingReserveByBody)
      || !Object.values(trackingReserveByBody).every(value => Number.isFinite(value) && value >= 0)) return invalid('invalid_reference_geometry');
  const polygon = stance ? sweep.terminalBodyXYHull : sweep.wholeBodyXYHull;
  if (!validPolygon(polygon)) return invalid('unsupported_reference_transform');
  if (!obstacles.every(rect => rect && [rect.minX, rect.minY, rect.maxX, rect.maxY].every(Number.isFinite)
      && rect.minX < rect.maxX && rect.minY < rect.maxY)) return invalid('invalid_obstacle_geometry');
  const angle = anchorYawRad, c = Math.cos(angle), s = Math.sin(angle);
  const worldHull = polygon.map(([x, y]) => [anchorXY[0] + c * x - s * y, anchorXY[1] + s * x + c * y]);
  if (heightAware) {
    if (!obstacles.every(rect => Number.isFinite(rect.minZ) && Number.isFinite(rect.maxZ) && rect.minZ < rect.maxZ)) {
      return invalid('invalid_obstacle_height');
    }
    if (!Array.isArray(sweep.collisionParts) || !sweep.collisionParts.length) return invalid('missing_reference_part_geometry');
    const parts = [];
    for (const part of sweep.collisionParts) {
      const shape = stance ? part.terminal : part.motion;
      if (!shape || !validPolygon(shape.xyHull) || !Number.isFinite(shape.minZ) || !Number.isFinite(shape.maxZ)
          || shape.minZ >= shape.maxZ || typeof part.body !== 'string') return invalid('invalid_reference_part_geometry');
      parts.push({ body: part.body, geom: part.geom, minZ: shape.minZ, maxZ: shape.maxZ,
        trackingReserve: trackingReserveByBody[part.body] ?? trackingReserve,
        worldHull: shape.xyHull.map(([x, y]) => [anchorXY[0] + c * x - s * y, anchorXY[1] + s * x + c * y]) });
    }
    let verticallyRelevantPairs = 0;
    for (let index = 0; index < obstacles.length; index++) {
      const rect = obstacles[index];
      for (const part of parts) {
        if (part.maxZ < rect.minZ - verticalTrackingReserve - 1e-9
            || part.minZ > rect.maxZ + verticalTrackingReserve + 1e-9) continue;
        verticallyRelevantPairs++;
        const radius = part.trackingReserve;
        if (!clearOfObstacle(part.worldHull, rect, radius)) return { supported: false, reason: 'reference_part_clearance',
          obstacleIndex: index, obstacleName: rect.name ?? rect.bodyName ?? null, body: part.body, geom: part.geom,
          heightAware: true, trackingReserve: radius, verticalTrackingReserve, worldHull,
          part, sweepName: sweep.name, stance };
      }
    }
    return { supported: true, reason: null, heightAware: true, trackingReserve, verticalTrackingReserve,
      worldHull, partCount: parts.length, ...(includePartGeometry ? { parts } : {}), verticallyRelevantPairs,
      anchorRootXY: anchorXY, anchorYawRad: angle,
      sweepName: sweep.name, stance };
  }
  for (let index = 0; index < obstacles.length; index++) {
    const rect = obstacles[index];
    if (!clearOfObstacle(worldHull, rect, trackingReserve)) {
      return { supported: false, reason: 'reference_sweep_clearance', obstacleIndex: index,
        obstacleName: rect.name ?? rect.bodyName ?? null, trackingReserve, worldHull,
        convexObstacle: obstacleHull(rect) !== null };
    }
  }
  return { supported: true, reason: null, trackingReserve, worldHull, anchorRootXY: anchorXY,
    anchorYawRad: angle, sweepName: sweep.name, stance };
}

/** sourceFrames===1 means the fixed terminal stance, anchored at that frame's
 * root XY/yaw. Otherwise the full source record is anchored at its first frame.
 * First/last root anchors must share one planar rigid transform. The owning
 * reference controller is responsible for preserving the recorded poses.
 * Obstacles are all current physical mesh XY bounds, before inflation. An
 * obstacle may also carry a convex `hull` (convexObstaclesFromProjections); the
 * hull then replaces the axis-aligned rectangle in the separation test.
 * A supported result covers intended source geometry plus the reserve only;
 * it does not certify incoming pose, actual transitions, or moving obstacles.
 * heightAware is experimental and defaults off. It uses per-geom XY hulls and
 * Z intervals; per-part tracking errors can exceed the old whole-outline error.
 * includePartGeometry is for bounded diagnostics, avoiding repeated large
 * geometry payloads in ordinary control-state snapshots. */
export function checkRestrictedReferenceSweep({ sweep, sourceFrames, alignedReferenceFrames, obstacles, trackingReserve = .1,
  heightAware = false, verticalTrackingReserve = .1, trackingReserveByBody = {}, includePartGeometry = false } = {}) {
  const invalid = reason => ({ supported: false, reason });
  if (!sweep || !Number.isFinite(trackingReserve) || trackingReserve < 0 || !Array.isArray(obstacles)
      || !Number.isInteger(sourceFrames)
      || ![1, sweep.sourceFrames].includes(sourceFrames) || !Array.isArray(alignedReferenceFrames)
      || alignedReferenceFrames.length < sourceFrames) return invalid('invalid_reference_geometry');
  if (typeof heightAware !== 'boolean' || typeof includePartGeometry !== 'boolean'
      || !Number.isFinite(verticalTrackingReserve) || verticalTrackingReserve < 0
      || !trackingReserveByBody || typeof trackingReserveByBody !== 'object' || Array.isArray(trackingReserveByBody)
      || !Object.values(trackingReserveByBody).every(value => Number.isFinite(value) && value >= 0)) return invalid('invalid_reference_geometry');
  const frames = alignedReferenceFrames.slice(0, sourceFrames);
  const source = sourceFrames === 1 ? [sweep.terminalRootPose] : [sweep.initialRootPose, sweep.terminalRootPose];
  const anchors = sourceFrames === 1 ? [frames[0]] : [frames[0], frames[sourceFrames - 1]];
  const polygon = sourceFrames === 1 ? sweep.terminalBodyXYHull : sweep.wholeBodyXYHull;
  if (!frames.every(frame => finite(frame, 747)) || !validPolygon(polygon)
      || !source.every(frame => finite(frame, 7)) || !alignedAnchorsMatch(anchors, source)) return invalid('unsupported_reference_transform');
  const anchor = frames[0];
  return checkAnchoredSweepGeometry({ sweep, stance: sourceFrames === 1, anchorXY: anchor.slice(0, 2),
    anchorYawRad: yaw(anchor.slice(3, 7)), obstacles, trackingReserve, heightAware, verticalTrackingReserve,
    trackingReserveByBody, includePartGeometry });
}
