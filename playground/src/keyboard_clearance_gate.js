// keyboard_clearance_gate.js — pure planar geometry for the hybrid keyboard gate.
//
// The student translator is only allowed to own a held keyboard command when the
// commanded motion keeps a clearance from every box collision AABB (the same
// `ObjectMeshBounds.read(data)` rectangles the recorded supervisor uses) for the
// next ~1 s of commanded travel. Two models:
//   * directional (W or S alone): the pelvis plus a forward footprint of
//     `footprintM` is swept along the commanded direction; the largest travel
//     that keeps >= `clearanceM` from every rectangle is returned exactly (ray
//     against rounded rectangles).
//   * omnidirectional (any A/D or Q/E component): the student's strafes and turns
//     are arcs of uncontrolled direction, so the whole disc pelvis + footprint +
//     drift must clear every rectangle.
// Nothing here touches physics, references or the policy.
export const HYBRID_GATE_DEFAULTS = Object.freeze({ clearanceM: .55, footprintM: .35, horizonS: 1, turnDriftMps: .5 });

const EPS = 1e-12;
const finiteXY = p => Array.isArray(p) || ArrayBuffer.isView(p) ? p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]) : false;
export function validRect(r) {
  return Boolean(r) && [r.minX, r.maxX, r.minY, r.maxY].every(Number.isFinite) && r.minX <= r.maxX && r.minY <= r.maxY;
}
function requireRects(rects) {
  if (!Array.isArray(rects) || !rects.every(validRect)) throw new Error('Finite collision rectangles are required');
}

/** Planar distance from a point to a closed rectangle (0 inside). */
export function pointRectDistance(p, r) {
  if (!finiteXY(p) || !validRect(r)) throw new Error('A finite point and rectangle are required');
  return Math.hypot(Math.max(r.minX - p[0], 0, p[0] - r.maxX), Math.max(r.minY - p[1], 0, p[1] - r.maxY));
}

function pointSegmentDistance(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
  const t = len2 < EPS ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

/** Liang–Barsky clip of segment a→b against a closed rectangle. */
export function segmentIntersectsRect(a, b, r) {
  let t0 = 0, t1 = 1;
  const d = [b[0] - a[0], b[1] - a[1]];
  for (const [axis, min, max] of [[0, r.minX, r.maxX], [1, r.minY, r.maxY]]) {
    if (Math.abs(d[axis]) < EPS) { if (a[axis] < min || a[axis] > max) return false; continue; }
    let ta = (min - a[axis]) / d[axis], tb = (max - a[axis]) / d[axis];
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    if (t0 > t1) return false;
  }
  return true;
}

/** Exact planar distance between a segment and a closed rectangle (0 when touching). */
export function segmentRectDistance(a, b, r) {
  if (!finiteXY(a) || !finiteXY(b) || !validRect(r)) throw new Error('Finite segment and rectangle are required');
  if (segmentIntersectsRect(a, b, r)) return 0;
  const corners = [[r.minX, r.minY], [r.maxX, r.minY], [r.maxX, r.maxY], [r.minX, r.maxY]];
  return Math.min(pointRectDistance(a, r), pointRectDistance(b, r), ...corners.map(c => pointSegmentDistance(c, a, b)));
}

// Ray o + t·d (t >= 0, d unit) entry parameter into a closed rectangle; 0 when the origin is inside; Infinity when missed.
function rayRectEntry(o, d, r) {
  let t0 = 0, t1 = Infinity;
  for (const [axis, min, max] of [[0, r.minX, r.maxX], [1, r.minY, r.maxY]]) {
    if (Math.abs(d[axis]) < EPS) { if (o[axis] < min || o[axis] > max) return Infinity; continue; }
    let ta = (min - o[axis]) / d[axis], tb = (max - o[axis]) / d[axis];
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    if (t0 > t1) return Infinity;
  }
  return t0;
}
// Ray entry into a closed disc; 0 when the origin is inside; Infinity when missed.
function rayDiscEntry(o, d, c, radius) {
  const f = [o[0] - c[0], o[1] - c[1]];
  const b = f[0] * d[0] + f[1] * d[1], cc = f[0] * f[0] + f[1] * f[1] - radius * radius;
  if (cc <= 0) return 0;
  const disc = b * b - cc;
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  return t >= 0 ? t : Infinity;
}
/** Ray entry into the rounded rectangle {p : dist(p, r) <= clearance}. */
export function rayRoundedRectEntry(o, d, r, clearance) {
  if (clearance < 0) throw new Error('Clearance must be non-negative');
  const entries = [
    rayRectEntry(o, d, { minX: r.minX - clearance, maxX: r.maxX + clearance, minY: r.minY, maxY: r.maxY }),
    rayRectEntry(o, d, { minX: r.minX, maxX: r.maxX, minY: r.minY - clearance, maxY: r.maxY + clearance }),
  ];
  if (clearance > 0) for (const c of [[r.minX, r.minY], [r.maxX, r.minY], [r.maxX, r.maxY], [r.minX, r.maxY]]) entries.push(rayDiscEntry(o, d, c, clearance));
  return Math.min(...entries);
}

/** Largest travel t in [0, maxTravelM] such that the footprint segment
 * root → root + dir·(t + footprintM) keeps >= clearanceM from every rectangle.
 * `entryM` is the travel of the footprint tip at which clearance would be lost. */
export function directionalFreeTravel(root, dirWorld, rects, { clearanceM, footprintM, maxTravelM = Infinity } = {}) {
  if (!finiteXY(root) || !finiteXY(dirWorld)) throw new Error('A finite root and direction are required');
  if (![clearanceM, footprintM, maxTravelM].every(v => Number.isFinite(v) || v === Infinity) || clearanceM < 0 || footprintM < 0 || maxTravelM < 0) {
    throw new Error('Non-negative clearance, footprint and travel bounds are required');
  }
  requireRects(rects);
  const norm = Math.hypot(dirWorld[0], dirWorld[1]);
  if (norm < 1e-9) throw new Error('A non-zero direction is required');
  const d = [dirWorld[0] / norm, dirWorld[1] / norm], o = [root[0], root[1]];
  let entry = Infinity, limitingIndex = null, minClearance = Infinity;
  rects.forEach((r, index) => {
    const e = rayRoundedRectEntry(o, d, r, clearanceM);
    if (e < entry) { entry = e; limitingIndex = index; }
    minClearance = Math.min(minClearance, pointRectDistance(o, r));
  });
  const freeTravelM = Math.max(0, Math.min(maxTravelM, entry - footprintM));
  return { freeTravelM, entryM: entry, limitingIndex, blockedNow: entry <= footprintM, minClearanceM: minClearance };
}

/** Minimum planar distance from the pelvis to every rectangle. */
export function omnidirectionalClearance(root, rects) {
  if (!finiteXY(root)) throw new Error('A finite root is required');
  requireRects(rects);
  let minClearance = Infinity, limitingIndex = null;
  rects.forEach((r, index) => { const d = pointRectDistance(root, r); if (d < minClearance) { minClearance = d; limitingIndex = index; } });
  return { minClearanceM: minClearance, limitingIndex };
}

/** Decide whether the student translator may own a keyboard command.
 * command: { forward: -1|0|1, lateral: -1|0|1, turn: -1|0|1 } after opposed keys cancel.
 * capSpeedMps: the envelope's current commanded translation cap. */
export function studentKeyboardAdmission({ root, yaw, command, rects, capSpeedMps, options = {} } = {}) {
  const o = { ...HYBRID_GATE_DEFAULTS, ...options };
  if (!command || ![command.forward, command.lateral, command.turn].every(v => [-1, 0, 1].includes(v))) throw new Error('A cancelled key command is required');
  if (!Number.isFinite(yaw) || !Number.isFinite(capSpeedMps) || capSpeedMps < 0) throw new Error('A finite heading and speed cap are required');
  if (!command.forward && !command.lateral && !command.turn) return { admissible: true, kind: 'none', minClearanceM: omnidirectionalClearance(root, rects).minClearanceM };
  const horizon = o.horizonS;
  if (command.forward && !command.lateral && !command.turn) {
    const dir = [Math.cos(yaw) * command.forward, Math.sin(yaw) * command.forward];
    const travel = directionalFreeTravel(root, dir, rects, { clearanceM: o.clearanceM, footprintM: o.footprintM, maxTravelM: capSpeedMps * horizon });
    const requiredTravelM = capSpeedMps * horizon;
    return { admissible: travel.freeTravelM + 1e-9 >= requiredTravelM, kind: 'directional', requiredTravelM,
      freeTravelM: travel.freeTravelM, entryM: travel.entryM, limitingIndex: travel.limitingIndex, minClearanceM: travel.minClearanceM, capSpeedMps };
  }
  const translating = Boolean(command.forward || command.lateral);
  const driftM = (translating ? capSpeedMps : o.turnDriftMps) * horizon;
  const requiredClearanceM = o.clearanceM + o.footprintM + driftM;
  const omni = omnidirectionalClearance(root, rects);
  return { admissible: omni.minClearanceM + 1e-9 >= requiredClearanceM, kind: translating ? 'omnidirectional_translation' : 'omnidirectional_turn',
    requiredClearanceM, minClearanceM: omni.minClearanceM, limitingIndex: omni.limitingIndex, capSpeedMps };
}
