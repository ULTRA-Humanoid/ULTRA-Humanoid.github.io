// Isolated teacher-first locomotion candidate.  It reuses one reviewed BONES
// source instead of inventing poses: source 0..263 supplies the measured
// stand-to-gait entry and first reviewed A034 period (184..263), later periods
// are rigid phase successors.  A 1.5x time dilation lowers the boundary speed
// from 0.742/0.751 m/s to about 0.495/0.501 m/s and scales every stored linear
// and angular reference velocity by the same factor.
import { quatMulXyzw, quatNormalize, quatRotateOne, yawQuat } from './math.js';
import { transformTeacherReference } from './teacher_reference.js';

const DIM = 747;
// Reference layout: root linear/angular, 29 joint velocities, object
// linear/angular, then 39 body linear/angular velocities.  Joint positions
// are time-interpolated at 13:42, so their corresponding 42:71 velocities
// must be time-dilated with every other derivative channel.
const REFERENCE_VELOCITY_RANGES = Object.freeze([[7, 13], [42, 71], [78, 84], [357, 591]]);
const QUATERNION_OFFSETS = Object.freeze([3, 74, ...Array.from({ length: 39 }, (_, i) => 201 + 4 * i)]);
const CONTACT_START = 591, CONTACT_END = 630;
const wrap = value => Math.atan2(Math.sin(value), Math.cos(value));
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));

function finiteFrame(frame, label) {
  if (!frame || frame.length !== DIM || !Array.from(frame).every(Number.isFinite)) {
    throw new Error(`${label} must contain ${DIM} finite values`);
  }
}
function slerp(a, b, t) {
  let right = Array.from(b), dot = a.reduce((sum, value, i) => sum + value * right[i], 0);
  if (dot < 0) { right = right.map(value => -value); dot = -dot; }
  dot = Math.max(-1, Math.min(1, dot));
  if (dot > .9995) return quatNormalize(a.map((value, i) => value + t * (right[i] - value)));
  const angle = Math.acos(dot), scale = Math.sin(angle);
  return a.map((value, i) => (Math.sin((1 - t) * angle) * value + Math.sin(t * angle) * right[i]) / scale);
}
function alignFrame(frame, sourceAnchor, targetAnchor) {
  const angle = wrap(yaw(targetAnchor.slice(3, 7)) - yaw(sourceAnchor.slice(3, 7)));
  const rotated = quatRotateOne(yawQuat(angle), sourceAnchor.slice(0, 3));
  return transformTeacherReference(frame, { yawRadians: angle,
    translation: [targetAnchor[0] - rotated[0], targetAnchor[1] - rotated[1], 0] });
}
function buildUnscaled(raw, count, { periodFirst, periodLast }) {
  const out = raw.slice(0, Math.min(raw.length, periodLast + 1)).map(frame => Float32Array.from(frame));
  const periodControls = periodLast - periodFirst;
  if (periodControls < 30) throw new Error('Periodic source must retain at least 30 phase controls');
  while (out.length < count) {
    const targetAnchor = out.at(-1), sourceAnchor = raw[periodFirst];
    for (let source = periodFirst + 1; source <= periodLast && out.length < count; source++) {
      out.push(alignFrame(raw[source], sourceAnchor, targetAnchor));
    }
  }
  return out;
}
function interpolateFrame(a, b, t, timeScale) {
  const out = Float32Array.from(a, (value, i) => value + t * (b[i] - value));
  for (const offset of QUATERNION_OFFSETS) out.set(slerp(Array.from(a.slice(offset, offset + 4)),
    Array.from(b.slice(offset, offset + 4)), t), offset);
  for (let i = CONTACT_START; i < CONTACT_END; i++) out[i] = t < .5 ? a[i] : b[i];
  for (const [start, end] of REFERENCE_VELOCITY_RANGES) for (let i = start; i < end; i++) out[i] /= timeScale;
  return out;
}
function timeDilate(unscaled, count, timeScale) {
  return Array.from({ length: count }, (_, index) => {
    const source = index / timeScale, left = Math.floor(source), alpha = source - left;
    if (left + 1 >= unscaled.length) throw new Error('Periodic source lacks requested lookahead');
    return interpolateFrame(unscaled[left], unscaled[left + 1], alpha, timeScale);
  });
}
function curveFrames(straight, sign, yawRateRadS) {
  if (!sign) return straight.map(Float32Array.from);
  const origin = straight[0].slice(0, 3), initialYaw = yaw(straight[0].slice(3, 7));
  const c0 = Math.cos(initialYaw), s0 = Math.sin(initialYaw), result = [];
  for (let i = 0; i < straight.length; i++) {
    const source = straight[i], dx = source[0] - origin[0], dy = source[1] - origin[1];
    const forward = c0 * dx + s0 * dy, lateral = -s0 * dx + c0 * dy;
    const theta = sign * yawRateRadS * i / 60;
    // Bounded intent scheduling: rotate the complete reference about its live
    // entry while preserving its measured translational displacement.  This is
    // a reference composition candidate, not a claim of a recorded strafe.
    const rotatedRoot = quatRotateOne(yawQuat(theta), source.slice(0, 3));
    const desired = [origin[0] + Math.cos(theta) * (c0 * forward - s0 * lateral)
      - Math.sin(theta) * (s0 * forward + c0 * lateral),
    origin[1] + Math.sin(theta) * (c0 * forward - s0 * lateral)
      + Math.cos(theta) * (s0 * forward + c0 * lateral), source[2]];
    const frame = transformTeacherReference(source, { yawRadians: theta,
      translation: desired.map((value, axis) => value - rotatedRoot[axis]) });
    // p'(t) = origin + R(theta(t)) (p(t) - origin), therefore its
    // derivative is R v + omega x (p' - origin).  The rigid transform above
    // already supplies R v; add the moving-frame term to every world-space
    // position/linear-velocity pair.  Without it, curved reference positions
    // and stored velocities describe different trajectories.
    const omega = sign * yawRateRadS;
    const addRotatingVelocity = (positionOffset, velocityOffset) => {
      const rx = frame[positionOffset] - origin[0], ry = frame[positionOffset + 1] - origin[1];
      frame[velocityOffset] += -omega * ry;
      frame[velocityOffset + 1] += omega * rx;
    };
    addRotatingVelocity(0, 7);
    addRotatingVelocity(71, 78);
    for (let body = 0; body < 39; body++) addRotatingVelocity(84 + 3 * body, 357 + 3 * body);
    result.push(frame);
  }
  // Progressive yaw contributes to world angular velocity.  Existing vectors
  // were already rotated by transformTeacherReference.
  for (const frame of result) {
    frame[12] += sign * yawRateRadS;
    frame[83] += sign * yawRateRadS;
    for (let body = 0; body < 39; body++) frame[474 + 3 * body + 2] += sign * yawRateRadS;
  }
  return result;
}

export function buildPeriodicTeacherSkills(rawSkill, { periodFirst = 184, periodLast = 263,
  timeScale = 1.5, sourceFrames = 436, yawRateRadS = .5, persistent = false } = {}) {
  if (!rawSkill || rawSkill.locomotionOnly !== true || !Array.isArray(rawSkill.frames)
      || rawSkill.frames.length <= periodLast + 1 || rawSkill.sourceFrames <= periodLast
      || !Number.isFinite(timeScale) || timeScale < 1 || !Number.isInteger(sourceFrames) || sourceFrames < 60
      || !Number.isFinite(yawRateRadS) || yawRateRadS <= 0 || yawRateRadS > .5
      || typeof persistent !== 'boolean') {
    throw new Error('A complete BONES periodic source and bounded timing are required');
  }
  const firstSuccessorControl = Math.ceil(periodLast * timeScale) + 1;
  if (periodFirst >= periodLast || firstSuccessorControl >= sourceFrames) {
    throw new Error('Periodic source bank must include entry, one reviewed period, and its successor');
  }
  rawSkill.frames.forEach((frame, index) => finiteFrame(frame, `Periodic source frame ${index}`));
  const periodControls = (periodLast - periodFirst) * timeScale;
  let loopControls = null, loopParts = null;
  if (persistent) {
    for (let cycles = 1; cycles <= 8; cycles++) {
      const candidate = periodControls * cycles;
      if (Math.abs(candidate - Math.round(candidate)) <= 1e-9) {
        loopControls = Math.round(candidate);
        loopParts = cycles === 1 ? [loopControls] : [Math.ceil(periodControls), loopControls - Math.ceil(periodControls)];
        break;
      }
    }
    if (!loopControls || loopParts.some(value => !Number.isInteger(value) || value < 30)) {
      throw new Error('Persistent periodic timing requires bounded integral phase segments');
    }
  }
  const persistentEnd = persistent ? Math.ceil(periodFirst * timeScale) + loopControls + 16 : 0;
  const totalFrames = Math.max(sourceFrames + 16, persistentEnd);
  const sourceNeeded = Math.ceil((totalFrames - 1) / timeScale) + 2;
  const unscaled = buildUnscaled(rawSkill.frames, sourceNeeded, { periodFirst, periodLast });
  const straight = timeDilate(unscaled, totalFrames, timeScale);
  const metadata = Object.freeze({ sourceName: rawSkill.name, sourcePhase: [periodFirst, periodLast],
    timeScale, sourceFrames, totalFrames, firstPeriodicControl: Math.ceil(periodFirst * timeScale),
    firstSuccessorControl,
    unscaledBoundarySpeedMps: [Math.hypot(...rawSkill.frames[periodFirst].slice(7, 9)),
      Math.hypot(...rawSkill.frames[periodLast].slice(7, 9))],
    scaledBoundarySpeedMps: [Math.hypot(...straight[Math.round(periodFirst * timeScale)].slice(7, 9)),
      Math.hypot(...straight[Math.round(periodLast * timeScale)].slice(7, 9))], yawRateRadS });
  const curves = { forward: straight.map(frame => Float32Array.from(frame)),
    turnLeft: curveFrames(straight, 1, yawRateRadS), turnRight: curveFrames(straight, -1, yawRateRadS) };
  const label = { forward: 'forward', turnLeft: 'forward+left-yaw', turnRight: 'forward+right-yaw' };
  const sign = { forward: 0, turnLeft: 1, turnRight: -1 };
  const skill = (kind, frames, count, extra = {}) => ({ ...rawSkill,
    name: `${rawSkill.name} lower-speed periodic ${label[kind]}${extra.role ? ` ${extra.role}` : ''}`,
    sourceFrames: count, frames, periodicTeacher: Object.freeze({ ...metadata, yawSign: sign[kind], ...extra }) });
  const ordinary = Object.fromEntries(Object.keys(curves).map(kind =>
    [kind, skill(kind, curves[kind].slice(0, sourceFrames + 16), sourceFrames)]));
  let persistentPlan = null;
  if (persistent) {
    // All segments use the same command-to-world rotation as the unsplit
    // candidate. Computing direction from each truncated endpoint rotates the
    // entry from control zero and independently straightens every loop piece.
    const alignmentDirection = Object.fromEntries(Object.keys(curves).map(kind => {
      const first = curves[kind][0], last = curves[kind][sourceFrames - 1];
      return [kind, Math.atan2(last[1] - first[1], last[0] - first[0])];
    }));
    const entryControls = metadata.firstPeriodicControl;
    const entry = Object.fromEntries(Object.keys(curves).map(kind => [kind,
      skill(kind, curves[kind].slice(0, entryControls + 16), entryControls,
        { role: 'persistent_entry', segmentIndex: 0, continuous: true,
          alignmentTravelDirectionRad: alignmentDirection[kind] })]));
    let anchor = entryControls;
    const loops = loopParts.map((count, index) => {
      const segment = Object.fromEntries(Object.keys(curves).map(kind => [kind,
        skill(kind, curves[kind].slice(anchor, anchor + count + 16), count,
          { role: 'persistent_loop', segmentIndex: index + 1, continuous: true,
            sourceAnchorControl: anchor, loopControls,
            alignmentTravelDirectionRad: alignmentDirection[kind] })]));
      anchor += count;
      return Object.freeze(segment);
    });
    persistentPlan = Object.freeze({ entry: Object.freeze(entry), loops: Object.freeze(loops),
      metadata: Object.freeze({ entryControls, loopControls, loopParts: Object.freeze([...loopParts]),
        switchPolicy: 'latest_eligible_intent_at_complete_phase_segment' }) });
  }
  return Object.freeze({ ...ordinary, metadata, persistent: persistentPlan });
}

// Conservative whole-body XY hull generated from the exact candidate frames.
// It feeds the unchanged restricted-reference obstacle rule and reserve.
export function buildPeriodicMotionSweep(skill, name = 'periodic_teacher_candidate') {
  if (!skill?.periodicTeacher || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames) {
    throw new Error('A prepared periodic teacher skill is required');
  }
  const hull = points => {
    const sorted = points.map(point => [point[0], point[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const unique = sorted.filter((point, i) => !i || point[0] !== sorted[i - 1][0] || point[1] !== sorted[i - 1][1]);
    const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const half = rows => { const out = []; for (const point of rows) {
      while (out.length > 1 && cross(out.at(-2), out.at(-1), point) <= 0) out.pop(); out.push(point);
    } return out; };
    const lower = half(unique), upper = half([...unique].reverse()); lower.pop(); upper.pop(); return [...lower, ...upper];
  };
  const localBodies = (frames, anchor) => {
    const heading = yaw(anchor.slice(3, 7)), c = Math.cos(heading), s = Math.sin(heading);
    return frames.flatMap(frame => Array.from({ length: 39 }, (_, body) => {
      const dx = frame[84 + body * 3] - anchor[0], dy = frame[85 + body * 3] - anchor[1];
      return [c * dx + s * dy, -s * dx + c * dy];
    }));
  };
  const source = skill.frames.slice(0, skill.sourceFrames), first = source[0], last = source.at(-1);
  return { name, sourceFrames: skill.sourceFrames, initialRootPose: Array.from(first.slice(0, 7)),
    terminalRootPose: Array.from(last.slice(0, 7)), wholeBodyXYHull: hull(localBodies(source, first)),
    terminalBodyXYHull: hull(localBodies([last], last)) };
}
