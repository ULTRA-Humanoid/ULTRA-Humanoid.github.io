// Experimental planar goal adaptation. Mirrors common_translation_warp in
// scripts/eval_teacher_carry_goal_warp.py; not connected to arbitrary UI goals.
import { TEACHER_REFERENCE_DIM } from './teacher_obs.js';
import { transformTeacherReference } from './teacher_reference.js';
import { quatMulXyzw, quatRotateOne, yawQuat } from './math.js';

const headingOf = q => { const f = quatRotateOne(q, [1, 0, 0]); return Math.atan2(f[1], f[0]); };
const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));

/** Rotate the reference object quaternion channels (74:78, xyzw) of every
 * frame by the multiple of (360 / symmetryTurns) degrees about the object's
 * own vertical axis that brings the reference object heading at
 * `referenceIndex` closest to the live object heading. This is a body-frame
 * symmetry operation, so the object's tilt (its body z axis in the world) is
 * unchanged and positions, velocities, body channels and the heading-relative
 * IG are untouched. `yawSymmetry` ({turns, stepRad}, object_profiles.js warp.yawSymmetry) is the object's measured yaw symmetry
 * and must come from its geometry, never from a guess: the default (null = 4
 * quarter turns) is valid only for a box whose horizontal footprint is
 * square - largebox_080_080_080 measures 0.377 x 0.367 m in XY (about 1 cm
 * apart) and 0.326 m tall. A standing suitcase (handle/wheels; Phase B B5) or a
 * 2:1 crate mirrors onto itself only under a half turn (symmetryTurns 2);
 * with 1 the reference is never rotated. Turns are only ever about the
 * vertical axis, never about X or Y. Default callers get the pre-B5 result
 * (same turn set, same step, same field values).
 */
export function snapReferenceObjectYaw(referenceFrames, objectQuaternion, { referenceIndex = 0, yawSymmetry = null } = {}) {
  if (!Array.isArray(referenceFrames) || referenceFrames.length === 0
      || referenceFrames.some(row => row.length !== TEACHER_REFERENCE_DIM)) throw new Error('Expected full747 reference frames');
  if (!objectQuaternion || objectQuaternion.length !== 4 || !Array.from(objectQuaternion).every(Number.isFinite)
      || Math.abs(Math.hypot(...objectQuaternion) - 1) > 1e-3) throw new Error('Object yaw snap requires a finite unit xyzw live object quaternion');
  if (!Number.isInteger(referenceIndex) || referenceIndex < 0 || referenceIndex >= referenceFrames.length) throw new Error('Snap reference index must lie inside the reference');
  // Body-frame yaw symmetry of the reference object (B4/B5 shared option). null => the v5 quarter-turn loop for the square largebox;
  // a profile symmetry {turns, stepRad} (object_profiles.js) restricts a standing suitcase / 2:1 crate to half turns. The default
  // stepRad Math.PI / 2 is exact, so `turns * stepRad` reproduces the former `turns * Math.PI / 2` bit for bit.
  // A provided symmetry object must declare BOTH fields: `{stepRad: x}` alone must not silently become 4 turns (fail closed).
  const symmetryTurns = yawSymmetry === null || yawSymmetry === undefined ? 4 : yawSymmetry.turns;
  const stepRad = yawSymmetry === null || yawSymmetry === undefined ? Math.PI / 2 : yawSymmetry.stepRad;
  if (!Number.isInteger(symmetryTurns) || symmetryTurns < 1 || symmetryTurns > 8) throw new Error('Object yaw symmetry must be a whole turn count between 1 and 8');
  if (!Number.isFinite(stepRad) || !(stepRad > 0)) throw new Error('Object yaw symmetry step must be a positive angle');
  const liveYaw = headingOf(objectQuaternion), anchor = Array.from(referenceFrames[referenceIndex].slice(74, 78));
  // `quarterTurns` keeps its recorded name (objectYawSnap contract) but counts
  // applied symmetry turns of `stepRad`; it is a quarter-turn count only when
  // symmetryTurns is 4.
  let quarterTurns = 0, residualYawRad = Infinity;
  for (let turns = 0; turns < symmetryTurns; turns++) {
    const residual = wrapAngle(liveYaw - headingOf(quatMulXyzw(anchor, yawQuat(turns * stepRad))));
    if (Math.abs(residual) < Math.abs(residualYawRad)) { quarterTurns = turns; residualYawRad = residual; }
  }
  const symmetry = yawQuat(quarterTurns * stepRad);
  const frames = referenceFrames.map(row => {
    const frame = Float32Array.from(row);
    if (quarterTurns !== 0) frame.set(quatMulXyzw(Array.from(row.slice(74, 78)), symmetry), 74);
    return frame;
  });
  return { frames, quarterTurns, symmetryTurns, stepRad, liveYawRad: liveYaw, referenceYawRad: headingOf(anchor), residualYawRad,
    snappedYawRad: headingOf(frames[referenceIndex].slice(74, 78)) };
}

/**
 * Translate the complete humanoid/object reference together through a quintic
 * profile, adding its analytic world velocity to every linear-velocity field.
 * Relative geometry and heading-frame IG remain unchanged. This pure math
 * helper does not establish that a chosen displacement is physically feasible.
 */
export function commonTranslationWarp(referenceFrames, displacement, startFrame, endFrame, controlHz = 60) {
  if (!Array.isArray(referenceFrames) || referenceFrames.length === 0
      || referenceFrames.some(row => row.length !== TEACHER_REFERENCE_DIM || !Array.from(row).every(Number.isFinite))) {
    throw new Error('Expected finite full747 reference frames');
  }
  if (!Number.isInteger(startFrame) || !Number.isInteger(endFrame)
      || startFrame < 0 || startFrame >= endFrame || endFrame >= referenceFrames.length) {
    throw new Error('Warp interval must lie inside the reference');
  }
  if (!displacement || displacement.length !== 3 || !Array.from(displacement).every(Number.isFinite)
      || displacement[2] !== 0 || !Number.isFinite(controlHz) || controlHz <= 0) {
    throw new Error('Warp requires finite planar displacement and positive control frequency');
  }
  const frames = [], offsets = [], velocities = [];
  const span = endFrame - startFrame;
  for (let index = 0; index < referenceFrames.length; index++) {
    const u = Math.max(0, Math.min(1, (index - startFrame) / span));
    const positionWeight = u ** 3 * (10 + u * (-15 + 6 * u));
    const velocityWeight = 30 * u ** 2 * (1 - u) ** 2 * controlHz / span;
    const offset = Float32Array.from(displacement, v => positionWeight * v);
    const velocity = Float32Array.from(displacement, v => velocityWeight * v);
    const frame = Float32Array.from(referenceFrames[index]);
    const add = (channel, vector) => {
      for (let axis = 0; axis < 3; axis++) frame[channel + axis] += vector[axis];
    };
    add(0, offset); add(71, offset); add(7, velocity); add(78, velocity);
    for (let body = 0; body < 39; body++) { add(84 + body * 3, offset); add(357 + body * 3, velocity); }
    frames.push(frame); offsets.push(offset); velocities.push(velocity);
  }
  return { frames, offsets, velocities };
}

/** Small correction toward a requested object XY goal. Keep the final user
 * goal outside this helper, because a bounded offset may not reach it. The
 * 0.25m default matches the small offsets evaluated for the carry reference.
 */
export function boundedObjectGoalOffset(finalReferenceFrame, goalWorld, maxDistance = 0.25) {
  if (!finalReferenceFrame || finalReferenceFrame.length !== TEACHER_REFERENCE_DIM
      || !Array.from(finalReferenceFrame).every(Number.isFinite)
      || !goalWorld || goalWorld.length !== 3 || !Array.from(goalWorld).every(Number.isFinite)
      || !Number.isFinite(maxDistance) || maxDistance <= 0) throw new Error('Invalid planar reference goal');
  const dx = goalWorld[0] - finalReferenceFrame[71], dy = goalWorld[1] - finalReferenceFrame[72];
  const requestedDistance = Math.hypot(dx, dy), scale = requestedDistance > 0 ? Math.min(1, maxDistance / requestedDistance) : 0;
  return { displacement: [dx * scale, dy * scale, 0], requestedDistance,
    remainingDistance: Math.max(0, requestedDistance - maxDistance) };
}

/** Orient an entire carry reference toward a planar object destination, then
 * correct its travel distance by a bounded common translation. The requested
 * goal is retained separately from the attainable reference endpoint. This
 * does not establish that the initial approach heading or path is reachable.
 */
export function planCarryToGoal(referenceFrames, sourceFrames, objectPosition, goalWorld,
  { startFrame, endFrame, maxCorrection = 0.25, controlHz = 60, snapObjectYaw = false, objectQuaternion = null,
    yawSymmetry = null } = {}) {
  if (!Array.isArray(referenceFrames) || !Number.isInteger(sourceFrames) || sourceFrames < 2
      || referenceFrames.length < sourceFrames + 16
      || !Number.isInteger(endFrame) || endFrame >= sourceFrames
      || !objectPosition || objectPosition.length !== 3 || !Array.from(objectPosition).every(Number.isFinite)
      || !goalWorld || goalWorld.length !== 3 || !Array.from(goalWorld).every(Number.isFinite)) {
    throw new Error('A complete carry reference and finite object positions are required');
  }
  const first = referenceFrames[0], last = referenceFrames[sourceFrames - 1];
  if (first.length !== TEACHER_REFERENCE_DIM || last.length !== TEACHER_REFERENCE_DIM) throw new Error('Expected full747 reference frames');
  const sourceDelta = [last[71] - first[71], last[72] - first[72]];
  const targetDelta = [goalWorld[0] - objectPosition[0], goalWorld[1] - objectPosition[1]];
  if (Math.hypot(...sourceDelta) < 1e-4 || Math.hypot(...targetDelta) < 1e-4) {
    throw new Error('Carry direction requires nonzero source and goal displacement');
  }
  const yawRadians = Math.atan2(targetDelta[1], targetDelta[0]) - Math.atan2(sourceDelta[1], sourceDelta[0]);
  const c = Math.cos(yawRadians), s = Math.sin(yawRadians);
  const transform = { yawRadians, translation: [objectPosition[0] - (c * first[71] - s * first[72]),
    objectPosition[1] - (s * first[71] + c * first[72]), 0] };
  const aligned = referenceFrames.map(frame => transformTeacherReference(frame, transform));
  const correction = boundedObjectGoalOffset(aligned[sourceFrames - 1], goalWorld, maxCorrection);
  const warped = commonTranslationWarp(aligned, correction.displacement, startFrame, endFrame, controlHz);
  // Optional (refObjYawSnap): the live object may sit one yaw-symmetry turn
  // from the clip's object; for the square largebox that is a quarter turn
  // (yawSymmetry null = the v5 default), for a standing suitcase a half turn
  // ({turns: 2, stepRad: PI}). The symmetry is the selected object's measured one.
  const snap = snapObjectYaw ? snapReferenceObjectYaw(warped.frames, objectQuaternion, { yawSymmetry }) : null;
  const frames = snap ? snap.frames : warped.frames;
  const final = frames[sourceFrames - 1];
  return { frames, transform, requestedGoalWorld: Array.from(goalWorld),
    referenceGoalWorld: Array.from(final.slice(71, 74)),
    remainingDistance: Math.hypot(goalWorld[0] - final[71], goalWorld[1] - final[72]),
    correctionWorld: correction.displacement,
    approachGoalWorld: [frames[0][0], frames[0][1], 0],
    objectYawSnap: snap ? { quarterTurns: snap.quarterTurns, symmetryTurns: snap.symmetryTurns, liveYawRad: snap.liveYawRad,
      referenceYawRad: snap.referenceYawRad, snappedYawRad: snap.snappedYawRad, residualYawRad: snap.residualYawRad } : null };
}
