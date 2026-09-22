// Experimental rigid alignment of teacher references to a live scene.
// A planar transform preserves floor height, DOF motion and local geometry.
import { quatMulXyzw, quatRotateOne, yawQuat } from './math.js';
import { TEACHER_REFERENCE_DIM } from './teacher_obs.js';

function finite(values, length, label) {
  if (!values || values.length !== length || !Array.from(values).every(Number.isFinite)) {
    throw new Error(`${label} requires ${length} finite values`);
  }
}
function heading(quaternion) {
  const forward = quatRotateOne(quaternion, [1, 0, 0]);
  return Math.atan2(forward[1], forward[0]);
}

/** Align the reference object's initial XY and heading to a live object. */
export function objectAlignedReferenceTransform(referenceFrame, objectPosition, objectQuaternion) {
  finite(referenceFrame, TEACHER_REFERENCE_DIM, 'Teacher reference');
  finite(objectPosition, 3, 'Object position');
  finite(objectQuaternion, 4, 'Object quaternion');
  const yaw = heading(objectQuaternion) - heading(referenceFrame.slice(74, 78));
  const rotation = yawQuat(yaw);
  const position = quatRotateOne(rotation, referenceFrame.slice(71, 74));
  return {
    yawRadians: yaw,
    translation: [objectPosition[0] - position[0], objectPosition[1] - position[1], 0],
  };
}

/** Anchor object XY while aligning the reference root to an approach heading.
 * The full reference rotates together. The physical object's rotation remains
 * unchanged, so its reference rotation mismatch must be measured by the caller.
 */
export function rootHeadingAlignedReferenceTransform(referenceFrame, objectPosition, desiredRootHeading) {
  finite(referenceFrame, TEACHER_REFERENCE_DIM, 'Teacher reference');
  finite(objectPosition, 3, 'Object position');
  if (!Number.isFinite(desiredRootHeading)) throw new Error('Root heading must be finite');
  const yaw = desiredRootHeading - heading(referenceFrame.slice(3, 7));
  const position = quatRotateOne(yawQuat(yaw), referenceFrame.slice(71, 74));
  return { yawRadians: yaw, translation: [objectPosition[0] - position[0], objectPosition[1] - position[1], 0] };
}

/**
 * Transform one full747 reference frame. World positions/quaternions and
 * linear/angular velocities rotate together. Stored IG is heading-relative,
 * so it and joint/contact channels remain unchanged. This changes references
 * only; it never teleports the simulated character or object.
 */
export function transformTeacherReference(referenceFrame, { yawRadians, translation }) {
  finite(referenceFrame, TEACHER_REFERENCE_DIM, 'Teacher reference');
  finite(translation, 3, 'Reference translation');
  if (!Number.isFinite(yawRadians) || translation[2] !== 0) {
    throw new Error('Teacher planar alignment requires finite yaw and zero vertical translation');
  }
  const rotation = yawQuat(yawRadians), out = Float32Array.from(referenceFrame);
  const vector = (offset, isPosition = false) => {
    const value = quatRotateOne(rotation, referenceFrame.slice(offset, offset + 3));
    out.set(value.map((v, i) => v + (isPosition ? translation[i] : 0)), offset);
  };
  const quaternion = offset => out.set(quatMulXyzw(rotation, referenceFrame.slice(offset, offset + 4)), offset);
  vector(0, true); quaternion(3); vector(7); vector(10);
  vector(71, true); quaternion(74); vector(78); vector(81);
  for (let body = 0; body < 39; body++) {
    vector(84 + 3 * body, true); quaternion(201 + 4 * body);
    vector(357 + 3 * body); vector(474 + 3 * body);
  }
  return out;
}
