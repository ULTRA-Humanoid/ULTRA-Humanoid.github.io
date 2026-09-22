// Phase B B4 v6e (2026-09-21, coordinator GO after pnat3 section 16.5): object-aware EXIT STAND-OFF. PURE GEOMETRY, no controller state.
// The recorded box-exit programme (teacher_box_exit_controller.js: hold 60 / retreat 199 / settle 180) was calibrated on the largebox
// (0.326 m tall at rest). An object that stands TALLER than that next to the human who has just set it down (suitcase standing 0.425 m by
// design; a plasticbox tipped onto an end face 0.444 m) is already inside the hold pose's contact envelope, and the exit preview refuses
// (`preview_contact`, pnat3 H079/H074: left_hip_yaw_link vs the crate). Rule: translate the exit hold/retreat reference AWAY from the object,
// along the horizontal object->root direction, by max(0, objectStandingHalfExtent - largeboxRestHalfHeight), where objectStandingHalfExtent =
// half of the object's mesh extent along whichever BODY axis is closest to world-up on the LIVE quaternion (rest.footprintWorld / footprint
// facts + live up-axis). The largebox is the calibration object: its offset is exactly 0 by definition (never a computed near-zero).
// Consumed by object_class_routing.js exitStandOff (OFF => {} so TeacherBoxExitController.start() sees no option: byte-identical v5 path).
import { OBJECT_PROFILES } from './object_profiles.js';

/** Half the largebox rest height (0.3263 / 2): the exit programme's calibration. */
export const LARGEBOX_REST_HALF_HEIGHT_M = OBJECT_PROFILES.largebox.rest.footprintWorld.z / 2;
const AXES = Object.freeze(['x', 'y', 'z']);
const unitXyzw = q => Array.isArray(q) || ArrayBuffer.isView(q) ? q.length === 4 && Array.from(q).every(Number.isFinite) && Math.abs(Math.hypot(...q) - 1) <= 1e-3 : false;
const finite3 = v => (Array.isArray(v) || ArrayBuffer.isView(v)) && v.length === 3 && Array.from(v).every(Number.isFinite);

/** World-z component of each BODY axis for an xyzw quaternion (third row of the rotation matrix). */
export function bodyAxesWorldZ(quatXyzw) {
  if (!unitXyzw(quatXyzw)) throw new Error('Exit stand-off requires a finite unit xyzw object quaternion');
  const [x, y, z, w] = quatXyzw;
  return { x: 2 * (x * z - w * y), y: 2 * (y * z + w * x), z: 1 - 2 * (x * x + y * y) };
}
/** The body axis closest to world-up (largest |world z|) and its world-z. */
export function verticalBodyAxis(quatXyzw) {
  const wz = bodyAxesWorldZ(quatXyzw);
  const axis = AXES.reduce((best, a) => Math.abs(wz[a]) > Math.abs(wz[best]) ? a : best, 'x');
  return { axis, worldZ: wz[axis] };
}
/** Half of the object's world-vertical extent for its live pose: mesh extent along the vertical body axis / 2 (profile.footprint = mesh bbox). */
export function standingHalfExtentM(profile, quatXyzw) {
  if (!profile?.footprint || !AXES.every(a => Number.isFinite(profile.footprint[a]) && profile.footprint[a] > 0)) throw new Error('Exit stand-off requires a profile with a positive mesh footprint');
  const { axis } = verticalBodyAxis(quatXyzw);
  return profile.footprint[axis] / 2;
}
/** Stand-off magnitude (m): 0 for the largebox by definition; else max(0, standingHalfExtent - largebox rest half height). */
export function exitStandOffM(profile, quatXyzw) {
  if (profile?.objectClass === 'largebox') { verticalBodyAxis(quatXyzw); return 0; }   // still validates the quaternion (fail closed on garbage)
  return Math.max(0, standingHalfExtentM(profile, quatXyzw) - LARGEBOX_REST_HALF_HEIGHT_M);
}
/** World stand-off vector [dx, dy, 0] pointing from the object to the root (horizontal), or null when the magnitude is 0.
 *  Throws when the object and root are horizontally coincident (< 1 mm): no direction => never a silent zero. */
export function exitStandOffWorld(profile, objectQuatXyzw, objectPosWorld, rootPosWorld) {
  const magnitude = exitStandOffM(profile, objectQuatXyzw);
  if (!finite3(objectPosWorld) || !finite3(rootPosWorld)) throw new Error('Exit stand-off requires finite object and root positions');
  if (magnitude === 0) return null;
  const dx = rootPosWorld[0] - objectPosWorld[0], dy = rootPosWorld[1] - objectPosWorld[1], d = Math.hypot(dx, dy);
  if (d < 1e-3) throw new Error('Exit stand-off direction undefined: root and object are horizontally coincident');
  return Object.freeze({ standOffWorld: Object.freeze([dx / d * magnitude, dy / d * magnitude, 0]), magnitudeM: magnitude,
    standingHalfExtentM: standingHalfExtentM(profile, objectQuatXyzw), verticalBodyAxis: verticalBodyAxis(objectQuatXyzw).axis,
    largeboxRestHalfHeightM: LARGEBOX_REST_HALF_HEIGHT_M });
}
