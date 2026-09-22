// camera_modes.js — fixed cameras for the demo canvas.
//
// Coordinates are three.js (y-up). The MuJoCo scene root is rotated -π/2
// about X, so three.js (x, y, z) = MuJoCo (x, z, -y).
//
//   wide  (public default) 60° fov from (1.5, 10, 2) toward (1.5, 0, 0):
//         a high three-quarter view centred on the working floor. Every one
//         of the 100 historical panel destinations projects inside the
//         720×480 canvas with ≥ 66 px margin, and the orbit distance
//         (10.2 m) stays below OrbitControls' maxDistance (12 m).
//   orbit (legacy)         40° fov from (3.2, 1.7, 3.2) toward (0, 0.85, 0):
//         the original close view. 40 of the 100 panel destinations project
//         outside the canvas and one (H085) lies behind the camera.
//   player (release)       low third-person view. It changes rendering only;
//         automation remains picker-free and physics is camera-independent.
//
// Selected by the `cameraMode` URL parameter (`wide` when absent or unknown).
export const CAMERA_MODES = Object.freeze({
  wide: Object.freeze({ fov: 60, position: Object.freeze([1.5, 10, 2]), target: Object.freeze([1.5, 0, 0]) }),
  orbit: Object.freeze({ fov: 40, position: Object.freeze([3.2, 1.7, 3.2]), target: Object.freeze([0, 0.85, 0]) }),
  player: Object.freeze({ fov: 55, position: Object.freeze([-4.2, 2.15, 2.8]), target: Object.freeze([0.45, 0.82, 0]) }),
});
export const DEFAULT_CAMERA_MODE = 'wide';

export function readCameraMode(params) {
  const value = params?.get?.('cameraMode');
  return typeof value === 'string' && Object.hasOwn(CAMERA_MODES, value) ? value : DEFAULT_CAMERA_MODE;
}

/** Apply a mode's literals to an existing camera (+ optional OrbitControls target). */
export function applyCameraMode(camera, controls, mode = DEFAULT_CAMERA_MODE) {
  const spec = CAMERA_MODES[mode] ?? CAMERA_MODES[DEFAULT_CAMERA_MODE];
  camera.fov = spec.fov;
  camera.position.set(...spec.position);
  camera.updateProjectionMatrix();
  if (controls) controls.target.set(...spec.target);
  else camera.lookAt(...spec.target);
  return spec;
}
