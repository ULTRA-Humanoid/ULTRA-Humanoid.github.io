// obs_builder.js — build 1422D student obs from MuJoCo state + UserState.
// Mirrors intermimic/sim2sim_vae_interactive.py:
//   - synthesize_new_cmd
//   - compute_obj_points_heading_frame
//   - build_mask_block
//
// CRITICAL: keep this in lockstep with the Python reference. Any change to
// obs layout, NEW_CMD synthesis, or mask binding MUST land in both files,
// or the JS deploy will produce out-of-distribution inputs.

import { Mode } from './state.js';
import {
  calcHeadingQuatInv,
  quatRotateOne,
  quatToRot6d,
  yawQuat,
  wxyzToXyzw,
} from './math.js';

// ---------------- Constants (must match training) ---------------- //

export const OBS_DIM = 1422;
export const NEW_CMD_DIM = 13;
export const BODY_DIM = 1012;
export const TASK_DIM = 192;    // 64 points × 3
export const MASK_DIM = 205;    // obj_points(192) + command(13)
export const NUM_POINTS = 64;
export const VAE_DIM = 64;
export const ACTION_DIM = 29;

// G_MAX_NORMALIZER matches G_MAX_NORMALIZER in interactive task class
const G_MAX_NORMALIZER = 240.0;

// WASD body-frame deltas — match Python WASD_* constants
const WASD_FORWARD_M = 1.0;
const WASD_BACKWARD_M = -0.5;
const WASD_STRAFE_M = 0.5;
const WASD_YAW_RAD = 0.5;
const WALK_SPEED_M_PER_S = 1.0;
const FPS = 30.0;

// ---------------- NEW_CMD synthesis ---------------- //

export function synthesizeNewCmd(user, rootPosWorld, rootQuatXyzwWorld, objPosWorld) {
  const out = new Float32Array(NEW_CMD_DIM);
  const headingInv = calcHeadingQuatInv(rootQuatXyzwWorld);
  const mode = user.mode;

  // human_target_pos (3) + human_target_rot (6 rot6d)
  const keepHuman = mode === Mode.LOCO || mode === Mode.HOI_FULL;
  let humanDx = 0, humanDy = 0, humanDyaw = 0;
  if (keepHuman) {
    if (user.humanGoalWorld) {
      const dx = user.humanGoalWorld[0] - rootPosWorld[0];
      const dy = user.humanGoalWorld[1] - rootPosWorld[1];
      const rotated = quatRotateOne(headingInv, [dx, dy, 0.0]);
      humanDx = rotated[0];
      humanDy = rotated[1];
      out[2] = rotated[2];
    } else {
      if (user.w) humanDx += WASD_FORWARD_M;
      if (user.s) humanDx += WASD_BACKWARD_M;
      if (user.a) humanDy += WASD_STRAFE_M;
      if (user.d) humanDy -= WASD_STRAFE_M;
      if (user.q) humanDyaw += WASD_YAW_RAD;
      if (user.e) humanDyaw -= WASD_YAW_RAD;
    }
    out[0] = humanDx;
    out[1] = humanDy;
    const rot6d = quatToRot6d(yawQuat(humanDyaw));
    out[3] = rot6d[0]; out[4] = rot6d[1]; out[5] = rot6d[2];
    out[6] = rot6d[3]; out[7] = rot6d[4]; out[8] = rot6d[5];
  }

  // obj_target_pos (3)
  const keepObj = mode === Mode.HOI_FULL || mode === Mode.HOI_OBJ_ONLY;
  let objDeltaNorm = 0;
  if (keepObj && objPosWorld && user.objGoalWorld) {
    const dx = user.objGoalWorld[0] - objPosWorld[0];
    const dy = user.objGoalWorld[1] - objPosWorld[1];
    const dz = user.objGoalWorld[2] - objPosWorld[2];
    const rotated = quatRotateOne(headingInv, [dx, dy, dz]);
    out[9] = rotated[0];
    out[10] = rotated[1];
    out[11] = rotated[2];
    objDeltaNorm = Math.sqrt(rotated[0]*rotated[0] + rotated[1]*rotated[1] + rotated[2]*rotated[2]);
  }

  // time_to_target (1) — always visible
  const longest = Math.max(
    keepHuman ? Math.abs(humanDx) : 0,
    keepHuman ? Math.abs(humanDy) : 0,
    objDeltaNorm,
  );
  if (longest > 1e-3) {
    const timeFrames = longest / WALK_SPEED_M_PER_S * FPS;
    out[12] = Math.min(Math.max(timeFrames / G_MAX_NORMALIZER, 0), 1);
  } else {
    out[12] = 0.0;
  }
  return out;
}

// ---------------- Point cloud → heading frame ---------------- //

export function computeObjPointsHeadingFrame(
  pointsObjLocal,        // Float32Array, length NUM_POINTS*3
  objPosWorld,            // [x,y,z]
  objQuatXyzwWorld,       // [x,y,z,w]
  rootPosWorld,
  rootQuatXyzwWorld,
) {
  const headingInv = calcHeadingQuatInv(rootQuatXyzwWorld);
  const out = new Float32Array(NUM_POINTS * 3);

  for (let i = 0; i < NUM_POINTS; i++) {
    const px = pointsObjLocal[i*3 + 0];
    const py = pointsObjLocal[i*3 + 1];
    const pz = pointsObjLocal[i*3 + 2];
    // Local → world
    const w = quatRotateOne(objQuatXyzwWorld, [px, py, pz]);
    const wx = w[0] + objPosWorld[0];
    const wy = w[1] + objPosWorld[1];
    const wz = w[2] + objPosWorld[2];
    // World → root translation
    const rx = wx - rootPosWorld[0];
    const ry = wy - rootPosWorld[1];
    const rz = wz - rootPosWorld[2];
    // Rotate to heading frame
    const h = quatRotateOne(headingInv, [rx, ry, rz]);
    out[i*3 + 0] = h[0];
    out[i*3 + 1] = h[1];
    out[i*3 + 2] = h[2];
  }
  return out;
}

// ---------------- Mask block ---------------- //

// Legacy binary mask used by the old WASD-derived synthesis path. Kept
// for fallback when the translator hasn't loaded yet.
export function buildMaskBlock(mode) {
  const out = new Float32Array(MASK_DIM);
  const keepPerception = (mode === Mode.HOI_FULL || mode === Mode.HOI_OBJ_ONLY);
  if (keepPerception) {
    out.fill(1.0, 0, 192);
  }
  const keepHuman = (mode === Mode.LOCO || mode === Mode.HOI_FULL);
  const keepObj = (mode === Mode.HOI_FULL || mode === Mode.HOI_OBJ_ONLY);
  if (keepHuman) {
    out.fill(1.0, 192, 201);   // human_target_pos(3) + human_target_rot(6)
  }
  if (keepObj) {
    out.fill(1.0, 201, 204);   // obj_target_pos(3)
  }
  out[204] = 1.0;              // time_to_target always
  return out;
}

// Float mask block driven by the goal translator's per-channel scalars
// in [0, 1]. Mask layout: 192 obj_point bits + 9 human_target + 3
// obj_target + 1 time_to_target.
export function buildMaskBlockFromTranslator(translatorMask, hasObjPerception) {
  const out = new Float32Array(MASK_DIM);
  // Object-points perception bit — driven by whether an object is
  // actually visible / selected (not by FSM directly).
  if (hasObjPerception && translatorMask.keepObjPoints !== 0) out.fill(1.0, 0, 192);
  // human_target_pos (3) + human_target_rot (6 rot6d)
  out.fill(translatorMask.keepHumanPos, 192, 195);
  out.fill(translatorMask.keepHumanRot, 195, 201);
  // obj_target_pos (3)
  out.fill(translatorMask.keepObjPos, 201, 204);
  // time_to_target (1) — always on in practice; translator may downscale
  out[204] = translatorMask.keepTime;
  return out;
}

// ---------------- NEW_CMD from translator output ---------------- //

// Pack a translator-produced goalSpec into the 13-D NEW_CMD slot.
// Layout matches synthesizeNewCmd exactly:
//   [0:3]   human_target_pos
//   [3:9]   human_target_rot (rot6d)
//   [9:12]  obj_target_pos
//   [12]    time_to_target
export function newCmdFromTranslator(goalSpec) {
  const out = new Float32Array(NEW_CMD_DIM);
  out[0] = goalSpec.humanTargetPos[0];
  out[1] = goalSpec.humanTargetPos[1];
  out[2] = goalSpec.humanTargetPos[2];
  out[3] = goalSpec.humanTargetRot[0];
  out[4] = goalSpec.humanTargetRot[1];
  out[5] = goalSpec.humanTargetRot[2];
  out[6] = goalSpec.humanTargetRot[3];
  out[7] = goalSpec.humanTargetRot[4];
  out[8] = goalSpec.humanTargetRot[5];
  out[9] = goalSpec.objTargetPos[0];
  out[10] = goalSpec.objTargetPos[1];
  out[11] = goalSpec.objTargetPos[2];
  out[12] = goalSpec.timeToTarget;
  return out;
}

// ---------------- Assemble final obs ---------------- //

export function buildObs(
  user,
  rootPosWorld, rootQuatXyzwWorld,
  objPosWorld, objQuatXyzwWorld,
  objPointsObjLocal,    // null when no active object
  bodyObs,              // Float32Array(1012) from proprio path
  translatorResult = null,   // { goalSpec, mask } from GoalTranslator.step (optional)
) {
  // NEW_CMD: prefer translator output, fall back to WASD-derived synthesis
  // (used briefly during DB load, or if the translator is intentionally
  // disabled for A/B comparison).
  const newCmd = translatorResult
      ? newCmdFromTranslator(translatorResult.goalSpec)
      : synthesizeNewCmd(user, rootPosWorld, rootQuatXyzwWorld, objPosWorld);

  let taskObs;
  if (objPosWorld && objQuatXyzwWorld && objPointsObjLocal
      && translatorResult?.mask.keepObjPoints !== 0) {
    taskObs = computeObjPointsHeadingFrame(
      objPointsObjLocal, objPosWorld, objQuatXyzwWorld,
      rootPosWorld, rootQuatXyzwWorld,
    );
  } else {
    taskObs = new Float32Array(TASK_DIM);  // zeros
  }

  // Mask block: float mask from translator (per-channel ramps for
  // smooth transitions), or legacy binary mask for fallback.
  const hasObjPerception = (objPosWorld !== null && objPosWorld !== undefined
                            && objPointsObjLocal !== null);
  const mask = translatorResult
      ? buildMaskBlockFromTranslator(translatorResult.mask, hasObjPerception)
      : buildMaskBlock(user.mode);

  // Concatenate: NEW_CMD | body | task | mask
  const out = new Float32Array(OBS_DIM);
  out.set(newCmd, 0);
  out.set(bodyObs, NEW_CMD_DIM);
  out.set(taskObs, NEW_CMD_DIM + BODY_DIM);
  out.set(mask, NEW_CMD_DIM + BODY_DIM + TASK_DIM);
  return out;
}
