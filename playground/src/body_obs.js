// body_obs.js — JS port of intermimic/utils/body_obs.py.
//
// Computes the 1012D 'body' slice of the interactive student obs (lives at
// obs_buf[13:1025]). Mirrors the math in obs_vae.py:_compute_humanoid_obs for
// the body portion only, decoupled from the full V2 obs builder.
//
// Layout (training):
//     body_obs (1012D) = concat([
//         current_frame (92D),
//         history (10 × 92D = 920D),
//     ])
//
//     current_frame (92D) = concat([
//         root_ang_vel (3D, world frame, via mj_jacBody @ qvel),
//         imu_obs (2D, roll & pitch from pelvis xquat),
//         dof_pos (29D, in G1_JOINT_ORDER),
//         dof_vel (29D, in G1_JOINT_ORDER),
//         last_action (29D, action computed at the PREVIOUS deploy step),
//     ])
//
// On first build() after reset(): all 10 history slots are filled with the
// current frame (mirrors training's `episode_length_buf <= 1` branch).
//
// JS↔Python parity test will live at scripts/test_body_obs_parity_js.py.

import { G1_JOINT_ORDER } from './pd_control.js';

export const HISTORY_LEN = 10;
export const FRAME_DIM = 92;          // 3 + 2 + 29 + 29 + 29
export const BODY_DIM = 1012;         // FRAME_DIM * (1 + HISTORY_LEN)


/**
 * Convert (x, y, z, w) quaternion → (roll, pitch, yaw) in radians.
 * Mirrors intermimic/utils/obs_vae.py:euler_from_quaternion exactly
 * (the IsaacGym/training convention).
 */
export function eulerFromQuatXyzw(x, y, z, w) {
  // roll (x-axis rotation)
  const t0 = 2.0 * (w * x + y * z);
  const t1 = 1.0 - 2.0 * (x * x + y * y);
  const roll = Math.atan2(t0, t1);

  // pitch (y-axis rotation), clamped before asin
  let t2 = 2.0 * (w * y - z * x);
  if (t2 > 1.0) t2 = 1.0;
  else if (t2 < -1.0) t2 = -1.0;
  const pitch = Math.asin(t2);

  // yaw (z-axis rotation)
  const t3 = 2.0 * (w * z + x * y);
  const t4 = 1.0 - 2.0 * (y * y + z * z);
  const yaw = Math.atan2(t3, t4);

  return [roll, pitch, yaw];
}


/**
 * Compute the 1012D body slice from MuJoCo state, maintaining a 10-frame
 * history buffer internally.
 *
 * Usage:
 *     const builder = new BodyObsBuilder(mujoco, model);
 *     builder.reset();
 *     const bodyObs = builder.build(data, lastAction);   // Float32Array(1012)
 *
 * `lastAction` is the 29D action from the previous deploy step (zeros at
 * the very first call after reset).
 */
export class BodyObsBuilder {
  /**
   * @param {object} mujoco   The mujoco-wasm Module.
   * @param {MjModel} model
   * @param {string} pelvisBodyName  Default 'pelvis'.
   */
  constructor(mujoco, model, pelvisBodyName = 'pelvis') {
    this.mujoco = mujoco;
    this.model = model;

    // --- Resolve pelvis body ID by name --------------------------------- //
    const namesBlob = model.names;
    const readName = (offset) => {
      let s = '';
      for (let k = offset; k < namesBlob.length; k++) {
        const c = namesBlob[k];
        if (c === 0) break;
        s += String.fromCharCode(c);
      }
      return s;
    };
    const bodyNameAdr = model.name_bodyadr;
    let pelvisId = -1;
    for (let bid = 0; bid < model.nbody; bid++) {
      if (readName(bodyNameAdr[bid]) === pelvisBodyName) {
        pelvisId = bid;
        break;
      }
    }
    if (pelvisId < 0) {
      throw new Error(`Body '${pelvisBodyName}' not found in MJCF.`);
    }
    this.pelvisId = pelvisId;

    // --- Resolve joint qpos/qvel addresses in G1_JOINT_ORDER ----------- //
    const jntNameAdr = model.name_jntadr;
    const jointNameToId = {};
    for (let jid = 0; jid < model.njnt; jid++) {
      jointNameToId[readName(jntNameAdr[jid])] = jid;
    }
    this.dofQposAddr = new Int32Array(29);
    this.dofQvelAddr = new Int32Array(29);
    for (let i = 0; i < 29; i++) {
      const name = G1_JOINT_ORDER[i];
      const jid = jointNameToId[name];
      if (jid === undefined) {
        throw new Error(`Joint '${name}' not found in MJCF.`);
      }
      this.dofQposAddr[i] = model.jnt_qposadr[jid];
      this.dofQvelAddr[i] = model.jnt_dofadr[jid];
    }

    // --- History buffer (10 × 92) -------------------------------------- //
    this.historyBuf = new Float32Array(HISTORY_LEN * FRAME_DIM);
    this.hasInitialized = false;

    // Native output buffers must live in WASM memory. The bundled binding
    // copies ordinary JS arrays into temporary buffers without copying the
    // Jacobian back, which silently made angular velocity always zero.
    this._jacp = new mujoco.DoubleBuffer(3 * model.nv);
    this._jacr = new mujoco.DoubleBuffer(3 * model.nv);

    // Reusable output buffers
    this._currentFrame = new Float32Array(FRAME_DIM);
    this._bodyObs = new Float32Array(BODY_DIM);
  }

  /** Private-history copy for isolated lookahead: copies only the observation
   *  history and its initialization flag from another builder of the same model. */
  copyHistoryFrom(other) {
    if (!(other instanceof BodyObsBuilder) || other.historyBuf.length !== this.historyBuf.length)
      throw new Error('History copy requires a same-model BodyObsBuilder');
    this.historyBuf.set(other.historyBuf);
    this.hasInitialized = other.hasInitialized;
  }

  /** Force history to re-fill on next build() call. */
  reset() {
    this.historyBuf.fill(0.0);
    this.hasInitialized = false;
  }

  /** Release native scratch buffers when destroying this builder. */
  dispose() {
    this._jacp?.delete();
    this._jacr?.delete();
    this._jacp = this._jacr = null;
  }

  /**
   * Compute the 92D current-frame vector and write it into this._currentFrame.
   * Returns the same Float32Array.
   */
  _computeCurrentFrame(data, lastAction) {
    const frame = this._currentFrame;

    // 1. Match Python exactly, including immediately after mj_step when
    // cached cvel still describes velocities from before integration.
    // Reacquire views because WASM memory growth can invalidate old views.
    if (this._jacp === null) throw new Error('BodyObsBuilder has been disposed');
    this._jacp.GetView().fill(0);
    this._jacr.GetView().fill(0);
    this.mujoco.mj_jacBody(this.model, data, this._jacp, this._jacr, this.pelvisId);
    const jacr = this._jacr.GetView();
    const qvel = data.qvel;
    // B9-PERF: `this.model.nv` is a compiled-model constant behind an Embind int getter; read it once (same value).
    const nv = this.model.nv;
    for (let axis = 0; axis < 3; axis++) {
      let value = 0;
      for (let k = 0; k < nv; k++) value += jacr[axis * nv + k] * qvel[k];
      frame[axis] = value;
    }

    // 2. IMU obs (roll, pitch) from pelvis xquat.
    //    MuJoCo data.xquat is (w, x, y, z); training expects (x, y, z, w).
    const qw = data.xquat[this.pelvisId * 4 + 0];
    const qx = data.xquat[this.pelvisId * 4 + 1];
    const qy = data.xquat[this.pelvisId * 4 + 2];
    const qz = data.xquat[this.pelvisId * 4 + 3];
    const [roll, pitch, _yaw] = eulerFromQuatXyzw(qx, qy, qz, qw);
    frame[3] = roll;
    frame[4] = pitch;

    // 3. dof_pos / dof_vel in G1_JOINT_ORDER.
    const qpos = data.qpos;
    for (let i = 0; i < 29; i++) {
      frame[5 + i] = qpos[this.dofQposAddr[i]];                // dof_pos[i]
      frame[5 + 29 + i] = qvel[this.dofQvelAddr[i]];           // dof_vel[i]
    }

    // 4. last_action (29D), as-is.
    if (lastAction.length !== 29) {
      throw new Error(
        `last_action must be length 29, got ${lastAction.length}`
      );
    }
    for (let i = 0; i < 29; i++) {
      frame[5 + 29 + 29 + i] = lastAction[i];
    }

    return frame;
  }

  /**
   * Compute the 1012D body obs slice. Update ordering mirrors training:
   *   1. Assemble body_obs using the CURRENT history (pre-update).
   *   2. Then update history for next step:
   *      - On the FIRST call after reset(): fill all 10 slots with current
   *        frame (matches training's `episode_length_buf <= 1` branch).
   *      - Otherwise: shift left, append current at the tail.
   *
   * Consequence: at frame 0 after reset, the policy sees
   *   `[current_frame, zeros × 10]`
   * which exactly matches training. From frame 1 onward, the buffer holds
   * the accumulated history.
   *
   * @param {MjData} data
   * @param {Float32Array|number[]} lastAction  29D
   * @returns {Float32Array}  (1012,) — the SAME internal buffer each call;
   *          caller must copy if they need to keep the value beyond next
   *          build() call.
   */
  build(data, lastAction) {
    const currentFrame = this._computeCurrentFrame(data, lastAction);

    // Assemble body_obs = [current_frame, history_buf]
    // body_obs uses the CURRENT history (pre-update) — matches training
    // ordering (obs_prop is computed before the history buffer is shifted).
    this._bodyObs.set(currentFrame, 0);
    this._bodyObs.set(this.historyBuf, FRAME_DIM);

    // Update history for next step.
    if (!this.hasInitialized) {
      // First call after reset → fill all 10 slots with current frame.
      for (let i = 0; i < HISTORY_LEN; i++) {
        for (let j = 0; j < FRAME_DIM; j++) {
          this.historyBuf[i * FRAME_DIM + j] = currentFrame[j];
        }
      }
      this.hasInitialized = true;
    } else {
      // Shift left by one frame, append current at tail.
      // historyBuf[0:9] = historyBuf[1:10]
      for (let i = 0; i < (HISTORY_LEN - 1) * FRAME_DIM; i++) {
        this.historyBuf[i] = this.historyBuf[i + FRAME_DIM];
      }
      // historyBuf[9] = current_frame
      for (let j = 0; j < FRAME_DIM; j++) {
        this.historyBuf[(HISTORY_LEN - 1) * FRAME_DIM + j] = currentFrame[j];
      }
    }

    return this._bodyObs;
  }
}
