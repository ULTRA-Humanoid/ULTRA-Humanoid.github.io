// pd_control.js — PD controller for the G1 robot in mujoco-wasm.
//
// MIRRORS the authoritative training PD setup in
// intermimic/utils/g1_control.py — which itself reflects what
// intermimic/env/tasks/humanoid_g1.py:231-271 hardcodes at training time
// (the yaml `control:` block is overridden). Each constant below has a
// matching named element in g1_control.py and the Python deploy
// intermimic/run_sim2sim_interactive.py.
//
// PD formula (from training humanoid.py:496-499 + post-clip):
//     tau = G1_STIFFNESS * (target_q - dof_pos) - G1_DAMPING * dof_vel
//     tau = clip(tau, -G1_TORQUE_LIMIT, +G1_TORQUE_LIMIT)
// where target_q = ACTION_SCALE * clamp(mu, -1, 1)   (NO default_pose offset!).
// Reason: humanoid.py:118 initializes `_initial_dof_pos` to zeros and never
// overwrites it; humanoid.py:497 reads `actions_scaled + _initial_dof_pos` =
// actions_scaled. Confirmed by intermimic/sim2sim_vae.py:1272 (the proven
// reference deploy path).
//
// Joint order: matches G1's 29-actuator MJCF declaration order, also the
// same as G1_JOINT_ORDER in intermimic/utils/body_obs.py.

export const G1_JOINT_ORDER = [
  'left_hip_pitch_joint',  'left_hip_roll_joint',  'left_hip_yaw_joint',
  'left_knee_joint',       'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint',      'right_ankle_pitch_joint','right_ankle_roll_joint',
  'waist_yaw_joint',       'waist_roll_joint',     'waist_pitch_joint',
  'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
  'left_shoulder_yaw_joint',   'left_elbow_joint',
  'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint','right_shoulder_roll_joint',
  'right_shoulder_yaw_joint',  'right_elbow_joint',
  'right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint',
];

// 29-element arrays. Values from utils/g1_control.py (single source of truth).
export const G1_STIFFNESS = new Float32Array([
  40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
  40.179238, 99.098428, 40.179238, 99.098428, 28.501246, 28.501246,
  40.179238, 28.501246, 28.501246,
  14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
  14.250623, 14.250623, 14.250623, 14.250623, 14.250623, 16.778327, 16.778327,
]);
export const G1_DAMPING = new Float32Array([
  2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
  2.557890, 6.308802, 2.557890, 6.308802, 1.814446, 1.814446,
  2.557890, 1.814446, 1.814446,
  0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
  0.907223, 0.907223, 0.907223, 0.907223, 0.907223, 1.068142, 1.068142,
]);
export const G1_ARMATURE = new Float32Array([
  0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
  0.010178, 0.025102, 0.010178, 0.025102, 0.007219, 0.007219,
  0.010178, 0.007219, 0.007219,
  0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
  0.003610, 0.003610, 0.003610, 0.003610, 0.003610, 0.004250, 0.004250,
]);
// Effort × 0.8 (matches humanoid_g1.py:270).
export const G1_TORQUE_LIMIT = new Float32Array([
   70.4, 111.2,  70.4, 111.2, 40.0, 40.0,
   70.4, 111.2,  70.4, 111.2, 40.0, 40.0,
   70.4,  40.0,  40.0,
   20.0,  20.0,  20.0,  20.0,  20.0,  4.0,  4.0,
   20.0,  20.0,  20.0,  20.0,  20.0,  4.0,  4.0,
]);
export const ACTION_SCALE = 3.0;

// Sim timing — matches training (60 Hz control, 1020 Hz physics).
export const SIM_DT = 1.0 / (60.0 * 17.0);
export const SIM_DECIMATION = 17;
export const CONTROL_HZ = 60.0;

// ---- Resolver: from joint names → qpos/qvel/actuator addresses ---- //

/**
 * Inspect the compiled mujoco model and return per-joint indices for the
 * 29 G1 joints in `G1_JOINT_ORDER`.
 *
 * @returns {{
 *   qposAddr: Int32Array, qvelAddr: Int32Array,
 *   actuatorOrder: Int32Array,
 *   armatureDofAddr: Int32Array,
 * }}
 */
export function resolveJointAddresses(model) {
  const N = 29;
  const qposAddr = new Int32Array(N);
  const qvelAddr = new Int32Array(N);
  const actuatorOrder = new Int32Array(N);

  // Read joint names from MuJoCo's name blob.
  const namesBlob = model.names;
  const jntNameAdr = model.name_jntadr;
  const readName = (offset) => {
    let s = '';
    for (let k = offset; k < namesBlob.length; k++) {
      const c = namesBlob[k];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  };
  const jointNameToId = {};
  for (let jid = 0; jid < model.njnt; jid++) {
    jointNameToId[readName(jntNameAdr[jid])] = jid;
  }

  // Joint id → actuator id (assumes 1 motor per joint, JOINT transmission).
  const jointToActuator = {};
  for (let aid = 0; aid < model.nu; aid++) {
    jointToActuator[model.actuator_trnid[aid * 2 + 0]] = aid;
  }

  for (let i = 0; i < N; i++) {
    const name = G1_JOINT_ORDER[i];
    const jid = jointNameToId[name];
    if (jid === undefined) {
      throw new Error(`Joint "${name}" not in MJCF; ` +
                      `have: ${Object.keys(jointNameToId).slice(0, 6)}...`);
    }
    qposAddr[i] = model.jnt_qposadr[jid];
    qvelAddr[i] = model.jnt_dofadr[jid];
    const aid = jointToActuator[jid];
    if (aid === undefined) throw new Error(`Joint "${name}" has no actuator`);
    actuatorOrder[i] = aid;
  }
  return {
    qposAddr, qvelAddr, actuatorOrder,
    armatureDofAddr: qvelAddr,   // alias — armature lives in dof_armature[qvelAddr]
  };
}

/**
 * Apply per-DOF armature from G1_ARMATURE to model.dof_armature, matching
 * training (humanoid_g1.py:260). Pass the addresses returned by
 * resolveJointAddresses.
 *
 * IMPORTANT: must be called AFTER model load but BEFORE the first physics step.
 * MJCF default armature is 0.01; we override with per-DOF tuned values.
 */
export function applyArmatureOverride(model, addresses) {
  for (let i = 0; i < 29; i++) {
    model.dof_armature[addresses.armatureDofAddr[i]] = G1_ARMATURE[i];
  }
}

/**
 * Extract the 29-element default-pose dof_pos from a keyframe.
 *
 * @param keyframeIdx default 0 (first keyframe in MJCF)
 * @returns Float32Array(29) in G1_JOINT_ORDER order
 */
export function defaultPoseFromKeyframe(model, addresses, keyframeIdx = 0) {
  const qpos0 = model.key_qpos;   // Float32Array, shape (nkey, nq)
  const nq = model.nq;
  const target = new Float32Array(29);
  for (let i = 0; i < 29; i++) {
    target[i] = qpos0[keyframeIdx * nq + addresses.qposAddr[i]];
  }
  return target;
}

/**
 * Apply PD torques to data.ctrl. Mirrors the Python `pd_torque` in
 * utils/g1_control.py.
 *
 * @param model    mujoco MjModel
 * @param data     mujoco MjData
 * @param target_q Float32Array(29) — target dof_pos in G1_JOINT_ORDER order
 * @param addresses returned by resolveJointAddresses()
 */
export function applyPDTorques(model, data, target_q, addresses) {
  const qpos = data.qpos;
  const qvel = data.qvel;
  const ctrl = data.ctrl;
  const { qposAddr, qvelAddr, actuatorOrder } = addresses;
  for (let i = 0; i < 29; i++) {
    const q = qpos[qposAddr[i]];
    const qd = qvel[qvelAddr[i]];
    let tau = G1_STIFFNESS[i] * (target_q[i] - q) - G1_DAMPING[i] * qd;
    const lim = G1_TORQUE_LIMIT[i];
    if (tau >  lim) tau =  lim;
    if (tau < -lim) tau = -lim;
    ctrl[actuatorOrder[i]] = tau;
  }
}
