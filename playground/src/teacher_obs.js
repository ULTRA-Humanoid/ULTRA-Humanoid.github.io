// Privileged reference teacher adapter for bounded whole-body skills.
// Uses the full training feature contract from intermimic_g1_retarget.py.
// Explicit diagnostic options also reproduce the older native sim2sim helper.
import { quatMulXyzw, quatNormalize, quatRotateOne } from './math.js';

export const TEACHER_OBS_DIM = 4052;
export const TEACHER_REFERENCE_DIM = 747;
export const TEACHER_HUMAN_BODY_NAMES = Object.freeze([
  'pelvis', 'imu_in_pelvis', 'left_hip_pitch_link', 'left_hip_roll_link',
  'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link',
  'pelvis_contour_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
  'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link',
  'waist_yaw_link', 'waist_roll_link', 'torso_link', 'd435_link', 'head_link', 'imu_in_torso',
  'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
  'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
  'left_rubber_hand', 'logo_link', 'mid360_link', 'right_shoulder_pitch_link',
  'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link',
  'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_rubber_hand',
]);

const SEGMENTS = [
  ['root_height', 1], ['body_position', 114], ['body_rotation', 234],
  ['body_velocity', 117], ['body_angular_velocity', 117], ['contact', 39],
  ['reference_position_error', 117], ['reference_rotation_error', 234], ['contact_error', 39],
  ['reference_position_with_root', 117], ['reference_rotation', 234],
  ['reference_velocity_error', 117], ['reference_angular_velocity_error', 117],
  ['actions', 29], ['dof_position', 29], ['dof_velocity', 29], ['torques', 29],
  ['previous_dof_position', 29], ['previous_dof_velocity', 29],
  ['object', 21], ['interaction_graph', 117], ['interaction_graph_error', 117],
];
let segmentOffset = 0;
export const TEACHER_HORIZON_SEGMENTS = Object.freeze(SEGMENTS.map(([name, length]) => {
  const entry = Object.freeze({ name, offset: segmentOffset, length });
  segmentOffset += length;
  return entry;
}));

const f32 = Math.fround;
const sub = (a, b) => a.map((v, i) => f32(v - b[i]));
const inverse = q => [-q[0], -q[1], -q[2], q[3]];
const rotate = (q, v) => quatRotateOne(q, v).map(f32);
const multiply = (a, b) => quatMulXyzw(a, b).map(f32);
const tanNorm = q => [...rotate(q, [1, 0, 0]), ...rotate(q, [0, 0, 1])];
function headingInverse(q) {
  // Native helper assumes normalized body quaternions; do not normalize again.
  const forward = rotate(q, [1, 0, 0]);
  const half = f32(-f32(Math.atan2(forward[1], forward[0])) / 2);
  return [0, 0, f32(Math.sin(half)), f32(Math.cos(half))];
}
function rotationError(reference, current, heading) {
  // quat_mul_norm normalizes the relative quaternion before change of basis.
  const relative = quatNormalize(multiply(inverse(reference), current)).map(f32);
  return tanNorm(multiply(multiply(heading, relative), inverse(heading)));
}
function softInteraction(vector) {
  const norm = f32(Math.hypot(...vector));
  const denominator = f32(norm + 1e-6), weight = f32(Math.exp(f32(-5 * norm)));
  return vector.map(v => f32(f32(v / denominator) * weight));
}
function finiteVector(value, length, name) {
  if (!value || value.length !== length || !Array.from(value).every(Number.isFinite)) {
    throw new Error(`${name} must contain ${length} finite values`);
  }
  return Float32Array.from(value);
}

/**
 * Full privileged teacher observation from the actual WASM model/data.
 * Pass FP32 747-channel reference frames at t+1 and t+16. No resampling or
 * reference advancement happens here; the caller owns the reference clock.
 * locomotionOnly applies the AMASS/BONES teacher's final object/IG masks.
 */
export class TeacherObsBuilder {
  constructor(mujoco, model, {
    objectBodyName, objectPointsLocal, diagnosticLegacyObjectOffsets = false,
    diagnosticNativeContract = false, locomotionOnly = false,
  }) {
    this.mujoco = mujoco;
    this.model = model;
    const names = model.names;
    const bodyNames = new Map();
    for (let id = 0; id < model.nbody; id++) {
      let name = '', offset = model.name_bodyadr[id];
      while (names[offset]) name += String.fromCharCode(names[offset++]);
      bodyNames.set(name, id);
    }
    this.bodyIds = TEACHER_HUMAN_BODY_NAMES.map(name => {
      if (!bodyNames.has(name)) throw new Error(`Missing teacher body ${name}`);
      return bodyNames.get(name);
    });
    if (!bodyNames.has(objectBodyName)) throw new Error(`Missing teacher object ${objectBodyName}`);
    this.objectId = bodyNames.get(objectBodyName);
    this.objectReferenceOffset = diagnosticLegacyObjectOffsets ? 313 : 71;
    this.diagnosticNativeContract = diagnosticNativeContract;
    if (typeof locomotionOnly !== 'boolean') throw new Error('Teacher locomotionOnly must be boolean');
    this.locomotionOnly = locomotionOnly;
    const points = Array.from(objectPointsLocal || []).flat();
    // The known-good native teacher uses all 256 sampled surface points.
    this.objectPoints = finiteVector(points, 256 * 3, 'Teacher object point cloud');
    if (model.nu !== 29) throw new Error(`Teacher requires 29 actuators, received ${model.nu}`);
    this.qposIds = new Int32Array(model.nu);
    this.qvelIds = new Int32Array(model.nu);
    for (let actuator = 0; actuator < model.nu; actuator++) {
      const joint = model.actuator_trnid[actuator * 2];
      if (joint < 0) throw new Error('Teacher requires joint actuators');
      this.qposIds[actuator] = model.jnt_qposadr[joint];
      this.qvelIds[actuator] = model.jnt_dofadr[joint];
    }
    // Ordinary JS Float64Array arguments do not receive native output writes.
    this.jacp = new mujoco.DoubleBuffer(3 * model.nv);
    this.jacr = new mujoco.DoubleBuffer(3 * model.nv);
    this.contactForce = new mujoco.DoubleBuffer(6);
    this.reset();
  }

  reset({ lastDofPos = new Float32Array(29), lastDofVel = new Float32Array(29) } = {}) {
    this.lastDofPos = finiteVector(lastDofPos, 29, 'Previous DOF position');
    this.lastDofVel = finiteVector(lastDofVel, 29, 'Previous DOF velocity');
  }

  dispose() {
    this.jacp?.delete(); this.jacr?.delete();
    this.jacp = this.jacr = null;
    this.contactForce?.delete(); this.contactForce = null;
  }

  _bodyState(data, id) {
    this.jacp.GetView().fill(0); this.jacr.GetView().fill(0);
    this.mujoco.mj_jacBody(this.model, data, this.jacp, this.jacr, id);
    const jp = this.jacp.GetView(), jr = this.jacr.GetView();
    // B9-PERF: `data.qvel` and `this.model.nv` are Embind getters that build a fresh memory view / marshal an int on
    // every access (5 getter calls per inner iteration, ~35k per control). Read once after mj_jacBody: no WASM call
    // happens before the loop ends, so the view aliases exactly the memory the per-access views aliased and every
    // value read is identical.
    const qvel = data.qvel, nv = this.model.nv;
    const velocity = [], angularVelocity = [];
    for (let axis = 0; axis < 3; axis++) {
      let linear = 0, angular = 0;
      for (let j = 0; j < nv; j++) {
        linear += jp[axis * nv + j] * qvel[j];
        angular += jr[axis * nv + j] * qvel[j];
      }
      velocity.push(f32(linear)); angularVelocity.push(f32(angular));
    }
    const q = data.xquat.slice(id * 4, id * 4 + 4);
    return {
      position: Array.from(data.xpos.slice(id * 3, id * 3 + 3), f32),
      rotation: [q[1], q[2], q[3], q[0]].map(f32), velocity, angularVelocity,
    };
  }

  _contacts(data) {
    const forces = Array.from({ length: this.model.nbody }, () => [0, 0, 0]);
    const contacts = data.contact;
    // B9-PERF: `data.ncon` is an Embind int getter; mj_contactForce never changes ncon, so one read is the same bound.
    const ncon = data.ncon;
    try {
      for (let i = 0; i < ncon; i++) {
        const contact = contacts.get(i);
        try {
          this.contactForce.GetView().fill(0);
          this.mujoco.mj_contactForce(this.model, data, i, this.contactForce);
          const force = this.contactForce.GetView(), frame = contact.frame;
          const body1 = this.model.geom_bodyid[contact.geom1], body2 = this.model.geom_bodyid[contact.geom2];
          for (let axis = 0; axis < 3; axis++) {
            const world = frame[axis] * force[0] + frame[3 + axis] * force[1] + frame[6 + axis] * force[2];
            forces[body1][axis] -= world;
            forces[body2][axis] += world;
          }
        } finally { contact.delete(); }
      }
    } finally { contacts.delete(); }
    return this.bodyIds.map(id => forces[id].reduce((s, v) => s + Math.abs(v), 0) > 0.1 ? 1 : 0);
  }

  /** Live forward-kinematics reference row (747): root pose, dof positions, the 39 body poses and the measured contact flags of
   *  the CURRENT physics state, all velocity blocks zero. Used as the source of a WS-C hold target (`planTeacherStandingReference`
   *  then re-anchors it to the live root/box). Read-only on `data`; mirrors the offline hold study's live-fk composition. */
  liveReferenceFrame(data) {
    if (!this.jacp) throw new Error('TeacherObsBuilder has been disposed');
    const frame = new Float32Array(TEACHER_REFERENCE_DIM);
    const pos = id => [data.xpos[id * 3], data.xpos[id * 3 + 1], data.xpos[id * 3 + 2]];
    const quat = id => [data.xquat[id * 4 + 1], data.xquat[id * 4 + 2], data.xquat[id * 4 + 3], data.xquat[id * 4]];
    frame.set(pos(this.bodyIds[0]), 0); frame.set(quat(this.bodyIds[0]), 3);
    frame.set(Float32Array.from(this.qposIds, id => data.qpos[id]), 13);
    this.bodyIds.forEach((id, i) => { frame.set(pos(id), 84 + i * 3); frame.set(quat(id), 201 + i * 4); });
    frame.set(this._contacts(data), 591);
    if (!frame.every(Number.isFinite)) throw new Error('Live reference frame must be finite');
    return frame;
  }

  build(data, references, actions = new Float32Array(29), torques = new Float32Array(29)) {
    if (!this.jacp) throw new Error('TeacherObsBuilder has been disposed');
    if (!references || references.length !== 2) throw new Error('Teacher needs +1 and +16 reference frames');
    const refs = references.map((r, i) => finiteVector(r, TEACHER_REFERENCE_DIM, `Reference ${i}`));
    const action = finiteVector(actions, 29, 'Teacher previous action');
    const torque = finiteVector(torques, 29, 'Teacher previous torque');
    const bodies = this.bodyIds.map(id => this._bodyState(data, id));
    const object = this._bodyState(data, this.objectId), root = bodies[0];
    const heading = headingInverse(root.rotation);
    const dofPos = Float32Array.from(this.qposIds, id => data.qpos[id]);
    const dofVel = Float32Array.from(this.qvelIds, id => data.qvel[id]);
    const contact = this.diagnosticNativeContract ? new Float32Array(39) : this._contacts(data);
    const observedContact = new Float32Array(39);
    const contactIndices = [6, 7, 13, 14, 28, 38];
    for (const i of contactIndices) observedContact[i] = contact[i];
    const points = [];
    for (let i = 0; i < this.objectPoints.length; i += 3) {
      points.push(rotate(object.rotation, this.objectPoints.subarray(i, i + 3))
        .map((v, j) => f32(v + object.position[j])));
    }
    const interaction = bodies.flatMap(body => {
      // B9-PERF: the same nearest-point search without allocating a delta Array per (body, point) pair (39 x 256 per
      // control). Components are the identical f32(body - point) values in the identical order; the squared distance is
      // the identical left-to-right double sum (((0 + d0*d0) + d1*d1) + d2*d2 === (d0*d0 + d1*d1) + d2*d2 because
      // 0 + x === x for x >= 0 or NaN); the first strict minimum wins exactly as before and only it is materialised.
      const [bx, by, bz] = body.position;
      let best = null, bestDistance = Infinity;
      for (const point of points) {
        const d0 = f32(bx - point[0]), d1 = f32(by - point[1]), d2 = f32(bz - point[2]);
        const distance = d0 * d0 + d1 * d1 + d2 * d2;
        if (distance < bestDistance) { bestDistance = distance; best = [d0, d1, d2]; }
      }
      return softInteraction(rotate(heading, best));
    });
    const out = new Float32Array(TEACHER_OBS_DIM);
    let offset = 0;
    const append = values => { out.set(values, offset); offset += values.length; };
    for (const ref of refs) {
      const slice = (start, length) => Array.from(ref.subarray(start, start + length));
      append([root.position[2]]);
      append(bodies.slice(1).flatMap(b => rotate(heading, sub(b.position, root.position))));
      append(bodies.flatMap(b => tanNorm(multiply(heading, b.rotation))));
      append(bodies.flatMap(b => rotate(heading, b.velocity)));
      append(bodies.flatMap(b => rotate(heading, b.angularVelocity)));
      append(observedContact);
      append(bodies.flatMap((b, i) => rotate(heading, sub(slice(84 + i * 3, 3), b.position))));
      append(bodies.flatMap((b, i) => rotationError(slice(201 + i * 4, 4), b.rotation, heading)));
      const contactError = new Float32Array(39);
      if (!this.diagnosticNativeContract) {
        for (const i of contactIndices) contactError[i] = f32(ref[591 + i] - contact[i]);
      }
      append(contactError);
      append(bodies.flatMap((b, i) => rotate(heading, sub(
        this.diagnosticNativeContract ? b.position : slice(84 + i * 3, 3), root.position))));
      append(bodies.flatMap((b, i) => tanNorm(multiply(heading, slice(201 + i * 4, 4)))));
      append(bodies.flatMap((b, i) => rotate(heading, sub(slice(357 + i * 3, 3), b.velocity))));
      append(bodies.flatMap((b, i) => rotate(heading, sub(slice(474 + i * 3, 3), b.angularVelocity))));
      for (const values of [action, dofPos, dofVel, this.diagnosticNativeContract ? torque : new Float32Array(29),
        this.lastDofPos, this.lastDofVel]) append(values);
      // 71:84 is the actual training reference object state. The native
      // sim2sim helper's legacy 313:326 behavior is available explicitly for
      // numerical comparison only; those offsets lie in body rotations.
      const objectOffset = this.objectReferenceOffset;
      append(rotate(heading, object.velocity));
      append(rotate(heading, object.angularVelocity));
      append(rotate(heading, sub(slice(objectOffset, 3), object.position)));
      append(rotationError(slice(objectOffset + 3, 4), object.rotation, heading));
      append(rotate(heading, sub(slice(objectOffset + 7, 3), object.velocity)));
      append(rotate(heading, sub(slice(objectOffset + 10, 3), object.angularVelocity)));
      append(interaction);
      const refInteraction = bodies.flatMap((b, i) => softInteraction(slice(630 + i * 3, 3)));
      append(sub(refInteraction, interaction));
      if (this.diagnosticNativeContract) {
        // Reproduce the older helper's sequential horizon mutation only when
        // explicitly comparing its contract. Training uses the same context.
        this.lastDofPos = dofPos; this.lastDofVel = dofVel;
      }
    }
    // Store this pre-physics state once for the next control observation.
    this.lastDofPos = dofPos; this.lastDofVel = dofVel;
    if (this.locomotionOnly) {
      // Actual RetargetAMASS teacher override: keep all humanoid features,
      // including foot contacts, then mask task21 + currentIG117 + deltaIG117
      // at the end of both horizons. This is independent of student masks.
      out.fill(0, 1771, 2026);
      out.fill(0, 3797, 4052);
    }
    if (offset !== TEACHER_OBS_DIM || !out.every(Number.isFinite)) {
      throw new Error(`Invalid teacher observation (${offset} channels)`);
    }
    return out;
  }
}
