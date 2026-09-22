// Preflight a complete, nonrigid carry reference against other live boxes.
// Kinematics and distance queries use a private MjData. The live scene,
// physics clock, model settings, controller state and histories are untouched.
import {resolveJointAddresses} from './pd_control.js';

export const CARRY_REFERENCE_TRACKING_RESERVE_M = 0.1;
// Float32 source rotations can differ slightly from unit length. This is a
// data-validity tolerance; it does not change any geometry or clearance.
const SOURCE_QUATERNION_NORM_TOLERANCE = 1e-5;
// MuJoCo returns distmax when it finds no nearer pair. Query just beyond the
// reserve so an actual distance exactly at the boundary can be distinguished.
const DISTANCE_QUERY_EPSILON_M = 1e-9;

function readName(model, offset) {
  let value = '';
  while (model.names[offset]) value += String.fromCharCode(model.names[offset++]);
  return value;
}

function finitePose(frame) {
  if (!frame || frame.length !== 747) return false;
  for (const [first, end] of [[0, 7], [13, 42], [71, 78]]) {
    for (let index = first; index < end; index++) if (!Number.isFinite(frame[index])) return false;
  }
  return Math.abs(Math.hypot(...frame.slice(3, 7)) - 1) <= SOURCE_QUATERNION_NORM_TOLERANCE
    && Math.abs(Math.hypot(...frame.slice(74, 78)) - 1) <= SOURCE_QUATERNION_NORM_TOLERANCE;
}

export class CarryReferenceClearance {
  constructor(mujoco, model) {
    if (typeof mujoco?.mj_geomDistance !== 'function' || typeof mujoco?.mj_kinematics !== 'function') {
      throw new Error('Carry reference clearance requires MuJoCo geometry distance and kinematics');
    }
    this.mujoco = mujoco;
    this.model = model;
    this.names = Array.from({length: model.nbody}, (_, id) => readName(model, model.name_bodyadr[id]));
    this.objectBodies = new Set(this.names.flatMap((name, id) => name.startsWith('active_') ? [id] : []));
    const pelvis = this.names.indexOf('pelvis');
    if (pelvis < 0) throw new Error('Carry reference clearance requires the humanoid pelvis');
    for (const body of [pelvis, ...this.objectBodies]) {
      const joint = model.body_jntadr[body];
      const address = model.jnt_qposadr[joint];
      if (!Number.isInteger(joint) || joint < 0 || joint >= model.njnt
        || model.jnt_type[joint] !== mujoco.mjtJoint.mjJNT_FREE.value
        || !Number.isInteger(address) || address < 0 || address + 7 > model.nq) {
        throw new Error('Carry reference bodies require free scene poses');
      }
    }
    this.rootQpos = model.jnt_qposadr[model.body_jntadr[pelvis]];
    this.jointQpos = resolveJointAddresses(model).qposAddr;
    const ownerOf = body => {
      let current = body;
      while (current > 0) {
        if (this.objectBodies.has(current)) return current;
        if (current === pelvis) return pelvis;
        current = model.body_parentid[current];
      }
      return -1;
    };
    this.geometries = [];
    for (let id = 0; id < model.ngeom; id++) {
      if (!(model.geom_contype[id] || model.geom_conaffinity[id])) continue;
      const body = model.geom_bodyid[id], owner = ownerOf(body);
      if (owner < 0) continue; // The support floor is not a movable obstacle.
      const type = model.geom_type[id], size = Array.from(model.geom_size.slice(id * 3, id * 3 + 3));
      const geometry = {id, body, owner, type, size, center: [0, 0, 0], half: null};
      if (type === mujoco.mjtGeom.mjGEOM_MESH.value) {
        const mesh = model.geom_dataid[id], first = model.mesh_vertadr[mesh] * 3;
        const count = model.mesh_vertnum[mesh] * 3, vertices = model.mesh_vert;
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let index = first; index < first + count; index += 3) {
          for (let axis = 0; axis < 3; axis++) {
            lo[axis] = Math.min(lo[axis], vertices[index + axis]);
            hi[axis] = Math.max(hi[axis], vertices[index + axis]);
          }
        }
        if (!count || ![...lo, ...hi].every(Number.isFinite)) throw new Error('Carry collision mesh is unavailable');
        geometry.center = lo.map((value, axis) => (value + hi[axis]) / 2);
        geometry.half = lo.map((value, axis) => (hi[axis] - value) / 2);
      } else if (![mujoco.mjtGeom.mjGEOM_SPHERE.value, mujoco.mjtGeom.mjGEOM_CAPSULE.value,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID.value, mujoco.mjtGeom.mjGEOM_CYLINDER.value,
        mujoco.mjtGeom.mjGEOM_BOX.value].includes(type)) {
        throw new Error('Unsupported collision geometry in carry clearance');
      }
      this.geometries.push(geometry);
    }
    this.scratch = new mujoco.MjData(model);
    this.fromTo = new mujoco.DoubleBuffer(6);
    this.bounds = new Float64Array(this.geometries.length * 6);
    this.obstacleBounds = new Float64Array(this.bounds.length);
    this.pelvis = pelvis;
  }

  _updateBounds() {
    const positions = this.scratch.geom_xpos, rotations = this.scratch.geom_xmat;
    const kinds = this.mujoco.mjtGeom;
    for (let index = 0; index < this.geometries.length; index++) {
      const geom = this.geometries[index], p = geom.id * 3, r = geom.id * 9, out = index * 6;
      for (let axis = 0; axis < 3; axis++) {
        const a = rotations[r + axis * 3], b = rotations[r + axis * 3 + 1], c = rotations[r + axis * 3 + 2];
        const center = positions[p + axis] + a * geom.center[0] + b * geom.center[1] + c * geom.center[2];
        let extent;
        if (geom.half) extent = Math.abs(a) * geom.half[0] + Math.abs(b) * geom.half[1] + Math.abs(c) * geom.half[2];
        else if (geom.type === kinds.mjGEOM_SPHERE.value) extent = geom.size[0];
        else if (geom.type === kinds.mjGEOM_CAPSULE.value) extent = geom.size[0] + Math.abs(c) * geom.size[1];
        else if (geom.type === kinds.mjGEOM_CYLINDER.value) extent = geom.size[0] * Math.sqrt(Math.max(0, 1 - c * c)) + Math.abs(c) * geom.size[1];
        else if (geom.type === kinds.mjGEOM_BOX.value) extent = Math.abs(a) * geom.size[0] + Math.abs(b) * geom.size[1] + Math.abs(c) * geom.size[2];
        else extent = Math.hypot(a * geom.size[0], b * geom.size[1], c * geom.size[2]);
        this.bounds[out + axis] = center - extent;
        this.bounds[out + 3 + axis] = center + extent;
      }
    }
  }

  check(liveData, {referenceFrames, sourceFrames, objectBodyName} = {}) {
    if (!this.scratch) throw new Error('Carry reference clearance has been disposed');
    if (!Number.isInteger(sourceFrames) || sourceFrames < 1 || !Array.isArray(referenceFrames)
      || referenceFrames.length < sourceFrames || !referenceFrames.slice(0, sourceFrames).every(finitePose)) {
      throw new Error('Carry clearance requires every complete source frame with finite poses and unit root/object quaternions');
    }
    const selected = this.names.indexOf(objectBodyName);
    if (!this.objectBodies.has(selected)) throw new Error('The carried scene object is unavailable');
    if (!liveData?.qpos || liveData.qpos.length !== this.model.nq || !Array.from(liveData.qpos).every(Number.isFinite)) {
      throw new Error('Finite live scene positions are required for carry clearance');
    }
    const started = performance.now(), reserve = CARRY_REFERENCE_TRACKING_RESERVE_M;
    const objectQpos = this.model.jnt_qposadr[this.model.body_jntadr[selected]];
    const initialQpos = Float64Array.from(liveData.qpos);
    this.scratch.qpos.set(initialQpos);
    this.mujoco.mj_kinematics(this.model, this.scratch);
    this._updateBounds(); this.obstacleBounds.set(this.bounds);
    const moving = [], obstacles = [];
    for (const [index, geometry] of this.geometries.entries()) {
      (geometry.owner === this.pelvis || geometry.owner === selected ? moving : obstacles).push(index);
    }
    let queries = 0;
    const result = (supported, checked, violation = null) => ({supported,
      reason: supported ? null : 'carry_reference_clearance', sourceFrames, framesChecked: checked,
      completeSourceChecked: supported, trackingReserveM: reserve, geomDistanceCalls: queries,
      firstViolation: violation, elapsedMs: performance.now() - started});
    for (let sourceIndex = 0; sourceIndex < sourceFrames; sourceIndex++) {
      const frame = referenceFrames[sourceIndex], qpos = this.scratch.qpos;
      qpos.set(initialQpos);
      qpos.set(frame.slice(0, 3), this.rootQpos);
      qpos.set([frame[6], frame[3], frame[4], frame[5]], this.rootQpos + 3);
      for (let joint = 0; joint < 29; joint++) qpos[this.jointQpos[joint]] = frame[13 + joint];
      qpos.set(frame.slice(71, 74), objectQpos);
      qpos.set([frame[77], frame[74], frame[75], frame[76]], objectQpos + 3);
      this.mujoco.mj_kinematics(this.model, this.scratch);
      this._updateBounds();
      for (const index of moving) for (const other of obstacles) {
        const a = index * 6, b = other * 6;
        let gap = -Infinity;
        for (let axis = 0; axis < 3; axis++) gap = Math.max(gap,
          this.bounds[a + axis] - this.obstacleBounds[b + 3 + axis],
          this.obstacleBounds[b + axis] - this.bounds[a + 3 + axis]);
        if (gap > reserve) continue;
        const geom = this.geometries[index], obstacle = this.geometries[other];
        this.fromTo.GetView().fill(0);
        const distance = this.mujoco.mj_geomDistance(this.model, this.scratch, geom.id, obstacle.id,
          reserve + DISTANCE_QUERY_EPSILON_M, this.fromTo);
        queries++;
        if (!Number.isFinite(distance)) throw new Error('Nonfinite carry geometry distance');
        if (distance <= reserve) return result(false, sourceIndex + 1, {sourceIndex,
          body: this.names[geom.body], bodyId: geom.body, geom: geom.id,
          object: this.names[obstacle.owner], objectBodyId: obstacle.owner, obstacleGeom: obstacle.id,
          distanceM: distance, fromToWorld: Array.from(this.fromTo.GetView())});
      }
    }
    return result(true, sourceFrames);
  }

  dispose() {
    this.fromTo?.delete(); this.scratch?.delete();
    this.fromTo = null; this.scratch = null;
  }
}
