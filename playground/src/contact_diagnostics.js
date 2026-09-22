// Read-only diagnostics from the actual WASM solver. These values do not
// participate in policy observations or task decisions.
export class ContactDiagnostics {
  constructor(mujoco, model, { leftHand, rightHand, feet, robotBodies = [], legs = [] }) {
    this.mujoco = mujoco; this.model = model;
    this.leftHand = leftHand; this.rightHand = rightHand;
    this.feet = new Set(feet);
    this.robotBodies = new Set([leftHand, rightHand, ...feet, ...robotBodies]);
    this.legs = new Set([...feet, ...legs]);
    this.force = new mujoco.DoubleBuffer(6);
    this.jacp = new mujoco.DoubleBuffer(3 * model.nv);
    this.jacr = new mujoco.DoubleBuffer(3 * model.nv);
  }

  dispose() {
    this.force?.delete(); this.jacp?.delete(); this.jacr?.delete();
    this.force = this.jacp = this.jacr = null;
  }

  read(data, objectBodyId = -1) {
    if (!this.force) throw new Error('Contact diagnostics have been disposed');
    const result = { leftHandObjectNormalForceN: 0, rightHandObjectNormalForceN: 0,
      nonHandObjectNormalForceN: 0, legObjectNormalForceN: 0,
      loadedFootGroundContacts: 0, loadedFootNormalForceN: 0,
      footTangentialSpeedMeanMps: null, footTangentialSpeedMaxMps: null };
    const velocities = new Map();
    let weightedSpeed = 0, maximumSpeed = 0;
    const contacts = data.contact;
    try {
      for (let index = 0; index < data.ncon; index++) {
        const contact = contacts.get(index);
        try {
          const a = this.model.geom_bodyid[contact.geom1], b = this.model.geom_bodyid[contact.geom2];
          const hand = a === objectBodyId ? b : b === objectBodyId ? a : -1;
          const foot = a === 0 && this.feet.has(b) ? b : b === 0 && this.feet.has(a) ? a : -1;
          if (!this.robotBodies.has(hand) && foot < 0) continue;
          this.force.GetView().fill(0);
          this.mujoco.mj_contactForce(this.model, data, index, this.force);
          const normalForce = Math.max(0, this.force.GetView()[0]);
          if (hand === this.leftHand) result.leftHandObjectNormalForceN += normalForce;
          if (hand === this.rightHand) result.rightHandObjectNormalForceN += normalForce;
          if (this.robotBodies.has(hand) && hand !== this.leftHand && hand !== this.rightHand) result.nonHandObjectNormalForceN += normalForce;
          if (this.legs.has(hand)) result.legObjectNormalForceN += normalForce;
          if (foot < 0 || normalForce <= 5) continue;
          if (!velocities.has(foot)) {
            this.jacp.GetView().fill(0); this.jacr.GetView().fill(0);
            this.mujoco.mj_jacBody(this.model, data, this.jacp, this.jacr, foot);
            const linear = [0, 0, 0], angular = [0, 0, 0];
            const jp = this.jacp.GetView(), jr = this.jacr.GetView();
            for (let axis = 0; axis < 3; axis++) for (let dof = 0; dof < this.model.nv; dof++) {
              linear[axis] += jp[axis * this.model.nv + dof] * data.qvel[dof];
              angular[axis] += jr[axis * this.model.nv + dof] * data.qvel[dof];
            }
            velocities.set(foot, { linear, angular });
          }
          const { linear, angular } = velocities.get(foot);
          const offset = Array.from(contact.pos, (value, axis) => value - data.xpos[foot * 3 + axis]);
          const v = [linear[0] + angular[1] * offset[2] - angular[2] * offset[1],
            linear[1] + angular[2] * offset[0] - angular[0] * offset[2],
            linear[2] + angular[0] * offset[1] - angular[1] * offset[0]];
          const normal = contact.frame;
          const projection = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
          const speed = Math.hypot(...v.map((value, axis) => value - projection * normal[axis]));
          result.loadedFootGroundContacts++;
          result.loadedFootNormalForceN += normalForce;
          weightedSpeed += speed * normalForce;
          maximumSpeed = Math.max(maximumSpeed, speed);
        } finally { contact.delete(); }
      }
    } finally { contacts.delete(); }
    if (result.loadedFootNormalForceN > 0) {
      result.footTangentialSpeedMeanMps = weightedSpeed / result.loadedFootNormalForceN;
      result.footTangentialSpeedMaxMps = maximumSpeed;
    }
    return result;
  }
}
