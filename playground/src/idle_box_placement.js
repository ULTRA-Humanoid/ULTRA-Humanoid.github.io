// A read-only request-time check for a box that is already at the destination.
import { checkAlreadyPlacedGoal } from './carry_goal_region_planner.js';

export function readIdleBoxPlacement({ mujoco, model, data, objectId, pelvisId, requestedGoalWorld, idle }) {
  const velocity = body => {
    const joint = model.body_jntadr[body];
    if (joint < 0 || model.jnt_type[joint] !== mujoco.mjtJoint.mjJNT_FREE.value)
      throw new Error('Placement measurement requires the current free-body velocity');
    const address = model.jnt_dofadr[joint];
    return Array.from(data.qvel.slice(address, address + 6));
  };
  const objectVelocity = velocity(objectId), rootVelocity = velocity(pelvisId);
  const force = new mujoco.DoubleBuffer(6), contacts = data.contact;
  let objectGrounded = false;
  try {
    for (let index = 0; index < data.ncon; index++) {
      const contact = contacts.get(index);
      try {
        const a = model.geom_bodyid[contact.geom1], b = model.geom_bodyid[contact.geom2];
        const floor = a === objectId ? contact.geom2 : b === objectId ? contact.geom1 : -1;
        if (floor < 0 || model.geom_bodyid[floor] !== 0 || model.geom_type[floor] !== 0) continue;
        force.GetView().fill(0);
        mujoco.mj_contactForce(model, data, index, force);
        // A light box distributes its support over several contacts, each
        // potentially below1 N. Require actual support, not an arbitrary
        // per-contact load that would reject an otherwise stationary box.
        if (force.GetView()[0] > 0) { objectGrounded = true; break; }
      } finally { contact.delete(); }
    }
  } finally { contacts.delete(); force.delete(); }
  const q = data.xquat.slice(pelvisId * 4, pelvisId * 4 + 4);
  return checkAlreadyPlacedGoal({ requestedGoalWorld, idle, objectGrounded,
    objectPositionWorld: Array.from(data.xpos.slice(objectId * 3, objectId * 3 + 3)),
    objectLinearSpeedMps: Math.hypot(...objectVelocity.slice(0, 3)),
    objectAngularSpeedRadps: Math.hypot(...objectVelocity.slice(3, 6)),
    rootLinearSpeedMps: Math.hypot(...rootVelocity.slice(0, 3)),
    rootHeightM: data.xpos[pelvisId * 3 + 2], upright: 1 - 2 * (q[1] ** 2 + q[2] ** 2) });
}
