// goal_viz.js — three.js markers that show what the goal translator is
// telling the policy each frame. Useful for verifying that the matched
// goal makes sense (correct direction, sensible magnitude) and that
// transitions feel smooth.
//
//   • Green sphere   — final clicked human target when one exists; otherwise
//                      the current policy human target
//   • Blue sphere    — current policy human target when it differs from the
//                      final clicked target
//   • Blue line      — heading direction of the current policy command
//   • Orange sphere  — object target position
//   • Orange line    — connector from current obj_pos to obj target
//
// All markers fade with the per-channel mask so an inactive goal field
// is dim/hidden. Toggle with the G keyboard shortcut.
//
// Coordinate system: this scene uses MuJoCo's Z-up convention directly
// (see scene_builder.js syncBodyTransforms — xpos/xquat are written
// straight into three.js group transforms with no axis swap).
//
// Goal-frame math: the translator returns goalSpec in the heading-
// aligned BODY frame at the moment of the query. To get a world point:
//
//   body_to_world = heading_quat(root_quat)   # yaw-only rotation
//   target_world  = root_pos + body_to_world · goalSpec.humanTargetPos
//
// `heading_quat` is the FORWARD-DIRECTION of heading_quat_inv. Since
// math.js exposes only the inverse, we compute it as the conjugate.

import * as THREE from 'three';
import { quatRotateOne } from './math.js';

// ------------------------------- Constants ------------------------------- //

const HUMAN_COLOR = 0x44ff88;   // soft green
const HUMAN_DIM_COLOR = 0x1f8048;
const CMD_COLOR = 0x66b7ff;
const CMD_DIM_COLOR = 0x1e4f80;
const OBJ_COLOR   = 0xff9933;   // warm orange
const OBJ_DIM_COLOR = 0x804a18;
const SPHERE_RADIUS_HUMAN = 0.13;
const SPHERE_RADIUS_CMD   = 0.08;
const SPHERE_RADIUS_OBJ   = 0.10;
const HEADING_ARROW_LEN   = 0.5;     // metres
const MASK_FADE_FLOOR     = 0.05;    // below this, hide entirely
const OPACITY_MAX         = 0.65;
const OPACITY_MIN         = 0.10;


// ------------------------------- Helpers --------------------------------- //

function makeSphere(color, dimColor, radius) {
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 18, 14),
    new THREE.MeshStandardMaterial({
      color,
      emissive: dimColor,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: OPACITY_MAX,
    }),
  );
  m.castShadow = false;
  m.receiveShadow = false;
  return m;
}

function makeLine(color, points) {
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: OPACITY_MAX, depthTest: true,
  }));
}

// Heading-only forward quaternion (yaw-only, inverse of
// calcHeadingQuatInv). We don't have a direct calcHeadingQuat in
// math.js, so derive: yaw = atan2(2(wz + xy), 1 - 2(y² + z²));
// forward yaw quat = (0, 0, sin(yaw/2), cos(yaw/2)).
function headingQuatForward(rootQuatXyzw) {
  const [x, y, z, w] = rootQuatXyzw;
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const half = 0.5 * yaw;
  return [0, 0, Math.sin(half), Math.cos(half)];
}

// goalSpec body-frame humanTargetRot is rot6d. Decode the heading
// (first column of the rotation matrix it represents — the
// "forward" axis in body frame).
function rot6dForwardVec(rot6d) {
  // rot6d layout from quatToRot6d (math.js): first 3 = column 0 of R,
  // next 3 = column 1 of R. Column 0 is the body-frame x-axis
  // (forward) expressed in… wait — actually rot6d encodes the rotation
  // FROM the body frame TO some target frame; the first column IS the
  // image of e_x. For our usage here goalSpec.humanTargetRot encodes
  // the goal's orientation relative to the current heading, so col 0
  // is where the goal-forward axis would point in the *current* body
  // frame.
  return [rot6d[0], rot6d[1], rot6d[2]];
}


// ------------------------------- Exported API --------------------------- //

export class GoalViz {
  constructor(scene, options = {}) {
    this.scene = scene;
    this.visible = options.visible !== false;

    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.humanSphere   = makeSphere(HUMAN_COLOR, HUMAN_DIM_COLOR, SPHERE_RADIUS_HUMAN);
    this.commandSphere = makeSphere(CMD_COLOR, CMD_DIM_COLOR, SPHERE_RADIUS_CMD);
    this.headingArrow  = makeLine(CMD_COLOR, [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(HEADING_ARROW_LEN, 0, 0),
    ]);
    this.objSphere     = makeSphere(OBJ_COLOR, OBJ_DIM_COLOR, SPHERE_RADIUS_OBJ);
    this.objConnector  = makeLine(OBJ_COLOR, [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 0),
    ]);

    this.group.add(this.humanSphere);
    this.group.add(this.commandSphere);
    this.group.add(this.headingArrow);
    this.group.add(this.objSphere);
    this.group.add(this.objConnector);

    this.group.visible = this.visible;
  }

  toggle() {
    this.visible = !this.visible;
    this.group.visible = this.visible;
    return this.visible;
  }

  // Recorded controllers already express their targets in world coordinates.
  // Keep an accepted destination visible while an earlier motion finishes.
  updateRecorded({ humanGoal = null, referenceFrame = null, objectGoal = null, objectPosition = null } = {}) {
    const point = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
    const show = (sphere, target) => {
      sphere.visible = point(target);
      if (sphere.visible) { sphere.position.set(...target); this._setOpacity(sphere.material, 1); }
    };
    this.group.visible = this.visible;
    const referenceRoot = referenceFrame?.length === 747 ? referenceFrame.slice(0, 3) : null;
    show(this.humanSphere, humanGoal); show(this.commandSphere, referenceRoot); show(this.objSphere, objectGoal);
    this.headingArrow.visible = point(referenceRoot);
    if (this.headingArrow.visible) {
      const forward = quatRotateOne(referenceFrame.slice(3, 7), [1, 0, 0]);
      this._updateLine(this.headingArrow, referenceRoot,
        Array.from(referenceRoot, (value, axis) => value + HEADING_ARROW_LEN * forward[axis]));
      this._setOpacity(this.headingArrow.material, 1);
    }
    this.objConnector.visible = point(objectGoal) && point(objectPosition);
    if (this.objConnector.visible) {
      this._updateLine(this.objConnector, objectPosition, objectGoal);
      this._setOpacity(this.objConnector.material, 1);
    }
  }

  // Update per frame. translatorResult is the {goalSpec, mask} dict
  // returned by GoalTranslator.step(). proprio supplies the current
  // root pose + (optional) object pose used to convert body-frame
  // goals into world coords. user supplies the final clicked world goals
  // when present.
  update(translatorResult, proprio, user = null) {
    if (!this.visible || translatorResult === null) return;
    const { goalSpec, mask } = translatorResult;

    // ------- Human command target (body → world) ------- //
    const headingFwd = headingQuatForward(proprio.rootQuatXyzwWorld);
    const humanBodyDelta = [
      goalSpec.humanTargetPos[0],
      goalSpec.humanTargetPos[1],
      goalSpec.humanTargetPos[2],
    ];
    const humanWorldDelta = quatRotateOne(headingFwd, humanBodyDelta);
    const hx = proprio.rootPosWorld[0] + humanWorldDelta[0];
    const hy = proprio.rootPosWorld[1] + humanWorldDelta[1];
    const hz = proprio.rootPosWorld[2] + humanWorldDelta[2];

    // For click-to-walk, users need the main green marker to be the static
    // final world goal, not the live motion-matched subgoal. The policy
    // subgoal is still shown as the smaller blue marker.
    const finalHuman = user && user.humanGoalWorld ? user.humanGoalWorld : null;
    if (finalHuman) {
      this.humanSphere.position.set(finalHuman[0], finalHuman[1], finalHuman[2]);
      this.commandSphere.position.set(hx, hy, hz);
      this._setOpacity(this.commandSphere.material, mask.keepHumanPos);
      this.commandSphere.visible = mask.keepHumanPos > MASK_FADE_FLOOR;
    } else {
      this.humanSphere.position.set(hx, hy, hz);
      this.commandSphere.visible = false;
    }
    this._setOpacity(this.humanSphere.material, mask.keepHumanPos);
    this.humanSphere.visible = mask.keepHumanPos > MASK_FADE_FLOOR;

    // ------- Heading arrow at the current policy command target ------- //
    // goal forward (in body frame) = first column of decoded rot
    const fwdBody = rot6dForwardVec(goalSpec.humanTargetRot);
    const fwdWorld = quatRotateOne(headingFwd, fwdBody);
    const ax = hx + HEADING_ARROW_LEN * fwdWorld[0];
    const ay = hy + HEADING_ARROW_LEN * fwdWorld[1];
    const az = hz + HEADING_ARROW_LEN * fwdWorld[2];
    this._updateLine(this.headingArrow, [hx, hy, hz], [ax, ay, az]);
    this._setOpacity(this.headingArrow.material, mask.keepHumanRot);
    this.headingArrow.visible = mask.keepHumanRot > MASK_FADE_FLOOR;

    // ------- Object target (body → world) ------- //
    const hasObj = proprio.objPosWorld !== null && mask.keepObjPos > MASK_FADE_FLOOR;
    if (hasObj) {
      const objBodyDelta = [
        goalSpec.objTargetPos[0],
        goalSpec.objTargetPos[1],
        goalSpec.objTargetPos[2],
      ];
      const objWorldDelta = quatRotateOne(headingFwd, objBodyDelta);
      const ox = proprio.objPosWorld[0] + objWorldDelta[0];
      const oy = proprio.objPosWorld[1] + objWorldDelta[1];
      const oz = proprio.objPosWorld[2] + objWorldDelta[2];
      this.objSphere.position.set(ox, oy, oz);
      this._setOpacity(this.objSphere.material, mask.keepObjPos);
      this.objSphere.visible = true;
      this._updateLine(this.objConnector,
        [proprio.objPosWorld[0], proprio.objPosWorld[1], proprio.objPosWorld[2]],
        [ox, oy, oz]);
      this._setOpacity(this.objConnector.material, mask.keepObjPos);
      this.objConnector.visible = true;
    } else {
      this.objSphere.visible = false;
      this.objConnector.visible = false;
    }
  }

  _setOpacity(material, maskScalar) {
    const m = Math.max(0, Math.min(1, maskScalar));
    material.opacity = OPACITY_MIN + (OPACITY_MAX - OPACITY_MIN) * m;
  }

  _updateLine(line, from, to) {
    const pos = line.geometry.attributes.position;
    pos.setXYZ(0, from[0], from[1], from[2]);
    pos.setXYZ(1, to[0], to[1], to[2]);
    pos.needsUpdate = true;
  }
}
