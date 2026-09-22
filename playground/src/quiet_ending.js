// Quiet-terminated endings (WS-G). A bounded hold ends once the robot and the
// placed box have been measurably still for a window of consecutive controls,
// never before a declared minimum and never after the original maximum.
// Measurements read the actual WASM solver state; nothing here steps physics.
//
// Harness note: the frozen joint100 execution proof reads each hold's expected
// duration at its FIRST control and requires the tracked count to equal it.
// Callers therefore run the declared minimum as the tracked phase and the
// quiet-terminated remainder as a separate, explicitly labelled phase.

export const QUIET_ENDING_LIMITS = Object.freeze({
  rootPlanarSpeedMps: .05, minUpright: .95, boxSpeedMps: .05,
  floorForceMinimumN: .1, humanForceMaximumN: .1,
});
export const QUIET_ENDING_DEFAULTS = Object.freeze({
  window: 30, settlingMinControls: 60, settlingMaxControls: 180,
  exitInitialHoldMinControls: 30, exitInitialHoldMaxControls: 60,
  exitFinalHoldMinControls: 30, exitFinalHoldMaxControls: 180,
});
export const PLACEMENT_CORRECTION_DEFAULTS = Object.freeze({
  toleranceM: .10, maxErrorM: .15, minBudgetControls: 1800, maxCorrections: 1,
  postClickBudgetControls: 5820, candidateIds: Object.freeze(['short_0184', 'short_0295']),
});

const finiteNonNegative = value => Number.isFinite(value) && value >= 0;

// Optional object-tilt criterion (Phase B / B5, standing suitcase). The frozen
// largebox limits above carry no `tippedObjectTiltDeg`, so classification and
// the solver measurement are unchanged unless an object profile supplies one.
// Tilt = angle between the object's local up axis and world +z on the LIVE
// body quaternion (xyzw). A tilt at or beyond `tippedObjectTiltDeg` is
// 'object_tipped'; an unmeasurable tilt under an active requirement is a
// failure, never a pass.
export const OBJECT_UP_AXES = Object.freeze(['x', 'y', 'z']);
const unitXyzw = q => q && q.length === 4 && Array.from(q).every(Number.isFinite) && Math.abs(Math.hypot(...q) - 1) <= 1e-3;
/** World-z component of the object's local up axis: 1 upright, 0 on its side, -1 upside down. */
export function objectUpAxisWorldZ(quatXyzw, upAxisLocal) {
  if (!OBJECT_UP_AXES.includes(upAxisLocal)) throw new Error('Object up axis must be x, y or z');
  if (!unitXyzw(quatXyzw)) throw new Error('Object tilt requires a finite unit xyzw quaternion');
  const [x, y, z, w] = quatXyzw;
  return upAxisLocal === 'x' ? 2 * (x * z - w * y) : upAxisLocal === 'y' ? 2 * (y * z + w * x) : 1 - 2 * (x * x + y * y);
}
export function objectTiltDeg(quatXyzw, upAxisLocal) {
  return Math.acos(Math.max(-1, Math.min(1, objectUpAxisWorldZ(quatXyzw, upAxisLocal)))) * 180 / Math.PI;
}
/** Tilt for live measurements: NaN (never a small number) when the quaternion is unusable, so `!(tilt < limit)` fails closed. */
export function measuredObjectTiltDeg(quatXyzw, upAxisLocal) {
  return unitXyzw(quatXyzw) ? objectTiltDeg(quatXyzw, upAxisLocal) : NaN;
}
/** Normalise the optional tilt requirement carried by outcome requirements or quiet limits: null when absent (largebox),
 * else a frozen {tippedObjectTiltDeg, objectUpAxisLocal}. A partial or malformed specification throws (never silently inert). */
export function objectTiltRequirement(spec) {
  const limit = spec?.tippedObjectTiltDeg, axis = spec?.objectUpAxisLocal;
  if (limit === undefined && axis === undefined) return null;
  if (!Number.isFinite(limit) || limit <= 0 || limit > 180) throw new Error('Object tilt limit must be within (0, 180] degrees');
  if (!OBJECT_UP_AXES.includes(axis)) throw new Error('Object tilt limit requires the object up axis (x, y or z)');
  return Object.freeze({ tippedObjectTiltDeg: limit, objectUpAxisLocal: axis });
}

/** Classify one measured control. Every condition is reported so a noisy hold
 * can be diagnosed from the recorded summary. */
export function classifyQuietSample(sample, limits = QUIET_ENDING_LIMITS) {
  const reasons = [];
  if (!sample || typeof sample !== 'object') return { quiet: false, reasons: ['missing_sample'] };
  const { rootPlanarSpeedMps, upright, boxSpeedMps, objectFloorNormalForceN, humanNormalForceN } = sample;
  if (!finiteNonNegative(rootPlanarSpeedMps) || !Number.isFinite(upright) || !finiteNonNegative(boxSpeedMps)
      || !finiteNonNegative(objectFloorNormalForceN) || !finiteNonNegative(humanNormalForceN)) {
    return { quiet: false, reasons: ['nonfinite_measurement'] };
  }
  const tilt = objectTiltRequirement(limits);
  if (tilt && !Number.isFinite(sample.objectTiltDeg)) return { quiet: false, reasons: ['object_tilt_unmeasured'] };
  if (rootPlanarSpeedMps > limits.rootPlanarSpeedMps) reasons.push('root_moving');
  if (upright < limits.minUpright) reasons.push('root_not_upright');
  if (boxSpeedMps > limits.boxSpeedMps) reasons.push('box_moving');
  if (objectFloorNormalForceN <= limits.floorForceMinimumN) reasons.push('box_not_floor_supported');
  if (humanNormalForceN > limits.humanForceMaximumN) reasons.push('human_box_force');
  if (tilt && sample.objectTiltDeg >= tilt.tippedObjectTiltDeg) reasons.push('object_tipped');
  return { quiet: reasons.length === 0, reasons };
}

/** Consecutive-quiet bookkeeping for one hold. Reset at every hold start. */
export class QuietHoldMonitor {
  constructor({ window = QUIET_ENDING_DEFAULTS.window, limits = QUIET_ENDING_LIMITS } = {}) {
    if (!Number.isInteger(window) || window < 1 || window > 1000) throw new Error('Quiet window must be 1–1000 controls');
    this.window = window; this.limits = limits; this.reset();
  }
  reset() { this.samples = 0; this.quietSamples = 0; this.consecutiveQuiet = 0; this.lastReasons = []; this.firstSatisfiedSample = null; }
  observe(sample) {
    const classified = classifyQuietSample(sample, this.limits);
    this.samples++;
    if (classified.quiet) { this.quietSamples++; this.consecutiveQuiet++; }
    else this.consecutiveQuiet = 0;
    this.lastReasons = classified.reasons;
    if (this.firstSatisfiedSample === null && this.consecutiveQuiet >= this.window) this.firstSatisfiedSample = this.samples;
    return classified;
  }
  get satisfied() { return this.consecutiveQuiet >= this.window; }
  summary() {
    return { window: this.window, samples: this.samples, quietSamples: this.quietSamples,
      consecutiveQuiet: this.consecutiveQuiet, satisfied: this.satisfied,
      firstSatisfiedSample: this.firstSatisfiedSample, lastReasons: [...this.lastReasons] };
  }
}

/** A hold is complete at its maximum, or at/after its minimum once quiet. */
export function quietHoldComplete({ controls, minControls, maxControls, satisfied }) {
  if (![controls, minControls, maxControls].every(Number.isInteger) || minControls < 0 || maxControls < minControls)
    throw new Error('Whole hold bounds with minimum <= maximum are required');
  return controls >= maxControls || (controls >= minControls && satisfied === true);
}

export function validateQuietOptions(options, label) {
  if (options === null || options === undefined) return null;
  if (typeof options !== 'object' || typeof options.measure !== 'function')
    throw new Error(`${label} requires a synchronous measure() function`);
  const window = options.window ?? QUIET_ENDING_DEFAULTS.window;
  if (!Number.isInteger(window) || window < 1) throw new Error(`${label} window must be a positive whole count`);
  if (options.limits !== undefined) {
    // Optional per-object quiet limits (B5). Every base limit must be present and the tilt requirement, if any, complete.
    if (!options.limits || typeof options.limits !== 'object') throw new Error(`${label} limits must be an object`);
    for (const key of Object.keys(QUIET_ENDING_LIMITS))
      if (!Number.isFinite(options.limits[key])) throw new Error(`${label} limits are missing ${key}`);
    objectTiltRequirement(options.limits);
  }
  return { ...options, window };
}

/** Actual solver measurement for the quiet criterion: free-body velocities,
 * pelvis upright score, box-floor and box-human normal forces. With an
 * `objectUpAxisLocal` (B5 profile switch, default null = unchanged sample)
 * the sample also carries the object's live tilt. */
export class QuietEndingMeasurement {
  constructor(mujoco, model, { rootBodyId, objectBodyId, objectUpAxisLocal = null }) {
    for (const id of [rootBodyId, objectBodyId])
      if (!Number.isInteger(id) || id <= 0 || id >= model.nbody) throw new Error('Known root and object body IDs are required');
    if (objectUpAxisLocal !== null && !OBJECT_UP_AXES.includes(objectUpAxisLocal)) throw new Error('Object up axis must be x, y, z or null');
    this.objectUpAxisLocal = objectUpAxisLocal;
    this.mujoco = mujoco; this.model = model; this.root = rootBodyId; this.object = objectBodyId;
    this.robotRoot = model.body_rootid[rootBodyId];
    const address = body => {
      const joint = model.body_jntadr[body];
      if (joint < 0 || model.jnt_type[joint] !== mujoco.mjtJoint.mjJNT_FREE.value)
        throw new Error('Quiet measurement requires free-body velocities');
      return model.jnt_dofadr[joint];
    };
    this.rootDof = address(rootBodyId); this.objectDof = address(objectBodyId);
    this.force = new mujoco.DoubleBuffer(6);
  }
  dispose() { this.force?.delete(); this.force = null; }
  read(data) {
    if (!this.force) throw new Error('Quiet measurement has been disposed');
    const { model } = this;
    const rootVelocity = Array.from(data.qvel.slice(this.rootDof, this.rootDof + 3));
    const objectVelocity = Array.from(data.qvel.slice(this.objectDof, this.objectDof + 3));
    const q = data.xquat.slice(this.root * 4, this.root * 4 + 4); // wxyz
    let floor = 0, human = 0, other = 0;
    const contacts = data.contact;
    try {
      for (let index = 0; index < data.ncon; index++) {
        const contact = contacts.get(index);
        try {
          const a = model.geom_bodyid[contact.geom1], b = model.geom_bodyid[contact.geom2];
          if (a !== this.object && b !== this.object) continue;
          const otherGeom = a === this.object ? contact.geom2 : contact.geom1, otherBody = model.geom_bodyid[otherGeom];
          this.force.GetView().fill(0);
          this.mujoco.mj_contactForce(model, data, index, this.force);
          const normal = Math.max(0, this.force.GetView()[0]);
          if (!Number.isFinite(normal)) throw new Error('Finite contact force required');
          if (otherBody === 0 && model.geom_type[otherGeom] === 0) floor += normal;
          else if (model.body_rootid[otherBody] === this.robotRoot) human += normal;
          else other += normal;
        } finally { contact.delete(); }
      }
    } finally { contacts.delete(); }
    const sample = {
      rootPlanarSpeedMps: Math.hypot(rootVelocity[0], rootVelocity[1]),
      upright: 1 - 2 * (q[1] ** 2 + q[2] ** 2),
      boxSpeedMps: Math.hypot(...objectVelocity),
      objectFloorNormalForceN: floor, humanNormalForceN: human, otherObjectNormalForceN: other,
      rootHeightM: data.xpos[this.root * 3 + 2], objectHeightM: data.xpos[this.object * 3 + 2],
      floorSupported: floor > QUIET_ENDING_LIMITS.floorForceMinimumN,
    };
    if (this.objectUpAxisLocal !== null) {
      const o = data.xquat.slice(this.object * 4, this.object * 4 + 4); // wxyz
      const objectQuatXyzwWorld = [o[1], o[2], o[3], o[0]];
      sample.objectQuatXyzwWorld = objectQuatXyzwWorld;
      sample.objectUpAxisWorldZ = unitXyzw(objectQuatXyzwWorld) ? objectUpAxisWorldZ(objectQuatXyzwWorld, this.objectUpAxisLocal) : NaN;
      sample.objectTiltDeg = measuredObjectTiltDeg(objectQuatXyzwWorld, this.objectUpAxisLocal);
    }
    return sample;
  }
}

/** Decide whether one bounded corrective carry may follow a missed final
 * placement. Pure; the caller still needs a supported single-segment plan. */
export function evaluatePlacementCorrection({ remainingDistanceM, setDown, floorSupported, remainingBudgetControls,
  correctionsUsed = 0, toleranceM = PLACEMENT_CORRECTION_DEFAULTS.toleranceM,
  maxErrorM = PLACEMENT_CORRECTION_DEFAULTS.maxErrorM, minBudgetControls = PLACEMENT_CORRECTION_DEFAULTS.minBudgetControls,
  maxCorrections = PLACEMENT_CORRECTION_DEFAULTS.maxCorrections } = {}) {
  if (toleranceM !== PLACEMENT_CORRECTION_DEFAULTS.toleranceM) throw new Error('The 10 cm placement tolerance is fixed');
  const reason = !Number.isFinite(remainingDistanceM) || remainingDistanceM < 0 ? 'invalid_distance'
    : !Number.isInteger(remainingBudgetControls) ? 'invalid_budget'
    : correctionsUsed >= maxCorrections ? 'correction_already_used'
    : remainingDistanceM <= toleranceM + 1e-12 ? 'within_tolerance'
    : remainingDistanceM > maxErrorM + 1e-12 ? 'error_too_large'
    : setDown !== true ? 'box_not_set_down'
    : floorSupported !== true ? 'box_not_floor_supported'
    : remainingBudgetControls < minBudgetControls ? 'insufficient_budget' : 'eligible';
  return Object.freeze({ eligible: reason === 'eligible', reason, remainingDistanceM, remainingBudgetControls,
    toleranceM, maxErrorM, minBudgetControls, correctionsUsed, maxCorrections });
}
