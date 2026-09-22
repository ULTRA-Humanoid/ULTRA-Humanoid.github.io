// Phase B B4 (2026-09-20): object-class routing behind the URL flag `objectClassRouting=1` (DEFAULT OFF).
// 2026-09-21 (phaseb-objects, B4/B5 shared): + carryOutcomeOverrides / quietSettlingLimits / quietMeasurementOptions (B5 suitcase overrides; OFF => {}).
// OFF  -> every routed site in main.js resolves to the v5 largebox literal it replaced (proved against the v5 source text by
//         benchmarks/phaseb-plasticbox/test_object_class_routing.mjs): body name, reference URLs, library keys, yaw
//         symmetry (null => teacher_goal_warp's built-in quarter-turn loop), setdown-clearance extents, refusal text.
// ON   -> the SELECTED body's profile (src/object_profiles.js) is used; bodies without a carry reference (suitcase,
//         smallbox) and unknown bodies are refused with the v5 refusal text; unknown bodies otherwise throw (fail closed).
// The push lane is NOT routed: it stays bound to the largebox (only the hand002 source exists) via PUSH_LANE_BODY.
import { OBJECT_PROFILES, DEFAULT_OBJECT_BODY, REFERENCE_STATUS, objectClassFromBodyName, profileForBody,
  referenceUrl as profileReferenceUrl, referenceStatus, releasedLibraryKeys } from './object_profiles.js';
import { exitStandOffWorld } from './object_exit_standoff.js';
import { exitReleaseOption, exitHandClampOption } from './object_exit_release.js';

export const OBJECT_CLASS_ROUTING_PARAM = 'objectClassRouting';
/** v5 main.js floor-click refusal text (kept verbatim for every refused class so the public carry surface is unchanged). */
export const LARGEBOX_ONLY_CARRY_REFUSAL = 'Carry controls currently use the large box.';
/** v5 main.js setdown clearance diagnostic literal `const half = [.377, .367, .326]` (labelled half extents there; the numbers
 * are the largebox mesh's FULL bbox extents 0.3769/0.3670/0.3263 rounded to 3 decimals). Kept verbatim when routing is off. */
export const SETDOWN_CLEARANCE_EXTENTS_V5 = Object.freeze([.377, .367, .326]);
/** The ground-push lane binds to this body unconditionally (v5 literal). */
export const PUSH_LANE_BODY = OBJECT_PROFILES.largebox.bodyName;

/** `?objectClassRouting=1` and nothing else enables routing (mirrors the other `=== '1'` flags in main.js). */
export function readObjectClassRouting(params) {
  return typeof params?.get === 'function' && params.get(OBJECT_CLASS_ROUTING_PARAM) === '1';
}

export class ObjectClassRouter {
  constructor({ enabled = false } = {}) {
    if (typeof enabled !== 'boolean') throw new Error('objectClassRouting must be boolean');
    this.enabled = enabled;
  }
  /** OFF: always the largebox profile (the body argument is ignored, exactly as the v5 literals ignored it).
   *  ON: the selected body's profile; throws on unknown bodies. */
  profileFor(selectedBodyName) {
    return this.enabled ? profileForBody(selectedBodyName) : OBJECT_PROFILES.largebox;
  }
  carryBodyName(selectedBodyName) { return this.profileFor(selectedBodyName).bodyName; }
  /** main.js floor-click gate (v5: `activeObjName !== 'active_largebox_080_080_080'` => refusal).
   *  Returns the status text to show when the click must be refused, else null. */
  carryRefusal(selectedBodyName) {
    if (!this.enabled) return selectedBodyName === DEFAULT_OBJECT_BODY ? null : LARGEBOX_ONLY_CARRY_REFUSAL;
    return this.taskRefusal(selectedBodyName, 'carry');
  }
  /** startBoxTask gate. v5 had none (OFF => never refuses). ON => refuse when the selected body is unknown or has no
   *  reference for `kind` ('pickup' | 'carry'); status text is the v5 refusal text. */
  taskRefusal(selectedBodyName, kind) {
    if (!this.enabled) return null;
    const cls = objectClassFromBodyName(selectedBodyName);
    if (!cls || referenceStatus(OBJECT_PROFILES[cls], kind) === REFERENCE_STATUS.MISSING) return LARGEBOX_ONLY_CARRY_REFUSAL;
    return null;
  }
  /** Reference URL for 'pickup' | 'carry' | 'carry_long' | library id of a resolved profile (null when the class has none). */
  referenceUrl(profile, key) { return profileReferenceUrl(profile, key); }
  /** loadedSkills cache key for the primary reference. OFF: the v5 key (`kind`). ON: per class, so a session that carries
   *  two different objects never serves one object's cached clip to the other. */
  skillCacheKey(profile, kind) { return this.enabled ? `${kind}@${profile.objectClass}` : kind; }
  /** Library keys in the caller's v5 order, restricted to the profile's released (ON: + exporter-qualified) clips. OFF: the caller's array, untouched. */
  libraryKeys(profile, orderedKeys) {
    if (!this.enabled) return orderedKeys;
    const released = releasedLibraryKeys(profile, { routingOn: this.enabled });   // ON: released + exporter-qualified (CR-6) clips
    return orderedKeys.filter(key => released.includes(key));
  }
  /** Yaw symmetry passed to the carry controller. OFF: null (teacher_goal_warp keeps its v5 quarter-turn loop). */
  yawSymmetry(skillObjectBodyName) { return this.enabled ? profileForBody(skillObjectBodyName).warp.yawSymmetry : null; }
  /** Setdown clearance AABB extents. OFF: the v5 literal. ON: the profile footprint (same full-extent convention as v5). */
  setdownClearanceExtents(skillObjectBodyName) {
    if (!this.enabled) return Array.from(SETDOWN_CLEARANCE_EXTENTS_V5);
    // ON: the object's WORLD extents at rest (rest.footprintWorld: x, y horizontal, z height) - for a body resting on +y the mesh bbox y/z swap.
    const profile = profileForBody(skillObjectBodyName), { x, y, z } = profile.rest?.footprintWorld ?? profile.footprint;
    return [x, y, z];
  }
  // ---- B5 carry-option overrides (object_profiles.js `carryOverrides`), 2026-09-21. Every accessor returns an object to SPREAD:
  // OFF => {} so no key is added (carryOptions are serialised into carryReferenceSelection.mixedSegments and the quiet sample is a
  // nine-key record: the OFF shapes must stay byte-identical to v5). ON => the skill object's declared overrides, or {} when the
  // profile declares none (largebox/plasticbox/smallbox: carryOverrides null). Unknown bodies throw (profileForBody).
  /** {outcomeRequirements} for CarryGoalSequenceController/CarryGoalController (teacher_carry_controller.js override merge). */
  carryOutcomeOverrides(skillObjectBodyName) {
    const requirements = this.enabled ? profileForBody(skillObjectBodyName).carryOverrides?.outcomeRequirements ?? null : null;
    return requirements ? { outcomeRequirements: requirements } : {};
  }
  /** {limits} for the quietSettling option object (QuietHoldMonitor limits; validateQuietOptions checks completeness). */
  quietSettlingLimits(skillObjectBodyName) {
    const limits = this.enabled ? profileForBody(skillObjectBodyName).carryOverrides?.quietLimits ?? null : null;
    return limits ? { limits } : {};
  }
  /** v6e (2026-09-21): option object to SPREAD into TeacherBoxExitController.start(): OFF => {} (start() sees no option, v5 path byte-identical);
   *  ON => {standOffWorld: [dx, dy, 0]} when the placed object stands taller than the largebox at rest (object_exit_standoff.js), else {}.
   *  The largebox always yields {} (offset exactly 0 by definition). Unknown bodies / unusable poses throw (fail closed). */
  exitStandOff(objectBodyName, proprio) {
    if (!this.enabled) return {};
    const result = exitStandOffWorld(profileForBody(objectBodyName), proprio?.objQuatXyzwWorld, proprio?.objPosWorld, proprio?.rootPosWorld);
    return result ? { standOffWorld: Array.from(result.standOffWorld) } : {};
  }
  /** v6g (2026-09-21, B5 section 11 item 1): option object to SPREAD into TeacherBoxExitController.start(): OFF => {} (v5/v6f path); ON => {releaseLift}
   *  when the placed object's profile declares `exitRelease` (suitcase), else {} (largebox/plasticbox/smallbox declare null). `measure` is the caller's
   *  synchronous contact measurement (QuietEndingMeasurement sample) that ends the micro-phase; the live standing half extent comes from the same
   *  geometry as the stand-off. Unknown bodies / unusable poses / a missing measure throw (fail closed). */
  exitRelease(objectBodyName, proprio, { measure } = {}) {
    if (!this.enabled) return {};
    const option = exitReleaseOption(profileForBody(objectBodyName), proprio, { measure });
    return option ? { releaseLift: option } : {};
  }
  /** v6h (2026-09-21, B5 section 14): option object to SPREAD into TeacherBoxExitController.start(): OFF => {} (v5 hold); ON => {holdHandClamp} when the placed
   *  object's profile declares `exitHandClamp` (suitcase), else {} (largebox/plasticbox/smallbox declare null). Unknown bodies / unusable poses throw (fail closed). */
  exitHandClamp(objectBodyName, proprio) {
    if (!this.enabled) return {};
    const option = exitHandClampOption(profileForBody(objectBodyName), proprio);
    return option ? { holdHandClamp: option } : {};
  }
  /** v6j (2026-09-21, B5 section 16): option object to SPREAD into TeacherBoxExitController.start(): OFF => {} (v5 hold); ON => {holdSource: 'retreat_row0'} when the placed
   *  object's profile declares `exitHoldSource` (suitcase), else {} (largebox/plasticbox/smallbox declare null). Any other declared value throws (fail closed). */
  exitHoldSource(objectBodyName) {
    if (!this.enabled) return {};
    const source = profileForBody(objectBodyName).exitHoldSource;
    if (source === null || source === undefined) return {};
    if (source !== 'retreat_row0') throw new Error('Exit hold source must be retreat_row0 or null: ' + String(source));
    return { holdSource: source };
  }
  /** {objectUpAxisLocal} for the QuietEndingMeasurement constructor of the placed body (adds the tilt channels to the sample). */
  quietMeasurementOptions(objectBodyName) {
    const axis = this.enabled ? profileForBody(objectBodyName).carryOverrides?.objectUpAxisLocal ?? null : null;
    return axis ? { objectUpAxisLocal: axis } : {};
  }
}
