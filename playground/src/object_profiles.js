// Phase B B4 (2026-09-20): per-object-class controller profile. DATA + PURE HELPERS ONLY.
// 2026-09-21 (B4/B5 shared shape, phaseb-objects): suitcase row carries B5's measured facts + the `carryOverrides` slot (null on every other row).
// Every largebox-bound site in main.js and the carry modules (see PHASEB_B4_PLASTICBOX_20260920.md, section 2) would read
// from the profile of the SELECTED body instead of the literal 'active_largebox_080_080_080'. Footprints are the mesh
// bounding-box extents measured from public/meshes/objects/*.obj (identical to object_pointclouds.json[key].bbox); the
// test in benchmarks/phaseb-plasticbox/test_object_profiles.mjs re-derives them from both files.
// Shape mirrors release-source-20260920/phaseb-multiobject-v6/web/src/object_skill_registry.js (frozen data, evidence strings).

export const OBJECT_CLASSES = Object.freeze(['largebox', 'plasticbox', 'suitcase', 'smallbox']);
export const DEFAULT_OBJECT_CLASS = 'largebox';
export const DEFAULT_OBJECT_BODY = 'active_largebox_080_080_080';
export const REFERENCE_STATUS = Object.freeze({
  RELEASED: 'released',                       // shipped v5 asset, qualified by P100/T20
  CANDIDATE: 'candidate_unqualified',         // exact teacher schema, source rows verified, NO native/goal-warp qualification yet
  // 2026-09-21 (CR-6, coordinator-approved): passed scripts/joint100_v2/export_carry_clip.py evaluate() VERBATIM (five checks; acceptance.json cited in
  // the row's evidence) but not shipped by v5. Admitted as a LIBRARY clip only when object-class routing is ON (releasedLibraryKeys routingOn).
  QUALIFIED: 'exporter_qualified',
  MISSING: 'missing',
});

const freeze = value => {
  if (value && typeof value === 'object') Object.values(value).forEach(freeze), Object.freeze(value);
  return value;
};
const footprint = (x, y, z) => freeze({ x, y, z, halfExtents: [x / 2, y / 2, z / 2] });
// Quarter-turn yaw symmetry (teacher_goal_warp.snapReferenceObjectYaw) is admissible only when |x - y| is within this
// tolerance: the largebox measures 0.377 x 0.367 m (1 cm apart). A 2:1 crate has only half-turn symmetry.
export const SQUARE_FOOTPRINT_TOLERANCE_M = 0.02;
const yawSymmetry = fp => (Math.abs(fp.x - fp.y) <= SQUARE_FOOTPRINT_TOLERANCE_M
  ? freeze({ turns: 4, stepRad: Math.PI / 2, kind: 'quarter_turn' })
  : freeze({ turns: 2, stepRad: Math.PI, kind: 'half_turn' }));
const ref = (file, status, evidence) => freeze({ file: file === null ? null : `public/${file}`, status, evidence });  // null file => no asset (never 'public/null')
// Rest pose of each object in the CANDIDATE scene = the rest pose of its source clips (2026-09-21, MEASURED with python mujoco 3.2.3 + the raw retargets):
// upAxisLocal = the body axis that is world-up at rest; footprintWorld = the mesh bbox re-ordered for that rest (x, y = horizontal, z = height);
// settledCentreZ = 5 s settle of the scene keyframe. The setdown-clearance diagnostic and any footprint-based rule must use footprintWorld, not the
// mesh-local bbox (`footprint`, kept for the B4 shape/tests). yawSymmetry stays the MEASURED mesh self-mirror symmetry about the up axis.
const rest = (upAxisLocal, x, y, z, settledCentreZ, evidence) => freeze({ upAxisLocal, footprintWorld: { x, y, z }, settledCentreZ, evidence });

// Per-object carry-option overrides (B5 patch parts A/B, applied behind default-OFF switches in teacher_carry_controller.js /
// teacher_controller.js / teacher_carry_sequence.js / quiet_ending.js). Consumed by main.js buildCarryOptions and quietEndingSample
// ONLY when objectClassRouting=1 (object_class_routing.js carryOutcomeOverrides / quietSettlingLimits / quietMeasurementOptions);
// with routing OFF nothing is spread and the v5 option shapes are unchanged. Suitcase values (B5, measured 2026-09-20):
//   outcomeRequirements = frozen largebox values + maxFinalObjectHeightM 0.30 (standing rest 0.222-0.241 m; a held suitcase is >= 0.57)
//     + tippedObjectTiltDeg 30 / objectUpAxisLocal 'y' => fail-closed 'object_tipped' outcome (static tip-over 37.6 deg);
//   quietLimits = v5 QUIET_ENDING_LIMITS verbatim + the same tilt requirement (validateQuietOptions requires every v5 key);
//   objectUpAxisLocal = the QuietEndingMeasurement axis (adds objectQuatXyzwWorld/objectUpAxisWorldZ/objectTiltDeg to the sample).
// Duplicated from suitcase_profile.js SUITCASE_SETTLING and asserted equal by benchmarks/phaseb-suitcase/test_suitcase_profile.mjs.
export const SUITCASE_CARRY_OVERRIDES = freeze({
  outcomeRequirements: { minLiftM: 0.35, maxFinalObjectHeightM: 0.30, minRootHeightM: 0.45, minUpright: 0.5, tippedObjectTiltDeg: 30, objectUpAxisLocal: 'y' },
  quietLimits: { rootPlanarSpeedMps: .05, minUpright: .95, boxSpeedMps: .05, floorForceMinimumN: .1, humanForceMaximumN: .1, tippedObjectTiltDeg: 30, objectUpAxisLocal: 'y' },
  objectUpAxisLocal: 'y',
});

// Plasticbox (2026-09-21, coordinator GO after pnat3 §16): the SAME B5 mechanism with the crate's MEASURED static tip-over angles. Resting body +y up
// (footprintWorld 0.4440 x 0.3082, height 0.2224 => centre 0.1112 above the floor): the crate tips over once its up axis leans past
// atan(0.1541 / 0.1112) = 54.2 deg about a LONG edge or atan(0.2220 / 0.1112) = 63.4 deg about a SHORT edge. tippedObjectTiltDeg = the SMALLER of the two
// (a resting crate tilted that far has fallen onto another face) - a geometry fact of the mesh, not a threshold fitted to any run. Outcome requirements
// otherwise = the frozen largebox values (a crate on its end is 0.222 m high, below maxFinalObjectHeightM 0.25: only the tilt rule separates it from a
// set-down). quietLimits = v5 QUIET_ENDING_LIMITS verbatim + the same tilt requirement (validateQuietOptions requires every v5 key).
// v6g (2026-09-21, B5 section 11 item 1): RELEASE-AND-LIFT exit micro-phase declaration (object_exit_release.js). Only the suitcase declares it;
// null everywhere else => TeacherBoxExitController.start() sees no option (v5/v6f exit path). Consumed by object_class_routing.js exitRelease ON only.
export const SUITCASE_EXIT_RELEASE = freeze({ hands: ['left_rubber_hand', 'right_rubber_hand'],
  evidence: 'B5 section 12.5 (pnat2, 4/4 cells): the releasing hand still touches the standing suitcase in the previewed exit step; a horizontal stand-off does not open it' });
// v6h (2026-09-21, B5 section 14): EXIT-HOLD HAND CLAMP declaration; replaces the release micro-phase (vacuous, B5 13.8: the hands come from ABOVE and the hold
// pulled them down through the top face). Only the suitcase declares it; null everywhere else => the v5 hold. Consumed by object_class_routing.js exitHandClamp ON only.
// v6i (2026-09-21, B5 section 14.6, coordinator ruling): the clamp margin is DATA-DERIVED from pnat4 (35238282). With the v6h margin 0.03 the policy tracked the clamped
// hold target with a consistent undershoot; measured per control by FK over the initial hold, ALL eight hands of the four cells (target z - min achieved hand z, m):
//   H052 L 0.0936  R 0.0843 | H055 L 0.0900  R 0.0842 | H056 L 0.0824  R 0.0727 | H058 L 0.0848  R 0.0673   (max 0.0936, mean 0.082, sd 0.008; file
//   benchmarks/phaseb-suitcase/suitcase_hold_undershoot_pnat4.json). Formula: margin = 0.03 (the v6h margin) + MAX undershoot over the eight hands = 0.1236 m (the max, not
// the mean: the clearance must hold for the worst hand; computed from all eight hands, not the three that touched). Raised hand target z = top + 0.1236 + hand half extent
// = 0.639 m: still 5-8 cm BELOW the exit-start hands (a descent, never a lift) and within 1-2 cm of the retreat clip's own hand height (0.636 / 0.653).
export const SUITCASE_EXIT_HAND_CLAMP_UNDERSHOOTS_M = freeze([0.0936, 0.0843, 0.0900, 0.0842, 0.0824, 0.0727, 0.0848, 0.0673]);
export const SUITCASE_EXIT_HAND_CLAMP_MARGIN_M = 0.03 + Math.max(...SUITCASE_EXIT_HAND_CLAMP_UNDERSHOOTS_M);   // 0.1236
export const SUITCASE_EXIT_HAND_CLAMP = freeze({ hands: ['left_rubber_hand', 'right_rubber_hand'], clearanceMarginM: SUITCASE_EXIT_HAND_CLAMP_MARGIN_M,
  evidence: 'B5 section 13.8 (pnat3, 4/4): hand at 0.70 m at the exit start, pulled to 0.38 m beside the 0.435 m suitcase; replay 14.1: clamped descent stays >= 3 cm above the top in 16/16; v6i margin 0.1236 = 0.03 + max pnat4 hold undershoot (section 14.6)' });
// v6j (2026-09-21, B5 section 15-16): EXIT HOLD SOURCE. pnat4/pnat5 showed the exit-hold hand height is the policy's posture prior for the carry's terminal crouch (target-
// independent, box bottom below the top plane in 8/8 episodes) and what separates completions from failures is the hand's horizontal clearance. The retreat clip's own
// row-0 pose IS a posture the policy holds (hands 0.61-0.67 m, standing) in every completed retreat; replay 14.1/16.1: hand boxes >= 0.14 m above the top, >= 0.23 m out.
// 'retreat_row0' = the initial hold reference is that frame aligned to the live root exactly as the retreat aligns itself (same transform, no new code path); the hold
// -> retreat seam is then the same aligned clip (frame 0 held, frames 1.. followed). Only the suitcase declares it; null => the v5 terminal-frame hold.
export const SUITCASE_EXIT_HOLD_SOURCE = 'retreat_row0';
export const PLASTICBOX_STATIC_TIP_OVER_DEG = freeze({
  aboutLongEdgeDeg: Math.round(Math.atan2(0.3082 / 2, 0.2224 / 2) * 1800 / Math.PI) / 10,    // 54.2
  aboutShortEdgeDeg: Math.round(Math.atan2(0.4440 / 2, 0.2224 / 2) * 1800 / Math.PI) / 10,   // 63.4
});
export const PLASTICBOX_TIPPED_TILT_DEG = Math.min(PLASTICBOX_STATIC_TIP_OVER_DEG.aboutLongEdgeDeg, PLASTICBOX_STATIC_TIP_OVER_DEG.aboutShortEdgeDeg);
export const PLASTICBOX_CARRY_OVERRIDES = freeze({
  outcomeRequirements: { minLiftM: 0.35, maxFinalObjectHeightM: 0.25, minRootHeightM: 0.45, minUpright: 0.5, tippedObjectTiltDeg: PLASTICBOX_TIPPED_TILT_DEG, objectUpAxisLocal: 'y' },
  quietLimits: { rootPlanarSpeedMps: .05, minUpright: .95, boxSpeedMps: .05, floorForceMinimumN: .1, humanForceMaximumN: .1, tippedObjectTiltDeg: PLASTICBOX_TIPPED_TILT_DEG, objectUpAxisLocal: 'y' },
  objectUpAxisLocal: 'y',
});

export const OBJECT_PROFILES = freeze({
  largebox: {
    objectClass: 'largebox',
    bodyName: 'active_largebox_080_080_080',
    pointCloudKey: 'largebox_080_080_080',                 // object_pointclouds.json key (64 rows); teacher refs carry 256 rows
    meshFile: 'public/meshes/objects/largebox_080_080_080.obj',
    footprint: footprint(0.3769, 0.3670, 0.3263),
    warp: { footprintM: 0.80, yawSymmetry: yawSymmetry({ x: 0.3769, y: 0.3670 }) },
    rest: rest('z', 0.3769, 0.3670, 0.3263, 0.1385, 'v5 scene keyframe (identity quat) = the largebox clips; settled centre 0.1385'),
    sceneMassKg: 0.172,                                     // wasm 3.3.8 + python 3.11 compile, density 10, legacy mesh inertia
    references: {
      pickup: ref('teacher_pickup_reference.json', REFERENCE_STATUS.RELEASED, 'v5 BOX_TASKS.pickup'),
      carry: ref('teacher_carry_reference.json', REFERENCE_STATUS.RELEASED, 'v5 BOX_TASKS.carry (1.0 m)'),
      carry_long: ref('teacher_carry_long_reference.json', REFERENCE_STATUS.RELEASED, 'v5 longCarryKey (2.33 m)'),
      library: {
        short_0184: ref('teacher_carry_short_0184_reference.json', REFERENCE_STATUS.RELEASED, 'v5 carry library'),
        short_0295: ref('teacher_carry_short_0295_reference.json', REFERENCE_STATUS.RELEASED, 'v5 carry library'),
        medium_1224: ref('teacher_carry_medium_1224_reference.json', REFERENCE_STATUS.RELEASED, 'v5 carry library'),
        alternate: ref('teacher_carry_alternate_reference.json', REFERENCE_STATUS.RELEASED, 'v5 alternateCarry opt-in'),
        long_sub16_010: ref('teacher_carry_long_sub16_010_reference.json', REFERENCE_STATUS.RELEASED, 'v5 longClipLibrary (OFF)'),
        long_sub8_042: ref('teacher_carry_long_sub8_042_reference.json', REFERENCE_STATUS.RELEASED, 'v5 longClipLibrary (OFF)'),
        mid_sub10_053_084: ref('teacher_carry_mid_sub10_053_084_reference.json', REFERENCE_STATUS.RELEASED, 'v5 midClipLibrary (OFF)'),
      },
      push_source: ref('task-assets/hand002_predecessor69_phase_reference.json', REFERENCE_STATUS.RELEASED, 'v5 default Push'),
    },
    matchedCarry: { sourceFrames: 366, bankRows: 386 },      // matched_carry_programme.js invariants (largebox only)
    carryOverrides: null,                                    // null => frozen v5 controller defaults (outcome requirements, quiet limits, nine-key sample)
    exitRelease: null,                                       // v6g: no release-and-lift (the exit programme was calibrated on the largebox)
    exitHandClamp: null,                                     // v6h: no hand clamp (the exit programme was calibrated on the largebox)
    exitHoldSource: null,                                    // v6j: the v5 terminal-frame hold
  },
  plasticbox: {
    objectClass: 'plasticbox',
    bodyName: 'active_plasticbox_080_080_080',
    pointCloudKey: 'plasticbox_080_080_080',
    meshFile: 'public/meshes/objects/plasticbox_080_080_080.obj',
    footprint: footprint(0.4440, 0.2224, 0.3082),
    // MEASURED mirror symmetry about the y-up axis: rms 6.5 mm under a half turn vs 46.7 mm under a quarter turn (18,920 OBJ vertices) => half turns.
    warp: { footprintM: 0.80, yawSymmetry: freeze({ turns: 2, stepRad: Math.PI, kind: 'half_turn' }) },
    // 446/447 source clips + the shipped reference rest the crate with body +y up (0.444 x 0.308 footprint, 0.222 tall, centre z 0.106); the candidate
    // scene now spawns it so (quat 0.7071 0.7071 0 0, settles to 0.1055). v5 spawned it +z up (0.308 tall, centre 0.1428) = a 90 deg roll vs every clip.
    rest: rest('y', 0.4440, 0.3082, 0.2224, 0.1055, 'B4 doc section 11; scene edit 2026-09-21'),
    sceneMassKg: 0.2644,                                    // candidate g1_scene.xml explicit <inertial> (density 10 x convex hull)
    references: {
      pickup: ref(null, REFERENCE_STATUS.MISSING, 'no plasticbox pickup source exported'),
      carry: ref('teacher_carry_plasticbox_073_reference.json', REFERENCE_STATUS.CANDIDATE,
        'plastic073 phase 45-335 (job 34871879 source reproduction); exact teacher schema; native/goal-warp qualification NOT run'),
      // Option A (coordinator 2026-09-21): the long slot is the SAME plastic073 clip as `carry`, so main.js builds the v5 four-slot carry library
      // (staged = long = plastic073; general 3 m plans stay one plastic073 segment) and the placement-correction / replan machinery gets a library context.
      carry_long: ref('teacher_carry_plasticbox_073_reference.json', REFERENCE_STATUS.CANDIDATE, 'Option A: same file as carry (B4 doc section 15.3)'),
      // CR-6 exporter-qualified SHORT clips (benchmarks/phaseb-plasticbox/candidate-qualification-*/acceptance.json; handoff_ready=true, failed=[]);
      // web JSONs written by convert_native_reference_to_web.py (refuses without a passing acceptance); sidecars in benchmarks/phaseb-plasticbox/.
      library: {
        short_0184: ref('teacher_carry_plasticbox_020_short_reference.json', REFERENCE_STATUS.QUALIFIED,
          'CR-6 job 35172539 sub1_plasticbox_020_127_087_082 rows 20..447 handoff_ready=true (in-place lift/set-down, travel 0.168 m; qualified alternate: job 35172540 sub1_plasticbox_020_127_087_080)'),
        short_0295: ref('teacher_carry_plasticbox_027_short_reference.json', REFERENCE_STATUS.QUALIFIED,
          'CR-6 job 35171615 sub16_plasticbox_027_084_097_074 rows 20..329 handoff_ready=true (short carry, travel 0.871 m)'),
      },
      push_source: ref(null, REFERENCE_STATUS.MISSING, 'none'),
    },
    matchedCarry: null,
    carryOverrides: PLASTICBOX_CARRY_OVERRIDES,                 // v6e (2026-09-21): tipped rule from the measured static tip-over; null in v6b
    exitRelease: null,                                          // v6g: no rule declared (design only for the crate; see B4 section 16.5)
    exitHandClamp: null,                                        // v6h: no rule declared for the crate
    exitHoldSource: null,                                       // v6j: no rule declared for the crate
  },
  suitcase: {
    objectClass: 'suitcase',
    bodyName: 'active_suitcase_080_080_080',
    pointCloudKey: 'suitcase_080_080_080',
    meshFile: 'public/meshes/objects/suitcase_080_080_080.obj',
    footprint: footprint(0.3279, 0.4252, 0.3306),          // mesh bbox extents in mesh-local axes (= v5 flat orientation; B4 semantics).
    // B5 (2026-09-20): the candidate scene rests the suitcase STANDING on body +y (0.328 x 0.331 footprint, 0.425 m tall). Its yaw
    // symmetry is the MEASURED mirror symmetry of the mesh (rms 5.8 mm under a half turn vs 34 mm under a quarter turn), not the
    // bbox-squareness rule: the standing footprint is 3 mm from square and that rule would wrongly grant quarter turns.
    warp: { footprintM: 0.80, yawSymmetry: freeze({ turns: 2, stepRad: Math.PI, kind: 'half_turn' }) },
    rest: rest('y', 0.3279, 0.3306, 0.4252, 0.2224, 'B5 section 1: standing on body +y (scene quat 0.7071 0.7071 0 0), settled centre 0.2224'),
    sceneMassKg: 0.2278,                                    // MuJoCo compile of the candidate scene (B5 verify_suitcase_physics.py; trimesh 0.2253)
    references: {
      pickup: ref(null, REFERENCE_STATUS.MISSING, 'no suitcase pickup source exported'),
      carry: ref('teacher_carry_suitcase_sub8_042_reference.json', REFERENCE_STATUS.CANDIDATE,
        'B5 suitcase042 phase 77-286 converted (carry_interval_frames [53,153], travel 2.0413 m); exact teacher schema; unqualified (no quiet exit, live entry not authorized)'),
      carry_long: ref(null, REFERENCE_STATUS.MISSING, 'none'),
      library: {},
      push_source: ref(null, REFERENCE_STATUS.MISSING, 'none'),
    },
    matchedCarry: null,
    carryOverrides: SUITCASE_CARRY_OVERRIDES,
    exitRelease: null,                                        // v6g release-and-lift RETIRED in v6h (vacuous: B5 section 13.8); the declaration constant is kept for the record
    exitHandClamp: null,                                      // v6h/v6i hand clamp SUPERSEDED in v6j (B5 section 15: the hold hand height is the policy's posture prior); constants kept for the record
    exitHoldSource: SUITCASE_EXIT_HOLD_SOURCE,                // v6j (2026-09-21): initial exit hold = the retreat clip's row-0 frame at the live root (ON only)
  },
  smallbox: {
    objectClass: 'smallbox',
    bodyName: 'active_smallbox_080_080_080',
    pointCloudKey: 'smallbox_080_080_080',
    meshFile: 'public/meshes/objects/smallbox_080_080_080.obj',
    footprint: footprint(0.3079, 0.3487, 0.1882),
    warp: { footprintM: 0.80, yawSymmetry: yawSymmetry({ x: 0.3079, y: 0.3487 }) },
    rest: rest('z', 0.3079, 0.3487, 0.1882, null, 'v5 scene keyframe (identity quat); out of scope, settle not measured'),
    sceneMassKg: 0.056,
    references: { pickup: ref(null, REFERENCE_STATUS.MISSING, 'out of scope (user)'), carry: ref(null, REFERENCE_STATUS.MISSING, 'out of scope (user)'),
      carry_long: ref(null, REFERENCE_STATUS.MISSING, 'none'), library: {}, push_source: ref(null, REFERENCE_STATUS.MISSING, 'none') },
    matchedCarry: null,
    carryOverrides: null,
    exitRelease: null,
    exitHandClamp: null,
    exitHoldSource: null,
  },
});

/** 'active_plasticbox_080_080_080' -> 'plasticbox'; null for anything that is not a known 080 scene body. */
export function objectClassFromBodyName(bodyName) {
  const m = /^active_([a-z]+)_080_080_080$/.exec(String(bodyName ?? ''));
  return m && OBJECT_CLASSES.includes(m[1]) ? m[1] : null;
}
/** Profile of a scene body. Fails closed: unknown bodies throw (never silently fall back to the largebox). */
export function profileForBody(bodyName) {
  const cls = objectClassFromBodyName(bodyName);
  if (!cls) throw new Error(`Unknown object body: ${bodyName}`);
  return OBJECT_PROFILES[cls];
}
export function profileForClass(objectClass) {
  if (!OBJECT_CLASSES.includes(objectClass)) throw new Error(`Unknown object class: ${objectClass}`);
  return OBJECT_PROFILES[objectClass];
}
/** Reference URL for a skill key ('pickup' | 'carry' | 'carry_long' | library id). Null when the class has no such asset. */
export function referenceUrl(profile, key) {
  const entry = profile.references[key] ?? profile.references.library[key] ?? null;
  return entry?.file ?? null;
}
export function referenceStatus(profile, key) {
  const entry = profile.references[key] ?? profile.references.library[key] ?? null;
  return entry?.status ?? REFERENCE_STATUS.MISSING;
}
/** Library keys whose asset is released for this class (v5 largebox: the 7 teacher carry clips; others: none).
 *  routingOn (2026-09-21): with object-class routing ON, exporter-QUALIFIED clips (CR-6) are admitted too; the default (OFF) admits
 *  RELEASED only, so the v5 largebox key set is unchanged and no unshipped clip can enter the OFF path. */
export function releasedLibraryKeys(profile, { routingOn = false } = {}) {
  if (typeof routingOn !== 'boolean') throw new Error('routingOn must be boolean');
  const admitted = routingOn ? [REFERENCE_STATUS.RELEASED, REFERENCE_STATUS.QUALIFIED] : [REFERENCE_STATUS.RELEASED];
  return Object.entries(profile.references.library).filter(([, r]) => admitted.includes(r.status)).map(([k]) => k);
}
/** Whether a loaded teacher skill (loadTeacherSkill output) belongs to this profile. */
export function skillMatchesProfile(skill, profile) {
  return typeof skill?.objectBodyName === 'string' && skill.objectBodyName === profile.bodyName;
}
/** Approximate AABB half extents used by the setdown clearance diagnostic (main.js:3693 hard-codes the largebox's). */
export function collisionHalfExtents(profile) {
  return Array.from(profile.footprint.halfExtents);
}
