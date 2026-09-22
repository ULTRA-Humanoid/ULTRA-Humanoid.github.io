// Phase B / B5 (suitcase carry) object profile. DORMANT: not imported by main.js. Dependency-free so a Node test can load a
// copy directly. Designed to plug into the shared object_profiles registry written for plasticbox (B4,
// release-source-20260920/phaseb-plasticbox-v6b/web/src/object_profiles.js): `suitcaseObjectProfileEntry()` returns an object whose
// B4 keys {objectClass, bodyName, pointCloudKey, meshFile, footprint{x,y,z,halfExtents}, warp{footprintM, yawSymmetry{turns,stepRad,kind}},
// sceneMassKg, references{pickup,carry,carry_long,library,push_source} (each {file,status,evidence}), matchedCarry} follow B4's shape and
// semantics exactly (footprint = MESH bbox extents in mesh-local axes, as B4 measures them, NOT the upright world footprint), plus B5
// extension keys that B4 does not define (geometryIdentity, referenceUrlFor, referenceLibrary, footprintStanding, rest, settling,
// physics, mesh, upright, classifySettling, yawSnap, footprintWorld, qualification, carryOptionOverrides). Known field mismatches vs
// B4's own placeholder `suitcase` row are listed in PHASEB_B5_SUITCASE_20260920.md (section 8) instead of being guessed away here.
// `carryOptionOverrides()` is the exact option set the routing switch (patch part C, NOT applied) would spread into
// main.js buildCarryOptions for this object: yawSymmetry {turns 2, stepRad PI}, outcomeRequirements (+ tilt), quiet limits (+ tilt), tilt sample.
// Every number is MEASURED on 2026-09-20 (trimesh 5.0 / MuJoCo 3.11 on web/public/meshes/objects/suitcase_080_080_080.obj and
// g1_scene.xml; 708 OMOMO retargets; see benchmarks/phaseb-suitcase/suitcase_physics.json). Nothing here is a tuned threshold:
// the settling proposal reuses the v5 QUIET_ENDING limits and only widens the object-height acceptance to the measured rest.
const freeze = Object.freeze;
export const SUITCASE_OBJECT_CLASS = 'suitcase';
export const SUITCASE_BODY_NAME = 'active_suitcase_080_080_080';
export const SUITCASE_GEOMETRY_IDENTITY = 'g1_scene.xml#active_suitcase_080_080_080';
export const SUITCASE_POINT_CLOUD_KEY = 'suitcase_080_080_080';          // public/object_pointclouds.json key (64 rows for the UI/goal; the teacher contract is the 256-row object_points256_local of each reference)
export const SUITCASE_REFERENCE_PREFIX = 'public/teacher_carry_suitcase_';
export const SUITCASE_MESH = freeze({
  file: 'meshes/objects/suitcase_080_080_080.obj', vertices: 38353, faces: 76718, watertight: true, volumeM3: 0.0225318,
  bboxMin: freeze([-0.16457, -0.22597, -0.16761]), bboxMax: freeze([0.16337, 0.19925, 0.16303]),
  extents: freeze([0.32794, 0.42522, 0.33063]), centreOfMassLocal: freeze([-0.00086, -0.01738, -0.00153]),
});
export const SUITCASE_PHYSICS = freeze({
  sceneClass: 'suitcase_080_080_080_phys', densityKgM3: 10, frictionSliding: 1.2, solref: '0.01 1', solimp: '0.99 0.99 0.01',
  massKgMujoco: 0.2278, massKgTrimesh: 0.2253, inertiaDiagKgM2: freeze([0.004420, 0.003607, 0.002283]),
  sceneClassChangeNeeded: false, // shared class is fine; the mass is a plausible rigid 0.23 kg suitcase and MuJoCo/trimesh agree within 1.1 %
});
// Rest pose. ALL 708 retargets (and the exported suitcase042 reference) rest the suitcase STANDING on body +y (the 0.425 m
// axis vertical, centre z 0.225-0.241). v5 spawned it flat (identity quat) which is not even a stable rest for this mesh
// (it rolls to a 45 deg edge and keeps creeping after 10 s). The candidate scene spawns it upright.
export const SUITCASE_REST = freeze({
  upAxisLocal: 'y', bodyQuatWxyz: freeze([0.70710678, 0.70710678, 0, 0]), spawnZ: 0.23,
  settledCentreZ: 0.2224, settledCentreXY: freeze([-1.4941, -1.5075]), dataRestCentreZRange: freeze([0.225, 0.241]),
  v5FlatPoseStableRest: false,
});
export const SUITCASE_FOOTPRINT = freeze({
  // standing (data + candidate scene): near-square 0.328 x 0.331 m but NOT quarter-turn symmetric (handle/wheels): the mesh
  // mirrors onto itself under a HALF turn about the vertical axis (rms 5.8 mm) and not under a quarter turn (rms 34 mm).
  standing: freeze({ xM: 0.32794, yM: 0.33063, heightM: 0.42522, yawSymmetryTurns: 2, halfTurnMirrorRmsM: 0.0058, quarterTurnMirrorRmsM: 0.0343 }),
  flatV5: freeze({ xM: 0.32794, yM: 0.42522, heightM: 0.33063, yawSymmetryTurns: 1 }),
  largeboxForComparison: freeze({ xM: 0.37692, yM: 0.36698, heightM: 0.32632, yawSymmetryTurns: 4 }),
});
// Terminal settling proposal for a tall/narrow standing object. Reuses v5 quiet limits verbatim (quiet_ending.js:12,16).
export const SUITCASE_SETTLING = freeze({
  settlingSteps: 180,
  quietSettling: freeze({ window: 30, minControls: 60, rootPlanarSpeedMps: 0.05, boxSpeedMps: 0.05, minUpright: 0.95 }),
  // teacher_carry_controller.js:31 uses maxFinalObjectHeightM 0.25 for the largebox (rest centre 0.138). A standing suitcase
  // rests at 0.222 (scene) / 0.225-0.241 (data): 9 mm of margin. 0.30 accepts the measured standing rest and a flat tipped
  // suitcase (0.165) alike, while a held suitcase (>= minLiftM above rest) is >= 0.57.
  // + tippedObjectTiltDeg/objectUpAxisLocal: the CarryGoalController `outcomeRequirements` override (patch part B) that adds the
  // fail-closed 'object_tipped' outcome (teacher_controller.js). Absent from the frozen largebox requirements.
  outcomeRequirements: freeze({ minLiftM: 0.35, maxFinalObjectHeightM: 0.30, minRootHeightM: 0.45, minUpright: 0.5,
    tippedObjectTiltDeg: 30, objectUpAxisLocal: 'y' }),
  // Quiet-ending limits for the settling monitor: v5 QUIET_ENDING_LIMITS verbatim (quiet_ending.js:11-14) + the same tilt requirement.
  quietLimits: freeze({ rootPlanarSpeedMps: 0.05, minUpright: 0.95, boxSpeedMps: 0.05, floorForceMinimumN: 0.1, humanForceMaximumN: 0.1,
    tippedObjectTiltDeg: 30, objectUpAxisLocal: 'y' }),
  // Object tilt monitor (new diagnostic, fail-closed classification): tilt = angle between body +y and world +z.
  // Static tip-over angle of the standing footprint = atan(0.164 / 0.2126) = 37.6 deg; warn at 15, classify tipped at 30.
  objectTilt: freeze({ upAxisLocal: 'y', warnDeg: 15, tippedDeg: 30, staticTipOverDeg: 37.6 }),
  // Source qualification (screening): release root speed <= 0.05 m/s and a 20-control quiet window after release.
  releaseTerminal: freeze({ referenceRowsAfterReleaseMin: 20, releaseRootSpeedMaxMps: 0.05, quietWindowControls: 20 }),
});
export const SUITCASE_REFERENCE_LIBRARY = freeze([
  freeze({ id: 'sub8_042', source: 'sub8_suitcase_042_095_101_076_080_080_080', sourcePhaseInclusive: freeze([77, 286]), sourceFrames: 210,
    carryIntervalFrames: freeze([53, 153]), qualification: 'unqualified_source_candidate', quietExit: false,
    evidence: 'source-reset reproduction PASS 35005466; live settling tipped/drifted 24.0-27.6 cm; exit 286 not quiet' }),
]);
const isFiniteVec = (v, n) => Array.isArray(v) || ArrayBuffer.isView(v) ? v.length === n && Array.from(v).every(Number.isFinite) : false;
export function isSuitcaseBody(bodyName) { return bodyName === SUITCASE_BODY_NAME; }
export function suitcaseReferenceUrl(libraryId) {
  if (typeof libraryId !== 'string' || !/^[a-z0-9]+(_[a-z0-9]+)*$/.test(libraryId)) throw new Error('Suitcase reference id must be a snake_case token');
  return `${SUITCASE_REFERENCE_PREFIX}${libraryId}_reference.json`;
}
/** World-z component of the body +y axis for an xyzw quaternion: 1 = standing on its base, 0 = lying on a side, -1 = upside down. */
export function suitcaseUpright(quatXyzw) {
  if (!isFiniteVec(quatXyzw, 4) || Math.abs(Math.hypot(...quatXyzw) - 1) > 1e-3) throw new Error('Suitcase upright requires a finite unit xyzw quaternion');
  const [x, y, z, w] = quatXyzw; return 2 * (y * z + w * x);
}
export function suitcaseTiltDeg(quatXyzw) { return Math.acos(Math.max(-1, Math.min(1, suitcaseUpright(quatXyzw)))) * 180 / Math.PI; }
/** Fail-closed settling classification for one measured control. Missing/non-finite inputs classify as NOT settled. */
export function classifySuitcaseSettling(sample, limits = SUITCASE_SETTLING) {
  const q = sample?.objectQuatXyzwWorld, h = sample?.objectHeightM, v = sample?.boxSpeedMps, r = sample?.rootPlanarSpeedMps;
  const finite = isFiniteVec(q, 4) && Number.isFinite(h) && Number.isFinite(v) && Number.isFinite(r) && v >= 0 && r >= 0;
  if (!finite) return freeze({ settled: false, tipped: true, reasons: ['invalid_sample'], tiltDeg: NaN });
  const tiltDeg = suitcaseTiltDeg(q), reasons = [];
  if (tiltDeg >= limits.objectTilt.tippedDeg) reasons.push('object_tipped');
  else if (tiltDeg >= limits.objectTilt.warnDeg) reasons.push('object_tilt_warning');
  if (h > limits.outcomeRequirements.maxFinalObjectHeightM) reasons.push('object_not_set_down');
  if (v > limits.quietSettling.boxSpeedMps) reasons.push('box_moving');
  if (r > limits.quietSettling.rootPlanarSpeedMps) reasons.push('root_moving');
  const tipped = reasons.includes('object_tipped');
  return freeze({ settled: reasons.filter(x => x !== 'object_tilt_warning').length === 0, tipped, reasons: freeze(reasons), tiltDeg,
    standingHeightPlausible: Math.abs(h - SUITCASE_REST.settledCentreZ) <= 0.03 });
}
/** Half-turn-only yaw snap (replaces the largebox quarter-turn assumption of teacher_goal_warp.snapReferenceObjectYaw).
 * Returns the number of half turns (0 or 1) that brings referenceYaw closest to liveYaw and the residual. */
export function suitcaseYawSnap(referenceYawRad, liveYawRad, symmetryTurns = SUITCASE_FOOTPRINT.standing.yawSymmetryTurns) {
  if (![referenceYawRad, liveYawRad].every(Number.isFinite) || !Number.isInteger(symmetryTurns) || symmetryTurns < 1) throw new Error('Finite yaws and a positive turn count are required');
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a)); let best = { turns: 0, residualYawRad: Infinity, yawRad: 0 };
  for (let turns = 0; turns < symmetryTurns; turns++) {
    const yawRad = turns * 2 * Math.PI / symmetryTurns, residual = wrap(liveYawRad - (referenceYawRad + yawRad));
    if (Math.abs(residual) < Math.abs(best.residualYawRad)) best = { turns, residualYawRad: residual, yawRad };
  }
  return freeze(best);
}
/** Planar extents (x, y) and height of the axis-aligned world box of the mesh bbox under a world xyzw quaternion. */
export function suitcaseFootprintWorld(quatXyzw) {
  if (!isFiniteVec(quatXyzw, 4)) throw new Error('Footprint requires a finite xyzw quaternion');
  const [x, y, z, w] = quatXyzw, rot = p => {
    const [px, py, pz] = p, tx = 2 * (y * pz - z * py), ty = 2 * (z * px - x * pz), tz = 2 * (x * py - y * px);
    return [px + w * tx + (y * tz - z * ty), py + w * ty + (z * tx - x * tz), pz + w * tz + (x * ty - y * tx)];
  };
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], { bboxMin: a, bboxMax: b } = SUITCASE_MESH;
  for (const cx of [a[0], b[0]]) for (const cy of [a[1], b[1]]) for (const cz of [a[2], b[2]]) {
    const p = rot([cx, cy, cz]); for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
  }
  return freeze({ xM: hi[0] - lo[0], yM: hi[1] - lo[1], heightM: hi[2] - lo[2], lowestOffsetM: lo[2] });
}
/** The carry-option switches for this object (all default-OFF/largebox in the controllers): spread into the CarryGoalSequenceController
 * options (main.js buildCarryOptions) by the routing switch of patch part C, which is NOT applied. `quietLimits` goes into
 * `quietSettling.limits`; `quietMeasurement` into the QuietEndingMeasurement constructor of quietEndingSample(). */
export function suitcaseCarryOptionOverrides() {
  return freeze({ yawSymmetry: { turns: SUITCASE_FOOTPRINT.standing.yawSymmetryTurns, stepRad: 2 * Math.PI / SUITCASE_FOOTPRINT.standing.yawSymmetryTurns, kind: 'half_turn' },
    outcomeRequirements: SUITCASE_SETTLING.outcomeRequirements,
    quietLimits: SUITCASE_SETTLING.quietLimits, quietMeasurement: freeze({ objectUpAxisLocal: SUITCASE_SETTLING.objectTilt.upAxisLocal }) });
}
// B4 REFERENCE_STATUS string values, copied verbatim (object_profiles.js:11-15) so the entry is comparable without importing B4.
export const B4_REFERENCE_STATUS = freeze({ RELEASED: 'released', CANDIDATE: 'candidate_unqualified', QUALIFIED: 'exporter_qualified', MISSING: 'missing' });   // QUALIFIED: 2026-09-21 CR-6 (shared object_profiles.js), admitted only with routing ON
const b4ref = (file, status, evidence) => freeze({ file, status, evidence });
/** Entry for the shared object_profiles registry (B4/B5 merge), B4 OBJECT_PROFILES shape + B5 extension keys (see header). */
export function suitcaseObjectProfileEntry() {
  const [x, y, z] = SUITCASE_MESH.extents;
  const carry = b4ref(suitcaseReferenceUrl('sub8_042'), B4_REFERENCE_STATUS.CANDIDATE,
    'B5 suitcase042 phase 77-286 converted (carry_interval_frames [53,153]); exact teacher schema; unqualified (no quiet exit, live entry not authorized)');
  return freeze({
    // --- B4 keys (shape + semantics of object_profiles.js OBJECT_PROFILES.<cls>) ---
    objectClass: SUITCASE_OBJECT_CLASS, bodyName: SUITCASE_BODY_NAME, pointCloudKey: SUITCASE_POINT_CLOUD_KEY,
    meshFile: `public/${SUITCASE_MESH.file}`,
    footprint: freeze({ x, y, z, halfExtents: freeze([x / 2, y / 2, z / 2]) }),          // mesh bbox extents (B4 semantics), not upright footprint
    warp: freeze({ footprintM: 0.80, yawSymmetry: freeze({ turns: 2, stepRad: Math.PI, kind: 'half_turn' }) }),  // measured mirror symmetry, not bbox squareness
    sceneMassKg: SUITCASE_PHYSICS.massKgMujoco,
    references: freeze({ pickup: b4ref(null, B4_REFERENCE_STATUS.MISSING, 'no suitcase pickup source exported'), carry,
      carry_long: b4ref(null, B4_REFERENCE_STATUS.MISSING, 'none'), library: freeze({ sub8_042: carry }),
      push_source: b4ref(null, B4_REFERENCE_STATUS.MISSING, 'none') }),
    matchedCarry: null,
    // --- B5 extension keys (not defined by B4) ---
    geometryIdentity: SUITCASE_GEOMETRY_IDENTITY, referenceUrlFor: suitcaseReferenceUrl, referenceLibrary: SUITCASE_REFERENCE_LIBRARY,
    footprintStanding: SUITCASE_FOOTPRINT.standing, rest: SUITCASE_REST, settling: SUITCASE_SETTLING, physics: SUITCASE_PHYSICS, mesh: SUITCASE_MESH,
    upright: suitcaseUpright, classifySettling: classifySuitcaseSettling, yawSnap: suitcaseYawSnap, footprintWorld: suitcaseFootprintWorld,
    carryOptionOverrides: suitcaseCarryOptionOverrides, qualification: 'unqualified_source_candidate',
    // shared object_profiles.js slot (2026-09-21): the same three overrides as data, read by object_class_routing.js when routing is ON
    carryOverrides: freeze({ outcomeRequirements: SUITCASE_SETTLING.outcomeRequirements, quietLimits: SUITCASE_SETTLING.quietLimits,
      objectUpAxisLocal: SUITCASE_SETTLING.objectTilt.upAxisLocal }) });
}
