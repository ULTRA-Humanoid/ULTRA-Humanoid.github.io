import {FirstPickupStandingRecovery} from './first_pickup_standing_recovery.js';
import {FirstPickupStudentRecoveryRegistry,readFirstPickupContacts,captureFirstPickupTeacherHandoff} from './first_pickup_student_recovery.js';
import {ReferenceStudentTurnController} from './reference_student_turn_controller.js';
import {HeightAwareApproachController} from './height_aware_approach_controller.js';
// main.js — browser scene, user controls and asynchronous policy/physics loop.
//
// What this version does:
//   1. Loads mujoco-wasm + builds three.js scene from the MJCF.
//   2. PD control + physics stepping at 60 Hz control / 1020 Hz physics.
//   3. Maps restricted commands to recorded motions and the 4052D teacher obs.
//      Student comparison/task settling use the 1422D student observation:
//        [NEW_CMD(13) | body(1012) | task(192) | mask(205)].
//   4. Runs the selected teacher or V2VAE student ONNX policy → mu (29).
//   5. Applies PD with target_q = 3.0 * clamp(mu, -1, 1)  (NO default-pose offset:
//      matches sim2sim_vae.py:1272 and humanoid.py:118 `_initial_dof_pos = 0`).
//
// See WASM_DEMO_PLAN.md for the full plan.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { loadMujocoScene } from './mujoco_loader.js';
import { buildSceneFromModel, syncBodyTransforms } from './scene_builder.js';
import {
  resolveJointAddresses, applyArmatureOverride,
  applyPDTorques,
  SIM_DECIMATION, SIM_DT, CONTROL_HZ,
} from './pd_control.js';
import { BodyObsBuilder } from './body_obs.js';
import { buildObs, computeObjPointsHeadingFrame, ACTION_DIM, VAE_DIM } from './obs_builder.js';
import { OnnxPolicy } from './policy.js';
import { UserState } from './state.js';
import { attachKeyboard } from './keyboard.js';
import { attachMousePicker } from './mouse_picker.js';
import { CAMERA_MODES, readCameraMode } from './camera_modes.js';
import {
  applyReleaseObjectPresentation,
  releaseCameraMode,
  releaseInteractiveNames,
  releasePresentationContract,
  releasePresentationEnabled,
} from './release_presentation.js';
import { GoalTranslator, FsmState } from './goal_translator.js';
import { GoalViz } from './goal_viz.js';
import { objectGoalOnFloor } from './goal_geometry.js';
import { createObjectSelection } from './object_selection.js';
import { TeacherPolicy } from './teacher_policy.js';
import { TeacherObsBuilder, TEACHER_HUMAN_BODY_NAMES } from './teacher_obs.js';
import { loadTeacherSkill, withInitialStance } from './teacher_skill.js';
import { TeacherSkillController } from './teacher_controller.js';
import { CarryGoalSequenceController, planCarrySegments, CARRY_PLACEMENT_TOLERANCE_M } from './teacher_carry_sequence.js';
import { chooseCarryReference } from './carry_reference_selection.js';
import { createCarrySkillLibrary, CARRY_RANKINGS, plannerRankingFor } from './carry_skill_library.js';
import { CARRY_CONTROL_BUDGET_CONTROLS, remainingCarryControls } from './carry_time_budget.js';
import { readCarryStylePreview, selectCarryStyleCandidates } from './carry_style_preview.js';
import { readCarryStyleSampling, CarryStyleRequestSampler, planCarryStyleChoices } from './carry_style_request_sampling.js';
import { withStudentLiftProfile, hasPreparedStudentLiftProfile, STUDENT_LIFT_PROFILE } from './student_lift_profile.js';
import { planMixedCarry, mixedCarryDistanceCoverage } from './mixed_carry_planner.js';
import { planCarryGoalRegion } from './carry_goal_region_planner.js';
import { planCarryWithPickupPoseSearch } from './pickup_pose_plan_search.js';
import { MixedCarryGoalSequenceController } from './mixed_carry_sequence.js';
import { checkMixedCarryClearance } from './mixed_carry_clearance.js';
import { readIdleBoxPlacement } from './idle_box_placement.js';
import { prepareTeacherFacingTurn } from './teacher_turn_controller.js';
import {RecoveredFacingTurnAdmission,RecoveredFacingTurnProbe,prepareRecoveredFacingTurn} from './recovered_facing_turn_admission.js';
import {PostTaskTargetContactAdmission} from './post_task_target_contact_admission.js';
import { planPickupFacingApproach } from './pickup_facing_approach.js';
import { planPickupFacingEntryRegion } from './pickup_facing_entry_region.js';
import { PickupFacingApproachController } from './pickup_facing_approach_controller.js';
import { NO_RESET_APPROACH_SOURCE, NoResetApproachPoseController,
  startNoResetApproachPoseRequest } from './no_reset_approach_pose.js';
import { GroundPushGoalSequenceController } from './ground_push_sequence.js';
import { bindHand002Predecessor69, prepareGroundPushGoalWarpEntry, prepareOutcomeBasedNoResetEntry,
  selectRetiredPushForwardPlanObstacles, startLargeboxPushLiveDiagnostic, shouldRunOutcomeLaneControlPreview, noResetActionAdvancedExactlyOnce } from './largebox_push_live_entry_runtime.js';
import { MatchedCarryHost } from './matched_carry_host.js';
import { TeacherDescentRematchOwner, TEACHER_DESCENT_PROFILE } from './teacher_descent_rematch.js';
import { StageFeedbackMeasurement, measureGroundingRelease } from './placement_feedback.js';
import { makeOriginal3mRequest, isFixedOriginal3mGoal } from './matched_carry_programme.js';
import { TeacherWaypointController } from './teacher_waypoint_controller.js';
import { TeacherRecordedApproachController } from './teacher_recorded_approach_controller.js';
import { TeacherApproachRecoveryController } from './teacher_approach_recovery_controller.js';
import { TeacherBoxExitController } from './teacher_box_exit_controller.js';
import { StagedStudentApproachController, STAGED_STUDENT_APPROACH_LIMITS } from './staged_student_approach_controller.js';
import { StagedStudentTransportController, STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES,
  STAGED_STUDENT_TRANSPORT_ADMISSION, STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS } from './staged_student_transport_controller.js';
import { ControlPreview } from './control_preview.js';
import { StudentClosedLoopPreview, decideClosedLoopCommit } from './student_closed_loop_preview.js';
import { TeacherDescentContactHold, holdOnsetDecision, originToBoxSurfaceM, boxCollisionGeometry, TEACHER_DESCENT_HOLD_LIMITS } from './teacher_descent_contact_hold.js';
import { TaskCoverageCapture, readTaskCoverageState } from './task_coverage_capture.js';
import { BoxTaskRequestLog } from './box_task_request_log.js';
import { planTeacherStandingReference } from './teacher_standing_reference.js';
import { ContactDiagnostics } from './contact_diagnostics.js';
import { QuietEndingMeasurement, QUIET_ENDING_DEFAULTS, PLACEMENT_CORRECTION_DEFAULTS, evaluatePlacementCorrection } from './quiet_ending.js';
import { BoxApproachPlanner, ObjectMeshBounds, planBoxApproach } from './box_approach.js';
import { ObjectCollisionMeshes } from './carry_destination_footprint.js';
import { CarryReferenceClearance } from './carry_reference_clearance.js';
import { checkCarryRequestClearance, checkCarrySegmentClearance } from './carry_request_clearance.js';
import { RestrictedLocomotionController,
  preservesPeriodicObservationHistory } from './restricted_locomotion_controller.js';
import { buildPeriodicTeacherSkills, buildPeriodicMotionSweep } from './periodic_teacher_reference.js';
import { HybridKeyboardArbiter, readHybridKeyboardOptions, heldKeysFromUser, studentUserView, fallbackKeyView } from './hybrid_keyboard_arbiter.js';
import { LATERAL_KEY_REFERENCES, LATERAL_SWEEP_ASSET, bindLateralKeyRecords } from './lateral_key_records.js';
import { planRestrictedFloorGoal } from './restricted_navigation.js';
import { bindMotionSweep, checkRestrictedReferenceSweep, convexObstaclesFromProjections } from './restricted_motion_geometry.js';
import { ApproachRecoveryCoordinator, ApproachRecoveryCycle, readApproachRecoveryFlags, RECOVERY_REFUSAL_MESSAGES } from './approach_recovery_loop.js';
import { commonTranslationWarp, planCarryToGoal } from './teacher_goal_warp.js';
import { measureChooseSkill, describePushRefusal, alignedPushApproachTarget,
  ARBITER_THRESHOLDS as SKILL_ARBITER_THRESHOLDS, PUSH_LANE as SKILL_ARBITER_PUSH_LANE,
  SKILL_ARBITER_VERSION } from './skill_arbiter.js';
import { skillMatchesProfile } from './object_profiles.js';
import { readObjectClassRouting, ObjectClassRouter, PUSH_LANE_BODY } from './object_class_routing.js';
import { spliceLoadedCarryGoal } from './loaded_reference_retarget.js';
import { StreamingCarryGoalUpdateOwner } from './streaming_carry_goal_update.js';
import { pendingRetargetReceipt, planarPickerRetargetGoal,
  runStepWithRetargetBarrier } from './mid_carry_retarget_runtime.js';
import { resolveQuietExitHoldOptions } from './quiet_exit_hold_policy.js';
import { SUITCASE_BODY, shouldActivateSuitcaseRestPose,
  standingSuitcaseQuaternionWxyz } from './suitcase_rest_activation.js';

const SCENE_URL = 'public/g1_scene.xml';
const POLICY_URL = 'public/policy.onnx';
const CLIP_DB_BIN_URL = 'public/clip_db.bin';
const CLIP_DB_JSON_URL = 'public/clip_db.json';
const BOX_TASKS = {
  pickup: { referenceUrl: 'public/teacher_pickup_reference.json', executionMessage: 'Lifting the box, then setting it back down…' },
  carry: { referenceUrl: 'public/teacher_carry_reference.json', executionMessage: 'Carrying the box about a metre, then setting it down…' },
};
const TURN_REFERENCES = ['turn', 'turn_clockwise'];
const STEP_REFERENCES = ['step_medium', 'step_short'];
const RESTRICTED_STEP_REFERENCES = ['walk_medium', 'walk_short', 'walk_small', 'walk_tiny'];
const RESTRICTED_SWEEP_KEYS = { walk_medium: 'step_608', walk_short: 'step_455',
  walk_small: 'step_260', walk_tiny: 'step_189', turn: 'turn_left', turn_clockwise: 'turn_right' };

// PD target = ACTION_SCALE * clamp(mu) (NO default-pose offset; see humanoid.py:118).
const ACTION_SCALE = 3.0;
// Fixed before merged-path physics from the exact-state paired policy probe
// (job34816825). It is not adjusted from the later physical outcome.
const MID_CARRY_RETARGET_ACTION_MAX_DELTA = 0.0089015;

const DEFAULT_BENCHMARK_DISTANCES = [0.5, 1.0, 1.5];
const DEFAULT_BENCHMARK_DIRECTIONS_DEG = [0, 45, -45, 90, -90, 135, -135, 180];

function parseNumberListParam(params, name, fallback) {
  const raw = params.get(name);
  if (!raw) return fallback.slice();
  const values = raw.split(/[,:]/)
    .map((x) => Number.parseFloat(x.trim()))
    .filter((x) => Number.isFinite(x));
  return values.length > 0 ? values : fallback.slice();
}

function readNumberParam(params, name, fallback, minimum = -Infinity, maximum = Infinity) {
  const value = Number.parseFloat(params.get(name) ?? 'NaN');
  return Number.isFinite(value) && value >= minimum && value <= maximum ? value : fallback;
}
function readIntegerParam(params, name, fallback, minimum, maximum) {
  const value = Number.parseInt(params.get(name) ?? '', 10);
  return Number.isSafeInteger(value) && value >= minimum && value <= maximum ? value : fallback;
}

function headingYawFromQuatXyzw(q) {
  const [x, y, z, w] = q;
  return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

function goalWorldFromRoot(rootPosWorld, rootQuatXyzwWorld, distanceM, directionDeg) {
  const yaw = headingYawFromQuatXyzw(rootQuatXyzwWorld);
  const angle = yaw + directionDeg * Math.PI / 180.0;
  return new Float32Array([
    rootPosWorld[0] + distanceM * Math.cos(angle),
    rootPosWorld[1] + distanceM * Math.sin(angle),
    0.0, // Match a real floor click, including its visualization height.
  ]);
}

// --- DOM + status ------------------------------------------------------- //
function setStatus(msg) {
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
  console.log('[status]', msg);
}

// Map internal FSM tags → user-facing labels. Keys must match
// FsmState in goal_translator.js (uppercase) AND legacy Mode in state.js
// (lowercase) — we accept both during transition.
const MODE_LABELS = {
  // FSM state names (from translator)
  'IDLE':         'Standing',
  'LOCO':         'Walking',
  'HOI_FULL':     'Walk + Object',
  'HOI_OBJ_ONLY': 'Object Task',
  'RECOVER':      'Recovering',
  'BOX_TASK':     'Lifting & setting down',
  'BOX_CARRY':    'Carrying & setting down',
  'TURN':         'Turning',
  'PAUSED':       'Reset needed',
  // Legacy lowercase Mode (pre-translator fallback)
  'idle':         'Standing',
  'loco':         'Walking',
  'hoi_full':     'Walk + Object',
  'hoi_obj_only': 'Object Task',
};

function setMode(modeStr) {
  const el = document.getElementById('mode-badge');
  if (!el) return;
  el.textContent = MODE_LABELS[modeStr] || modeStr;
  // CSS uses kebab-case lowercase (mode-idle / mode-loco / mode-hoi-full /
  // mode-hoi-obj-only / mode-recover). Strip ALL underscores → hyphens
  // (`replace('_', '-')` only replaces the first occurrence).
  el.className = 'mode-' + modeStr.toLowerCase().replace(/_/g, '-');
}

function setInfoPanel(lines) {
  const el = document.getElementById('info-panel');
  if (el) el.textContent = lines.join('\n');
}

// --- three.js setup ----------------------------------------------------- //

/** Build a 2×N CanvasTexture used as the scene's gradient sky. */
function makeSkyGradientTexture() {
  const c = document.createElement('canvas');
  c.width = 2; c.height = 512;
  const ctx = c.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, 0, c.height);
  // Subtle navy → near-black, with a hint of warm at the horizon line.
  grad.addColorStop(0.00, '#1a2236');   // upper sky
  grad.addColorStop(0.55, '#11151e');   // mid
  grad.addColorStop(0.85, '#0b0d12');   // horizon
  grad.addColorStop(1.00, '#0a0c10');   // floor band
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, c.width, c.height);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function setupThreeJs(canvas, { cameraMode = 'wide', maxPixelRatio = 2 } = {}) {
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: false,
    powerPreference: 'high-performance',
  });
  // `maxPixelRatio` (URL, default 2 = v16) lets an embedding page trade backbuffer pixels for frame time.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, maxPixelRatio));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = makeSkyGradientTexture();
  // Soft distance fade so the far reaches of the gradient blend nicely
  // with whatever ground geometry the MJCF declares.
  scene.fog = new THREE.Fog(0x0b0d12, 9, 24);

  // `wide` (public default) keeps the whole working floor in view so every
  // reachable destination is clickable; `orbit` is the legacy close view.
  // Literals live in camera_modes.js (shared with the click replay tests).
  const cameraSpec = CAMERA_MODES[cameraMode] ?? CAMERA_MODES.wide;
  const camera = new THREE.PerspectiveCamera(
    cameraSpec.fov, canvas.clientWidth / canvas.clientHeight, 0.05, 100
  );
  camera.position.set(...cameraSpec.position);
  scene.add(camera);

  // --- Lighting --------------------------------------------------------- //
  // Hemisphere fill (cool sky + warm ground bounce) — replaces a flat
  // ambient. Gives the model subtle volume even where the directional light
  // doesn't reach.
  const hemi = new THREE.HemisphereLight(0x9fb6e0, 0x2a2118, 0.55);
  hemi.position.set(0, 1, 0);
  scene.add(hemi);

  // Key directional light, warm white.
  const dir = new THREE.DirectionalLight(0xfff1d6, 1.6);
  dir.position.set(4, 7, 3);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 2048;
  dir.shadow.mapSize.height = 2048;
  dir.shadow.camera.left = -4;
  dir.shadow.camera.right = 4;
  dir.shadow.camera.top = 4;
  dir.shadow.camera.bottom = -4;
  dir.shadow.camera.near = 0.5;
  dir.shadow.camera.far = 20;
  dir.shadow.bias = -0.0008;
  dir.shadow.radius = 4;            // softens shadow edges (PCFSoft)
  scene.add(dir);

  // Subtle rim/back light from the opposite side, cool tint.
  const rim = new THREE.DirectionalLight(0x6e8ed8, 0.35);
  rim.position.set(-4, 3, -3);
  scene.add(rim);

  // --- Camera + interaction -------------------------------------------- //
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(...cameraSpec.target);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 12;
  controls.maxPolarAngle = Math.PI * 0.495;   // can't look from below the floor
  controls.update();

  window.addEventListener('resize', () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  });

  return { renderer, scene, camera, controls };
}

// --- Standing init ----------------------------------------------------- //

/**
 * Reset robot to default standing keyframe, then override pelvis z = 0.8.
 * Mirrors run_sim2sim_interactive.py's reset: the MJCF keyframe puts pelvis
 * at z=0.95 (URDF tall pose), but the default standing DOF pose (knees bent
 * 0.3 rad) makes natural feet-on-ground pelvis ~0.78. Starting at 0.95 would
 * cause a 17 cm free-fall before contact — an OOD condition.
 */
function resetRobotToStanding(mujoco, model, data, keyId = 0, rootYawRad = null) {
  mujoco.mj_resetDataKeyframe(model, data, keyId);
  data.qpos[2] = 0.8;                // pelvis z
  if (rootYawRad !== null) {
    data.qpos[3] = Math.cos(rootYawRad / 2);
    data.qpos[4] = data.qpos[5] = 0;
    data.qpos[6] = Math.sin(rootYawRad / 2);
  }
  mujoco.mj_forward(model, data);    // recompute kinematics
}

// --- Pelvis ID resolver ------------------------------------------------ //

// B9-PERF: body names are fixed by the compiled model, so the (model, name) -> id answer is memoised per model object.
// getState() re-resolved 4 names per call, each a full scan building every body-name string; the memo returns the same id.
const bodyIdMemo = new WeakMap();
function findBodyIdByName(model, targetName) {
  let memo = bodyIdMemo.get(model);
  if (!memo) { memo = new Map(); bodyIdMemo.set(model, memo); }
  if (memo.has(targetName)) return memo.get(targetName);
  const id = scanBodyIdByName(model, targetName);
  memo.set(targetName, id);
  return id;
}
function scanBodyIdByName(model, targetName) {
  const namesBlob = model.names;
  const bodyNameAdr = model.name_bodyadr;
  const readName = (offset) => {
    let s = '';
    for (let k = offset; k < namesBlob.length; k++) {
      const c = namesBlob[k];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  };
  for (let bid = 0; bid < model.nbody; bid++) {
    if (readName(bodyNameAdr[bid]) === targetName) return bid;
  }
  return -1;
}

// --- Body-obs parity test --------------------------------------------- //

/**
 * JS↔Python parity validation. Run by hitting `?parity=1` in the URL.
 *
 * Loads `public/body_obs_parity_state.json` (produced by
 * `scripts/dump_body_obs_parity_state.py`), overrides MuJoCo qpos/qvel to
 * the saved values, runs `BodyObsBuilder.build()` twice (to exercise both
 * the first-call full-fill history path and the shift+append path), and
 * diffs each output element against the saved Python-expected values.
 *
 * Pass criterion: max |Δ| < 1e-4 across all 1012 floats of both frames.
 * fp32 noise from atan2/asin and floating-point matmul typically gives
 * max |Δ| < 1e-6.
 */
async function runParityTest(mujoco, model, data) {
  setStatus('Parity test: loading body_obs_parity_state.json...');
  const resp = await fetch('public/body_obs_parity_state.json');
  if (!resp.ok) throw new Error(`Failed to fetch parity state: ${resp.status}`);
  const dump = await resp.json();
  console.log('[parity] loaded dump:', dump.schema, 'frame_dim=', dump.frame_dim);

  // Override MuJoCo state to the dump's qpos/qvel.
  if (data.qpos.length !== dump.qpos.length) {
    throw new Error(`qpos length mismatch: js=${data.qpos.length} py=${dump.qpos.length}`);
  }
  if (data.qvel.length !== dump.qvel.length) {
    throw new Error(`qvel length mismatch: js=${data.qvel.length} py=${dump.qvel.length}`);
  }
  for (let i = 0; i < dump.qpos.length; i++) data.qpos[i] = dump.qpos[i];
  for (let i = 0; i < dump.qvel.length; i++) data.qvel[i] = dump.qvel[i];
  mujoco.mj_forward(model, data);

  // Run body_obs.js builds.
  const builder = new BodyObsBuilder(mujoco, model);
  builder.reset();
  const la0 = new Float32Array(dump.last_action_0);
  const la1 = new Float32Array(dump.last_action_1);
  const got0 = new Float32Array(builder.build(data, la0));   // copy
  const got1 = new Float32Array(builder.build(data, la1));

  const expected0 = new Float32Array(dump.body_obs_0_expected);
  const expected1 = new Float32Array(dump.body_obs_1_expected);

  function diffStats(a, b, label) {
    let maxAbs = 0;
    let argmax = 0;
    let sumSq = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      sumSq += d * d;
      if (d > maxAbs) { maxAbs = d; argmax = i; }
    }
    const rms = Math.sqrt(sumSq / a.length);
    console.log(`[parity] ${label}: max|Δ|=${maxAbs.toExponential(3)}  ` +
                `rms=${rms.toExponential(3)}  argmax=${argmax} ` +
                `(js[${argmax}]=${a[argmax].toFixed(6)}, py[${argmax}]=${b[argmax].toFixed(6)})`);
    return { maxAbs, rms, argmax };
  }

  const r0 = diffStats(got0, expected0, 'body_obs_0');
  const r1 = diffStats(got1, expected1, 'body_obs_1');

  // Per-segment breakdown for frame 0 to localize any mismatch.
  if (r0.maxAbs > 1e-4) {
    console.log('[parity] frame 0 — per-segment diff:');
    const segs = [
      ['root_ang_vel', 0, 3],
      ['imu_obs', 3, 5],
      ['dof_pos', 5, 34],
      ['dof_vel', 34, 63],
      ['last_action', 63, 92],
      ['history_buf', 92, 1012],
    ];
    for (const [name, lo, hi] of segs) {
      let segMax = 0, segArg = -1;
      for (let i = lo; i < hi; i++) {
        const d = Math.abs(got0[i] - expected0[i]);
        if (d > segMax) { segMax = d; segArg = i; }
      }
      console.log(`   ${name.padEnd(14)} [${String(lo).padStart(4)}:${String(hi).padStart(4)}]  max|Δ|=${segMax.toExponential(3)}  argmax=${segArg}`);
    }
  }

  const pass = r0.maxAbs < 1e-4 && r1.maxAbs < 1e-4;
  const msg = `Parity: ${pass ? '✓ PASS' : '✗ FAIL'}  ` +
              `frame0 max|Δ|=${r0.maxAbs.toExponential(2)}  ` +
              `frame1 max|Δ|=${r1.maxAbs.toExponential(2)}`;
  setStatus(msg);
  console.log('[parity]', msg);
  return pass;
}

// --- Public entry presentation ---------------------------------------- //
// `publicUi` (default on) hides development-only controls: the fixed-distance
// / example buttons, the review toolbar and the Space / 1 2 3 legend entries.
// What remains: click box · click floor · Deselect · W A S D · Q E · V · R ·
// Esc, Reset and the status line. `publicUi=0` restores everything.
const PUBLIC_UI_HIDDEN_IDS = Object.freeze(['pickup-button', 'carry-button', 'long-carry-example-button',
  'short-carry-example-button', 'medium-carry-example-button', 'mixed-carry-example-button',
  'style-carry-example-button', 'angled-carry-example-button', 'angled-medium-carry-example-button',
  'entry-region-carry-example-button', 'review-toolbar', 'review-recording-note',
  'style-controls', 'smoothing-controls']);
function applyPublicUi() {
  for (const id of PUBLIC_UI_HIDDEN_IDS) {
    const element = document.getElementById(id);
    if (element) element.hidden = true;
  }
}

// Status text for a carry planner refusal. The planner's summary reason
// (`planning_budget_exhausted` / `no_clear_plan` / `unsupported_distance`) is
// owned by the planner and reported unchanged in the request log; the status
// line additionally names the dominant (most frequent) reason among the
// rejected candidate plans so "budget exhausted" over an occupied destination
// reads as what it is. Reads `plan.attempts` only; never raises the budget.
function dominantPlanRefusalReason(plan) {
  const counts = new Map();
  for (const attempt of plan?.attempts ?? []) {
    if (!attempt || attempt.supported === true || typeof attempt.reason !== 'string') continue;
    counts.set(attempt.reason, (counts.get(attempt.reason) ?? 0) + 1);
  }
  let dominant = null, best = 0;
  for (const [reason, count] of counts) if (count > best) { best = count; dominant = reason; }
  return dominant;
}
const OBJECT_DISPLAY_NAMES = Object.freeze({ largebox: 'large box', suitcase: 'suitcase', plasticbox: 'plastic crate', smallbox: 'small box' });
/** 'active_suitcase_080_080_080' -> 'suitcase' (user-facing). Unknown bodies read as 'box'. */
function objectDisplayName(bodyName) {
  const match = /^active_([a-z]+)_/.exec(String(bodyName ?? ''));
  return (match && OBJECT_DISPLAY_NAMES[match[1]]) || 'box';
}
/** "1.8–2.3 m, 3.6–4.6 m or 5.4–6.9 m" from planner distance intervals. */
function describeReachIntervals(intervals) {
  const parts = intervals.map(([lo, hi]) => `${lo.toFixed(1)}–${hi.toFixed(1)} m`);
  return parts.length > 1 ? `${parts.slice(0, -1).join(', ')} or ${parts.at(-1)}` : parts[0] ?? '';
}
/** Why a distance was refused: too close, too far, or in a gap between the
 * supported carries. v16 said "needs a shorter move" for every case, which sent
 * suitcase users (single 2.0 m clip, nothing shorter) the wrong way. */
function unsupportedDistanceStatus(plan, label) {
  const intervals = (plan.supportedDistanceIntervalsM ?? []).filter(i => Array.isArray(i) && i.length === 2 && i.every(Number.isFinite));
  const distance = plan.distanceM;
  if (!intervals.length || !Number.isFinite(distance)) return `That distance is not supported for the ${label}. Try a different destination.`;
  const shortest = intervals[0][0], longest = intervals.at(-1)[1];
  if (distance < shortest) return `Too close: the ${label} carry needs at least ${shortest.toFixed(1)} m. Click farther away in the same direction.`;
  if (distance > longest) return `Too far: the ${label} can be carried at most ${longest.toFixed(1)} m in one request. Click closer in the same direction.`;
  return `That distance falls between the supported ${label} carries (${describeReachIntervals(intervals)}). Click a little closer or farther.`;
}
function carryPlanRefusalStatus(plan, objectLabel = 'box') {
  if (plan.reason === 'unsupported_distance') return unsupportedDistanceStatus(plan, objectLabel);
  const dominant = plan.reason === 'occupied_carry_destination' ? plan.reason : dominantPlanRefusalReason(plan);
  if (dominant === 'occupied_carry_destination')
    return 'That destination is occupied by another box. Choose a clearer spot.';
  if (dominant === 'carry_reference_clearance')
    return 'Every carry route to that point passes too close to another box. Choose a different direction.';
  return 'I could not find a clear approach and carry route to that point. Try a different direction.';
}

// --- Main --------------------------------------------------------------- //

async function main() {
  setMode('idle');
  const canvas = document.getElementById('mujoco_canvas');
  if (!canvas) throw new Error('No #mujoco_canvas in DOM');

  // Parse URL params (e.g. ?parity=1 → run parity test then halt).
  const urlParams = new URLSearchParams(window.location.search);
  const parityMode = urlParams.get('parity') === '1';
  const clickPositionSource = urlParams.get('clickPositionSource') || 'matched';
  const liveGoalMaxStepParam = Number.parseFloat(urlParams.get('liveGoalMaxStepM') || 'NaN');
  const benchmarkMode = urlParams.get('benchmark') || '';
  const benchmarkDurationS = Number.parseFloat(urlParams.get('benchDurationS') || '5.0');
  const benchmarkSuccessThresholdM = Number.parseFloat(urlParams.get('benchSuccessThresholdM') || '0.25');
  const benchmarkFallZM = Number.parseFloat(urlParams.get('benchFallZM') || '0.45');
  // Public-entry presentation flags (all default ON; each switchable off by
  // URL so evaluation arms can be compared):
  //   cameraMode=orbit     legacy close camera (default `wide`, camera_modes.js)
  //   destinationPicker=0  legacy click-on-body selection toggle
  //   publicUi=0           show the development buttons / legend / toolbar
  const presentationEnabled = releasePresentationEnabled(urlParams);
  const presentationCameraMode = releaseCameraMode(urlParams, presentationEnabled);
  const cameraMode = presentationCameraMode && Object.hasOwn(CAMERA_MODES, presentationCameraMode)
    ? presentationCameraMode : readCameraMode(urlParams);
  const destinationPickerEnabled = urlParams.get('destinationPicker') !== '0';
  const publicUi = urlParams.get('publicUi') !== '0';
  if (publicUi) applyPublicUi();
  if (presentationEnabled) document.body?.classList?.add('release-presentation');

  setStatus('Initializing renderer…');
  const maxPixelRatioParam = Number.parseFloat(urlParams.get('maxPixelRatio') ?? '');
  const maxPixelRatio = Number.isFinite(maxPixelRatioParam) && maxPixelRatioParam >= 1 ? Math.min(maxPixelRatioParam, 2) : 2;
  const { renderer, scene, camera, controls } = setupThreeJs(canvas, { cameraMode, maxPixelRatio });

  setStatus('Loading physics engine and scene…');
  const { mujoco, model, data } = await loadMujocoScene(SCENE_URL, setStatus);

  if (parityMode) {
    setStatus('Running body_obs JS↔Python parity test...');
    const pass = await runParityTest(mujoco, model, data);
    console.log(`[parity] final: ${pass ? 'PASS' : 'FAIL'}`);
    return;   // do not start the policy loop
  }

  setStatus('Building 3D scene…');
  const { root, bodyGroups } = buildSceneFromModel(model);
  const presentationHiddenObjects = applyReleaseObjectPresentation(bodyGroups, presentationEnabled);
  scene.add(root);

  // --- PD control + physics ------------------------------------------ //
  setStatus('Configuring robot joints…');
  const addresses = resolveJointAddresses(model);
  applyArmatureOverride(model, addresses);
  // Force the training-time SIM_DT. The MJCF declares timestep=0.001s, but
  // training uses 1/(60·17) ≈ 0.000980 s (matches sim2sim_vae.py:542). Even
  // a 2% mismatch on dt means the policy's torque integration accumulates
  // a tiny drift per step → visible as jitter over a few seconds.
  // We unconditionally write `model.opt.timestep`; the earlier guard
  // `'timestep' in model.opt` was unreliable for Emscripten ClassHandles.
  const dtBefore = model.opt.timestep;
  model.opt.timestep = SIM_DT;
  const dtAfter = model.opt.timestep;
  console.log(`[main] SIM_DT target=${SIM_DT.toExponential(6)}s, MJCF default=${dtBefore.toExponential(6)}s, actual after set=${dtAfter.toExponential(6)}s, decim=${SIM_DECIMATION}, ctrl=${CONTROL_HZ.toFixed(2)}Hz`);
  if (Math.abs(dtAfter - SIM_DT) > 1e-9) {
    console.warn(`[main] timestep override FAILED: requested ${SIM_DT}, got ${dtAfter}. Physics will run at MJCF default rate (off by ${((dtAfter / SIM_DT - 1) * 100).toFixed(1)}%).`);
  }

  // --- Reset to standing (pelvis z = 0.8) --------------------------- //
  setStatus('Placing robot…');
  resetRobotToStanding(mujoco, model, data, 0);
  const pelvisId = findBodyIdByName(model, 'pelvis');
  if (pelvisId < 0) throw new Error("Body 'pelvis' not found in scene.");

  // Foot body IDs for proprio-derived foot-contact bits fed to the
  // goal-translator. We min over each side's ankle_pitch + ankle_roll
  // to match the offline DB build (scripts/build_browser_clip_db.py
  // uses min(body[6].z, body[7].z) for left, min(body[13], body[14])
  // for right — those are the pitch+roll pair per side).
  const leftFootIds  = [
    findBodyIdByName(model, 'left_ankle_pitch_link'),
    findBodyIdByName(model, 'left_ankle_roll_link'),
  ].filter(id => id >= 0);
  const rightFootIds = [
    findBodyIdByName(model, 'right_ankle_pitch_link'),
    findBodyIdByName(model, 'right_ankle_roll_link'),
  ].filter(id => id >= 0);
  if (leftFootIds.length === 0 || rightFootIds.length === 0) {
    throw new Error(`Foot body lookup failed: left=${leftFootIds} right=${rightFootIds}`);
  }
  console.log(`[main] foot body IDs: L=[${leftFootIds.join(',')}] R=[${rightFootIds.join(',')}]`);

  // --- Body obs builder --------------------------------------------- //
  const bodyObsBuilder = new BodyObsBuilder(mujoco, model);
  bodyObsBuilder.reset();

  // --- Object point clouds ------------------------------------------ //
  // Loaded for the mouse picker / obs assembly. Format:
  //   { object_name: { points: [[x,y,z], ...], bbox: ..., ... }, ... }
  setStatus('Loading objects…');
  const pcResp = await fetch('public/object_pointclouds.json');
  if (!pcResp.ok) throw new Error(`Failed to fetch point clouds: ${pcResp.status}`);
  const pointCloudDb = await pcResp.json();
  const ownedStudentObjectSelections = new Map();
  // The MJCF prefixes each scene-body with `active_` (e.g. body
  // `active_largebox_080_080_080`), but the point-cloud JSON keys are the
  // bare object names (`largebox_080_080_080`). Build a Set of BODY names
  // (with prefix) so the raycaster recognizes the scene-body when clicked.
  const selectableNames = new Set();
  for (const key of Object.keys(pointCloudDb)) {
    if (key.startsWith('_')) continue;             // skip _meta etc.
    const bodyName = `active_${key}`;
    selectableNames.add(bodyName);
  }
  // Keep `selectableNames` complete for validated physics/planning monitors.
  // Only the mouse/API interaction surface is narrowed for the v16 release.
  const interactiveSelectableNames = releaseInteractiveNames(selectableNames, presentationEnabled);
  console.log(`[main] loaded ${selectableNames.size} object point clouds. ` +
              `Selectable body names in scene: [${[...selectableNames].slice(0, 3).join(', ')}...]`);

  // --- Policy ------------------------------------------------------- //
  setStatus('Loading neural policy (≈40 MB, cached after first visit)…');
  const policy = new OnnxPolicy();
  await policy.load(POLICY_URL);

  // --- Goal translator (motion matching) ---------------------------- //
  // Replaces the keyboard's constant-Δ goal synthesis with KNN over real
  // AMASS+BONES+OMOMO clips, gated by a state machine that includes a
  // proprio-derived RECOVER state for fall handling. See
  // PLAN_GOAL_TRANSLATOR.md and web/src/goal_translator.js.
  setStatus('Loading motion-matching DB (≈2 MB)…');
  const translator = new GoalTranslator({
    clickPositionSource,
    liveGoalWorldTargetMaxStepM: Number.isFinite(liveGoalMaxStepParam)
      ? liveGoalMaxStepParam
      : undefined,
  });
  await translator.load(CLIP_DB_BIN_URL, CLIP_DB_JSON_URL);

  // --- User state + keyboard ---------------------------------------- //
  const user = new UserState();
  let episodeVersion = 0;
  let controlStep = 0;
  let episodeControlStep = 0;
  let lastTranslatorResult = null;
  let lastObservation = null;
  let lastRawAction = null;
  let teacherPolicy = null, teacherObs = null, skillController = null;
  let activeCarryController = null, pendingCarryController = null, turnSkills = [];
  let pendingWaypointCarryController = null, stepSkills = [], waypointApproaches = [];
  let pendingSegmentCarryController = null;
  let stagedStudentApproach = null, lastStudentApproachEntry = null;
  const studentApproaches = [];
  const firstPickupStudentRegistry=new FirstPickupStudentRecoveryRegistry();

  const firstPickupStudentDecisions=[],firstPickupStudentHandoffs=[];
  let firstPickupStanding=null;
  const firstPickupStandings=[],firstPickupStandingUsed=new WeakSet();
  const firstPickupStandingContext=()=>({...firstPickupCommandContext(),student:stagedStudentApproach});
  const teacherDescentHoldContext = () => ({ parent: skillController, episode: episodeVersion, requestId: activeBoxTaskRequestId,
    latestRequestId: latestBoxTaskRequestId, queuedRequestId: queuedBoxTask?.requestId ?? null, physicalControl: episodeControlStep });
  function teacherDescentBoxGeometry(objectBodyName) {
    if (!teacherDescentGeometry.has(objectBodyName)) {
      const geometry = boxCollisionGeometry(model, findBodyIdByName(model, objectBodyName));
      teacherDescentGeometry.set(objectBodyName, { ...geometry, triggers: TEACHER_DESCENT_HOLD_LIMITS.triggerBodies.map(name => ({ name, id: findBodyIdByName(model, name) })) });
    }
    return teacherDescentGeometry.get(objectBodyName);
  }
  const firstPickupStudentContext=()=>({parent:skillController,episode:episodeVersion,
    requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep});
  const firstPickupCommandContext=()=>({parent:skillController===activeCarryController?skillController:pendingWaypointCarryController,
    activeParent:activeCarryController,episode:episodeVersion,physicalControl:episodeControlStep,requestId:activeBoxTaskRequestId,
    latestRequestId:latestBoxTaskRequestId,queuedRequestId:queuedBoxTask?.requestId??null,
    originalGoalWorld:taskDestinationWorld,selectedObjectBodyName:user.activeObjName,
    request:boxTaskRequestLog.records.get(activeBoxTaskRequestId),
    commandRevision:restrictedController?.requestedIntent?.revision,paused,suspended:restrictedSuspended});
  let stagedStudentTransport = null, lastStudentTransportEntry = null, transportTeacherResumePending = false;
  let carryReferenceSelection = null;
  // WS-B: request-time library inputs (reused by the remaining-goal replan),
  // the last truthful refusal, replan decisions and the per-task lineage of
  // completed sequences (pickup accounting across replans).
  let carryLibraryContext = null, lastCarryRefusal = null;
  const carryReplans = [], carryTaskLineage = [];
  let finalCarryPlacement = null;
  const studentTransports = [];
  let recordedApproachHold = null;
  let executedApproachTerminal = null;
  let terminalFacingProbe=null,terminalFacingPending=null;
  const terminalFacingEvents=[];
  const recoveredFacingAdmission=new RecoveredFacingTurnAdmission({
    enabled:urlParams.get('recoveredFacingTurn')==='1',substepsPerControl:SIM_DECIMATION});
  const terminalFacingContext=()=>({owner:skillController,parent:pendingCarryController,
    episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep,teacherBuilder:teacherObs});
  const recoveredFacingRecoveryContext=controller=>({controller,owner:skillController,parent:pendingWaypointCarryController,
    activeParent:activeCarryController,episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep});
  let terminalRefusalRecoveryAttempt = null;
  function readExecutedApproachTerminalReview() {
    const saved = executedApproachTerminal;
    return saved ? {
      episode:saved.episode, executedAtControl:saved.executedAtControl,
      sourceFrames:saved.skill.sourceFrames, ownerIsCurrent:saved.owner===skillController,
      parentIsCurrent:saved.parent===pendingWaypointCarryController,
      ownerCancelRequested:Boolean(saved.owner.cancelRequested),
      parentFinishRequested:Boolean(saved.parent.finishRequested),
      endpointDistanceM:saved.endpointDistanceM, parentArrivalRadiusM:saved.parent.arrivalRadius,
      originalGoalWorld:Array.from(saved.parent.requestedGoalWorld),
      terminal:Array.from(saved.terminal), planFrame:Array.from(saved.plan.frame)
    } : null;
  }
  let pickupFacingSweeps = null;
  const pickupFacingPlans = [], pickupFacingOwners = [];
  const pickupFacingContext = () => ({ owner: skillController, parent: pendingWaypointCarryController,
    episode: episodeVersion, requestId: activeBoxTaskRequestId, physicalControl: episodeControlStep });
  let noResetApproachOwner = null, lastNoResetApproachReview = null;
  let lastLargeboxPushLiveDiagnostic = null, pendingNormalGroundPush = null;
  const retiredLargeboxPushFollowupCurrent=()=>Boolean(lastLargeboxPushLiveDiagnostic
    && lastBoxTaskRetirement?.requestId===lastLargeboxPushLiveDiagnostic.requestId
    && lastBoxTaskRetirement.task==='push'&&lastBoxTaskRetirement.reason==='box_exit_complete'
    && lastBoxTaskRetirement.episodeVersion===episodeVersion
    && lastLargeboxPushLiveDiagnostic.token?.episodeVersion===episodeVersion
    && latestBoxTaskRequestId===lastLargeboxPushLiveDiagnostic.requestId
    && activeBoxTask===null&&activeBoxTaskRequestId===null
    && user.activeObjName===PUSH_LANE_BODY);
  const noResetApproachContext = () => ({ owner: skillController, episode: episodeVersion,
    physicalControl: episodeControlStep, selectedObjectBodyName: user.activeObjName,
    selectedObjectBodyId: activeObjBodyId });
  const approachRecoveries = [];
  let boxExitController = null, boxExitResults = [], boxTaskResults = [];
  let controlPreview = null, lastControlPreview = null, previewControls = 0;
  let headingPreparations = [], lastControlPhase = null;
  const recordReferenceApproachInputs=urlParams.get('recordReferenceApproachInputs')==='1';
  let referenceStudentTurn=null,heightAwareApproach=null,approvedReferenceTurn=null,referenceTeacherResumePending=false;
  let referenceStudentTurns=[],heightAwareApproaches=[],usedReferenceStudentTurns=new WeakMap();
  let preserveCompletionCommand = false;
  // keyboardMode=recorded|student|hybrid. Without the parameter the evaluated default applies
  // unchanged: restrictedControl=0 selects the student translator, anything else the recorded
  // supervisor. `hybrid` keeps the recorded supervisor (records, box tasks, floor clicks) and
  // lets a clearance-gated arbiter hand held keys to the student translator (hybrid_keyboard_arbiter.js).
  const keyboardModeParam = urlParams.get('keyboardMode');
  const keyboardMode = ['recorded', 'student', 'hybrid'].includes(keyboardModeParam) ? keyboardModeParam
    : urlParams.get('restrictedControl') === '0' ? 'student' : 'recorded';
  const restrictedMode = keyboardMode !== 'student';
  const hybridKeyboardEnabled = keyboardMode === 'hybrid';
  const hybridArbiter = hybridKeyboardEnabled ? new HybridKeyboardArbiter(readHybridKeyboardOptions(urlParams)) : null;
  // Opt-in: after a runaway-backward fallback under a W+turn chord, the recorded supervisor runs a
  // forward step instead of the pure turn (hybrid_keyboard_arbiter.js fallbackKeyView). Default OFF.
  const hybridFallbackStepEnabled = hybridKeyboardEnabled && urlParams.get('hybridFallbackStep') === '1';
  let lastHybridDecision = null, lastHybridStatus = null;
  let restrictedController = null, restrictedObs = null, lastRestrictedStep = null;
  let restrictedWalkSkills = [], restrictedTurnSkills = [], approveRestrictedReference = null;
  let restrictedPlan = null, queuedBoxTask = null, restrictedAfterBox = false;
  let restrictedSuspended = null;
  let lastRestrictedGeometry = null;
  let restrictedGeometryCheckStep = null;
  let restrictedReferenceAttempts = [];
  // Safe-recovery lane (WS-D1): never-suspend retirement, bounded retreat/turn
  // recovery loop, preview-owned hull fallback and convex obstacle footprints.
  const recoveryFlags = readApproachRecoveryFlags(urlParams);
  let approachRecovery = null, pendingRecoveryParent = null, standingPreviewOwned = null;
  let transientSuspension = null, pendingOwnedRetirement = null;
  const approachRecoveryDecisions = [];
  const debugControls = urlParams.get('debug') === '1';
  // Default OFF. This changes no combined/default behaviour unless the exact
  // query flag is present, and it never replaces the selected object or task.
  const midCarryRetargetEnabled = urlParams.get('midCarryRetarget') === '1';
  // B9 unified interface (Phase B, default OFF): skillArbiter=1 routes a selected-object floor click through the
  // deterministic skill arbiter (push / carry / pickup, skill_arbiter.js). OFF = v5 carry-only routing, byte-identical.
  // Set to "1" only by combined_settings.js releaseProfile (profile=release); an explicit skillArbiter=0 in the query
  // forces the v5 path for A/B comparisons and identity replays.
  const skillArbiterEnabled = urlParams.get('skillArbiter') === '1';
  // R1 (v14a): when the arbiter is ON, prefetch and parse the push lane's two task assets at startup so a push-routed click has no
  // fetch/parse between its boundary capture and admission. Static asset reads only: inert for physics, controls, inference and
  // history; nothing is prefetched when skillArbiter is OFF. On failure the lane falls back to its v5 on-demand fetch.
  const PUSH_TASK_ASSET_URLS = Object.freeze(['public/task-assets/hand002_predecessor69_phase_reference.json', 'public/task-assets/native-summary.json']);
  let pushAssetPrefetch = null, pushAssetPrefetchState = 'off';
  if (skillArbiterEnabled) {
    pushAssetPrefetchState = 'pending';
    pushAssetPrefetch = Promise.all(PUSH_TASK_ASSET_URLS.map(url => fetch(url))).then(async responses => {
      if (!responses.every(response => response.ok)) throw new Error('Ground-push task assets are unavailable');
      const parsed = []; for (const response of responses) parsed.push(await response.json());
      pushAssetPrefetchState = 'ready'; return parsed;
    }).catch(error => { pushAssetPrefetchState = 'failed'; console.warn('[arbiter] push task-asset prefetch failed; the push lane fetches on demand', error); return null; });
  }
  // R2 (v14a) routing policy: after the push lane REFUSES an arbiter-routed click, route the same click to carry (v5's lane) so the
  // destination is still served. Alternative = false: status-only (`[push] Push unavailable: <reason>`, no carry). Product decision.
  const PUSH_REFUSAL_FALLBACK_TO_CARRY = true;
  // Diagnostic-only continuous teacher locomotion. Keep the physical A039
  // contract explicit: older phase defaults were not validated and must not
  // activate this bank accidentally.
  const teacherPeriodicRequested = urlParams.get('teacherPeriodic') === '1';
  const teacherPeriodicPersistentRequested = urlParams.get('teacherPeriodicPersistent') === '1';
  const teacherPeriodicA039FlagsExact = urlParams.get('teacherPeriodicFirst') === '283'
    && urlParams.get('teacherPeriodicLast') === '362'
    && urlParams.get('teacherPeriodicTimeScale') === '1.5'
    && urlParams.get('teacherPeriodicSourceFrames') === '600';
  const teacherPeriodicPersistentEnabled = restrictedMode && debugControls
    && teacherPeriodicRequested && teacherPeriodicPersistentRequested && teacherPeriodicA039FlagsExact;
  const periodicBoundaryLogEnabled = teacherPeriodicPersistentEnabled
    && urlParams.get('periodicMeasurementLog') === '1';
  const teacherPeriodicOptions = Object.freeze({ periodFirst: 283, periodLast: 362,
    timeScale: 1.5, sourceFrames: 600 });
  let periodicTeacherSkills = null;
  const restrictedBackwardEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedBackward') === '1';
  const restrictedKeyTerminalEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedKeyTerminal') === '1';
  const restrictedStartupStandingEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedStartupStanding') === '1';
  // Complete lateral teacher records on A/D in the recorded regime (default off until measured).
  const recordedLateralEnabled = restrictedMode && debugControls
    && urlParams.get('recordedLateral') === '1';
  const restrictedApproachTerminalEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedApproachTerminal') === '1';
  const restrictedApproachRoutingEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedApproachRouting') === '1';
  const restrictedApproachRecoveryEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedApproachRecovery') === '1';
  const restrictedStudentApproachEnabled = restrictedMode && debugControls
    && urlParams.get('restrictedStudentApproach') === '1';
  // The restricted entrypoint includes its measured box approach and exit.
  // Legacy comparisons must explicitly opt out through the debug controls.
  const recordedApproachEnabled = restrictedMode
    && !(debugControls && urlParams.get('teacherRecordedApproach') === '0');
  const requestedBoxExit = debugControls ? urlParams.get('teacherBoxExit') : null;
  const boxExitMode = !recordedApproachEnabled || requestedBoxExit === '0' ? null
    : requestedBoxExit === 'long' ? 'long' : '1';
  const boxExitEnabled = recordedApproachEnabled && boxExitMode !== null;
  const restrictedStudentTransportEnabled = restrictedMode && debugControls && boxExitEnabled
    && urlParams.get('restrictedStudentTransport') === '1';
  // WS-C transport lane (Sep 2026). transportAdmission=instant restores the
  // one-state hand-load test; window (default) needs transportAdmissionMinLoaded
  // of the last transportAdmissionWindow control boundaries loaded in both hands.
  // transportDivergenceExit=0 disables the mid-window exit against the
  // interpolated reference. refObjYawSnap=1 quarter-turns the clip's box
  // quaternion to the live box. restrictedStudentTransport=0 removes the
  // student window entirely (teacher carries 240-330) as the comparison arm.
  const transportAdmissionMode = urlParams.get('transportAdmission') === 'instant' ? 'instant' : 'window';
  const transportAdmissionWindow = readIntegerParam(urlParams, 'transportAdmissionWindow',
    STAGED_STUDENT_TRANSPORT_ADMISSION.window.windowControls, 1, 90);
  const transportAdmission = transportAdmissionMode === 'instant' ? { ...STAGED_STUDENT_TRANSPORT_ADMISSION.instant }
    : { mode: 'window', windowControls: transportAdmissionWindow, minLoadedControls: readIntegerParam(urlParams,
      'transportAdmissionMinLoaded', Math.min(STAGED_STUDENT_TRANSPORT_ADMISSION.window.minLoadedControls, transportAdmissionWindow), 1, transportAdmissionWindow) };
  const transportDivergenceExitEnabled = urlParams.get('transportDivergenceExit') !== '0';
  // Diagnostic only (default OFF): per-control knee/box clearance trace during the
  // teacher-owned setdown descent (reference index >= 330) — H064/H067 stop with the
  // right knee pressing into the carried box at index 368-369. No behaviour change.
  const setdownClearanceDiagEnabled = urlParams.get('setdownClearanceDiag') === '1';
  // Bounded teacher reference-clock experiment. Default OFF; the controller
  // owns its frozen margin and per-segment control budget.
  const carryDescentSagHoldEnabled = urlParams.get('carryDescentSagHold') === '1';
  const setdownClearanceTrace = [];
  const transportDivergenceLimits = transportDivergenceExitEnabled ? {
    maxRootXYErrorM: readNumberParam(urlParams, 'transportExitRootXYM', STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS.maxRootXYErrorM, 1e-3, 10),
    maxYawErrorRad: readNumberParam(urlParams, 'transportExitYawDeg', 25, 1e-3, 180) * Math.PI / 180,
    minRootHeightM: readNumberParam(urlParams, 'transportExitMinRootZM', STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS.minRootHeightM, 1e-3, 2),
    maxUnloadedControls: readIntegerParam(urlParams, 'transportExitUnloadedControls', STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS.maxUnloadedControls, 1, 90),
  } : null;
  const refObjYawSnapEnabled = urlParams.get('refObjYawSnap') === '1';
  // Phase B B4: objectClassRouting=1 routes the carry lane through the SELECTED body's profile (object_profiles.js).
  // Default OFF: every routed site below resolves to the v5 largebox literal it replaced
  // (benchmarks/phaseb-plasticbox/test_object_class_routing.mjs). The push lane stays largebox-only (PUSH_LANE_BODY).
  const objectClassRoutingEnabled = readObjectClassRouting(urlParams);
  const objectRouter = new ObjectClassRouter({ enabled: objectClassRoutingEnabled });
  // The integrated scene retains v15's passive flat Suitcase so an unselected
  // object cannot perturb Largebox solver trajectories.  Suitcase routing
  // activates the certified standing rest pose exactly once, at selection and
  // before click geometry, goal normalization, or task inference.  XY and yaw
  // are preserved so the frozen Suitcase panel's rigid pose construction is
  // still authoritative.
  let suitcaseStandingPoseActivated = false;
  function activateSelectedSuitcaseRestPose(bodyName) {
    if (!shouldActivateSuitcaseRestPose({ routingEnabled: objectClassRoutingEnabled,
      bodyName, alreadyActivated: suitcaseStandingPoseActivated })) return false;
    const body = findBodyIdByName(model, bodyName);
    const joint = body >= 0 ? model.body_jntadr[body] : -1;
    if (joint < 0 || model.jnt_type[joint] !== mujoco.mjtJoint.mjJNT_FREE.value) {
      throw new Error('Selected Suitcase requires its free joint');
    }
    const qpos = model.jnt_qposadr[joint], qvel = model.jnt_dofadr[joint];
    const standing = standingSuitcaseQuaternionWxyz(data.qpos.slice(qpos + 3, qpos + 7));
    data.qpos[qpos + 2] = 0.23;
    data.qpos.set(standing, qpos + 3);
    data.qvel.fill(0, qvel, qvel + 6);
    data.qacc_warmstart.fill(0, qvel, qvel + 6);
    mujoco.mj_forward(model, data);
    suitcaseStandingPoseActivated = true;
    return true;
  }
  let transportHandForceHistory = null;
  const restrictedLongCarryEnabled = restrictedMode && debugControls && boxExitEnabled
    && urlParams.get('restrictedLongCarry') === '1';
  // Existing long-carry bookmarks receive the expanded controller. The old
  // two-reference selection remains available through an explicit opt-out.
  const restrictedCarryLibraryEnabled = restrictedLongCarryEnabled
    && urlParams.get('restrictedCarryLibrary') !== '0';
  const restrictedCarryGoalRegionEnabled = restrictedCarryLibraryEnabled
    && urlParams.get('restrictedCarryGoalRegion') !== '0';
  // maxPickups=1|2|3 (default 3 = v16): the most pickups a carry request may plan.
  // The site embeds with 1. Probe of 12 largebox floor clicks (2026-09-23, same
  // runtime): with 3, the 7 multi-pickup plans v16 chose completed 3 times, fell
  // twice and stalled twice (7/12 clicks completed overall); with 1, 10/12
  // completed and none stalled, 5 clicks being moved onto single-pickup coverage
  // (longest delivered carry 2.57 m). The reach ring and destination snapping
  // use the same coverage, so a click still turns into one supported carry.
  const maxCarryPickupsParam = Number.parseInt(urlParams.get('maxPickups') ?? '', 10);
  const maxCarryPickups = [1, 2, 3].includes(maxCarryPickupsParam) ? maxCarryPickupsParam : 3;
  // WS-B planner honesty (2026-09-15). Each behaviour has its own URL parameter:
  // replanRemaining=1 (default on): after a refused live distance, replan the
  //   remaining goal from the live box toward the ORIGINAL destination.
  // carryRanking=default|excludeLong|reliability (default `default`).
  // carryBudgetGuard=1 (default on): rank plans estimated to fit the remaining
  //   post-click controls first and record budgetRisk; never adds a refusal.
  // carryControlBudget=<controls> (default 5820): the panel's post-click budget.
  const replanRemainingEnabled = restrictedCarryLibraryEnabled && urlParams.get('replanRemaining') !== '0';
  const carryRankingParam = urlParams.get('carryRanking') ?? 'default';
  const carryRanking = CARRY_RANKINGS.includes(carryRankingParam) ? carryRankingParam : 'default';
  if (carryRanking !== carryRankingParam) console.warn(`[carry planner] unknown carryRanking=${carryRankingParam}; using default`);
  const carryBudgetGuardEnabled = urlParams.get('carryBudgetGuard') !== '0';
  const carryControlBudgetParam = Number.parseInt(urlParams.get('carryControlBudget') ?? '', 10);
  const carryControlBudgetControls = Number.isInteger(carryControlBudgetParam) && carryControlBudgetParam > 0
    ? carryControlBudgetParam : CARRY_CONTROL_BUDGET_CONTROLS;
  // WS-G endings. quietEndings=1 (default OFF): the fixed 180-control settling and
  // the exit's 60/180 holds end once the robot and box have been measurably
  // quiet for `quietWindow` consecutive controls, never before their minimums
  // (settling 60, exit holds 30) and never after the original maxima.
  // placementCorrection=1 (default OFF): after a near-miss final setdown
  // (10 cm < error <= 15 cm) with enough post-click budget, ONE corrective
  // short carry to the ORIGINAL destination is queued through the ordinary
  // request path. The 10 cm tolerance itself never changes.
  const wholeParam = (name, fallback, minimum, maximum) => {
    const value = Number.parseInt(urlParams.get(name) ?? '', 10);
    return Number.isInteger(value) && value >= minimum && value <= maximum ? value : fallback;
  };
  const quietEndingsEnabled = urlParams.get('quietEndings') === '1';   // default OFF until the WS-G gate passes
  const quietEndingConfig = Object.freeze({
    window: wholeParam('quietWindow', QUIET_ENDING_DEFAULTS.window, 1, 180),
    settlingMinControls: wholeParam('quietSettlingMin', QUIET_ENDING_DEFAULTS.settlingMinControls, 1, 180),
    exitInitialHoldMinControls: wholeParam('quietExitHoldMin', QUIET_ENDING_DEFAULTS.exitInitialHoldMinControls, 1, 60),
    exitFinalHoldMinControls: wholeParam('quietExitSettlingMin', QUIET_ENDING_DEFAULTS.exitFinalHoldMinControls, 1, 180),
  });
  // B2 (Phase B). quietExitSettling=onRequest|quiet (default OFF = 'fixed' = v5): only the FINAL exit's settling hold
  // (teacher_exit_settling, 180) may end early, after the 30-control tracked minimum and a full quiet window;
  // 'onRequest' ends it ONLY for a user request that arrived after the settling started (a request already queued
  // when the settling starts runs the full 180, so every frozen P100/T20 control stream is unchanged). The initial
  // exit hold stays fixed at 60, inter-segment exits keep quietHolds:false, and the carry-sequence settling (180)
  // is never armed by this flag. quietEndings=1 keeps the legacy WS-G semantics untouched ('legacy').
  const quietExitSettlingMode = quietEndingsEnabled ? 'legacy'
    : ({ onRequest: 'quiet_on_request', quiet: 'quiet' }[urlParams.get('quietExitSettling')] ?? 'fixed');
  const placementCorrectionEnabled = restrictedCarryLibraryEnabled && urlParams.get('placementCorrection') === '1';   // default OFF
  const placementCorrectionConfig = Object.freeze({
    postClickBudgetControls: wholeParam('postClickBudget', PLACEMENT_CORRECTION_DEFAULTS.postClickBudgetControls, 1, 1000000),
    minBudgetControls: wholeParam('correctionMinBudget', PLACEMENT_CORRECTION_DEFAULTS.minBudgetControls, 0, 1000000),
    maxErrorM: PLACEMENT_CORRECTION_DEFAULTS.maxErrorM,
  });
  const quietEndingMeasurements = new Map();
  let lastCarryLibrary = null;
  const placementCorrections = [];
  function quietEndingSample(objectBodyName) {
    const name = objectBodyName || user.activeObjName;
    if (!name) throw new Error('Quiet ending measurement requires the placed object');
    if (!quietEndingMeasurements.has(name)) quietEndingMeasurements.set(name, new QuietEndingMeasurement(mujoco, model,
      { rootBodyId: pelvisId, objectBodyId: findBodyIdByName(model, name), ...objectRouter.quietMeasurementOptions(name) }));   // OFF: {} (nine-key v5 sample)
    return quietEndingMeasurements.get(name).read(data);
  }
  const studentLiftPreviewEnabled = restrictedCarryLibraryEnabled && restrictedStudentTransportEnabled
    && urlParams.get('studentLiftPreview') === '1';
  const carryStylePreview = readCarryStylePreview(urlParams, {
    libraryEnabled: restrictedCarryLibraryEnabled, studentLiftPreviewEnabled });
  const carryStyleSamplingConfig = readCarryStyleSampling(urlParams, {
    libraryEnabled: restrictedCarryLibraryEnabled, studentLiftPreviewEnabled });
  const carryStyleRequestSampler = carryStyleSamplingConfig
    ? new CarryStyleRequestSampler({ seed: carryStyleSamplingConfig.seed, episodeVersion }) : null;
  const referenceStudentTurnsEnabled = restrictedCarryLibraryEnabled && urlParams.get('referenceStudentTurns') !== '0';
  const heightAwareApproachEnabled = restrictedCarryLibraryEnabled && urlParams.get('heightAwareApproach') !== '0';
  // Isolated experiment (root card 05:5xZ): for a LATER carry segment's approach (segmentIndex >= 1) whose complete
  // record the legacy whole-body XY hull refuses, re-check the SAME aligned reference with the existing 32-part
  // swept geometry (authoritative part hulls + heights, same 0.10 m reserve, all scene obstacles). Opt-in only.
  const laterSegmentPartAdmissionEnabled = urlParams.get('laterSegmentPartAdmission') === '1';
  const laterSegmentPartAdmissions = [];
  const pickupFacingApproachEnabled = restrictedCarryLibraryEnabled && recordedApproachEnabled
    && urlParams.get('pickupFacingApproach') === '1';
  const pickupFacingEntryRegionEnabled = pickupFacingApproachEnabled
    && urlParams.get('pickupFacingEntryRegion') === '1';
  // Plan-time pickup-pose reachability ranking (lane WS-D2). Default on with the
  // carry library; ?pickupPoseSearch=0 restores the pre-search ranking exactly.
  const pickupPoseSearchEnabled = restrictedCarryLibraryEnabled && urlParams.get('pickupPoseSearch') !== '0';
  // Ranking passes: 'first' (default) ranks on the first pickup pose from the
  // actual root; 'all' additionally requires nominal later segments (opt-in).
  const pickupPoseSearchTiers = (urlParams.get('pickupPoseSearchTiers') ?? 'first,none').split(',').map(v => v.trim())
    .filter(v => ['all', 'first', 'none'].includes(v));
  if (!pickupPoseSearchTiers.length || pickupPoseSearchTiers.at(-1) !== 'none') pickupPoseSearchTiers.splice(0, pickupPoseSearchTiers.length, 'first', 'none');
  // The alternate one-metre carry joins the library as a different pickup side.
  const alternateCarryEnabled = restrictedCarryLibraryEnabled && urlParams.get('alternateCarry') !== '0';
  // M5 opt-in (default OFF): two holdout-qualified longer clips join the library with maxUses 1.
  const longClipLibraryEnabled = restrictedCarryLibraryEnabled && urlParams.get('longClipLibrary') === '1';
  const longClipSingleSegmentEnabled = longClipLibraryEnabled && urlParams.get('longClipSingleSegment') === '1'; // opt-in: long clips only as one-segment plans
  const longClipAfterStagedEnabled = longClipLibraryEnabled && !longClipSingleSegmentEnabled && urlParams.get('longClipAfterStaged') === '1'; // opt-in: long clip alone or as last leg after `staged` only
  const LONG_CLIP_KEYS = Object.freeze(['long_sub16_010']);
  // Opt-in mid-range clip candidate (root card 2026-09-16): fills the one-pickup coverage hole 0.395–0.79 m with an existing
  // complete OMOMO carry (sub10_053 sibling _084_083_085, 0.46 m). Requires the exported asset public/teacher_carry_mid_sub10_053_084_reference.json;
  // never part of the default library. maxCorrection .10 like the short clips.
  const midClipLibraryEnabled = restrictedCarryLibraryEnabled && urlParams.get('midClipLibrary') === '1';
  const MID_CLIP_KEYS = Object.freeze(['mid_sub10_053_084']);
  // Opt-in (2026-09-16 breadth losses H029/H046): the mid clip only as the single or FINAL carry segment of a plan, never an
  // intermediate pickup. NOTE: replanRemaining can still start a NEW sequence after a short set-down, so an executed final leg
  // may be followed by another pickup across plans; this flag constrains plan structure only.
  const midClipFinalOnlyEnabled = midClipLibraryEnabled && urlParams.get('midClipFinalOnly') === '1';
  // Opt-in experimental (headless evaluation): 32-control STUDENT-only closed-loop lookahead on private physics/history,
  // requested only while an owned first-pickup/staged student approach has left the entry envelope (root < .70 m or
  // upright < .95). Only the existing hard preview criteria can refuse; default off is behaviour-identical.
  const studentClosedLoopPreviewEnabled = restrictedMode && debugControls && urlParams.get('studentClosedLoopPreview') === '1';
  let studentClosedLoopPreview = null; const studentClosedLoopPreviews = [];
  // Opt-in experimental WS-C teacher-descent contact hold: the ordinary teacher carry action is previewed (six distal support
  // bodies allowed on the selected box); when the predicted knee/hip origin comes within the evidence threshold of the box
  // surface, the NEXT control starts a bounded hold on the live-FK target (actual box pose, recomputed interaction graph)
  // while the carry source clock is held, then the clip resumes. Only hard preview criteria refuse; default off is inert.
  const teacherDescentHoldEnabled = restrictedMode && debugControls && urlParams.get('teacherDescentHold') === '1';
  let teacherDescentHold = null, teacherDescentHoldPending = null; const teacherDescentHolds = []; const teacherDescentGeometry = new Map();
  const teacherDescentHoldUsed = new WeakMap(); // parent -> Set(segmentIndex): one hold per carry segment, as in the offline study // root option B: sibling sub10_053_084_083_085, whole-source 0.462 m → reach 0.362–0.562 m covers the 0.400 m request // sub8_042 removed from the candidate after H067 balance stop (int9); asset retained
  const pickupPoseSearchReviews = [];
  const matchedCarryPreviewEnabled = restrictedStudentTransportEnabled && boxExitMode === '1'
    && urlParams.get('matchedCarryPreview') === '1';
  const teacherDescentRematchEnabled = restrictedCarryLibraryEnabled && boxExitMode === '1'
    && urlParams.get('teacherDescentRematch') === '1';
  let teacherDescentAttempts = new WeakMap();
  const teacherDescentDecisions = [];
  // Measured duration experiment: +60 fixed initial poses, preserving the
  // original carry and its +1/+16 horizons. Default remains the30-frame prefix.
  const carryInitialStanceFrames = restrictedMode && debugControls
    && urlParams.get('teacherCarryEntry') === '1' ? 90 : 30;
  const teacherApproachEnabled = recordedApproachEnabled
    || (debugControls && urlParams.get('teacherApproach') === '1');
  const teacherStandingParam = debugControls ? urlParams.get('teacherStanding') : null;
  const teacherStandingMode = ['original', 'live-root', 'neutral'].includes(teacherStandingParam) ? teacherStandingParam : null;
  const teacherStandingAlignment = teacherStandingMode === 'neutral' ? 'live-root' : teacherStandingMode;
  let teacherStandingPlan = null, teacherStandingSteps = 0;
  const boxApproachPlanner = recordedApproachEnabled
    || (debugControls && (urlParams.get('boxApproachRouting') === '1' || teacherApproachEnabled))
    ? new BoxApproachPlanner({ respectTransitClearance: restrictedApproachRoutingEnabled }) : null;
  const boxCollisionBounds = boxApproachPlanner || restrictedMode ? new ObjectMeshBounds(model,
    Array.from(selectableNames, name => findBodyIdByName(model, name)).filter(id => id > 0)) : null;
  const carryDestinationGeometry = restrictedMode ? new ObjectCollisionMeshes(model,
    Array.from(selectableNames, name => findBodyIdByName(model, name)).filter(id => id > 0)) : null;
  const carryReferenceClearance = restrictedMode ? new CarryReferenceClearance(mujoco, model) : null;
  const contactDiagnostics = debugControls || benchmarkMode || midCarryRetargetEnabled ? new ContactDiagnostics(mujoco, model, {
    leftHand: findBodyIdByName(model, 'left_rubber_hand'), rightHand: findBodyIdByName(model, 'right_rubber_hand'),
    feet: ['left_ankle_roll_link', 'right_ankle_roll_link'].map(name => findBodyIdByName(model, name)),
    robotBodies: TEACHER_HUMAN_BODY_NAMES.map(name => findBodyIdByName(model, name)),
    legs: TEACHER_HUMAN_BODY_NAMES.filter(name => /_(hip|knee|ankle)_/.test(name)).map(name => findBodyIdByName(model, name)),
  }) : null;
  let lastCarryRequestClearance = null;
  let lastCarryEntryClearance = null;
  let lastApproachRoute = null;
  let skillLoading = false, skillLoadPromise = null;
  let skillLoadsInFlight = 0;
  let skillRequestVersion = 0;
  const postTaskTargetContactAdmission = new PostTaskTargetContactAdmission();
  // Diagnostic request identity survives queuing and the post-carry exit.
  // It does not select a reference or participate in control decisions.
  let boxTaskRequestSerial = 0, latestBoxTaskRequestId = null, activeBoxTaskRequestId = null;
  const boxTaskRequestLog = new BoxTaskRequestLog();
  const loadedRetargetRequestLog = new BoxTaskRequestLog();
  let loadedRetargetRequestSerial = 0, loadedRetargetCommandGeneration = 0;
  let pendingLoadedRetarget = null, activeLoadedRetargetPromise = null;
  let loadedRetargetReceiptSerial = 0, latestLoadedRetargetReceipt = null;
  const loadedRetargetReviews = [];
  const boxRequestClock = () => ({ episodeVersion, episodeControlStep, controlStep });
  function beginBoxTaskRequest(kind, goal, semanticGoalType = null) {
    if (pendingNormalGroundPush) {
      const superseded = pendingNormalGroundPush; pendingNormalGroundPush = null;
      boxTaskRequestLog.transition(superseded.requestId, 'superseded', 'newer_box_request', boxRequestClock());
      if (activeBoxTaskRequestId === superseded.requestId) retireBoxTaskOwnership('superseded_before_push', noResetApproachOwner);
    }
    postTaskTargetContactAdmission.revoke('new_box_request');
    const requestId = ++boxTaskRequestSerial;
    latestBoxTaskRequestId = requestId;
    boxTaskRequestLog.begin(requestId, kind, goal, boxRequestClock(), semanticGoalType);
    taskCoverageCapture?.capture({stage:'request_received', requestId, task:kind,
      originalGoalWorld:goal, clock:boxRequestClock()});
    return requestId;
  }
  function discardQueuedBoxTask(reason) {
    if (queuedBoxTask) {
      carryStyleRequestSampler?.discard(queuedBoxTask.requestId, reason);
      boxTaskRequestLog.transition(queuedBoxTask.requestId, 'superseded', reason, boxRequestClock());
    }
    queuedBoxTask = null;
  }
  // Promise cache keyed exactly as v16 (kind, 'carry_long', library ids, step/turn ids). `resolved`
  // mirrors the settled values so the reach guide and destination snapping can read
  // them synchronously; a rejected or deleted entry never leaves a stale value behind.
  const loadedSkills = new (class extends Map {
    resolved = new Map();
    set(key, promise) {
      super.set(key, promise);
      Promise.resolve(promise).then(skill => { if (this.get(key) === promise) this.resolved.set(key, skill); }, () => {});
      return this;
    }
    delete(key) { this.resolved.delete(key); return super.delete(key); }
  })();
  let activeBoxTask = null;
  // Shared normal-terminal retirement. Called once the completed controller has
  // already published its terminal receipt and any required box exit has finished.
  // Preserves latestBoxTaskRequestId and diagnostics; clears active ownership only.
  let lastBoxTaskRetirement = null;
  function retireBoxTaskOwnership(reason, completedController) {
    if (activeBoxTask === null && activeBoxTaskRequestId === null) return null;
    lastBoxTaskRetirement = { requestId: activeBoxTaskRequestId, task: activeBoxTask, reason,
      episodeVersion, episodeControlStep, orphanPendingControllerCleared: pendingCarryController !== null
        && pendingCarryController === completedController,
      completedWasActiveCarry: completedController === activeCarryController,
      skillControllerRetained: skillController !== null && skillController === completedController };
    if (pendingCarryController === completedController) pendingCarryController = null;
    activeBoxTask = null;
    activeBoxTaskRequestId = null;
    return lastBoxTaskRetirement;
  }
  let taskDestinationWorld = null;
  const matchedCarryHost = matchedCarryPreviewEnabled || teacherDescentRematchEnabled ? new MatchedCarryHost({mujoco, model,
    readContext:() => ({episode:episodeVersion, physicalControl:episodeControlStep,
      requestId:activeBoxTaskRequestId, latestRequestId:latestBoxTaskRequestId,
      queuedRequestId:queuedBoxTask?.requestId ?? null, parent:skillController,
      activeCarryController, approachParent:pendingWaypointCarryController ?? pendingCarryController,
      originalGoalWorld:taskDestinationWorld}),
    getTeacherBuilder:() => teacherObs, setTeacherBuilder:builder => { teacherObs = builder; },
    allowOrdinaryLoadedSupport:teacherDescentRematchEnabled ? (context, phase) =>
      context.episode === episodeVersion && context.requestId != null
      && context.parent === activeCarryController && activeCarryController != null
      && activeCarryController.skill?.objectBodyName === TEACHER_DESCENT_PROFILE.objectBodyName
      && ['teacher', 'student_transport'].includes(phase) : null,
    runtimeEnvironment:() => ({mujoco, model, data, addresses,
      body:bodyObsBuilder, action:lastAction, target:targetQ, torque:lastTorque,
      previousDofPos, previousDofVel, pointCloudDb, diagnostic:contactDiagnostics, policy, teacherPolicy, zeroNoise,
      onBeforePhysical:({sample, obs, mu, checked}) => {
        lastControlPhase = sample.phase; lastObservation = obs; lastRawAction = Float32Array.from(mu);
        lastControlPreview = checked;
        lastTranslatorResult = {fsmState:obs.length === 1422 && sample.phase !== 'settling' ? 'HOI_FULL' : 'IDLE',
          debug:{matchedCarry:true}};
      },
      onPhysicalCommitted:() => { controlStep++; episodeControlStep++; },
      onAfterControl:() => { syncBodyTransforms(data, bodyGroups); updateMatchedCarryMarker(); },
    }),
  }) : null;
  matchedCarryHost?.reset({episode:episodeVersion, simulationTime:data.time});
  function updateMatchedCarryMarker() {
    const goal = matchedCarryHost?.originalGoalWorld;
    if (!goal) return;
    const objectId = findBodyIdByName(model, matchedCarryHost.owner.raw.objectBodyName);
    goalViz.updateRecorded({objectGoal:goal,
      objectPosition:Array.from(data.xpos.slice(objectId * 3, objectId * 3 + 3))});
  }
  function matchedCarryCommand(command) {
    return matchedCarryHost?.command(command)?.suppressLegacyCarryCancel === true;
  }
  function tryTeacherDescentRematch() {
    const parent = activeCarryController, profile = TEACHER_DESCENT_PROFILE;
    if (!teacherDescentRematchEnabled || matchedCarryHost.inProgress || parent !== skillController
      || parent?.phase !== 'teacher' || parent.rawSkill?.name !== profile.sourceName
      || parent.referenceIndex !== parent.skill.sourceFrames - parent.rawSkill.sourceFrames + profile.firstRaw
      || teacherDescentAttempts.get(parent) === parent.segmentIndex) return;
    teacherDescentAttempts.set(parent, parent.segmentIndex);
    const decision = {episode:episodeVersion, requestId:activeBoxTaskRequestId,
      physicalControl:episodeControlStep, segmentIndex:parent.segmentIndex, supported:false};
    let owner;
    try {
      const objectId = findBodyIdByName(model, profile.objectBodyName);
      const measurement = new StageFeedbackMeasurement(mujoco, model, {rootBodyId:pelvisId, objectBodyId:objectId,
        leftHandBodyId:findBodyIdByName(model, 'left_rubber_hand'), rightHandBodyId:findBodyIdByName(model, 'right_rubber_hand')});
      const force = new mujoco.DoubleBuffer(6);
      let live;
      try { live = measureGroundingRelease({mujoco, model, data, measurement, diagnostic:contactDiagnostics, objectId,
        distalBodyIds:STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.map(name => findBodyIdByName(model, name)), forceBuffer:force}); }
      finally { measurement.dispose(); force.delete(); }
      owner = new TeacherDescentRematchOwner({context:matchedCarryHost.readContext(), prefixParent:parent, live,
        prefixSafety:matchedCarryHost.monitor.snapshot(), readCommandContext:() => matchedCarryHost.context()});
      matchedCarryHost.attachContinuation(owner);
      decision.supported = true; decision.transform = owner.transform;
      decision.entryMismatch = owner.entryMismatch;
      decision.finalReferenceGoalResidualM = owner.finalReferenceGoalResidualM;
    } catch (error) {
      // Admission is optional. Its failure cannot cancel the ordinary task or
      // change the user's goal, controller, histories or physical state.
      owner?.cancel('optional_descent_not_attached');
      decision.reason = String(error?.message ?? error);
    }
    if (decision.supported) {
      const previousTeacher = teacherObs; teacherObs = null;
      skillController = activeCarryController = owner.parent;
      stagedStudentTransport = null; transportTeacherResumePending = false;
      recordedApproachHold = null; teacherStandingPlan = null;
      // Ownership is already transferred. A resource-cleanup failure must not
      // masquerade as a declined admission or cancel the accepted controller.
      try { previousTeacher?.dispose(); }
      catch (error) { console.warn('[teacher descent cleanup]', error); }
    }
    teacherDescentDecisions.push(decision);
    if (teacherDescentDecisions.length > 32) teacherDescentDecisions.shift();
  }
  let previousSkillClickSource = null;
  const pickupButton = document.getElementById('pickup-button');
  const carryButton = document.getElementById('carry-button');
  const longCarryExampleButton = document.getElementById('long-carry-example-button');
  const carryLibraryExamples = (carryStylePreview ? [['style-carry-example-button', 1.15]] : [
    ...(carryStyleSamplingConfig ? [['style-carry-example-button', 1.15]] : []),
    ['short-carry-example-button', .36], ['medium-carry-example-button', 1.4], ['mixed-carry-example-button', 3.3],
  ]).concat(pickupFacingApproachEnabled ? [
    ['angled-carry-example-button', 1.15, -Math.PI / 12],
    ['angled-medium-carry-example-button', 1.4, -Math.PI / 12],
  ] : []).concat(pickupFacingEntryRegionEnabled ? [
    ['entry-region-carry-example-button', 1.4, Math.PI / 12],
  ] : []).map(([id, distanceM, headingRad = 0]) => ({ button: document.getElementById(id), distanceM, headingRad }));
  const resetButton = document.getElementById('reset-button');
  const deselectButton = document.getElementById('deselect-button');
  const taskStatus = document.getElementById('task-status');
  const skillDecisionStatus = document.getElementById('skill-decision-status');
  const previousDofPos = new Float32Array(ACTION_DIM);
  const previousDofVel = new Float32Array(ACTION_DIM);
  const lastTorque = new Float32Array(ACTION_DIM);
  const taskCoverageCapture = (urlParams.has('review') || urlParams.get('taskCoverageCapture') === '1')
    && urlParams.get('taskCoverageCapture') !== '0' ? new TaskCoverageCapture({
      readState:() => readTaskCoverageState({mujoco, model, data, rootId:pelvisId,
        objectIds:[...selectableNames].map(name => [name, findBodyIdByName(model, name)]).filter(([, id]) => id >= 0),
        history:readPolicyHistoryState(), context:{phase:lastControlPhase,
          activeRequestId:activeBoxTaskRequestId, latestRequestId:latestBoxTaskRequestId,
          selectedObject:user.activeObjName, queuedCommand:queuedBoxTask,
          controllerPhase:skillController?.phase ?? null, sourceIndex:skillController?.referenceIndex ?? null,
          requestedCarryGoal:taskDestinationWorld, configuration:Object.fromEntries(urlParams)},
      }),
    }) : null;
  function readPolicyHistoryState() {
    const teacherHistory = builder => builder ? {lastDofPos:Array.from(builder.lastDofPos),
      lastDofVel:Array.from(builder.lastDofVel)} : null;
    return {bodyInitialized:bodyObsBuilder.hasInitialized, bodyHistory:Array.from(bodyObsBuilder.historyBuf),
      boxTeacher:teacherHistory(teacherObs), locomotionTeacher:teacherHistory(restrictedObs),
      previousDofPos:Array.from(previousDofPos), previousDofVel:Array.from(previousDofVel),
      lastAction:Array.from(lastAction), targetQ:Array.from(targetQ), lastTorque:Array.from(lastTorque)};
  }
  // B9 unified interface: transparent `[push]` / `[carry]` / `[pickup]` status prefix and the
  // ring buffer of skill-arbiter decisions (exposed via getState / getSkillDecisionReview).
  let taskStatusSkillLabel = null, selectedSkill = null, skillDecisionSerial = 0;
  let pendingPushAdmission = null;   // R1b (v14a): a push-routed click's lane admission; steps wait for it like the retarget barrier
  const skillDecisions = [];
  function setTaskStatus(message) { if (taskStatus) taskStatus.textContent = taskStatusSkillLabel ? `[${taskStatusSkillLabel}] ${message}` : message; }
  // Plain-language HUD copy for the arbiter's decision. The raw reason string
  // stays in the element's title (and, unchanged, in every record/getState).
  const PUSH_BLOCKER_TEXT = Object.freeze({
    no_push_capability: 'no push clip for this object', task_active: 'a task is still running',
    below_push_range: 'closer than the 0.95 m push clip', above_push_range: 'farther than the 0.95 m push clip',
    above_delivered_reach: 'beyond the push clip\'s measured reach', goal_not_on_floor: 'destination is not on the floor',
    object_yaw_missing: 'object heading unknown', off_face: 'box face is not square to the push direction',
    off_cone: 'robot is not behind the object', entry_state_unvalidated: null, approach_unknown: 'approach not evaluated',
    approach_blocked: 'approach line is blocked', live_entry_unavailable: 'robot is not standing still yet',
    approach_too_far: 'approach is too long', robot_pose_missing: 'robot pose unknown',
    distance_below_pickup_tolerance: null, unsupported_object: 'this object has no carry or push skill', invalid_input: 'invalid destination',
  });
  function describeSkillDecision(decision) {
    const label = String(decision.skill ?? '').toUpperCase(), reason = String(decision.reason ?? '');
    if (decision.skill === 'refuse') return `No skill: ${decision.message ?? reason.replace(/_/g, ' ')}`;
    if (reason.startsWith('push_blocked:')) {
      const parts = reason.slice('push_blocked:'.length).split(',')
        .map(code => PUSH_BLOCKER_TEXT[code] === undefined ? code.replace(/_/g, ' ') : PUSH_BLOCKER_TEXT[code]).filter(Boolean);
      return `Chosen skill: ${label} — push ruled out (${parts.join('; ')})`;
    }
    if (reason.startsWith('push_preconditions_satisfied')) return `Chosen skill: ${label} — straight ahead, within the 0.95 m push clip`;
    if (reason.startsWith('goal_within_placement_tolerance')) return `Chosen skill: ${label} — the destination is where the object already stands`;
    return `Chosen skill: ${label} — ${reason.replace(/_/g, ' ')}`;
  }
  let destinationAdjustmentNote = null;   // set by snapCarryDestination for the current click; shown after the decision
  let lastGoalSource = null;              // 'click' (picker) or 'api' (__interactiveDemo) for the destination being routed
  function showSkillDecision(decision = null, note = null) {
    if (!skillDecisionStatus) return;
    if (!decision) destinationAdjustmentNote = null;
    // The adjustment note leads: a one-line HUD truncates the tail, and the moved
    // destination is what the user must learn first.
    skillDecisionStatus.textContent = decision
      ? (destinationAdjustmentNote ? `${destinationAdjustmentNote} · ` : '') + describeSkillDecision(decision)
      : `Chosen skill: none — ${note ?? 'select an object and click a floor target.'}`;
    skillDecisionStatus.title = decision ? `${decision.skill}: ${decision.reason}` : '';
  }
  function cancelPendingBoxTask(reason = 'user_command') {
    if (!skillLoading) return;
    carryStyleRequestSampler?.discardPending(reason);
    boxTaskRequestLog.closeUnstarted('superseded', reason, boxRequestClock(), { statuses: ['loading'] });
    skillRequestVersion++;
    setTaskStatus('Box task cancelled.');
  }
  function teacherActive() { return pendingCarryController !== null || ['teacher', 'teacher_turn', 'teacher_step'].includes(skillController?.phase); }
  function skillActive() { return Boolean(matchedCarryHost?.inProgress) || pendingCarryController !== null || pendingWaypointCarryController !== null
    || pendingSegmentCarryController !== null || pendingRecoveryParent !== null
    || Boolean(boxExitController?.isBusy())
    || ['approach', 'teacher', 'teacher_turn', 'teacher_step', 'teacher_settling', 'settling', 'settling_quiet'].includes(skillController?.phase); }
  const capitalize = text => text ? text[0].toUpperCase() + text.slice(1) : text;
  /** 'large box' / 'suitcase' for the object a running or just-finished task handles. */
  function carriedObjectLabel() {
    return objectDisplayName(skillController?.skill?.objectBodyName ?? activeCarryController?.skill?.objectBodyName
      ?? boxExitController?.objectBodyName ?? user.activeObjName);
  }
  function finishingTaskMessage() {
    return boxExitController?.isBusy() ? `Stepping clear of the ${carriedObjectLabel()} before following the new command.`
      : skillController?.phase === 'teacher_step' ? 'Finishing this step, then cancelling the box task.'
      : pendingCarryController || skillController?.phase === 'teacher_turn' ? 'Finishing the turn, then cancelling the box task.'
      : 'Finishing the setdown. Movement resumes when the box is down.';
  }
  function updateTaskControls() {
    if (carryStylePreview || carryStyleSamplingConfig) {
      if (pickupButton) pickupButton.hidden = true;
      if (carryButton) carryButton.hidden = true;
    }
    if (pickupButton) pickupButton.disabled = skillLoading || Boolean(restrictedSuspended) || (!restrictedMode && skillActive());
    if (carryButton) carryButton.disabled = skillLoading || Boolean(restrictedSuspended) || (!restrictedMode && skillActive());
    if (longCarryExampleButton) {
      longCarryExampleButton.hidden = !restrictedLongCarryEnabled || Boolean(carryStylePreview);
      longCarryExampleButton.disabled = !restrictedLongCarryEnabled || skillLoading || Boolean(restrictedSuspended);
    }
    for (const { button } of carryLibraryExamples) if (button) {
      button.hidden = !restrictedCarryLibraryEnabled;
      button.disabled = !restrictedCarryLibraryEnabled || skillLoading || Boolean(restrictedSuspended);
    }
    if (deselectButton) deselectButton.disabled = user.activeObjName === null;
    // The public entry keeps the fixed-distance / example buttons hidden even
    // when their debug flags are on (the combined defaults enable them).
    if (publicUi) for (const button of [pickupButton, carryButton, longCarryExampleButton,
      ...carryLibraryExamples.map(example => example.button)]) if (button) button.hidden = true;
  }
  function restoreSkillCommandStyle() {
    if (previousSkillClickSource !== null) {
      translator.setClickPositionSource(previousSkillClickSource);
      previousSkillClickSource = null;
    }
  }
  // VAE latent: deterministic by default. Space samples one vector and holds
  // it until the next reset/resample; the control loop never samples noise.
  user.vaeNoise = new Float32Array(VAE_DIM);
  user.deterministic = true;
  function resampleVaeNoise() {
    // Box-Muller for standard-normal in pure JS.
    for (let i = 0; i < VAE_DIM; i += 2) {
      const u1 = Math.max(Math.random(), 1e-12);
      const u2 = Math.random();
      const mag = Math.sqrt(-2.0 * Math.log(u1));
      user.vaeNoise[i] = mag * Math.cos(2.0 * Math.PI * u2);
      if (i + 1 < VAE_DIM) user.vaeNoise[i + 1] = mag * Math.sin(2.0 * Math.PI * u2);
    }
    user.deterministic = false;
    console.log('[main] vae_noise sampled once and held');
  }
  function doReset({ rootYawRad = null } = {}) {
    if (rootYawRad !== null && !Number.isFinite(rootYawRad)) throw new Error('Reset heading must be finite');
    if (activeLoadedRetargetPromise) {
      setTaskStatus('Goal update in progress. Reset is available after this zero-control check.');
      return false;
    }
    console.log('[main] reset robot + history buffer');
    postTaskTargetContactAdmission.revoke('episode_reset');
    boxTaskRequestLog.reset(boxRequestClock());
    loadedRetargetRequestLog.reset(boxRequestClock());
    pendingLoadedRetarget?.owner?.cancelPending('episode_reset');
    pendingLoadedRetarget = null;
    episodeVersion += 1;
    terminalFacingProbe?.finish('episode_reset');terminalFacingPending=null;recoveredFacingAdmission.reset();terminalFacingEvents.length=0;
    carryStyleRequestSampler?.resetEpisode(episodeVersion);
    skillRequestVersion++;
    episodeControlStep = 0;
    lastTranslatorResult = lastObservation = lastRawAction = null;
    resetRobotToStanding(mujoco, model, data, 0, rootYawRad);
    suitcaseStandingPoseActivated = false;
    matchedCarryHost?.reset({episode:episodeVersion, simulationTime:data.time});
    teacherDescentAttempts = new WeakMap(); teacherDescentDecisions.length = 0;
    bodyObsBuilder.reset();
    restoreSkillCommandStyle();
    translator.reset();
    skillController?.reset();
    skillController = null;
    noResetApproachOwner = null; lastNoResetApproachReview = null;
    lastLargeboxPushLiveDiagnostic = pendingNormalGroundPush = null;
    taskStatusSkillLabel = null; selectedSkill = null; showSkillDecision(null, 'scene reset; choose a new target.');
    activeCarryController?.reset();
    activeCarryController = pendingCarryController = null;
    pendingSegmentCarryController = null;
    stagedStudentApproach = lastStudentApproachEntry = null; studentApproaches.length = 0;
    stagedStudentTransport = lastStudentTransportEntry = null; studentTransports.length = 0;
    transportTeacherResumePending = false;
    pendingWaypointCarryController = null; waypointApproaches = [];
    headingPreparations = []; lastControlPhase = null;
    referenceStudentTurn?.cancel('episode_reset',{owner:null,parent:null,episode:-1,requestId:null,physicalControl:0});
    heightAwareApproach?.cancel('episode_reset');
    referenceStudentTurn=heightAwareApproach=approvedReferenceTurn=null;referenceTeacherResumePending=false;
    referenceStudentTurns=[];heightAwareApproaches=[];usedReferenceStudentTurns=new WeakMap();
    pickupFacingPlans.length = 0; pickupFacingOwners.length = 0;
    preserveCompletionCommand = false;
    teacherStandingPlan = null; teacherStandingSteps = 0;
    restrictedController?.reset(); restrictedObs?.reset();
    hybridArbiter?.reset(); lastHybridDecision = null; lastHybridStatus = null;
    lastRestrictedStep = restrictedPlan = queuedBoxTask = null; restrictedAfterBox = false;
    if (restrictedSuspended) paused = urlParams.get('paused') === '1';
    restrictedSuspended = null;
    lastRestrictedGeometry = null;
    restrictedGeometryCheckStep = null;
    restrictedReferenceAttempts = [];
    approachRecovery?.reset(); pendingRecoveryParent = null; standingPreviewOwned = null;
    transientSuspension = null; pendingOwnedRetirement = null; approachRecoveryDecisions.length = 0;
    recordedApproachHold = null;
    executedApproachTerminal = null; terminalRefusalRecoveryAttempt = null; approachRecoveries.length = 0;
    boxExitController?.reset(); boxExitResults = []; boxTaskResults = [];
    carryLibraryContext = null; lastCarryRefusal = null; carryTaskLineage.length = 0;
    lastControlPreview = null; previewControls = 0;
    boxApproachPlanner?.reset(); lastApproachRoute = null;
    activeBoxTask = null;
    latestBoxTaskRequestId = activeBoxTaskRequestId = null;
    taskDestinationWorld = null;
    lastCarryRequestClearance = null;
    lastCarryEntryClearance = null;
    carryReferenceSelection = null;
    finalCarryPlacement = null;
    teacherObs?.reset();
    previousDofPos.fill(0); previousDofVel.fill(0); lastTorque.fill(0);
    setTaskStatus('Choose a box task, or use the walking controls.');
    setMode('IDLE');
    updateTaskControls();
    user.releaseKeys();
    for (let i = 0; i < ACTION_DIM; i++) lastAction[i] = 0.0;
    for (let i = 0; i < ACTION_DIM; i++) targetQ[i] = 0.0;   // wipe EMA history
    if (user.vaeNoise) user.vaeNoise.fill(0.0);
    user.deterministic = true;
    user.activeObjName = null;
    user.humanGoalWorld = null;
    user.objGoalWorld = null;
    // picker.syncFromUserState() (called below each loop iter) will clear highlight + cache.
  }
  function toggleDeterministic() {
    user.deterministic = !user.deterministic;
    console.log(`[main] deterministic = ${user.deterministic}`);
  }

  const benchmark = (() => {
    if (benchmarkMode !== 'click' && benchmarkMode !== 'click-grid') return null;
    const durationSteps = Math.max(1, Math.round(
      (Number.isFinite(benchmarkDurationS) ? benchmarkDurationS : 5.0) * CONTROL_HZ,
    ));
    const successThresholdM = Number.isFinite(benchmarkSuccessThresholdM)
      ? benchmarkSuccessThresholdM
      : 0.25;
    const fallZM = Number.isFinite(benchmarkFallZM) ? benchmarkFallZM : 0.45;
    let cases;
    if (benchmarkMode === 'click-grid') {
      const distances = parseNumberListParam(urlParams, 'benchDistances', DEFAULT_BENCHMARK_DISTANCES);
      const directions = parseNumberListParam(urlParams, 'benchDirsDeg', DEFAULT_BENCHMARK_DIRECTIONS_DEG);
      cases = [];
      for (const distanceM of distances) {
        for (const directionDeg of directions) {
          cases.push({ distanceM, directionDeg });
        }
      }
    } else {
      cases = [{
        distanceM: Number.parseFloat(urlParams.get('benchDist') || '1.0'),
        directionDeg: Number.parseFloat(urlParams.get('benchDirDeg') || '0.0'),
      }];
    }
    cases = cases.filter((c) => Number.isFinite(c.distanceM) && Number.isFinite(c.directionDeg));
    if (cases.length === 0) return null;
    return {
      mode: benchmarkMode,
      cases,
      caseIndex: 0,
      durationSteps,
      successThresholdM,
      fallZM,
      pendingStart: true,
      done: false,
      current: null,
      results: [],
    };
  })();

  function startBenchmarkCase(rootPosWorld, rootQuatXyzwWorld) {
    if (!benchmark || benchmark.done || !benchmark.pendingStart) return;
    const spec = benchmark.cases[benchmark.caseIndex];
    const goal = goalWorldFromRoot(
      rootPosWorld,
      rootQuatXyzwWorld,
      spec.distanceM,
      spec.directionDeg,
    );
    user.humanGoalWorld = goal;
    user.activeObjName = null;
    user.objGoalWorld = null;
    benchmark.current = {
      ...spec,
      goalWorld: Array.from(goal),
      steps: 0,
      minPelvisZ: Infinity,
      minGoalDistM: Infinity,
      finalGoalDistM: Infinity,
      successStep: -1,
    };
    benchmark.pendingStart = false;
    console.log('[benchmark] start', JSON.stringify({
      index: benchmark.caseIndex,
      total: benchmark.cases.length,
      mode: benchmark.mode,
      distance_m: spec.distanceM,
      direction_deg: spec.directionDeg,
      duration_steps: benchmark.durationSteps,
      goal_world: benchmark.current.goalWorld,
    }));
  }

  function finishBenchmarkCase() {
    if (!benchmark || !benchmark.current) return;
    const c = benchmark.current;
    const stable = c.minPelvisZ >= benchmark.fallZM;
    const goalSuccess = c.successStep >= 0;
    const goalHold = stable && goalSuccess && c.finalGoalDistM <= benchmark.successThresholdM;
    const result = {
      index: benchmark.caseIndex,
      distance_m: c.distanceM,
      direction_deg: c.directionDeg,
      duration_s: benchmark.durationSteps / CONTROL_HZ,
      success_threshold_m: benchmark.successThresholdM,
      fall_z_m: benchmark.fallZM,
      stable,
      goal_success: goalSuccess,
      goal_arrival_hold_success: goalHold,
      success_step: c.successStep,
      min_goal_dist_m: c.minGoalDistM,
      final_goal_dist_m: c.finalGoalDistM,
      min_pelvis_z: c.minPelvisZ,
      goal_world: c.goalWorld,
    };
    benchmark.results.push(result);
    console.log('[benchmark] result', JSON.stringify(result));
    benchmark.current = null;
    user.humanGoalWorld = null;
    benchmark.caseIndex += 1;
    if (benchmark.caseIndex >= benchmark.cases.length) {
      benchmark.done = true;
      window.__interactiveBenchmarkResults = benchmark.results;
      const clickRows = benchmark.results;
      const mean = (key) => clickRows.reduce((s, r) => s + (r[key] ? 1 : 0), 0) / Math.max(clickRows.length, 1);
      const summary = {
        mode: benchmark.mode,
        cases: clickRows.length,
        stable_rate: mean('stable'),
        goal_success_rate: mean('goal_success'),
        goal_arrival_hold_success_rate: mean('goal_arrival_hold_success'),
        min_pelvis_z: Math.min(...clickRows.map((r) => r.min_pelvis_z)),
        mean_final_goal_dist_m: clickRows.reduce((s, r) => s + r.final_goal_dist_m, 0) / Math.max(clickRows.length, 1),
      };
      window.__interactiveBenchmarkSummary = summary;
      console.log('[benchmark] summary', JSON.stringify(summary));
      setStatus(`Benchmark done: hold ${(summary.goal_arrival_hold_success_rate * 100).toFixed(0)}%, min z ${summary.min_pelvis_z.toFixed(2)} m`);
      return;
    }
    doReset();
    benchmark.pendingStart = true;
  }

  function updateBenchmark(rootPosWorld, pelvisZ) {
    if (!benchmark || benchmark.done || !benchmark.current) return;
    const c = benchmark.current;
    const dx = c.goalWorld[0] - rootPosWorld[0];
    const dy = c.goalWorld[1] - rootPosWorld[1];
    const dist = Math.hypot(dx, dy);
    c.steps += 1;
    c.minPelvisZ = Math.min(c.minPelvisZ, pelvisZ);
    c.minGoalDistM = Math.min(c.minGoalDistM, dist);
    c.finalGoalDistM = dist;
    if (c.successStep < 0 && dist <= benchmark.successThresholdM) {
      c.successStep = c.steps;
    }
    if (c.steps >= benchmark.durationSteps) {
      finishBenchmarkCase();
    }
  }

  // RPG-style follow camera. When enabled:
  //   * On toggle, snap the camera to an over-the-shoulder pose: 4 m behind
  //     the robot in three.js world frame (robot's MuJoCo +X = three.js +X
  //     = forward, so behind = −X) and 1.6 m above the pelvis.
  //   * Each step, `controls.target` lerps toward the pelvis world pos so
  //     the view trails the robot. Horizontal lerp is fast (smooth walk
  //     tracking); vertical lerp is slow and floor-clamped at 0.55 m so a
  //     fall doesn't drag the camera down with the body.
  // OrbitControls still works — mouse drag rotates around the moving target.
  let followMode = false;
  const followTarget = new THREE.Vector3();
  const FOLLOW_OFFSET_BEHIND = -4.0;    // metres on three.js −X (= behind robot)
  const FOLLOW_OFFSET_UP     =  1.6;    // metres on three.js +Y (above pelvis)
  const FOLLOW_TARGET_Y_FLOOR = 0.55;   // never let view Y drop below this
  function toggleFollow() {
    followMode = !followMode;
    if (followMode) {
      const pelvisGroup = bodyGroups[pelvisId];
      if (pelvisGroup) {
        pelvisGroup.getWorldPosition(followTarget);
        // Pin target Y to the standing-height floor so a falling robot
        // doesn't immediately drop the camera at toggle time.
        followTarget.y = Math.max(FOLLOW_TARGET_Y_FLOOR, followTarget.y);
        // Snap the camera to the RPG pose so the toggle is *visible*.
        camera.position.set(
          followTarget.x + FOLLOW_OFFSET_BEHIND,
          followTarget.y + FOLLOW_OFFSET_UP,
          followTarget.z,
        );
        controls.target.copy(followTarget);
        controls.update();
      } else {
        followTarget.copy(controls.target);
      }
    }
    console.log(`[main] follow camera → ${followMode ? 'on (over-the-shoulder)' : 'off'}`);
  }

  // Goal-translator visualization. Default on so the first impression is
  // "you can see exactly what the policy is being told." Toggle with G.
  // IMPORTANT: attach to the MuJoCoRoot group (rotated -90° about X to
  // convert MuJoCo Z-up → three.js Y-up). Adding directly to `scene`
  // would put markers in raw MuJoCo coords interpreted as three.js
  // coords, so the robot's MuJoCo Y (sideways) would render as
  // three.js Y (vertical) — sphere appears floating in the sky.
  const goalViz = new GoalViz(root, { visible: true });

  attachKeyboard(user, {
    onUserCommand: (code) => {
      const movement = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Escape'].includes(code);
      if (restrictedMode) {
        if (['Space', 'F1', 'Digit1', 'Digit2', 'Digit3'].includes(code)) return false;
        if (movement) {
          cancelPendingBoxTask('keyboard_command'); discardQueuedBoxTask('keyboard_command');
          if (code === 'Escape') user.releaseKeys();
          preserveCompletionCommand = true;
          if (skillActive()) {
            if (!matchedCarryCommand({kind:code === 'Escape' ? 'escape' : 'movement'})) {
              pendingSegmentCarryController?.requestCancel();
              skillController.requestCancel();
            }
            setTaskStatus(finishingTaskMessage());
          }
        }
        return true;
      }
      if (teacherStandingPlan && ['Space', 'F1', 'Digit1', 'Digit2', 'Digit3'].includes(code)) return false;
      if (movement && teacherStandingPlan) { teacherStandingPlan = null; translator.reset(); }
      if (movement && !teacherActive() && skillController?.phase === 'complete') preserveCompletionCommand = true;
      if (movement && !teacherActive()) taskDestinationWorld = null;
      if (movement) cancelPendingBoxTask('keyboard_command');
      if (!skillActive()) return true;
      if (movement) {
        skillController.requestCancel();
        if (teacherActive()) {
          setTaskStatus(finishingTaskMessage());
          return false;
        }
      }
      return !['Space', 'F1', 'Digit1', 'Digit2', 'Digit3'].includes(code);
    },
    onKeysChanged: () => {
      if (restrictedMode) { restrictedController?.requestKeys(user); restrictedPlan = null; }
    },
    onReset: doReset,
    onResampleNoise: resampleVaeNoise,
    onToggleDeterministic: toggleDeterministic,
    onSetSmoothing: (a) => {
      smoothingAlpha = a;
      console.log(`[main] PD-target EMA alpha → ${a.toFixed(2)} (1=pass-through, lower=smoother)`);
    },
    onToggleFollow: toggleFollow,
    onToggleGoalViz: () => {
      const on = goalViz.toggle();
      console.log(`[main] goal viz ${on ? 'on' : 'off'}`);
    },
  });

  // --- Mouse picker (single-button: click body to toggle select, click
  // ground to set obj_target_pos when an object is selected) ---------- //
  // Cache the flat point cloud array (192 floats = 64 × 3) of whichever
  // object is currently selected. Used in the obs builder each frame.
  let activeObjPointsFlat = null;       // Float32Array(192) | null
  let activeObjBodyId = -1;              // MuJoCo body id of selected obj
  function objectGoalFromGround(point) {
    if (activeObjBodyId < 0 || !activeObjPointsFlat) throw new Error('Select an object first');
    const q = data.xquat.slice(activeObjBodyId * 4, activeObjBodyId * 4 + 4);
    return objectGoalOnFloor(point, activeObjPointsFlat, [q[1], q[2], q[3], q[0]]);
  }
  const picker = attachMousePicker({
    canvas, camera, scene,
    rootGroup: root,
    bodyGroups,
    selectableNames: interactiveSelectableNames,
    user,
    selectionOwnsDestination: destinationPickerEnabled,
    canInteract: () => {
      if (activeLoadedRetargetPromise) return false;
      if (restrictedMode) return true;
      cancelPendingBoxTask();
      if (teacherActive()) {
        skillController.requestCancel();
        setTaskStatus(finishingTaskMessage());
        return false;
      }
      if (skillController?.phase === 'complete') preserveCompletionCommand = true;
      if (['approach', 'settling', 'settling_quiet'].includes(skillController?.phase)) skillController.requestCancel();
      return true;
    },
    objectGoalFromGround,
    onFloorGoal: goal => { if (restrictedMode) submitRestrictedFloorGoal(goal); },
    getReachGuide: () => {
      if (user.activeObjName === null || activeObjBodyId < 0) return null;
      const intervals = carryReachIntervals(user.activeObjName);
      return intervals ? { center: [data.xpos[activeObjBodyId * 3], data.xpos[activeObjBodyId * 3 + 1]], intervals } : null;
    },
    getTaskGoal: () => taskDestinationWorld,
    getTaskGoalRefused: () => {
      // Read request outcomes only to color the user's marker. This has no
      // effect on reference selection, controller state, or the destination.
      const request = boxTaskRequestLog.records.get(latestBoxTaskRequestId);
      return Boolean(taskDestinationWorld && request?.goalWorld && request.episodeVersion === episodeVersion
        && Math.hypot(request.goalWorld[0] - taskDestinationWorld[0], request.goalWorld[1] - taskDestinationWorld[1]) < 1e-6
        && (request.disposition === 'refused' || (request.disposition === 'outcome'
          && !['finished', 'cancelled'].includes(request.reason))));
    },
    onObjectGoal: (goal, { source = 'api', hitBody = null } = {}) => {
      // The selected floor destination goes to the carry planner. Keep the
      // student in standing/locomotion while optional task assets load.
      user.objGoalWorld = null;
      lastGoalSource = source;
      // Retarget binds to the live controller's selected object, including a
      // Plastic-box carry started by the existing multi-object sequence.  The
      // ordinary new-task path below intentionally remains Large-box only.
      if (midCarryRetargetEnabled && activeBoxTask === 'carry' && activeCarryController) {
        const planarGoal = planarPickerRetargetGoal(goal, activeCarryController.requestedGoalWorld);
        const operation = requestLoadedCarryRetarget(planarGoal);
        const receipt = pendingRetargetReceipt({ receiptId: ++loadedRetargetReceiptSerial,
          goalWorld: planarGoal, selectedObject: user.activeObjName, operation });
        latestLoadedRetargetReceipt = receipt;
        void receipt.completion.then(outcome => setTaskStatus(outcome.disposition === 'started'
          ? `Carry destination updated without releasing ${outcome.selectedObject}.`
          : `Carry destination unchanged: ${outcome.reason}`));
        return;
      }
      // A person's click on bare floor that no carry can serve is moved onto the
      // nearest supported distance (e.g. the suitcase's single 2.0 m clip); the HUD
      // says so after the decision. Automation requests (__interactiveDemo), clicks
      // that land on an object body, and clicks while a task is running, loading or
      // queued keep the v16 route unchanged (those are measured from a moving object).
      destinationAdjustmentNote = null;
      if (source === 'click' && hitBody === null && !skillActive() && !skillLoading && queuedBoxTask === null && pendingNormalGroundPush === null) {
        const adjusted = snapCarryDestination(goal);
        if (adjusted.note) { destinationAdjustmentNote = adjusted.note; goal = adjusted.goal; }
      }
      // B9 (skillArbiter=1): one interface. The deterministic skill arbiter (skill_arbiter.js) picks the
      // existing lane for this click (push / carry / pickup) or the existing refusal text.
      if (skillArbiterEnabled) { void routeFloorGoalThroughSkillArbiter(goal); return; }
      const carryRefusal = objectRouter.carryRefusal(user.activeObjName);   // OFF: v5 largebox-only refusal
      if (carryRefusal) {
        setTaskStatus(carryRefusal);
        return;
      }
      void startCarryToGoal(goal);
    },
    onSelect: (bodyName) => {
      if (matchedCarryHost?.inProgress) {
        discardQueuedBoxTask('selection_changed');
        if (!matchedCarryCommand({kind:'selection', objectBodyName:bodyName})) {
          pendingSegmentCarryController?.requestCancel();
          skillController?.requestCancel();
        }
      } else taskDestinationWorld = null;
      try {
        activateSelectedSuitcaseRestPose(bodyName);
        const selection = createObjectSelection(
          bodyName,
          bodyName === null ? -1 : findBodyIdByName(model, bodyName),
          pointCloudDb,
        );
        activeObjPointsFlat = selection.pointsLocal;
        activeObjBodyId = selection.bodyId;
        if (bodyName === null) setMode('idle');
      } catch (error) {
        // Clear both fields together so a failed second selection cannot
        // accidentally retain another object's pose or perception data.
        activeObjPointsFlat = null;
        activeObjBodyId = -1;
        user.activeObjName = user.objGoalWorld = null;
        console.warn(`[main] ${error.message}`);
        setStatus(error.message);
      }
      updateTaskControls();   // Deselect button follows the selection
    },
  });

  // --- Per-frame buffers -------------------------------------------- //
  const lastAction = new Float32Array(ACTION_DIM);   // zeros for first frame
  const zeroNoise = new Float32Array(VAE_DIM);       // for deterministic mode
  const targetQ = new Float32Array(ACTION_DIM);      // persistent → enables EMA

  // PD target smoothing: target_q[i] = alpha * (3·clamp(mu[i])) + (1-alpha) * target_q[i].
  // alpha=1.0 (default) = pass-through, matches training. Lower → smoother
  // visuals at the cost of responsiveness. Toggled by keys 1 / 2 / 3.
  let smoothingAlpha = 1.0;

  function skillProprio(controller = skillController) {
    const objectBodyName = controller?.skill?.objectBodyName || activeCarryController?.skill?.objectBodyName;
    // A locomotion preparation may name a different dummy object. Resuming
    // its parent must read that parent's live object before changing owners.
    const objectId = controller === skillController ? teacherObs?.objectId
      : findBodyIdByName(model, objectBodyName);
    const readQuat = id => {
      const q = data.xquat.slice(id * 4, id * 4 + 4);
      return [q[1], q[2], q[3], q[0]];
    };
    const q = readQuat(pelvisId);
    return {
      rootPosWorld: Array.from(data.xpos.slice(pelvisId * 3, pelvisId * 3 + 3)),
      rootQuatXyzwWorld: q, rootVelWorld: Array.from(data.qvel.slice(0, 3)),
      uprightScore: 1 - 2 * (q[0] ** 2 + q[1] ** 2),
      objectBodyName,
      objPosWorld: objectId === undefined ? null : Array.from(data.xpos.slice(objectId * 3, objectId * 3 + 3)),
      objQuatXyzwWorld: objectId === undefined ? null : readQuat(objectId),
    };
  }

  const vectorDelta = (before, after) => {
    if (before.length !== after.length) throw new Error('Retarget comparison vectors must match');
    let changedCount = 0, maxAbs = 0, maxIndex = 0, squared = 0;
    for (let i = 0; i < before.length; i++) {
      const delta = Math.abs(after[i] - before[i]);
      if (delta !== 0) changedCount++;
      if (delta > maxAbs) { maxAbs = delta; maxIndex = i; }
      squared += delta * delta;
    }
    return { length: before.length, changedCount, maxAbs, maxIndex,
      rmse: Math.sqrt(squared / before.length), l2: Math.sqrt(squared) };
  };
  const exactArray = (a, b) => a.length === b.length
    && Array.from(a).every((value, index) => value === b[index]);
  function loadedRetargetContext() {
    const sequence = activeCarryController, child = sequence?.child, builder = teacherObs;
    const request = boxTaskRequestLog.records.get(activeBoxTaskRequestId);
    if (!midCarryRetargetEnabled) throw new Error('Mid-carry retarget is disabled');
    if (matchedCarryHost?.inProgress || stagedStudentTransport?.active || teacherDescentHold?.active)
      throw new Error('A private carry/hold owner currently owns this control');
    if (!sequence || skillController !== sequence || sequence.phase !== 'teacher' || child?.phase !== 'teacher'
        || activeBoxTask !== 'carry' || queuedBoxTask !== null || !request
        || ['refused', 'superseded', 'reset', 'outcome'].includes(request.disposition))
      throw new Error('The active ordinary carry teacher must own loaded transport');
    if (!builder || builder.objectId < 1 || child.skill.objectBodyName !== sequence.rawSkill.objectBodyName
        || child.skill.objectBodyName !== user.activeObjName)
      throw new Error('The current selected object must remain the teacher-owned carry object');
    return { sequence, child, builder, request };
  }
  // Read-only public timing for the exact ordinary carry owner used by loadedRetargetContext.
  // MixedCarryGoalSequenceController inherits CarryGoalSequenceController; both expose their
  // live CarryGoalController as child. Do not fall back to skillController.sourceFrames: the
  // sequence intentionally has no sourceFrames getter, while its child owns the padded source.
  function activeCarryTimingSnapshot() {
    const sequence = activeCarryController, child = sequence?.child;
    if (!(sequence instanceof CarryGoalSequenceController) || skillController !== sequence
        || sequence.phase !== 'teacher' || child?.phase !== 'teacher') return null;
    const snapshot = { segmentIndex: sequence.segmentIndex,
      referenceIndex: child.referenceIndex, sourceFrames: child.skill?.sourceFrames,
      warpStartFrame: child.warpStartFrame, warpEndFrame: child.warpEndFrame };
    if (!Object.values(snapshot).every(Number.isInteger) || snapshot.segmentIndex < 0
        || snapshot.referenceIndex < 0 || snapshot.sourceFrames < 1
        || snapshot.warpStartFrame < 0 || snapshot.warpStartFrame >= snapshot.warpEndFrame
        || snapshot.warpEndFrame >= snapshot.sourceFrames) return null;
    return snapshot;
  }
  async function previewLoadedCarryRetarget(value) {
    if (pendingLoadedRetarget) throw new Error('Finish or cancel the pending loaded retarget first');
    const goal = Array.from(value ?? []);
    if (goal.length !== 3 || !goal.every(Number.isFinite)) throw new Error('Retarget goal must contain finite XYZ');
    const { sequence, child, builder } = loadedRetargetContext();
    const loads = contactDiagnostics.read(data, builder.objectId), live = skillProprio(sequence);
    const graspRetained = loads.leftHandObjectNormalForceN > 1 && loads.rightHandObjectNormalForceN > 1
      && live.objPosWorld[2] > .5;
    if (!graspRetained) throw new Error('Measured two-hand loaded grasp is required');
    const physicalControl = episodeControlStep, commandGeneration = ++loadedRetargetCommandGeneration;
    const historyGeneration = physicalControl;
    const originalRequestId = ++loadedRetargetRequestSerial;
    loadedRetargetRequestLog.begin(originalRequestId, 'carry', sequence.requestedGoalWorld,
      boxRequestClock(), 'loaded_retarget_origin');
    const owner = new StreamingCarryGoalUpdateOwner({ requestLog: loadedRetargetRequestLog,
      readClock: boxRequestClock });
    owner.attach({ requestId: originalRequestId, episodeVersion,
      objectBodyName: child.skill.objectBodyName, skillName: child.skill.name,
      goalWorld: sequence.requestedGoalWorld, controllerOwner: child,
      teacherBuilder: builder, commandGeneration, historyGeneration });
    const updateRequestId = ++loadedRetargetRequestSerial;
    loadedRetargetRequestLog.begin(updateRequestId, 'carry', goal, boxRequestClock(), 'loaded_retarget_goal');
    const submitted = owner.submit({ requestId: updateRequestId, episodeVersion,
      objectBodyName: child.skill.objectBodyName, skillName: child.skill.name, goalWorld: goal });
    if (submitted.behavior !== 'latest_mid_carry_goal')
      throw new Error(`Retarget command was not admitted: ${submitted.behavior}`);
    const proposal = spliceLoadedCarryGoal({ sequence, requestedGoalWorld: goal,
      objectBodyName: child.skill.objectBodyName, controllerOwner: child, expectedControllerOwner: child,
      teacherBuilder: builder, expectedTeacherBuilder: builder,
      commandGeneration, expectedCommandGeneration: commandGeneration,
      historyGeneration, expectedHistoryGeneration: historyGeneration,
      graspRetained, loadedPhase: 'teacher_loaded_transport', commonTranslationWarp, publish: false });
    const qpos = Array.from(data.qpos), qvel = Array.from(data.qvel), ctrl = Array.from(data.ctrl);
    const action = Array.from(lastAction), torque = Array.from(lastTorque), target = Array.from(targetQ);
    const previousPosition = Array.from(previousDofPos), previousVelocity = Array.from(previousDofVel);
    const history = { lastDofPos: Array.from(builder.lastDofPos), lastDofVel: Array.from(builder.lastDofVel) };
    const build = references => {
      builder.reset(history);
      try { return builder.build(data, references, lastAction, lastTorque); }
      finally { builder.reset(history); }
    };
    let oldObservation, newObservation, oldAction, newAction;
    try {
      oldObservation = build([proposal.oldFrames[child.referenceIndex + 1],
        proposal.oldFrames[child.referenceIndex + 16]]);
      oldAction = await teacherPolicy.infer(oldObservation);
      if (episodeVersion !== owner.active.episodeVersion || episodeControlStep !== physicalControl
          || child !== sequence.child || builder !== teacherObs)
        throw new Error('Retarget ownership changed during old-action inference');
      newObservation = build(proposal.referenceFrames);
      newAction = await teacherPolicy.infer(newObservation);
    } finally { builder.reset(history); }
    const stateUnchanged = exactArray(qpos, data.qpos) && exactArray(qvel, data.qvel)
      && exactArray(ctrl, data.ctrl) && exactArray(action, lastAction)
      && exactArray(torque, lastTorque) && exactArray(target, targetQ)
      && exactArray(previousPosition, previousDofPos) && exactArray(previousVelocity, previousDofVel)
      && exactArray(history.lastDofPos, builder.lastDofPos)
      && exactArray(history.lastDofVel, builder.lastDofVel)
      && episodeControlStep === physicalControl && child.referenceIndex === proposal.record.referenceIndex;
    const observationDelta = vectorDelta(oldObservation, newObservation);
    const actionDelta = vectorDelta(oldAction, newAction);
    const passed = stateUnchanged && proposal.record.positionBoundaryMaxAbs === 0
      && proposal.record.rotationBoundaryMaxAbs === 0 && proposal.record.velocityBoundaryMaxAbs === 0
      && proposal.record.relativeGeometryRemainderMaxAbs <= 2e-7
      && actionDelta.maxAbs <= MID_CARRY_RETARGET_ACTION_MAX_DELTA;
    const review = { schema: 'web_mid_carry_retarget_preview_v1',
      previewId: `${episodeVersion}:${physicalControl}:${updateRequestId}`,
      episodeVersion, physicalControl, activeBoxTaskRequestId, requestId: updateRequestId,
      objectBodyName: child.skill.objectBodyName, referenceIndex: child.referenceIndex,
      sourceFrames: child.skill.sourceFrames, oldGoalWorld: Array.from(sequence.requestedGoalWorld),
      requestedGoalWorld: goal, handNormalForceN: [loads.leftHandObjectNormalForceN,
        loads.rightHandObjectNormalForceN], objectPositionWorld: Array.from(live.objPosWorld),
      commandGeneration, historyGeneration, proposal: structuredClone(proposal.record),
      observationDelta, actionDelta, actionContinuityLimit: MID_CARRY_RETARGET_ACTION_MAX_DELTA,
      stateUnchanged, passed, physicsControlsConsumed: 0, physicalGoalUpdateQualified: false };
    loadedRetargetReviews.push(review);
    pendingLoadedRetarget = { owner, token: submitted.token, preview: review, sequence, child, builder,
      goal, commandGeneration, historyGeneration, physicalControl, episodeVersion,
      activeBoxTaskRequestId, qpos, qvel, action, torque, target, previousPosition, previousVelocity,
      teacherLastDofPos: Array.from(builder.lastDofPos), teacherLastDofVel: Array.from(builder.lastDofVel) };
    return structuredClone(review);
  }
  function applyLoadedCarryRetarget(previewId) {
    const pending = pendingLoadedRetarget;
    if (!pending || pending.preview.previewId !== previewId || !pending.preview.passed)
      throw new Error('A current passed zero-control retarget preview is required');
    if (pending.episodeVersion !== episodeVersion || pending.physicalControl !== episodeControlStep
        || pending.activeBoxTaskRequestId !== activeBoxTaskRequestId
        || pending.sequence !== activeCarryController || pending.child !== activeCarryController.child
        || pending.builder !== teacherObs || pending.child.skill.objectBodyName !== user.activeObjName)
      throw new Error('Retarget owner, selected object, request or physical clock changed after preview');
    if (!exactArray(pending.qpos, data.qpos) || !exactArray(pending.qvel, data.qvel)
        || !exactArray(pending.action, lastAction) || !exactArray(pending.torque, lastTorque)
        || !exactArray(pending.target, targetQ) || !exactArray(pending.previousPosition, previousDofPos)
        || !exactArray(pending.previousVelocity, previousDofVel)
        || !exactArray(pending.teacherLastDofPos, pending.builder.lastDofPos)
        || !exactArray(pending.teacherLastDofVel, pending.builder.lastDofVel))
      throw new Error('Retarget action, physics or teacher history changed after preview');
    const loads = contactDiagnostics.read(data, pending.builder.objectId);
    const live = skillProprio(pending.sequence);
    const graspRetained = loads.leftHandObjectNormalForceN > 1 && loads.rightHandObjectNormalForceN > 1
      && live.objPosWorld[2] > .5;
    const committed = pending.owner.commitControllerSplice({ token: pending.token, episodeVersion,
      controllerOwner: pending.child, teacherBuilder: pending.builder,
      commandGeneration: pending.commandGeneration, historyGeneration: pending.historyGeneration,
      phase: 'teacher_loaded_transport', splice: request => spliceLoadedCarryGoal({
        sequence: pending.sequence, requestedGoalWorld: request.goalWorld,
        objectBodyName: pending.child.skill.objectBodyName,
        controllerOwner: pending.child, expectedControllerOwner: pending.child,
        teacherBuilder: pending.builder, expectedTeacherBuilder: pending.builder,
        commandGeneration: pending.commandGeneration,
        expectedCommandGeneration: pending.commandGeneration,
        historyGeneration: pending.historyGeneration,
        expectedHistoryGeneration: pending.historyGeneration,
        graspRetained, loadedPhase: 'teacher_loaded_transport', commonTranslationWarp, publish: true }) });
    if (committed.accepted !== true) throw new Error(`Retarget splice refused: ${committed.reason}`);
    taskDestinationWorld = Array.from(pending.goal);
    boxTaskRequestLog.transition(activeBoxTaskRequestId, 'started', 'mid_carry_goal_update',
      boxRequestClock(), { loadedRetargetRequestId: committed.requestId,
        oldGoalWorld: pending.preview.oldGoalWorld, requestedGoalWorld: Array.from(pending.goal),
        objectBodyName: pending.child.skill.objectBodyName,
        referenceIndex: pending.child.referenceIndex, physicsControlsConsumed: 0 });
    pending.preview.applied = true;
    pending.preview.appliedPhysicalControl = episodeControlStep;
    pending.preview.splice = structuredClone(committed.result.record);
    pendingLoadedRetarget = null;
    return structuredClone(pending.preview);
  }
  function cancelLoadedCarryRetarget(reason = 'user_cancel') {
    if (!pendingLoadedRetarget) return false;
    pendingLoadedRetarget.owner.cancelPending(reason);
    pendingLoadedRetarget = null;
    return true;
  }
  function requestLoadedCarryRetarget(value) {
    if (!midCarryRetargetEnabled) return Promise.reject(new Error('Mid-carry retarget is disabled'));
    if (activeLoadedRetargetPromise) return Promise.reject(new Error('A retarget update is already in progress'));
    activeLoadedRetargetPromise = (async () => {
      if (activeStepPromise) await activeStepPromise;
      const preview = await previewLoadedCarryRetarget(value);
      if (!preview.passed) {
        cancelLoadedCarryRetarget('zero_control_continuity_refused');
        throw new Error(`Retarget continuity refused: action delta ${preview.actionDelta.maxAbs}`);
      }
      return applyLoadedCarryRetarget(preview.previewId);
    })().catch(error => {
      cancelLoadedCarryRetarget('request_error');
      throw error;
    }).finally(() => { activeLoadedRetargetPromise = null; });
    return activeLoadedRetargetPromise;
  }

  // --- Safe recovery helpers (WS-D1) --------------------------------------- //
  function readSweepObstacles(scope = 'recovery') {
    const bounds = boxCollisionBounds.read(data);
    const convex = recoveryFlags.convexObstacles === 'all' || (recoveryFlags.convexObstacles === 'recovery' && scope === 'recovery');
    if (!convex || !carryDestinationGeometry) return bounds;
    try {
      const convex = convexObstaclesFromProjections(carryDestinationGeometry.read(data));
      return bounds.map(rect => { const match = convex.find(o => o.bodyId === rect.bodyId); return match?.hull ? { ...rect, name: match.name, hull: match.hull } : rect; });
    } catch (error) { console.warn('[sweep obstacles]', error); return bounds; }
  }
  function postTaskContactObstacles(obstacles, intent) {
    return postTaskTargetContactAdmission.selectObstacles(obstacles, {
      episodeVersion, activeBoxTask, boxTaskBusy: skillActive(), skillLoading, intent,
    });
  }
  function robotUnloaded() {
    // UNLOADED = selected box on the floor (z <= 0.25) with no hand support.
    const live = skillProprio(activeCarryController ?? skillController);
    if (!live.objPosWorld) return !skillActive();
    if (!live.objPosWorld.every(Number.isFinite) || live.objPosWorld[2] > .25) return false;
    if (!contactDiagnostics) return true;
    const objectId = findBodyIdByName(model, live.objectBodyName);
    if (objectId < 0) return false;
    const loads = contactDiagnostics.read(data, objectId);
    return !(loads.leftHandObjectNormalForceN > .1 || loads.rightHandObjectNormalForceN > .1);
  }
  function balanceLost() {
    const q = data.xquat.slice(pelvisId * 4, pelvisId * 4 + 4);
    return data.xpos[pelvisId * 3 + 2] < .45 || 1 - 2 * (q[1] ** 2 + q[2] ** 2) < .5;
  }
  function neverSuspendEligible(reason) {
    if (!recoveryFlags.neverSuspend || reason === 'lost_balance' || reason === 'preview_balance' || balanceLost()) return false;
    try { return robotUnloaded(); } catch { return false; }
  }
  function recoveryContext() {
    const request = boxTaskRequestLog.records.get(activeBoxTaskRequestId);
    const received = request?.events?.find(event => event.disposition === 'received')?.episodeControlStep ?? null;
    return { episode: episodeVersion, requestId: activeBoxTaskRequestId, physicalControl: episodeControlStep,
      requestReceivedControl: received, deadlineControl: received === null ? null : received + recoveryFlags.recoveryBudgetControls };
  }
  /** Hand a refused unloaded approach to the recovery loop. Returns the cycle's first step or null. */
  function startApproachRecovery(kind, carry, refusal, skillStep) {
    if (!approachRecovery || !carry || !teacherObs) return null;
    let unloaded = false; try { unloaded = robotUnloaded(); } catch { unloaded = false; }
    const started = approachRecovery.tryStart({ kind, parent: carry, live: skillProprio(carry), context: recoveryContext(), refusal, unloaded });
    approachRecoveryDecisions.push({ ...structuredClone(started.decision), originalRefusal: { mode: skillStep?.mode ?? null, supported: skillStep?.supported ?? null,
      justCompleted: skillStep?.justCompleted ?? null, completionReason: skillStep?.completionReason ?? null } });
    if (approachRecoveryDecisions.length > 32) approachRecoveryDecisions.shift();
    if (!started.supported) return null;
    executedApproachTerminal = null; recordedApproachHold = null; teacherStandingPlan = null;
    pendingWaypointCarryController = null; pendingRecoveryParent = carry; skillController = started.controller;
    teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, skillController.skill);
    translator.reset(); user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
    boxApproachPlanner?.reset(); lastApproachRoute = null;
    setTaskStatus(skillController.statusMessage);
    return skillController.step(skillProprio());
  }
  /** Owned retirement of a refused, unloaded approach in the same control (no suspension). */
  function ownedApproachRetirement(skillStep, carry) {
    executedApproachTerminal = null; recordedApproachHold = null; teacherStandingPlan = null;
    pendingWaypointCarryController = null; skillController = null;
    waypointApproaches.push({ completionReason: skillStep.completionReason, outcome: skillStep.outcome ?? null, ownedRetirement: true });
    return { ...skillStep, phase: 'complete', mode: 'student', supported: false, justCompleted: true, referenceFrames: null,
      requestedGoalWorld: carry?.requestedGoalWorld ? Array.from(carry.requestedGoalWorld) : skillStep.requestedGoalWorld ?? null, ownedRetirement: true };
  }
  /** A refusal found after inference cannot execute in this control. Instead of
   * PAUSED until Reset, suspend for exactly this control; the next control
   * retires the owner with an owned reason and resumes restricted standing. */
  function beginTransientSuspension(reason) {
    if (!neverSuspendEligible(reason)) return false;
    const same = transientSuspension && transientSuspension.episode === episodeVersion && transientSuspension.physicalControl === episodeControlStep;
    const count = (same ? transientSuspension.count : 0) + 1;
    if (count > 3) return false; // repeated refusals at one control: protective stop
    transientSuspension = { episode: episodeVersion, physicalControl: episodeControlStep, reason, count, released: false };
    setTaskStatus(RECOVERY_REFUSAL_MESSAGES[reason] ?? 'That movement is not supported from here. Choose another direction or destination.');
    return true;
  }
  function releaseTransientSuspension() {
    const suspension = transientSuspension;
    if (!suspension || suspension.released || suspension.episode !== episodeVersion || suspension.physicalControl !== episodeControlStep) return false;
    suspension.released = true; restrictedSuspended = null; lastRestrictedStep = null;
    const parent = pendingRecoveryParent ?? pendingWaypointCarryController ?? pendingSegmentCarryController ?? pendingCarryController ?? activeCarryController;
    const goal = parent?.requestedGoalWorld ? Array.from(parent.requestedGoalWorld) : taskDestinationWorld ? Array.from(taskDestinationWorld) : null;
    if (!(skillController instanceof ApproachRecoveryCycle && pendingRecoveryParent)) {
      if (skillController === boxExitController) boxExitController.reset();
      if (pendingSegmentCarryController?.phase === 'awaiting_exit') { try { pendingSegmentCarryController.abandonAwaitingExit(suspension.reason); } catch (error) { console.warn('[transient suspension]', error); } }
      const completed = pendingSegmentCarryController?.phase === 'complete' ? pendingSegmentCarryController : null;
      pendingWaypointCarryController = pendingCarryController = pendingSegmentCarryController = pendingRecoveryParent = null;
      recordedApproachHold = null; executedApproachTerminal = null; teacherStandingPlan = null;
      terminalFacingProbe?.finish(suspension.reason); terminalFacingPending = null;
      skillController = completed;
      if (!completed) pendingOwnedRetirement = { completionReason: suspension.reason, requestedGoalWorld: goal, outcome: null };
    }
    restrictedController?.reanchor();
    updateTaskControls();
    return true;
  }

  function submitRestrictedFloorGoal(goal) {
    if (!restrictedController) return false;
    if (goal === null) { restrictedController.requestCancel(); restrictedPlan = null; return true; }
    const live = skillProprio();
    restrictedPlan = planRestrictedFloorGoal(live.rootPosWorld, goal, boxCollisionBounds.read(data));
    if (!restrictedPlan.supported) {
      setTaskStatus(restrictedPlan.reason === 'goal_distance' ? 'Choose a walking destination within 2 metres.'
        : 'That destination needs more clearance or a shorter route. Choose another point.');
      return false;
    }
    cancelPendingBoxTask('floor_command'); discardQueuedBoxTask('floor_command');
    user.releaseKeys(); preserveCompletionCommand = true;
    if (skillActive()) {
      if (!matchedCarryCommand({kind:'movement'})) {
        pendingSegmentCarryController?.requestCancel();
        skillController.requestCancel();
      }
    }
    restrictedController.requestFloorGoal(restrictedPlan.finalGoalWorld, { waypoints: restrictedPlan.waypoints });
    setTaskStatus(skillActive() ? finishingTaskMessage() : 'Walking to the destination in measured steps…');
    return true;
  }

  function finishWaypointApproach(skillStep) {
    const firstPickupEnabled=restrictedMode&&debugControls&&urlParams.get('firstPickupStudentRecovery')==='1';
    const firstPickupTerminalRefusal=firstPickupEnabled
      &&skillController instanceof TeacherRecordedApproachController
      &&skillStep?.mode==='none'&&skillStep.supported===false&&skillStep.justCompleted===false
      &&skillStep.completionReason==='reference_sweep_clearance';
    const terminalRefusal = restrictedApproachRecoveryEnabled
      && urlParams.get('terminalRefusalRecovery') === '1'
      && skillController instanceof TeacherRecordedApproachController
      && skillStep?.mode === 'none' && skillStep.supported === false
      && skillStep.justCompleted === false && skillStep.completionReason === 'reference_sweep_clearance';
    const latchedRefusal = skillStep?.mode === 'none' && skillStep.supported === false && skillStep.justCompleted === false;
    if (!pendingWaypointCarryController || (!skillStep?.justCompleted && !terminalRefusal && !firstPickupTerminalRefusal
        && !(latchedRefusal && recoveryFlags.neverSuspend))) return skillStep;
    if (terminalRefusal) terminalRefusalRecoveryAttempt = {
      episode:episodeVersion, physicalControl:episodeControlStep,
      encoding:{mode:skillStep.mode,supported:skillStep.supported,
        justCompleted:skillStep.justCompleted,completionReason:skillStep.completionReason},
      saved:readExecutedApproachTerminalReview()
    };
    const carry = pendingWaypointCarryController;
    if(skillController instanceof TeacherApproachRecoveryController)
      recoveredFacingAdmission.completeRecovery({...recoveredFacingRecoveryContext(skillController),step:skillStep});
    const saved = executedApproachTerminal;
    if (restrictedApproachRecoveryEnabled && skillStep.completionReason === 'reference_sweep_clearance'
        && saved && saved.episode === episodeVersion && saved.owner === skillController && saved.parent === carry
        && !saved.owner.cancelRequested && !carry.finishRequested && saved.endpointDistanceM <= carry.arrivalRadius) {
      const recovery = new TeacherApproachRecoveryController({ skill: saved.skill, plan: saved.plan,
        goalWorld: Array.from(carry.approachGoalWorld), radius: carry.arrivalRadius,
        approveReference: approveRestrictedReference });
      recoveredFacingAdmission.registerRecovery({...recoveredFacingRecoveryContext(recovery),
        recovery,saved,parent:carry,owner:skillController});
      approachRecoveries.push({ startControl: episodeControlStep, sourceFrames: saved.skill.sourceFrames,
        executedAtControl: saved.executedAtControl, endpointDistanceM: saved.endpointDistanceM,
        originalRefusal: structuredClone(skillStep.outcome), controller: recovery,
        ...(terminalRefusal ? {originalRefusalEncoding:{mode:skillStep.mode,
          supported:skillStep.supported, justCompleted:skillStep.justCompleted}} : {}) });
      executedApproachTerminal = null;
      waypointApproaches.push({ completionReason: 'approach_recovery_after_refusal', outcome: skillStep.outcome });
      skillController = recovery;
      // Preserve the DOF and body histories from the actual approach. This
      // bounded hold uses the original terminal; it never reanchors the goal.
      user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
      setTaskStatus(`Settling closer to the ${carriedObjectLabel()} before lifting…`);
      return recovery.step(skillProprio());
    }

    if(firstPickupEnabled&&skillController instanceof TeacherRecordedApproachController
        &&skillStep.completionReason==='reference_sweep_clearance') {
      const live=skillProprio(carry),objectId=findBodyIdByName(model,carry.skill.objectBodyName);
      // A fresh direct-path check at the actual reached pose; no stale start route.
      const route=new BoxApproachPlanner({respectTransitClearance:false}).step(live.rootPosWorld,
        carry.approachGoalWorld,boxCollisionBounds.read(data));
      const contact=readFirstPickupContacts({mujoco,model,data,objectId,rootId:pelvisId,
        feet:['left_ankle_roll_link','right_ankle_roll_link'].map(name=>findBodyIdByName(model,name)),
        objectIds:Array.from(selectableNames,name=>findBodyIdByName(model,name)).filter(id=>id>0)});
      const command=firstPickupCommandContext();
      const options={...command,parent:carry,activeParent:activeCarryController,pendingParent:pendingWaypointCarryController,
        owner:skillController,step:skillStep,live,route,contact,episode:episodeVersion,
        physicalControl:episodeControlStep,requestId:activeBoxTaskRequestId,readContext:firstPickupCommandContext};
      globalThis.__combinedFirstPickupBeforeClaim?.({options});
      const claim=firstPickupStudentRegistry.tryClaim(options);
      globalThis.__combinedFirstPickupAfterClaim?.({options,supported:claim.supported,
        reason:claim.reason??null,entry:claim.entry??null});
      firstPickupStudentDecisions.push({episode:episodeVersion,requestId:activeBoxTaskRequestId,
        physicalControl:episodeControlStep,supported:claim.supported,reason:claim.reason,
        originalGoalWorld:Array.from(carry.requestedGoalWorld),live:structuredClone(live),
        route:structuredClone(route),contact:structuredClone(contact),
        originalRefusal:{mode:skillStep.mode,supported:skillStep.supported,justCompleted:skillStep.justCompleted,
          completionReason:skillStep.completionReason,outcome:structuredClone(skillStep.outcome)},
        entry:claim.entry??null});
      if(claim.supported) {
        const attempt=StagedStudentApproachController.tryStart({...options,firstPickupLease:claim.lease});
        if(!attempt.supported)throw Error('Admitted first pickup student construction failed: '+attempt.reason);
        const actualTeacherHistory={lastDofPos:Float32Array.from(teacherObs.lastDofPos),
          lastDofVel:Float32Array.from(teacherObs.lastDofVel)};
        waypointApproaches.push({completionReason:'first_pickup_student_after_clearance_refusal',outcome:skillStep.outcome});
        pendingWaypointCarryController=null;executedApproachTerminal=null;recordedApproachHold=null;teacherStandingPlan=null;
        skillController=carry;stagedStudentApproach=attempt.controller;studentApproaches.push(stagedStudentApproach);
        lastStudentApproachEntry={episode:episodeVersion,physicalControl:episodeControlStep,segmentIndex:0,
          supported:true,reason:null,distanceM:claim.entry.distanceM,origin:'owned_first_pickup_refusal'};
        teacherObs.dispose();teacherObs=new TeacherObsBuilder(mujoco,model,carry.skill);teacherObs.reset(actualTeacherHistory);
        translator.reset();translator.setClickPositionSource('stable_receding');
        user.humanGoalWorld=user.objGoalWorld=null;user.releaseKeys();
        boxApproachPlanner?.reset();lastApproachRoute=null;
        boxTaskRequestLog.transition(activeBoxTaskRequestId,'started','first_pickup_student_recovery',boxRequestClock());
        setTaskStatus('Approaching the grasp before continuing to your destination…');
        return carry.step(skillProprio(carry));
      }
    }
    // Safe recovery (unloaded only): a refused recorded approach hands to the
    // bounded retreat/turn loop; otherwise a latched refusal retires the task in
    // this control instead of suspending. A loaded robot keeps the old behavior.
    if (recoveryFlags.approachRecoveryLoop && skillStep?.completionReason === 'reference_sweep_clearance'
        && (skillStep.justCompleted || latchedRefusal) && skillController instanceof TeacherRecordedApproachController) {
      const recovered = startApproachRecovery('approach', carry, { reason: skillStep.completionReason }, skillStep);
      if (recovered) return recovered;
    }
    if (latchedRefusal && neverSuspendEligible(skillStep.completionReason)) return ownedApproachRetirement(skillStep, carry);
    // A latched refusal with no eligible actual terminal keeps the old result,
    // paused parent, histories and ordinary suspension behavior.
    if (terminalRefusal || firstPickupTerminalRefusal || latchedRefusal) return skillStep;
    executedApproachTerminal = null;
    pendingWaypointCarryController = null;
    waypointApproaches.push({ completionReason: skillStep.completionReason, outcome: skillStep.outcome });
    if (skillStep.completionReason === 'finished') {
      if (skillStep.handoffReferenceFrames) {
        recordedApproachHold = { parent: carry, goalWorld: skillController.requestedGoalWorld,
          referenceFrames: skillStep.handoffReferenceFrames, episode: episodeVersion,
          requiresPreview: skillController instanceof TeacherApproachRecoveryController
            || skillController instanceof PickupFacingApproachController };
        restrictedObs.reset({ lastDofPos: previousDofPos, lastDofVel: previousDofVel });
      }
      // The carry clock was paused in approach. Resume it from the actual
      // reached pose, retaining its original box destination and segment.
      skillController = carry;
      teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, carry.skill);
      translator.reset(); translator.setClickPositionSource('stable_receding');
      user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
      boxApproachPlanner.reset(); lastApproachRoute = null;
      skillStep = carry.step(skillProprio());
    } else {
      skillStep = { ...skillStep, requestedGoalWorld: carry.requestedGoalWorld };
    }
    return skillStep;
  }

  function loadTeacherActor() {
    if (!skillLoadPromise) skillLoadPromise = (async () => {
      const actor = new TeacherPolicy();
      try { await actor.load('public/teacher_policy.onnx'); teacherPolicy = actor; }
      catch (error) { await actor.dispose(); throw error; }
    })().catch(error => { skillLoadPromise = null; throw error; });
    return skillLoadPromise;
  }

  function queuedRestrictedIntent() {
    if (!restrictedController) return null;
    const intent = restrictedController.requestedIntent;
    return intent.revision > 0 && (skillActive() || intent.revision !== lastRestrictedStep?.activeIntent?.revision) ? intent : null;
  }

  // ---- WS-B planner honesty helpers -------------------------------------
  /** Controls left before the post-click deadline of a request, measured on
   * the same control clock the request log stamped at `received`. */
  function carryRequestControlBudget(requestId = activeBoxTaskRequestId ?? latestBoxTaskRequestId) {
    const record = requestId == null ? null : boxTaskRequestLog.records.get(requestId);
    const received = record?.events?.find(event => event.disposition === 'received') ?? record?.events?.[0];
    if (!received || record.episodeVersion !== episodeVersion || !Number.isInteger(received.controlStep)
        || controlStep < received.controlStep) return null;
    return { requestId, ...remainingCarryControls({ requestControlStep: received.controlStep,
      currentControlStep: controlStep, budgetControls: carryControlBudgetControls }) };
  }
  /** Lane WS-D2: the loaded walking/turning library and per-request memo for
   * plan-time pickup-pose reachability; null when the search is off or the
   * measured locomotion library is not loaded. */
  function pickupPoseSearchContext() {
    if (!(pickupPoseSearchEnabled && restrictedWalkSkills.length > 0 && pickupFacingSweeps instanceof Map)) return null;
    return { walkSkills: restrictedWalkSkills, turnSkills: restrictedTurnSkills, sweeps: pickupFacingSweeps,
      cache: new Map(), referenceCache: new Map(), checkCache: new Map() };
  }
  function makeCarryPlanChecker(pickupPoseSearch = null) {
    if (!carryReferenceClearance) throw new Error('Carry planning requires complete reference geometry');
    const rootWxyz = Array.from(data.xquat.slice(pelvisId * 4, pelvisId * 4 + 4));
    return plan => {
      const checked = checkMixedCarryClearance({ plan, liveData: data,
        destinationGeometry: carryDestinationGeometry, pathChecker: carryReferenceClearance,
        rootPositionWorld: Array.from(data.xpos.slice(pelvisId * 3, pelvisId * 3 + 3)),
        rootQuaternionXyzwWorld: [rootWxyz[1], rootWxyz[2], rootWxyz[3], rootWxyz[0]],
        exitSkill: boxExitController?.retreatSkill, pickupPoseSearch });
      return { ...checked, approachCostM: checked.predictedApproachCostM };
    };
  }
  function buildCarryLibrary(context, requestedDistanceM, { alternate = false } = {}) {
    const { defaultSkill, longCarrySkill, librarySkills } = context;
    return selectCarryStyleCandidates(createCarrySkillLibrary({ shortPlacement: librarySkills[0], smallPlacement: librarySkills[1],
        mediumPlacement: studentLiftPreviewEnabled ? withStudentLiftProfile(librarySkills[2], {
          libraryId: STUDENT_LIFT_PROFILE.libraryId, referenceUrl: STUDENT_LIFT_PROFILE.referenceUrl,
        }) : librarySkills[2], stagedCarry: defaultSkill, longCarry: longCarrySkill,
        alternateCarry: alternate ? librarySkills[3] ?? null : null,
        longClips: [...(longClipLibraryEnabled ? LONG_CLIP_KEYS.map(id => ({ id, skill: librarySkills[context.libraryKeys?.indexOf(id) ?? -1] ?? null })) : []),
          ...(midClipLibraryEnabled ? MID_CLIP_KEYS.map(id => ({ id, skill: librarySkills[context.libraryKeys?.indexOf(id) ?? -1] ?? null, maxCorrection: .10, finalSegmentOnly: midClipFinalOnlyEnabled })) : [])],
        longClipsSingleSegmentOnly: longClipSingleSegmentEnabled, longClipsLastLegAfter: longClipAfterStagedEnabled ? ['staged'] : null },
      { stagedInitialStanceFrames: carryInitialStanceFrames, ranking: carryRanking, requestedDistanceM }), carryStylePreview);
  }
  /** The request-time planning path (exact mixed plan, then the bounded goal
   * region), shared verbatim with the remaining-goal replan. */
  function planCarryDestination({ requestId, stage, originalGoalWorld, initialObjectPositionWorld, goalWorld, library, checkPlan, maxSegments = 3 }) {
    const controlBudget = carryRequestControlBudget(requestId);
    const plannerOptions = { maxSegments, checkPlan, ranking: plannerRankingFor(carryRanking),
      budget: controlBudget ? { remainingControls: controlBudget.remainingControls, guard: carryBudgetGuardEnabled } : null };
    const initial = Array.from(initialObjectPositionWorld), goal = Array.from(goalWorld);
    const exactPlan = planMixedCarry(initial, goal, library, plannerOptions);
    let plan = exactPlan, region = null;
    if (restrictedCarryGoalRegionEnabled) {
      region = planCarryGoalRegion(initial, goal, library, { ...plannerOptions, exactFallbackPlan: exactPlan.supported ? exactPlan : null });
      // Leave most of the10 cm success tolerance for physical tracking
      // error. A tiny endpoint offset can remove unnecessary pickups
      // without changing the user's original click or source warp.
      const priorRegionSelection=region.supported&&region.nominalFinalGoalResidualM<=.01+1e-12?region:exactPlan;
      globalThis.__combinedRegionBeforeSelection?.({requestId,stage,episodeVersion,episodeControlStep,
        originalGoalWorld:Array.from(originalGoalWorld),conditioningGoal:Array.from(goal),
        initialObjectPositionWorld:Array.from(initial),exactPlan,region,
        priorSelectedPlan:priorRegionSelection});
      if (region.supported && region.nominalFinalGoalResidualM <=
          (debugControls&&urlParams.get('combinedPickupRegion')==='1' ? .03 : .01) + 1e-12) plan = region;
      globalThis.__combinedRegionAfterSelection?.({requestId,stage,episodeVersion,episodeControlStep,
        originalGoalWorld:Array.from(originalGoalWorld),selectedPlan:plan});
    }
    return { plan, exactPlan, region, controlBudget, plannerOptions };
  }
  // A planner search outcome (budget exhausted / nothing clear) is not the
  // user's reason. Record the dominant failing geometry reason instead.
  const PLANNER_SEARCH_REFUSALS = ['planning_budget_exhausted', 'no_clear_plan'];
  function truthfulRefusalReason(plan) {
    return PLANNER_SEARCH_REFUSALS.includes(plan?.reason) && plan.dominantReason ? plan.dominantReason : plan?.reason ?? null;
  }
  function buildCarryOptions(rawSkill, { requestId, longCarrySkill, useMatchedCarry = false }) {
    const [warpStartFrame, warpEndFrame] = rawSkill.carryInterval || [76, 293];
    return {
      warpStartFrame, warpEndFrame, snapObjectYaw: refObjYawSnapEnabled,
      ...(objectClassRoutingEnabled ? { yawSymmetry: objectRouter.yawSymmetry(rawSkill.objectBodyName) } : {}),
      carryDescentSagHold: carryDescentSagHoldEnabled,
      ...objectRouter.carryOutcomeOverrides(rawSkill.objectBodyName),   // OFF: {} (v5 key set); ON+suitcase: {outcomeRequirements} incl. tilt
      maxSegments: rawSkill === longCarrySkill ? 1 : maxCarryPickups,
      initialStanceFrames: useMatchedCarry || rawSkill === longCarrySkill ? 90 : carryInitialStanceFrames,
      requireSegmentExit: boxExitEnabled,
      quietSettling: quietEndingsEnabled ? { window: quietEndingConfig.window,
        minControls: quietEndingConfig.settlingMinControls,
        measure: () => quietEndingSample(rawSkill.objectBodyName),
        ...objectRouter.quietSettlingLimits(rawSkill.objectBodyName) } : null,   // OFF: {} => frozen QUIET_ENDING_LIMITS
      maxApproachSteps: boxApproachPlanner ? 720 : 360,
      checkEntryReference: carryReferenceClearance ? entry => {
        const check = checkCarrySegmentClearance({ ...entry, liveData: data,
          destinationGeometry: carryDestinationGeometry, pathChecker: carryReferenceClearance });
        lastCarryEntryClearance = { requestId, segmentIndex: entry.segmentIndex,
          episodeVersion, episodeControlStep, ...check };
        return check;
      } : null,
    };
  }
  function carryTaskPickupCount(carry) {
    const finished = results => (results ?? []).filter(segment => segment.completionReason === 'finished').length;
    return carryTaskLineage.reduce((sum, entry) => sum + finished(entry.segmentResults), 0) + finished(carry?.segmentResults);
  }
  /** Consume the completed sequence's remaining-goal request and, when a
   * supported plan is estimated to fit the controls left before the post-click
   * deadline, start a fresh sequence from the LIVE box toward the ORIGINAL
   * destination. The request id, goal object and control accounting are kept;
   * at most three pickups run per task across replans. */
  function tryReplanRemainingGoal(carry, completionStep) {
    const record = { episodeVersion, requestId: activeBoxTaskRequestId, physicalControl: episodeControlStep, controlStep,
      completionReason: completionStep.completionReason, replanIndex: carryTaskLineage.length, started: false, reason: null,
      remainingControls: null, estimatedControls: null, plannedCandidateIds: null, pickupsUsed: null, maxSegments: null, request: null };
    carryReplans.push(record); if (carryReplans.length > 16) carryReplans.shift();
    const refuse = reason => { record.reason = reason; return record; };
    try {
      if (matchedCarryHost?.inProgress || !taskDestinationWorld || activeBoxTask !== 'carry') return refuse('replan_unavailable');
      if (!carryLibraryContext || carryLibraryContext.requestId !== activeBoxTaskRequestId
          || carryLibraryContext.episodeVersion !== episodeVersion) return refuse('library_unavailable');
      const live = skillProprio(carry);
      const request = carry.remainingGoalRequest(live);
      record.request = { ...request };
      if (!request.needsPlan) return refuse(request.reason);
      const controlBudget = carryRequestControlBudget(activeBoxTaskRequestId);
      record.remainingControls = controlBudget?.remainingControls ?? null;
      if (!controlBudget) return refuse('request_clock_unavailable');
      const pickupsUsed = carryTaskPickupCount(carry), maxSegments = maxCarryPickups - pickupsUsed;
      record.pickupsUsed = pickupsUsed; record.maxSegments = maxSegments;
      if (maxSegments < 1) return refuse('pickup_budget_exhausted');
      const library = buildCarryLibrary(carryLibraryContext, request.remainingDistanceM);
      // Lane WS-D2: the replan starts from the measured post-exit root, where the
      // approach dry-run is most reliable; reachable first pickup poses rank first.
      const pickupSearch = pickupPoseSearchContext();
      const extendedLibrary = pickupSearch && alternateCarryEnabled && carryLibraryContext.librarySkills[3] && !carryStylePreview
        ? buildCarryLibrary(carryLibraryContext, request.remainingDistanceM, { alternate: true }) : null;
      const searched = planCarryWithPickupPoseSearch({ initialObjectPositionWorld: request.objectPositionWorld, goalWorld: taskDestinationWorld,
        library, extendedLibrary, tiers: pickupSearch ? pickupPoseSearchTiers : ['none'],
        makeCheckPlan: require => makeCarryPlanChecker(pickupSearch ? { ...pickupSearch, require } : null),
        planOnce: ({ library: passLibrary, checkPlan }) => planCarryDestination({ requestId: activeBoxTaskRequestId, stage: 'replan',
          originalGoalWorld: taskDestinationWorld, initialObjectPositionWorld: request.objectPositionWorld, goalWorld: taskDestinationWorld,
          library: passLibrary, checkPlan, maxSegments }) });
      const planning = searched.chosen.planned, plan = searched.plan;
      record.pickupPoseSearchTier = plan.pickupPoseSearchTier ?? null; record.pickupPoseReachable = plan.pickupPoseReachable ?? null;
      record.pickupPoseSearchTiers = searched.tierResults;
      record.plannerReason = plan.reason; record.dominantReason = plan.dominantReason ?? null; record.dominantObstacle = plan.dominantObstacle ?? null;
      record.checkedPlans = plan.checkedPlans ?? null; record.estimatedControls = plan.budget?.estimatedControls ?? null;
      record.plannedCandidateIds = plan.supported ? plan.segments.map(segment => segment.candidateId) : null;
      if (!plan.supported) return refuse(truthfulRefusalReason(plan));
      if (plan.budget && !plan.budget.fits) return refuse('not_enough_time_left');
      const controller = new MixedCarryGoalSequenceController(plan, buildCarryOptions(plan.segments[0].skill,
        { requestId: activeBoxTaskRequestId, longCarrySkill: carryLibraryContext.longCarrySkill }));
      controller.start(live);
      if (controller.phase !== 'approach') return refuse(controller.completionReason ?? 'replan_start_refused');
      // The completed sequence's records stay in the task lineage for accounting.
      carryTaskLineage.push({ replanIndex: carryTaskLineage.length, completionReason: completionStep.completionReason,
        plannedCandidateIds: carry.plan?.segments?.map(segment => segment.candidateId) ?? null,
        segmentResults: structuredClone(carry.segmentResults), segmentExitResults: structuredClone(carry.segmentExitResults),
        placementStatus: carry.placementStatus ? structuredClone(carry.placementStatus) : null });
      boxTaskResults.push({ requestId: activeBoxTaskRequestId, episodeVersion, episodeControlStep, task: activeBoxTask,
        goalWorld: Array.from(carry.requestedGoalWorld), completionReason: completionStep.completionReason, replanned: true,
        outcome: completionStep.outcome ?? null, segments: structuredClone(carry.segmentResults),
        segmentExits: structuredClone(carry.segmentExitResults) });
      // Same live-state handover as a fresh request, without a new request id.
      translator.reset(); translator.setClickPositionSource('stable_receding');
      teacherObs?.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, controller.skill);
      skillController = activeCarryController = controller;
      pendingCarryController = null; headingPreparations = []; pendingSegmentCarryController = null;
      stagedStudentApproach = null; lastStudentApproachEntry = null;
      stagedStudentTransport = null; lastStudentTransportEntry = null; transportTeacherResumePending = false;
      pendingWaypointCarryController = null; waypointApproaches = [];
      recordedApproachHold = null; executedApproachTerminal = null; teacherStandingPlan = null; teacherStandingSteps = 0;
      boxExitController?.reset(); preserveCompletionCommand = false;
      boxApproachPlanner?.reset(); lastApproachRoute = null;
      user.releaseKeys(); user.objGoalWorld = null; user.activeObjName = controller.skill.objectBodyName;
      user.deterministic = true; user.vaeNoise.fill(0); smoothingAlpha = 1;
      user.humanGoalWorld = controller.approachGoalWorld; picker.syncFromUserState();
      carryReferenceSelection = { requestId: activeBoxTaskRequestId, episodeVersion, episodeControlStep, stage: 'replan', replanIndex: record.replanIndex,
        carryStylePreview: carryStylePreview ? { ...carryStylePreview } : null, carryStyleSampling: null,
        supported: true, reason: null, selectedId: plan.segments.length === 1 ? plan.segments[0].candidateId : 'mixed',
        originalGoalWorld: Array.from(taskDestinationWorld), attempts: plan.attempts,
        supportedDistanceIntervalsM: plan.supportedDistanceIntervalsM, ranking: carryRanking,
        mixedSegments: plan.segments.map(segment => ({ candidateId: segment.candidateId, goalWorld: Array.from(segment.goalWorld),
          nominalTravelM: segment.nominalTravelM, plannedTravelM: segment.plannedTravelM, carryOptions: { ...segment.carryOptions } })),
        mixedScore: plan.score ?? null, checkedPlans: plan.checkedPlans ?? null, plannedFinalGoalWorld: plan.plannedFinalGoalWorld ?? null,
        plannedOriginalGoalResidualM: plan.nominalFinalGoalResidualM ?? 0, goalRegionSelection: plan.selection ?? null,
        dominantReason: plan.dominantReason ?? null, dominantObstacle: plan.dominantObstacle ?? null, refusalReason: null,
        budget: plan.budget ?? null, budgetRisk: plan.budgetRisk === true, remainingControls: controlBudget.remainingControls,
        replanFromObjectPositionWorld: Array.from(request.objectPositionWorld), remainingDistanceM: request.remainingDistanceM };
      // Keep the request open: the panel keeps measuring this same request.
      boxTaskRequestLog.transition(activeBoxTaskRequestId, 'started', 'replan_remaining_goal', boxRequestClock(), {
        completionReason: completionStep.completionReason, replanIndex: record.replanIndex, remainingDistanceM: request.remainingDistanceM,
        remainingControls: controlBudget.remainingControls, estimatedControls: record.estimatedControls,
        plannedCandidateIds: record.plannedCandidateIds, pickupsUsed, budgetRisk: plan.budgetRisk === true });
      record.started = true;
      setTaskStatus(`${capitalize(carriedObjectLabel())} set down ${Math.round(request.remainingDistanceM * 100)} cm short. Continuing the carry to your destination…`);
      return record;
    } catch (error) {
      console.warn('[replan remaining goal]', error);
      return refuse(`replan_error:${error.message}`);
    }
  }

  /** The reference files a box task of `kind` for `carryProfile` needs, with the
   * exact v16 cache keys and URLs (primary, optional long carry, carry library,
   * turn and step records). Used by startBoxTask, the startup prefetch and the
   * reach guide, so all three agree on what "loaded" means. */
  function carryReferencePlan(carryProfile, kind, { hasGoal = true, useMatchedCarry = false } = {}) {
    const primarySkillKey = useMatchedCarry ? 'matched_carry_prefix' : objectRouter.skillCacheKey(carryProfile, kind);
    const longCarryUrl = objectRouter.referenceUrl(carryProfile, 'carry_long');   // OFF: 'public/teacher_carry_long_reference.json'
    const primaryReferenceUrl = useMatchedCarry ? longCarryUrl : objectRouter.referenceUrl(carryProfile, kind);   // OFF: task.referenceUrl
    const longCarryKey = !useMatchedCarry && hasGoal && kind === 'carry' && restrictedLongCarryEnabled && longCarryUrl ? 'carry_long' : null;
    const libraryKeys = longCarryKey && restrictedCarryLibraryEnabled
      ? objectRouter.libraryKeys(carryProfile, ['short_0184', 'short_0295', 'medium_1224', ...(alternateCarryEnabled ? ['alternate'] : []), ...(longClipLibraryEnabled ? LONG_CLIP_KEYS : []), ...(midClipLibraryEnabled ? MID_CLIP_KEYS : [])]) : [];
    const turnKeys = hasGoal ? TURN_REFERENCES : [];
    const stepKeys = hasGoal ? (teacherApproachEnabled ? STEP_REFERENCES
      : teacherStandingMode === 'neutral' ? [STEP_REFERENCES[0]] : []) : [];
    const entries = [{ key: primarySkillKey, url: primaryReferenceUrl },
      ...(longCarryKey ? [{ key: longCarryKey, url: longCarryUrl }] : []),
      ...libraryKeys.map(key => ({ key, url: objectRouter.referenceUrl(carryProfile, key) })),   // OFF: `public/teacher_carry_${key}_reference.json`
      ...[...turnKeys, ...stepKeys].map(key => ({ key, url: `public/teacher_${key}_reference.json` }))];
    return { primarySkillKey, primaryReferenceUrl, longCarryUrl, longCarryKey, libraryKeys, turnKeys, stepKeys, entries };
  }
  function ensureSkillLoaded(key, url) {
    if (!loadedSkills.has(key)) loadedSkills.set(key,
      Promise.resolve().then(() => loadTeacherSkill(url)).catch(error => { loadedSkills.delete(key); throw error; }));
    return loadedSkills.get(key);
  }
  /** Distances (m from the object) the carry planner can serve for `bodyName`,
   * from the same library / single-reference rules startBoxTask applies, or
   * null until the references are loaded. Cached per loaded-set. */
  let reachIntervalsCache = null;
  function carryReachIntervals(bodyName) {
    if (!bodyName) return null;
    let profile;
    try { profile = objectRouter.profileFor(bodyName); } catch { return null; }
    if (objectRouter.taskRefusal(bodyName, 'carry')) return null;
    const references = carryReferencePlan(profile, 'carry', { hasGoal: true });
    const primary = loadedSkills.resolved.get(references.primarySkillKey);
    if (!primary) return null;
    const stamp = `${bodyName}|${loadedSkills.resolved.size}`;
    if (reachIntervalsCache?.stamp === stamp) return reachIntervalsCache.intervals;
    let intervals = null;
    try {
      if (references.longCarryKey) {
        const longCarrySkill = loadedSkills.resolved.get(references.longCarryKey);
        const librarySkills = references.libraryKeys.map(key => loadedSkills.resolved.get(key));
        if (restrictedCarryLibraryEnabled && longCarrySkill && librarySkills.length && librarySkills.every(Boolean)) {
          // requestedDistanceM null: no distance-dependent exclusion (carryRanking=excludeLong), so the
          // coverage is the union of what a request at any distance can plan.
          const library = buildCarryLibrary({ defaultSkill: primary, longCarrySkill, librarySkills, libraryKeys: references.libraryKeys }, null);
          intervals = mixedCarryDistanceCoverage(library, { maxSegments: maxCarryPickups }).supportedDistanceIntervalsM;
        }
      } else {
        // Single-reference object (suitcase): the same planCarrySegments defaults startBoxTask falls back to.
        intervals = planCarrySegments([0, 0, 0], [10, 0, 0], primary, { maxSegments: maxCarryPickups }).supportedDistanceIntervalsM;
      }
    } catch (error) { console.warn('[reach guide]', error); intervals = null; }
    reachIntervalsCache = { stamp, intervals: intervals ? Object.freeze(intervals.map(([lo, hi]) => Object.freeze([lo, hi]))) : null };
    return reachIntervalsCache.intervals;
  }
  /** Move a floor click that no carry can serve onto the nearest supported
   * distance along the same bearing (1 cm inside the band). Clicks inside a band,
   * on the object (pickup / already placed) or before the references are loaded
   * are returned unchanged. This changes only the requested destination, never
   * a planner rule or tolerance; the request log records the moved goal. */
  function snapCarryDestination(goal) {
    const unchanged = { goal, note: null };
    const bodyName = user.activeObjName;
    if (!bodyName || activeObjBodyId < 0 || !goal || goal.length !== 3) return unchanged;
    const intervals = carryReachIntervals(bodyName);
    if (!intervals?.length) return unchanged;
    const ox = data.xpos[activeObjBodyId * 3], oy = data.xpos[activeObjBodyId * 3 + 1];
    const dx = goal[0] - ox, dy = goal[1] - oy, distance = Math.hypot(dx, dy);
    if (!(distance > CARRY_PLACEMENT_TOLERANCE_M + 1e-12)) return unchanged;
    if (intervals.some(([lo, hi]) => distance >= lo - 1e-9 && distance <= hi + 1e-9)) return unchanged;
    let target = null, gap = Infinity;
    for (const [lo, hi] of intervals) {
      const inset = Math.min(0.01, (hi - lo) / 2);
      for (const [edge, inside] of [[lo, lo + inset], [hi, hi - inset]]) {
        const g = Math.abs(distance - edge);
        if (g < gap) { gap = g; target = inside; }
      }
    }
    if (target === null || !(target > CARRY_PLACEMENT_TOLERANCE_M)) return unchanged;
    const scale = target / distance;
    const label = objectDisplayName(bodyName);
    return { goal: [ox + dx * scale, oy + dy * scale, goal[2]],
      note: `Destination moved to ${target.toFixed(2)} m along your click: ${distance.toFixed(2)} m is ${distance < target ? 'too close for' : 'beyond'} the ${label} carry (${describeReachIntervals(intervals)})` };
  }
  /** Load every carry reference the release objects can need right after
   * startup, one file at a time, so the first click does not download and
   * parse ~40 MB of JSON. Same cache and keys as startBoxTask; failures only log. */
  function warmCarryReferences() {
    const jobs = [];
    for (const bodyName of interactiveSelectableNames) {
      let profile;
      try { profile = objectRouter.profileFor(bodyName); } catch { continue; }
      if (objectRouter.taskRefusal(bodyName, 'carry')) continue;
      for (const entry of carryReferencePlan(profile, 'carry', { hasGoal: true }).entries)
        if (entry.url && !jobs.some(job => job.key === entry.key)) jobs.push(entry);
    }
    void (async () => {
      const started = performance.now();
      for (const { key, url } of jobs) {
        try { await ensureSkillLoaded(key, url); }
        catch (error) { console.warn('[prefetch]', key, error.message); }
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      console.log(`[prefetch] ${jobs.length} carry references ready in ${((performance.now() - started) / 1000).toFixed(1)} s`);
    })();
  }
  async function startBoxTask(kind, requestedGoal = null, queuedRequestId = null) {
    const task = BOX_TASKS[kind];
    if (!task) throw new Error('Unknown box task');
    // Only a person's click (routed synchronously from onObjectGoal) gets the painted frame below.
    const interactiveRequest = queuedRequestId === null && lastGoalSource === 'click';
    lastGoalSource = null;
    const requestId = queuedRequestId ?? beginBoxTaskRequest(kind, requestedGoal);
    const styleRequestToken = carryStyleRequestSampler?.reserve({ episodeVersion, requestId, kind,
      originalGoalWorld: requestedGoal }) ?? null;
    latestBoxTaskRequestId = requestId;
    if (!matchedCarryHost?.inProgress) {
      lastCarryRequestClearance = null;
      lastCarryEntryClearance = null;
      carryReferenceSelection = null;
      finalCarryPlacement = null;
    }
    if (restrictedSuspended) {
      carryStyleRequestSampler?.discard(requestId, 'reset_required');
      boxTaskRequestLog.transition(requestId, 'refused', 'reset_required', boxRequestClock());
      setTaskStatus('Reset the scene before starting another task.'); return false;
    }
    const unsupportedObject = objectRouter.taskRefusal(user.activeObjName, kind);   // OFF: null (v5 had no gate here)
    if (unsupportedObject) {
      carryStyleRequestSampler?.discard(requestId, 'unsupported_object');
      boxTaskRequestLog.transition(requestId, 'refused', 'unsupported_object', boxRequestClock());
      setTaskStatus(unsupportedObject); return false;
    }
    const carryProfile = objectRouter.profileFor(user.activeObjName);   // OFF: the largebox profile (v5 literals)
    if (skillActive()) {
      if (!restrictedMode) {
        boxTaskRequestLog.transition(requestId, 'refused', 'task_busy', boxRequestClock());
        return false;
      }
      discardQueuedBoxTask('newer_box_request');
      queuedBoxTask = { kind, requestedGoal: requestedGoal ? Array.from(requestedGoal) : null, requestId };
      boxTaskRequestLog.transition(requestId, 'queued', 'finish_current_task', boxRequestClock());
      restrictedController.requestCancel();
      if (!matchedCarryCommand({kind:'box', taskKind:kind, requestId,
          originalGoalWorld:requestedGoal, objectBodyName:carryProfile.bodyName})) {
        pendingSegmentCarryController?.requestCancel(); skillController.requestCancel();
      }
      setTaskStatus('Finishing the current task before starting the new box request.');
      return true;
    }
    if (restrictedMode && !restrictedController.isSettled) {
      discardQueuedBoxTask('newer_box_request');
      queuedBoxTask = { kind, requestedGoal: requestedGoal ? Array.from(requestedGoal) : null, requestId };
      boxTaskRequestLog.transition(requestId, 'queued', 'finish_current_motion', boxRequestClock());
      restrictedController.requestCancel();
      setTaskStatus('Finishing this movement before starting the box task.');
      return true;
    }
    const version = episodeVersion;
    // Keep this measured composition an explicit experiment. All other goals
    // continue through the ordinary planner; the nearby 3.1 m trial missed.
    const useMatchedCarry = matchedCarryPreviewEnabled && kind === 'carry' && isFixedOriginal3mGoal(requestedGoal);
    const requestVersion = ++skillRequestVersion;
    boxTaskRequestLog.closeUnstarted('superseded', 'newer_box_request', boxRequestClock(), {
      exceptRequestId: requestId, statuses: ['loading'],
    });
    boxTaskRequestLog.transition(requestId, 'loading', 'reference_load', boxRequestClock());
    skillLoadsInFlight++; skillLoading = true; updateTaskControls();
    setTaskStatus('Loading the box task…');
    try {
      loadTeacherActor();
      // Same keys/URLs as v16 (see carryReferencePlan); the startup prefetch fills the same cache.
      const references = carryReferencePlan(carryProfile, kind, { hasGoal: Boolean(requestedGoal), useMatchedCarry });
      const { primarySkillKey, longCarryKey, libraryKeys, turnKeys, stepKeys } = references;
      for (const { key, url } of references.entries) ensureSkillLoaded(key, url);
      const [, defaultSkill, longCarrySkill, librarySkills, ...loadedLocomotionSkills] = await Promise.all([skillLoadPromise, loadedSkills.get(primarySkillKey),
        longCarryKey ? loadedSkills.get(longCarryKey) : null,
        Promise.all(libraryKeys.map(key => loadedSkills.get(key))),
        ...turnKeys.map(key => loadedSkills.get(key)), ...stepKeys.map(key => loadedSkills.get(key))]);
      let rawSkill = defaultSkill, mixedPlan = null, carryStyleSamplingSelection = null;
      const loadedTurnSkills = loadedLocomotionSkills.slice(0, turnKeys.length);
      if (requestedGoal && interactiveRequest && !paused) {
        // The geometric planner below is synchronous and can block for 0.1–10 s on a
        // far click. Let the browser paint the marker and this line first; the
        // loop may run one more standing control in between (the version guards
        // below still apply). Automation requests and a paused loop never wait.
        setTaskStatus('Planning the carry…');
        await new Promise(resolve => { let done = false; const finish = () => { if (!done) { done = true; resolve(); } };
          requestAnimationFrame(finish); setTimeout(finish, 60); });
      }
      if (activeStepPromise) await activeStepPromise;
      if (version !== episodeVersion || requestVersion !== skillRequestVersion || restrictedSuspended) {
        carryStyleRequestSampler?.discard(requestId, version !== episodeVersion ? 'episode_reset'
          : restrictedSuspended ? 'reset_required' : 'newer_request_or_cancel');
        boxTaskRequestLog.transition(requestId, version !== episodeVersion ? 'reset'
          : restrictedSuspended ? 'refused' : 'superseded', version !== episodeVersion ? 'episode_reset'
          : restrictedSuspended ? 'reset_required' : 'newer_request_or_cancel', boxRequestClock());
        return false;
      }
      if (objectClassRoutingEnabled && !skillMatchesProfile(defaultSkill, carryProfile))
        throw new Error(`Reference object ${defaultSkill.objectBodyName} does not match the selected ${carryProfile.bodyName}`);
      taskCoverageCapture?.capture({stage:'planning_input', requestId, task:kind,
        originalGoalWorld:requestedGoal, clock:boxRequestClock()});
      const matchedRequest = useMatchedCarry ? makeOriginal3mRequest({episode:episodeVersion,
        requestId, physicalControl:episodeControlStep, originalGoalWorld:requestedGoal}) : null;
      const conditioningGoal = matchedRequest?.firstIntermediateGoalWorld ?? requestedGoal;
      if (conditioningGoal) {
        const id = findBodyIdByName(model, defaultSkill.objectBodyName);
        if (id < 0) throw new Error('The carry box is missing from the scene');
        if ((restrictedCarryLibraryEnabled || useMatchedCarry) && Math.hypot(data.xpos[id * 3] - requestedGoal[0],
            data.xpos[id * 3 + 1] - requestedGoal[1]) <= CARRY_PLACEMENT_TOLERANCE_M + 1e-12) {
          const placement = readIdleBoxPlacement({ mujoco, model, data, objectId: id, pelvisId,
            requestedGoalWorld: requestedGoal, idle: !skillActive() && restrictedController.isSettled });
          boxTaskRequestLog.transition(requestId, 'loading', 'placement_check', boxRequestClock(), { placement });
          if (placement.canSkipCarry) {
            carryStyleRequestSampler?.discard(requestId, 'already_placed');
            finalCarryPlacement = { requestId, checkedAtIdleRequest: true, goalReached: true,
              originalGoalWorld: Array.from(requestedGoal), measuredObjectPositionWorld: placement.objectPositionWorld,
              remainingDistanceM: placement.remainingDistanceM, toleranceM: CARRY_PLACEMENT_TOLERANCE_M };
            boxTaskRequestLog.transition(requestId, 'outcome', 'already_placed', boxRequestClock(), finalCarryPlacement);
            setTaskStatus('Already within 10 cm of your destination. Choose your next movement.');
            return true;
          }
        }
        let selected = null;
        if (longCarrySkill) {
          if (longCarrySkill.objectBodyName !== defaultSkill.objectBodyName) throw new Error('Carry references require the same selected box');
          if (restrictedCarryLibraryEnabled) {
            carryLibraryContext = { requestId, episodeVersion, defaultSkill, longCarrySkill, librarySkills, libraryKeys };
            const regionInitial = Array.from(data.xpos.slice(id * 3, id * 3 + 3));
            const requestedDistanceM = Math.hypot(conditioningGoal[0] - regionInitial[0], conditioningGoal[1] - regionInitial[1]);
            const regionLibrary = buildCarryLibrary(carryLibraryContext, requestedDistanceM);
            // Lane WS-D2: reachable first pickup poses rank first. The unchanged
            // library is searched before the extended one (alternate one-metre
            // carry); the last pass is the unchanged ranking, whose plan is marked
            // pickupPoseReachable:false rather than refused.
            const pickupSearch = pickupPoseSearchContext();
            const extendedLibrary = pickupSearch && alternateCarryEnabled && librarySkills[3] && !carryStylePreview
              ? buildCarryLibrary(carryLibraryContext, requestedDistanceM, { alternate: true }) : null;
            const planningStartedMs = performance.now();
            const searched = planCarryWithPickupPoseSearch({ initialObjectPositionWorld: regionInitial, goalWorld: conditioningGoal,
              library: regionLibrary, extendedLibrary, tiers: pickupSearch ? pickupPoseSearchTiers : ['none'],
              makeCheckPlan: require => makeCarryPlanChecker(pickupSearch ? { ...pickupSearch, require } : null),
              planOnce: ({ library, checkPlan }) => planCarryDestination({ requestId, stage: 'request', originalGoalWorld: requestedGoal,
                initialObjectPositionWorld: regionInitial, goalWorld: conditioningGoal, library, checkPlan, maxSegments: maxCarryPickups }) });
            const { chosen, tierResults } = searched, planning = chosen.planned, checkPlan = chosen.checkPlan;
            mixedPlan = searched.plan;
            lastCarryLibrary = { requestId, library: regionLibrary, checkPlan };
            if (styleRequestToken) {
              const proposal = planCarryStyleChoices({ requestToken: styleRequestToken,
                initialObjectPositionWorld: regionInitial, originalGoalWorld: conditioningGoal,
                library: regionLibrary, baselinePlan: mixedPlan, checkPlan,
                goalRegionEnabled: restrictedCarryGoalRegionEnabled });
              const sampled = carryStyleRequestSampler.select(styleRequestToken, proposal);
              mixedPlan = sampled.plan;
              carryStyleSamplingSelection = sampled.diagnostic;
            }
            // The style sampler may replace the plan; re-check the final selection.
            const selectedClearance = mixedPlan === searched.plan ? searched.clearance : mixedPlan.supported ? checkPlan(mixedPlan) : null;
            pickupPoseSearchReviews.push({ requestId, episodeVersion, episodeControlStep, enabled: Boolean(pickupSearch),
              alternateCarryInLibrary: Boolean(extendedLibrary), selectedFromExtendedLibrary: chosen.extendedLibrary ?? false,
              tiers: tierResults, selectedTier: chosen.require, supported: mixedPlan.supported, reason: mixedPlan.reason,
              selectedCandidateIds: mixedPlan.segments?.map(segment => segment.candidateId) ?? [],
              pickupPoseReachable: mixedPlan.pickupPoseReachable ?? null,
              reachability: selectedClearance?.pickupPoseReachability ?? null,
              reachabilityEvaluations: pickupSearch?.cache.size ?? 0, planningMs: performance.now() - planningStartedMs });
            if (pickupPoseSearchReviews.length > 16) pickupPoseSearchReviews.shift();
            const first = mixedPlan.segments[0];
            selected = { supported: mixedPlan.supported, reason: mixedPlan.reason,
              id: mixedPlan.segments.length === 1 ? first.candidateId : 'mixed',
              skill: first?.skill ?? defaultSkill, plan: mixedPlan, attempts: mixedPlan.attempts,
              supportedDistanceIntervalsM: mixedPlan.supportedDistanceIntervalsM,
              clearance: selectedClearance,
              dominantReason: mixedPlan.dominantReason ?? null, dominantObstacle: mixedPlan.dominantObstacle ?? null,
              budget: mixedPlan.budget ?? null, budgetRisk: mixedPlan.budgetRisk === true,
              remainingControls: planning.controlBudget?.remainingControls ?? null };
          } else selected = chooseCarryReference(Array.from(data.xpos.slice(id * 3, id * 3 + 3)), conditioningGoal,
            [{ id: 'staged', skill: defaultSkill }, { id: 'long', skill: longCarrySkill, maxSegments: 1 }], {
              checkReference: carryReferenceClearance ? candidate => checkCarryRequestClearance({ skill: candidate.skill,
                objectPositionWorld: Array.from(data.xpos.slice(id * 3, id * 3 + 3)), requestedGoalWorld: conditioningGoal,
                liveData: data, destinationGeometry: carryDestinationGeometry, pathChecker: carryReferenceClearance }) : null,
            });
          carryReferenceSelection = { requestId, episodeVersion, episodeControlStep,
            carryStylePreview: carryStylePreview ? { ...carryStylePreview } : null,
            carryStyleSampling: carryStyleSamplingSelection,
            supported: selected.supported, reason: selected.reason, selectedId: selected.id,
            originalGoalWorld: Array.from(requestedGoal), attempts: selected.attempts,
            supportedDistanceIntervalsM: selected.supportedDistanceIntervalsM,
            mixedSegments: mixedPlan?.segments.map(segment => ({ candidateId: segment.candidateId,
              goalWorld: Array.from(segment.goalWorld), nominalTravelM: segment.nominalTravelM,
              plannedTravelM: segment.plannedTravelM, carryOptions: { ...segment.carryOptions } })) ?? null,
            mixedScore: mixedPlan?.score ?? null, checkedPlans: mixedPlan?.checkedPlans ?? null,
            plannedFinalGoalWorld: mixedPlan?.plannedFinalGoalWorld ?? null,
            plannedOriginalGoalResidualM: mixedPlan?.nominalFinalGoalResidualM ?? 0,
            goalRegionSelection: mixedPlan?.selection ?? null,
            stage: 'request', ranking: carryRanking, dominantReason: selected.dominantReason ?? null, dominantObstacle: selected.dominantObstacle ?? null,
            refusalReason: selected.supported ? null : truthfulRefusalReason(selected.plan ?? selected),
            budget: selected.budget ?? null, budgetRisk: selected.budgetRisk === true, remainingControls: selected.remainingControls ?? null,
            pickupPoseReachable: mixedPlan?.pickupPoseReachable ?? null,
            pickupPoseSearchTier: mixedPlan?.pickupPoseSearchTier ?? null,
            pickupPoseSearch: pickupPoseSearchReviews.at(-1)?.requestId === requestId ? structuredClone(pickupPoseSearchReviews.at(-1)) : null };
          if (selected.supported) rawSkill = selected.skill;
        }
        const plan = selected?.plan ?? planCarrySegments(Array.from(data.xpos.slice(id * 3, id * 3 + 3)), conditioningGoal, rawSkill, { maxSegments: maxCarryPickups });
        if (!plan.supported) {
          const intervals = selected?.supportedDistanceIntervalsM ?? plan.supportedDistanceIntervalsM;
          const refusalReason = truthfulRefusalReason(plan);
          lastCarryRefusal = { requestId, episodeVersion, episodeControlStep, stage: 'request', reason: refusalReason,
            plannerReason: plan.reason, dominantReason: plan.dominantReason ?? null, dominantObstacle: plan.dominantObstacle ?? null,
            checkedPlans: plan.checkedPlans ?? null, distanceM: plan.distanceM,
            remainingControls: carryRequestControlBudget(requestId)?.remainingControls ?? null };
          boxTaskRequestLog.transition(requestId, 'refused', refusalReason, boxRequestClock(), {
            distanceM: plan.distanceM, supportedDistanceIntervalsM: intervals, plannerReason: plan.reason,
            dominantReason: plan.dominantReason ?? null, dominantObstacle: plan.dominantObstacle ?? null,
            checkedPlans: plan.checkedPlans ?? null,
          });
          setTaskStatus(carryPlanRefusalStatus(plan, objectDisplayName(carryProfile.bodyName)));
          return false;
        }
        if (carryReferenceClearance) {
          lastCarryRequestClearance = { requestId, ...(selected?.clearance ?? checkCarryRequestClearance({ skill: rawSkill,
            objectPositionWorld: Array.from(data.xpos.slice(id * 3, id * 3 + 3)), requestedGoalWorld: conditioningGoal,
            liveData: data, destinationGeometry: carryDestinationGeometry, pathChecker: carryReferenceClearance })) };
          if (!lastCarryRequestClearance.supported) {
            const failed = lastCarryRequestClearance.checks.at(-1);
            boxTaskRequestLog.transition(requestId, 'refused', lastCarryRequestClearance.reason, boxRequestClock(), {
              segmentIndex: lastCarryRequestClearance.segmentIndex,
              obstacles: failed?.destination?.collisions?.map(contact => contact.obstacleName) ?? [],
              pathViolation: failed?.path?.firstViolation ?? null,
            });
            setTaskStatus(lastCarryRequestClearance.reason === 'occupied_carry_destination'
              ? 'There is not enough room for the box at that destination. Choose a clearer spot.'
              : 'The carry would pass too close to another box. Choose a different direction.');
            return false;
          }
        }
      }
      const skill = conditioningGoal ? rawSkill : withInitialStance(rawSkill, 30);
      if (restrictedMode) {
        restrictedController.reset(); lastRestrictedStep = null; restrictedPlan = null;
        restrictedAfterBox = true;
      }
      teacherStandingPlan = null; teacherStandingSteps = 0;
      restoreSkillCommandStyle();
      translator.reset();
      teacherObs?.dispose();
      teacherObs = new TeacherObsBuilder(mujoco, model, skill);
      // The student often strafes without turning. Aligning the clip to
      // its current heading avoids asking the teacher to begin sideways.
      const carryOptions = buildCarryOptions(rawSkill, { requestId, longCarrySkill, useMatchedCarry });
      skillController = conditioningGoal ? mixedPlan
        ? new MixedCarryGoalSequenceController(mixedPlan, carryOptions)
        : new CarryGoalSequenceController(skill, conditioningGoal, carryOptions)
        : new TeacherSkillController(skill, {
        alignmentMode: 'preserve-heading', arrivalRadius: 0.25, maxReferenceStartDistance: 0.40, maxApproachSteps: 480,
        outcomeRequirements: { minLiftM: 0.35, maxFinalObjectHeightM: 0.25,
          minRootHeightM: 0.45, minUpright: 0.5 },
      });
      activeCarryController = conditioningGoal ? skillController : null;
      pendingCarryController = null; headingPreparations = [];
      pendingSegmentCarryController = null;
      stagedStudentApproach = null; lastStudentApproachEntry = null;
      stagedStudentTransport = null; lastStudentTransportEntry = null; transportTeacherResumePending = false;
      pendingWaypointCarryController = null; waypointApproaches = [];
      recordedApproachHold = null;
      executedApproachTerminal = null;
      boxExitController?.reset();
      preserveCompletionCommand = false;
      boxApproachPlanner?.reset(); lastApproachRoute = null;
      if (loadedTurnSkills.length) turnSkills = loadedTurnSkills;
      if (stepKeys.length) stepSkills = loadedLocomotionSkills.slice(turnKeys.length);
      activeBoxTask = kind;
      activeBoxTaskRequestId = requestId;
      carryTaskLineage.length = 0;
      user.releaseKeys(); user.objGoalWorld = null;
      user.activeObjName = skillController.skill.objectBodyName;
      user.deterministic = true; user.vaeNoise.fill(0); smoothingAlpha = 1;
      previousSkillClickSource = translator.clickPositionSource;
      // A nearby pregrasp target should not inherit a multi-metre matched
      // clip endpoint. Use the existing bounded 0.5 m receding command.
      translator.setClickPositionSource('stable_receding');
      skillController.start(skillProprio());
      user.humanGoalWorld = skillController.approachGoalWorld;
      picker.syncFromUserState();
      taskDestinationWorld = requestedGoal ? Array.from(requestedGoal) : null;
      if (matchedRequest) {
        matchedCarryHost.attach({request:matchedRequest, rawSkill, prefixParent:activeCarryController});
        carryReferenceSelection = {requestId, episodeVersion, episodeControlStep, supported:true,
          selectedId:'matched_carry_experiment', originalGoalWorld:Array.from(requestedGoal),
          intermediateConditioningGoalWorld:Array.from(conditioningGoal),
          sourceFrames:activeCarryController.skill.sourceFrames};
        updateMatchedCarryMarker();
      }
      setTaskStatus(`Walking to the ${objectDisplayName(skillController.skill?.objectBodyName)}…`);
      boxTaskRequestLog.transition(requestId, 'started', 'approach_started', boxRequestClock());
      taskCoverageCapture?.capture({stage:'execution_start', requestId, task:kind,
        originalGoalWorld:requestedGoal, clock:boxRequestClock()});
      return true;
    } catch (error) {
      carryStyleRequestSampler?.discard(requestId, 'task_load_error');
      boxTaskRequestLog.transition(requestId, 'refused', 'task_load_error', boxRequestClock(), { message: error.message });
      if (requestVersion === skillRequestVersion) {
        restoreSkillCommandStyle();
        setTaskStatus(`Box task unavailable: ${error.message}`);
      }
      console.error(error);
      return false;
    } finally { skillLoadsInFlight--; skillLoading = skillLoadsInFlight > 0; updateTaskControls(); }
  }
  /** WS-G: after a near-miss final setdown and a complete exit, queue ONE
   * corrective single-segment carry from the live box to the ORIGINAL
   * destination through the ordinary request path (new request id, same
   * goal). The 10 cm tolerance never changes; when ineligible the task keeps
   * reporting placement_missed exactly as before. Queued, not started, because
   * startBoxTask awaits the running control step. */
  function maybeQueuePlacementCorrection({ measuredObject, placementDistanceM }) {
    if (!placementCorrectionEnabled || !taskDestinationWorld) return null;
    const parentRequestId = activeBoxTaskRequestId;
    const parent = boxTaskRequestLog.records.get(parentRequestId);
    const decision = { parentRequestId, episodeVersion, episodeControlStep, remainingDistanceM: placementDistanceM, queued: false };
    const decline = reason => { decision.reason = reason; placementCorrections.push(decision); return decision; };
    if (activeCarryController?.completionReason !== 'placement_missed') return decline('not_placement_missed');
    if (!parent || parent.episodeVersion !== episodeVersion) return decline('unknown_parent_request');
    if (parent.semanticGoalType === 'placement_correction'
        || placementCorrections.some(c => c.queued && c.parentRequestId === parentRequestId)) return decline('correction_already_used');
    if (queuedBoxTask || queuedRestrictedIntent()) return decline('user_command_queued');
    const received = parent.events[0]?.episodeControlStep;
    if (!Number.isInteger(received)) return decline('unknown_request_clock');
    let sample;
    try { sample = quietEndingSample(activeCarryController.skill.objectBodyName); }
    catch (error) { return decline('measurement_unavailable:' + error.message); }
    const remainingBudgetControls = placementCorrectionConfig.postClickBudgetControls - (episodeControlStep - received);
    const eligibility = evaluatePlacementCorrection({ remainingDistanceM: placementDistanceM,
      setDown: measuredObject[2] <= .25, floorSupported: sample.floorSupported, remainingBudgetControls,
      minBudgetControls: placementCorrectionConfig.minBudgetControls, maxErrorM: placementCorrectionConfig.maxErrorM });
    Object.assign(decision, { remainingBudgetControls, eligibility, sample });
    if (!eligibility.eligible) return decline(eligibility.reason);
    if (!lastCarryLibrary || lastCarryLibrary.requestId !== parentRequestId) return decline('library_unavailable');
    let plan;
    try { plan = planMixedCarry(Array.from(measuredObject), taskDestinationWorld, lastCarryLibrary.library,
      { maxSegments: 1, checkPlan: lastCarryLibrary.checkPlan }); }
    catch (error) { return decline('planning_error:' + error.message); }
    decision.plan = { supported: plan.supported, reason: plan.reason, distanceM: plan.distanceM,
      candidateIds: plan.segments.map(segment => segment.candidateId) };
    if (!plan.supported) return decline(plan.reason ?? 'no_clear_plan');
    if (plan.segments.length !== 1 || !PLACEMENT_CORRECTION_DEFAULTS.candidateIds.includes(plan.segments[0].candidateId))
      return decline('unexpected_correction_plan');
    const requestId = beginBoxTaskRequest('carry', Array.from(taskDestinationWorld), 'placement_correction');
    queuedBoxTask = { kind: 'carry', requestedGoal: Array.from(taskDestinationWorld), requestId };
    boxTaskRequestLog.transition(requestId, 'queued', 'placement_correction', boxRequestClock(),
      { parentRequestId, remainingDistanceM: placementDistanceM, remainingBudgetControls, candidateId: plan.segments[0].candidateId });
    decision.queued = true; decision.requestId = requestId; decision.reason = 'queued';
    placementCorrections.push(decision);
    return decision;
  }
  const startPickup = () => startBoxTask('pickup');
  const startCarry = () => startBoxTask('carry');
  const startCarryToGoal = value => {
    if (!value || value.length !== 3 || !Array.from(value).every(Number.isFinite)) {
      const requestId = beginBoxTaskRequest('carry', null, 'carry_destination');
      boxTaskRequestLog.transition(requestId, 'refused', 'invalid_goal', boxRequestClock());
      throw new Error('Carry destination must contain finite XYZ');
    }
    if (skillActive() && !restrictedMode) {
      const requestId = beginBoxTaskRequest('carry', value);
      boxTaskRequestLog.transition(requestId, 'refused', 'task_busy', boxRequestClock());
      return Promise.resolve(false);
    }
    if (!matchedCarryHost?.ownsDestination) taskDestinationWorld = Array.from(value);
    return startBoxTask('carry', Array.from(value));
  };

  // --- B9 unified interface: deterministic skill arbiter ------------------ //
  // Pure decision (skill_arbiter.js) + mechanical routing to the EXISTING
  // lanes. No physics, controller or tolerance changes live here.
  const SKILL_DECISION_RING = 32;
  const yawFromWxyz = q => Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
  function pushLiveEntryAvailable() {
    // Same runtime gate as requestNoResetApproachPose `available()` + the busy facts
    // startGroundPushToGoal refuses on. The push lane is a closure-level function (no debug-API dependency).
    // Integration v2: also require the controller's settled standing boundary. In rejected v1/job 35298988_1,
    // the exact D x2376 trace ended in teacher_settling; the Suitcase scene perturbation moved its planar cone
    // error barely inside 45 degrees, Push was admitted, and it lost balance (root .244 m, upright -.056).
    // Ordinary forward Push/job 35298988_0 entered from teacher_standing and passed. This binds admission to
    // the two physically observed entry classes without changing the certified arbiter or any controller.
    return Boolean(restrictedController && restrictedObs && approveRestrictedReference && restrictedWalkSkills.length
      && restrictedTurnSkills.length && contactDiagnostics)
      && lastControlPhase === 'teacher_standing'
      && !noResetApproachOwner && !pendingNormalGroundPush && !activeLoadedRetargetPromise && !pendingLoadedRetarget;
  }
  function readSkillArbiterRequest(goal, requestId) {
    const objectBodyName = user.activeObjName;
    const objectId = activeObjBodyId >= 0 ? activeObjBodyId : (objectBodyName ? findBodyIdByName(model, objectBodyName) : -1);
    const objectPosWorld = objectId >= 0 ? Array.from(data.xpos.slice(objectId * 3, objectId * 3 + 3)) : null;
    const objectYawRad = objectId >= 0 ? yawFromWxyz(data.xquat.slice(objectId * 4, objectId * 4 + 4)) : null;
    const rootPosWorld = Array.from(data.xpos.slice(pelvisId * 3, pelvisId * 3 + 3));
    const rootYawRad = yawFromWxyz(data.xquat.slice(pelvisId * 4, pelvisId * 4 + 4));
    let approach = { clearLine: false, liveEntryAvailable: pushLiveEntryAvailable(), distanceM: null,
      directBlocked: null, clearanceBlocked: null, reason: 'collision_bounds_unavailable', alignedTarget: null };
    if (boxCollisionBounds && objectPosWorld) {
      try {
        // The push lane's own approach planner and clearance (largebox_push_live_entry_runtime.js
        // prepareOutcomeBasedNoResetEntry -> planBoxApproach(..., {clearance:.55})) applied to the
        // robot -> selected-object line, with the selected object itself excluded (it is the target).
        const bounds = boxCollisionBounds.read(data);
        const others = bounds.filter(bound => bound.bodyId !== objectId);
        const diagnostic = planBoxApproach([rootPosWorld[0], rootPosWorld[1]], [objectPosWorld[0], objectPosWorld[1]], others,
          { clearance: SKILL_ARBITER_PUSH_LANE.approachClearanceM, respectTransitClearance: false });
        approach = { ...approach, clearLine: diagnostic.supported && !diagnostic.directBlocked && !diagnostic.clearanceBlocked,
          distanceM: diagnostic.pathLengthM, directBlocked: diagnostic.directBlocked, clearanceBlocked: diagnostic.clearanceBlocked,
          reason: diagnostic.reason ?? null };
        // Phase C v4: preflight the exact frame-0 stance that the push lane will request after loading its source.
        // Unlike the legacy centre proxy above, this uses every compiled object bound, including the selected box.
        // Off-cone widening is admitted only for a supported, physically direct-clear route to this exact target.
        const targetWorld = alignedPushApproachTarget(objectPosWorld, goal);
        if (targetWorld) {
          const aligned = planBoxApproach([rootPosWorld[0], rootPosWorld[1]], targetWorld, bounds,
            { clearance: SKILL_ARBITER_PUSH_LANE.approachClearanceM, respectTransitClearance: false });
          approach.alignedTarget = { directClear: aligned.supported && !aligned.directBlocked,
            supported: aligned.supported, directBlocked: aligned.directBlocked, clearanceBlocked: aligned.clearanceBlocked,
            pathLengthM: aligned.pathLengthM, targetWorld: Array.from(targetWorld), reason: aligned.reason ?? null };
        }
      } catch (error) { approach = { ...approach, clearLine: false, reason: error.message }; }
    }
    // B2: any task in flight (running, loading, queued or a push awaiting its handoff) is reported as activeTask so the
    // arbiter routes the click to carry = v5's synchronous queued-carry path (startBoxTask queues while busy).
    const taskInFlight = skillActive() || skillLoading || pendingNormalGroundPush !== null || queuedBoxTask !== null;
    const activeTask = taskInFlight ? { kind: activeBoxTask ?? queuedBoxTask?.kind ?? (skillLoading ? 'loading' : 'push_pending'),
      sequencePhase: skillController?.phase ?? null, segmentIndex: activeCarryController?.segmentIndex ?? null,
      objectBodyName: skillController?.skill?.objectBodyName ?? activeCarryController?.skill?.objectBodyName ?? null } : null;
    // The standalone arbiter remains fail-closed for every non-Largebox body.
    // In this combined release only the certified v6j router may grant the
    // Suitcase its single qualified lane. It never grants pickup or Push, and
    // Plasticbox/smallbox retain the arbiter's empty capability table.
    const routedCapability = objectClassRoutingEnabled
      && objectBodyName === 'active_suitcase_080_080_080'
      && objectRouter.taskRefusal(objectBodyName, 'carry') === null ? ['carry'] : null;
    return { objectBodyName, ...(routedCapability ? { capability: routedCapability } : {}),
      objectPosWorld, objectYawRad, goalWorld: Array.from(goal), rootPosWorld, rootYawRad, approach, requestId, activeTask };
  }
  function routeFloorGoalThroughSkillArbiter(goal) {
    const decisionId = ++skillDecisionSerial;
    const readStartedMs = performance.now();
    const request = readSkillArbiterRequest(goal, decisionId);   // state read + planBoxApproach proxy (not the arbiter)
    const requestMs = performance.now() - readStartedMs;
    const { decision, decisionMs } = measureChooseSkill(request); // pure arbiter call only
    const record = { decisionId, skill: decision.skill, reason: decision.reason, pushBlockers: Array.from(decision.pushBlockers),
      features: structuredClone(decision.features), seed: decision.features.seed, decisionMs, requestMs, version: decision.version,
      control: episodeControlStep, episodeVersion, goalWorld: Array.from(goal), objectBodyName: request.objectBodyName,
      requestId: null, routed: null, routeResult: null };
    selectedSkill = record; skillDecisions.push(record); showSkillDecision(decision);
    if (skillDecisions.length > SKILL_DECISION_RING) skillDecisions.shift();
    console.log(`[arbiter] #${decisionId} ${decision.skill} (${decision.reason}) in ${decisionMs.toFixed(3)} ms`);
    if (decision.skill === 'refuse') {
      taskStatusSkillLabel = null; record.routed = 'refused';
      setTaskStatus(decision.message ?? 'Carry controls currently use the large box.');
      return Promise.resolve(false);
    }
    taskStatusSkillLabel = decision.skill;
    const serialBefore = boxTaskRequestSerial;
    let outcome;
    if (decision.skill === 'push') {
      record.routed = 'startGroundPushToGoal';
      const admission = startGroundPushToGoal(Array.from(goal));
      pendingPushAdmission = admission.then(() => null, () => null).finally(() => { if (pendingPushAdmission === barrier) pendingPushAdmission = null; });
      const barrier = pendingPushAdmission;
      outcome = admission.then(result => {
        record.requestId = result?.requestId ?? record.requestId;
        record.routeResult = { supported: result?.supported ?? null, reason: result?.reason ?? null, phase: result?.phase ?? null };
        const refusal = describePushRefusal(result);
        if (!refusal.refused) return true;
        // R2 (v14a): the lane refused this click — say so where the user looks, then (policy) serve the destination with carry.
        record.refusal = refusal;
        setTaskStatus(refusal.statusText);
        if (!PUSH_REFUSAL_FALLBACK_TO_CARRY) return false;
        taskStatusSkillLabel = 'carry'; record.fallback = 'startCarryToGoal';
        const serialBeforeFallback = boxTaskRequestSerial;
        return Promise.resolve(startCarryToGoal(Array.from(goal))).then(carried => {
          record.fallbackRequestId = boxTaskRequestSerial > serialBeforeFallback ? latestBoxTaskRequestId : null;
          record.fallbackResult = { supported: carried === true };
          return carried === true;
        });
      });
    } else if (decision.skill === 'pickup') {
      record.routed = "startBoxTask('pickup')"; outcome = startPickup();
    } else {
      record.routed = 'startCarryToGoal'; outcome = startCarryToGoal(goal);
    }
    if (boxTaskRequestSerial > serialBefore) record.requestId = latestBoxTaskRequestId;
    return Promise.resolve(outcome).then(result => {
      if (record.routeResult === null) record.routeResult = { supported: result === true };
      return result;
    }, error => {
      record.routeResult = { supported: false, reason: 'route_error', message: error.message };
      console.error('[arbiter] route failed', error); return false;
    });
  }
  const startCarryExample = (distanceM = 2.3, headingRad = 0) => {
    if (!restrictedLongCarryEnabled || longCarryExampleButton?.disabled) return;
    try {
      const name = objectRouter.carryBodyName(user.activeObjName);   // OFF: 'active_largebox_080_080_080'
      const selection = createObjectSelection(name, findBodyIdByName(model, name), pointCloudDb);
      const position = data.xpos.slice(selection.bodyId * 3, selection.bodyId * 3 + 3);
      const quaternion = data.xquat.slice(selection.bodyId * 4, selection.bodyId * 4 + 4);
      const goal = objectGoalOnFloor([position[0] + distanceM * Math.cos(headingRad),
        position[1] + distanceM * Math.sin(headingRad), 0], selection.pointsLocal,
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]);
      return startCarryToGoal(goal);
    } catch (error) {
      setTaskStatus(`Box task unavailable: ${error.message}`);
      return Promise.resolve(false);
    }
  };
  const startLongCarryExample = () => startCarryExample(2.3);
  // Explicit selection change: the destination picker no longer toggles
  // selection on box clicks. Same selection effect as Esc (keyboard.js) via
  // the picker's out-of-band sync, without cancelling a running box task.
  deselectButton?.addEventListener('click', () => {
    user.activeObjName = null;
    user.humanGoalWorld = user.objGoalWorld = null;
    picker.syncFromUserState();
    taskStatusSkillLabel = null;
    showSkillDecision(null, 'selection cleared.');
    setTaskStatus('Selection cleared. Click the large box or the suitcase to select it.');
  });
  pickupButton?.addEventListener('click', startPickup);
  carryButton?.addEventListener('click', startCarry);
  longCarryExampleButton?.addEventListener('click', startLongCarryExample);
  for (const { button, distanceM, headingRad } of carryLibraryExamples) button?.addEventListener('click', () => {
    if (restrictedCarryLibraryEnabled && !button.disabled) return startCarryExample(distanceM, headingRad);
  });
  resetButton?.addEventListener('click', doReset);
  if (resetButton) resetButton.disabled = false;
  updateTaskControls();

  // --- Sync three.js once -------------------------------------------- //
  syncBodyTransforms(data, bodyGroups);

  setStatus('Ready — try the keyboard or click a box.');

  // --- Async-friendly control loop ---------------------------------- //
  // Use requestAnimationFrame for rendering; inference is async (await
  // policy.infer()). Maintain a "busy" flag so a slow inference does not
  // re-enter itself.
  let activeStepPromise = null;
  let paused = urlParams.get('paused') === '1';
  // realtime=1: the browser loop advances one 1/60 s control per 1/60 s of wall
  // time instead of one per animation frame, so the robot moves at the same
  // speed on a 120 Hz display as on a 60 Hz one (v16 ran 1.5-2x fast there and
  // its speed followed the frame rate). At most one control per frame either
  // way; a slow machine still runs slower than real time, exactly as before.
  const realtimePacing = urlParams.get('realtime') === '1';
  const controlPeriodMs = 1000 / CONTROL_HZ;
  let stepDebtMs = 0, lastLoopMs = null;
  let frameCount = 0;
  let lastFpsT = performance.now();

  if (restrictedMode) {
    setStatus('Loading measured walking and turning motions…');
    const note = document.getElementById('control-mode-note');
    if (note) { note.hidden = false; note.textContent = 'Measured steps: changes wait for the current step or turn to finish. Green marks your walking destination, blue the current motion target, and orange the box destination. Walking destinations must be within 2 metres. Select the large box to request a longer box move. Reset restarts the scene.'; }
    if (carryStylePreview && note) note.textContent = `${carryStylePreview.label} preview: try Move 1.15 m after Reset, or select the large box and click a floor destination. The higher lift can need a longer approach and some destinations may miss or be refused. ` + note.textContent;
    if (carryStyleSamplingConfig && note) note.textContent = 'Sampled carry styles: try Move 1.15 m, then Reset and try again. Choices can repeat. Reload this page to repeat the same choice sequence. Other destinations keep the ordinary carry planner when neither style fits. Higher lifts need a longer approach and can miss the destination. ' + note.textContent;
    if (pickupFacingEntryRegionEnabled && note) note.textContent = 'Pickup region preview: try Move 1.4 m at +15° after Reset, then your own destinations. Higher carries can use nearby pickup positions and approach from the required side. This can take longer; some commands still miss or refuse. ' + note.textContent;
    else if (pickupFacingApproachEnabled && note) note.textContent = 'Pickup approach preview: higher lifts can walk around to their pickup heading. This can take longer, and placement can still miss. Other carry styles keep their existing approach. ' + note.textContent;
    if (teacherDescentRematchEnabled && note) note.textContent = 'Placement adjustment preview: the higher carry can adjust its descent toward your original destination. Try your own box destinations and Download session to retain a command record. ' + note.textContent;
    if (recoveredFacingAdmission.enabled && urlParams.get('terminalRefusalRecovery') === '1' && note)
      note.textContent = 'Approach recovery preview: the robot can finish settling and turn toward pickup when an ordinary approach stops. Select the large box and click a destination. Some approaches still need more room; Download session retains your command and the approach result. ' + note.textContent;
    for (const id of ['style-controls', 'smoothing-controls']) {
      const element = document.getElementById(id); if (element) element.hidden = true;
    }
    const keys = [...RESTRICTED_STEP_REFERENCES, ...TURN_REFERENCES];
    for (const key of keys) if (!loadedSkills.has(key)) loadedSkills.set(key,
      loadTeacherSkill(`public/teacher_${key}_reference.json`).catch(error => { loadedSkills.delete(key); throw error; }));
    const [, ...skills] = await Promise.all([loadTeacherActor(), ...keys.map(key => loadedSkills.get(key))]);
    const sweepResponse = await fetch('public/restricted_motion_sweeps.json');
    if (!sweepResponse.ok) throw new Error('Could not load walking collision geometry');
    const sweepGeometry = await sweepResponse.json();
    const sweeps = new Map(skills.map((skill, index) => [skill,
      bindMotionSweep(skill, sweepGeometry, RESTRICTED_SWEEP_KEYS[keys[index]])]));
    const additionalKeySkills = [];
    let backwardSkill = null;
    if (restrictedBackwardEnabled || recoveryFlags.approachRecoveryLoop) {
      const [loadedBackwardSkill, backwardSweepResponse] = await Promise.all([
        loadTeacherSkill('public/teacher_box_exit_reference.json'),
        fetch('public/restricted_backward_sweep.json'),
      ]);
      if (!backwardSweepResponse.ok) throw new Error('Could not load backward collision geometry');
      backwardSkill = loadedBackwardSkill;
      sweeps.set(backwardSkill, bindMotionSweep(backwardSkill, await backwardSweepResponse.json(), 'retreat_292'));
      if (restrictedBackwardEnabled) additionalKeySkills.push(backwardSkill);
    }
    if (recordedLateralEnabled) {
      // Two complete lateral records (left: ACCAD side step 0.56 m, right: KIT recovery step 0.27 m)
      // join the directional key bank with their private collision sweeps.
      const [lateralLeft, lateralRight, lateralSweepResponse] = await Promise.all([
        loadTeacherSkill(`public/teacher_${LATERAL_KEY_REFERENCES[0]}_reference.json`),
        loadTeacherSkill(`public/teacher_${LATERAL_KEY_REFERENCES[1]}_reference.json`),
        fetch(LATERAL_SWEEP_ASSET),
      ]);
      if (!lateralSweepResponse.ok) throw new Error('Could not load lateral collision geometry');
      const lateralKeyRecords = bindLateralKeyRecords({ left: lateralLeft, right: lateralRight }, await lateralSweepResponse.json());
      for (const record of lateralKeyRecords) { sweeps.set(record.skill, record.sweep); additionalKeySkills.push(record.skill); }
    }
    const walkSkills = skills.slice(0, RESTRICTED_STEP_REFERENCES.length);
    if (teacherPeriodicPersistentEnabled) {
      const rawPeriodic = await loadTeacherSkill('public/teacher_periodic_raw_reference.json');
      periodicTeacherSkills = buildPeriodicTeacherSkills(rawPeriodic,
        { ...teacherPeriodicOptions, persistent: true });
      const groups = [periodicTeacherSkills.persistent.entry, ...periodicTeacherSkills.persistent.loops];
      groups.forEach((group, segment) => {
        for (const [name, skill] of Object.entries(group)) {
          sweeps.set(skill, buildPeriodicMotionSweep(skill,
            `teacher_periodic_persistent_${segment}_${name}`));
        }
      });
    }
    pickupFacingSweeps = sweeps;
    restrictedWalkSkills = walkSkills;
    restrictedTurnSkills = skills.slice(RESTRICTED_STEP_REFERENCES.length);
    const referenceEntryMetrics = (live, frame) => {
      // Read the same cached body transforms seen by the teacher. Raw source
      // body targets can disagree with asset FK; these are diagnostics, not
      // entry approval thresholds or a reason to change the reference.
      const bodyErrors = restrictedObs.bodyIds.map((id, index) => Math.hypot(
        ...[0, 1, 2].map(axis => data.xpos[id * 3 + axis] - frame[84 + index * 3 + axis])));
      const jointErrors = Array.from(addresses.qposAddr, (id, index) => data.qpos[id] - frame[13 + index]);
      const rms = values => Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
      const headingError = headingYawFromQuatXyzw(live.rootQuatXyzwWorld) - headingYawFromQuatXyzw(frame.slice(3, 7));
      return {
        bodyPositionMeanErrorM: bodyErrors.reduce((sum, value) => sum + value, 0) / bodyErrors.length,
        bodyPositionMaxErrorM: Math.max(...bodyErrors),
        anklePositionErrorM: Object.fromEntries(['left_ankle_roll_link', 'right_ankle_roll_link']
          .map(name => [name, bodyErrors[TEACHER_HUMAN_BODY_NAMES.indexOf(name)]])),
        jointPositionRmsErrorRad: rms(jointErrors), legJointPositionRmsErrorRad: rms(jointErrors.slice(0, 12)),
        rootHeightErrorM: live.rootPosWorld[2] - frame[2],
        rootHeadingErrorRad: Math.atan2(Math.sin(headingError), Math.cos(headingError)),
        rootPlanarSpeedMps: Math.hypot(...live.rootVelWorld.slice(0, 2)),
      };
    };
    approveRestrictedReference = (live, request) => {
      const sweep = sweeps.get(request.skill);
      const noResetPoseReference = skillController instanceof NoResetApproachPoseController;
      const postTaskSelection = postTaskContactObstacles(readSweepObstacles('approach'), request.activeIntent);
      if (restrictedGeometryCheckStep !== episodeControlStep) restrictedReferenceAttempts = [];
      restrictedGeometryCheckStep = episodeControlStep;
      lastRestrictedGeometry = checkRestrictedReferenceSweep({ sweep,
        sourceFrames: request.sourceFrames, alignedReferenceFrames: request.alignedReferenceFrames,
        obstacles: postTaskSelection.obstacles, trackingReserve: sweep?.name.startsWith('turn_') ? .15 : .1,
        ...(noResetPoseReference ? { heightAware: true, verticalTrackingReserve: .1 } : {}) });
      if (postTaskSelection.admitted) lastRestrictedGeometry = {
        ...lastRestrictedGeometry, postTaskTargetContact: postTaskSelection.admission,
      };
      if (laterSegmentPartAdmissionEnabled && request.sourceFrames > 1 && lastRestrictedGeometry.supported === false
          && lastRestrictedGeometry.reason === 'reference_sweep_clearance' && Array.isArray(sweep?.collisionParts) && sweep.collisionParts.length
          && pendingWaypointCarryController?.phase === 'approach' && pendingWaypointCarryController.segmentIndex >= 1) {
        const partGeometry = checkRestrictedReferenceSweep({ sweep,
          sourceFrames: request.sourceFrames, alignedReferenceFrames: request.alignedReferenceFrames,
          obstacles: readSweepObstacles('approach'), trackingReserve: sweep.name.startsWith('turn_') ? .15 : .1,
          heightAware: true, verticalTrackingReserve: .1 });
        laterSegmentPartAdmissions.push({ episode: episodeVersion, physicalControl: episodeControlStep, requestId: activeBoxTaskRequestId,
          segmentIndex: pendingWaypointCarryController.segmentIndex, phase: request.phase, sweep: sweep.name, sourceFrames: request.sourceFrames,
          referenceAnchor: Array.from(request.alignedReferenceFrames[0].slice(0, 7)),
          legacy: { supported: false, reason: lastRestrictedGeometry.reason, obstacleIndex: lastRestrictedGeometry.obstacleIndex ?? null },
          part: { supported: partGeometry.supported === true, reason: partGeometry.reason ?? null, obstacleName: partGeometry.obstacleName ?? null,
            body: partGeometry.body ?? null, partCount: partGeometry.partCount ?? null, verticallyRelevantPairs: partGeometry.verticallyRelevantPairs ?? null },
          admitted: partGeometry.supported === true });
        if (laterSegmentPartAdmissions.length > 64) laterSegmentPartAdmissions.shift();
        if (partGeometry.supported === true) lastRestrictedGeometry = { ...partGeometry, laterSegmentPartAdmitted: true,
          legacyRefusal: { reason: 'reference_sweep_clearance', obstacleIndex: lastRestrictedGeometry.obstacleIndex ?? null } };
      }
      if (request.sourceFrames === 1) {
        standingPreviewOwned = null;
        // A stance refused while a recorded approach owns the motion keeps its
        // original refusal: that latched refusal is what the existing terminal
        // refusal recovery / recovered facing turn, the first-pickup student and
        // the recovery loop consume in finishWaypointApproach (J04 depends on it).
        const approachOwned = skillController instanceof TeacherRecordedApproachController
          || skillController instanceof NoResetApproachPoseController || pendingWaypointCarryController !== null;
        if (!approachOwned && lastRestrictedGeometry.supported === false && lastRestrictedGeometry.reason === 'reference_sweep_clearance'
            && controlPreview && neverSuspendEligible(lastRestrictedGeometry.reason)) {
          // Unloaded standing refused by the hull: hold it under the per-control
          // physical preview instead of suspending (never-suspend lane).
          standingPreviewOwned = { episode: episodeVersion, physicalControl: episodeControlStep, hullRefusal: structuredClone(lastRestrictedGeometry),
            anchorRootXY: Array.from(request.alignedReferenceFrames[0].slice(0, 2)) };
          lastRestrictedGeometry = { ...lastRestrictedGeometry, supported: true, reason: null, previewOwned: true };
        }
      } else standingPreviewOwned = null;

      if(heightAwareApproachEnabled && !(skillController instanceof PickupFacingApproachController)
          && !(skillController instanceof NoResetApproachPoseController)
          && !heightAwareApproach?.active && request.phase==='teacher_step'){
        const attempt=HeightAwareApproachController.tryStart({owner:skillController,parent:pendingWaypointCarryController,
          eligibleSkill:restrictedWalkSkills[3],request,sweep,wholeGeometry:lastRestrictedGeometry,
          obstacles:readSweepObstacles('approach'),live,episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep});
        if(attempt.supported){heightAwareApproach=attempt.controller;heightAwareApproaches.push(heightAwareApproach);
          if(heightAwareApproaches.length>32)heightAwareApproaches.shift();
          lastRestrictedGeometry=attempt.geometry;}
      }
      if(referenceStudentTurnsEnabled && !(skillController instanceof PickupFacingApproachController)
          && !(skillController instanceof NoResetApproachPoseController)
          && request.phase==='teacher_turn' && lastRestrictedGeometry.supported)
        approvedReferenceTurn={owner:skillController,parent:pendingWaypointCarryController,request,
          geometry:structuredClone(lastRestrictedGeometry),physicalControl:episodeControlStep};
      const retiredDiagnosticPushFollowup = retiredLargeboxPushFollowupCurrent();
      if (retiredDiagnosticPushFollowup && lastRestrictedGeometry?.supported === false
          && ['reference_sweep_clearance', 'reference_part_clearance'].includes(lastRestrictedGeometry.reason)) {
        const geometryDiagnostic = { supported: false, reason: lastRestrictedGeometry.reason ?? null,
          obstacleName: lastRestrictedGeometry.obstacleName ?? null, body: lastRestrictedGeometry.body ?? null,
          obstacleIndex: lastRestrictedGeometry.obstacleIndex ?? null };
        lastRestrictedGeometry = { ...lastRestrictedGeometry, supported: true, reason: null,
          retiredDiagnosticPushFollowup: { requestId: lastLargeboxPushLiveDiagnostic.requestId,
            episodeVersion, geometryDiagnostic } };
      }
      restrictedReferenceAttempts.push({ phase: request.phase, skillName: request.skill.name,
        sweepName: sweep?.name, sourceFrames: request.sourceFrames,
        referenceAnchor: Array.from(request.alignedReferenceFrames[0].slice(0, 7)),
        entry: referenceEntryMetrics(live, request.alignedReferenceFrames[0]),
        geometry: lastRestrictedGeometry });
      return lastRestrictedGeometry;
    };
    restrictedObs = new TeacherObsBuilder(mujoco, model, { ...walkSkills[0], locomotionOnly: true });
    restrictedController = new RestrictedLocomotionController(walkSkills, {
      turnSkills: restrictedTurnSkills, approveReference: approveRestrictedReference,
      retainKeyTerminal: restrictedKeyTerminalEnabled,
      settleOnStart: restrictedStartupStandingEnabled,
      ...(restrictedBackwardEnabled || recordedLateralEnabled ? {
        additionalKeySkills, directionalKeySteps: true,
        maxDirectionalKeyHeadingOffsetRad: Math.PI / 12,
        keyDirectionLine: true, keyDirectionLineAdditionalOnly: true,
        retainAdditionalKeyTerminal: true,
      } : {}),
      planKeyboardGoal: (root, goal, context = {}) => {
        const targetBodyId=findBodyIdByName(model,PUSH_LANE_BODY);
        const selection=selectRetiredPushForwardPlanObstacles(boxCollisionBounds.read(data),{
          eligible:retiredLargeboxPushFollowupCurrent(),intent:context.intent,targetBodyId});
        const planned=planRestrictedFloorGoal(root,goal,selection.obstacles);
        return selection.admitted?{...planned,retiredDiagnosticPushForwardPlan:{requestId:lastLargeboxPushLiveDiagnostic.requestId,
          episodeVersion,ignoredTargetObstacleCount:selection.ignoredTargetObstacleCount}}:planned;
      },
      approveKeyboardTurn: (live, request) => {
        const selection = postTaskContactObstacles(boxCollisionBounds.read(data), request.intent);
        const clear = selection.obstacles.every(bound => Math.hypot(
          Math.max(bound.minX - live.rootPosWorld[0], 0, live.rootPosWorld[0] - bound.maxX),
          Math.max(bound.minY - live.rootPosWorld[1], 0, live.rootPosWorld[1] - bound.maxY)) >= .55);
        return { supported: clear, reason: clear ? null : 'turn_clearance',
          ...(selection.admitted ? { postTaskTargetContact: selection.admission } : {}) };
      },
      ...(periodicTeacherSkills?.persistent
        ? { persistentPeriodicKeySkills: periodicTeacherSkills.persistent } : {}),
    });
    if (boxExitEnabled) {
      boxExitController = new TeacherBoxExitController(await loadTeacherSkill(boxExitMode === 'long'
        ? 'public/teacher_box_exit_long_reference.json' : 'public/teacher_box_exit_reference.json'),
        quietEndingsEnabled ? { quietHold: { window: quietEndingConfig.window,
          initialHoldMinControls: quietEndingConfig.exitInitialHoldMinControls,
          finalHoldMinControls: quietEndingConfig.exitFinalHoldMinControls,
          measure: () => quietEndingSample(boxExitController?.objectBodyName) } }
        : quietExitSettlingMode === 'fixed' ? {}
        // B2: request-terminated final settling; the pending-request predicate is the synchronous queue state a click
        // during the exit leaves behind (queuedBoxTask, startCarryToGoal) or a queued restricted intent.
        : { quietHold: resolveQuietExitHoldOptions({ mode: quietExitSettlingMode,
            measure: () => quietEndingSample(boxExitController?.objectBodyName),
            requestQueued: () => queuedBoxTask !== null || Boolean(queuedRestrictedIntent()) }) });
      controlPreview = new ControlPreview(mujoco, model, { addresses, minRootHeightM: .45, minUpright: .5 });
      if (studentClosedLoopPreviewEnabled) studentClosedLoopPreview = new StudentClosedLoopPreview({ mujoco, model, controlPreview,
        createBodyObsBuilder: () => new BodyObsBuilder(mujoco, model), policy, actionDim: ACTION_DIM, actionScale: ACTION_SCALE, pelvisId });
    }
    if (recoveryFlags.approachRecoveryLoop && backwardSkill && controlPreview) {
      try {
        approachRecovery = new ApproachRecoveryCoordinator({ retreatSkill: backwardSkill, turnSkills: restrictedTurnSkills, sweeps,
          readObstacles: readSweepObstacles, flags: recoveryFlags });
      } catch (error) { console.warn('[approach recovery unavailable]', error); approachRecovery = null; }
    }
    setTaskStatus(hybridKeyboardEnabled ? 'Hybrid controls: student policy in open floor, measured steps and turns near boxes.'
      : 'Measured controls: short steps, complete turns, and standing between motions.');
    setStatus(hybridKeyboardEnabled ? 'Ready — hybrid keyboard controls enabled.' : 'Ready — measured walking controls enabled.');
  }

  async function step() {
    // Explicit debug stepping obeys the same unsupported-state suspension as
    // the browser loop. Do not advance history or apply a cached refused stance.
    if (restrictedSuspended && !releaseTransientSuspension()) return;
    tryTeacherDescentRematch();
    if (matchedCarryHost?.inProgress && matchedCarryHost.owner.role === 'prefix' && stagedStudentTransport?.ended) {
      try {
        skillController = activeCarryController = matchedCarryHost.beginAfter90(stagedStudentTransport);
        stagedStudentTransport = null; transportTeacherResumePending = false;
        recordedApproachHold = null; teacherStandingPlan = null;
      } catch (error) {
        restrictedSuspended = 'matched_carry_entry_refused';
        setTaskStatus('The next carry stage is not supported from the reached pose.');
        console.warn('[matched carry]', error);
        return;
      }
    }
    if (matchedCarryHost?.owner?.active) {
      const owner = matchedCarryHost.owner;
      try { await matchedCarryHost.step(); }
      catch (error) {
        if (matchedCarryHost.owner === owner && owner.request.episode === episodeVersion)
          owner.cancel('matched_carry_runtime_error');
        console.warn('[matched carry]', error);
      }
      // Reset or a replacement task may have changed the host while inference
      // was pending. Never publish the old outcome into the new scene.
      if (matchedCarryHost.owner !== owner || owner.request.episode !== episodeVersion) return;
      if (owner.ended) {
        const outcome = owner.ended;
        finalCarryPlacement = {requestId:owner.request.requestId,
          originalGoalWorld:Array.from(owner.request.originalGoalWorld), ...outcome,
          toleranceM:CARRY_PLACEMENT_TOLERANCE_M,
          checkedAfterCompleteProgramme:owner.counts.postplacement === 739};
        boxTaskResults.push({requestId:owner.request.requestId, episodeVersion, episodeControlStep,
          task:'carry', goalWorld:Array.from(owner.request.originalGoalWorld),
          completionReason:outcome.completionReason, outcome:finalCarryPlacement,
          matchedProgramme:owner.review()});
        boxTaskRequestLog.transition(owner.request.requestId, 'outcome', outcome.completionReason,
          boxRequestClock(), finalCarryPlacement);
        setTaskStatus(outcome.goalReached ? 'Complete. The box is within 10 cm of your original destination.'
          : 'The carry experiment stopped before completing your destination.');
        if (outcome.goalReached) matchedCarryHost.returnToOrdinary(() => {
          skillController = null; restrictedAfterBox = true;
          recordedApproachHold = null; teacherStandingPlan = null;
          executedApproachTerminal = null; transportTeacherResumePending = false;
          restoreSkillCommandStyle(); translator.reset();
          if (!preserveCompletionCommand) { user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys(); }
          preserveCompletionCommand = false;
          restrictedObs.reset({lastDofPos:previousDofPos, lastDofVel:previousDofVel});
          updateTaskControls();
        });
        else { restrictedSuspended = outcome.completionReason; matchedCarryHost.disposeRuntime(); updateTaskControls(); }
      }
      return;
    }
    matchedCarryHost?.observePrefixCommandState();
    const stepEpisodeVersion = episodeVersion;
    const stepPhysicalControl = episodeControlStep;
    const teacherHistory = builder => builder ? {
      lastDofPos: Float32Array.from(builder.lastDofPos), lastDofVel: Float32Array.from(builder.lastDofVel),
    } : null;
    const controlHistory = restrictedMode ? {
      body: Float32Array.from(bodyObsBuilder.historyBuf), initialized: bodyObsBuilder.hasInitialized,
    } : null;
    let queriedTeacherHistory = null;
    const restoreControlHistory = () => {
      if (!controlHistory || stepEpisodeVersion !== episodeVersion || stepPhysicalControl !== episodeControlStep) return;
      bodyObsBuilder.historyBuf.set(controlHistory.body);
      bodyObsBuilder.hasInitialized = controlHistory.initialized;
      if (queriedTeacherHistory) {
        const { builder, saved, locomotion } = queriedTeacherHistory;
        if (builder === (locomotion ? restrictedObs : teacherObs)) builder.reset(saved);
      }
    };
    // 1. Build body obs from current state.
    const bodyObs = bodyObsBuilder.build(data, lastAction);

    // 2. Read root pose for NEW_CMD heading-frame transforms.
    const rootPosWorld = [
      data.xpos[pelvisId * 3 + 0],
      data.xpos[pelvisId * 3 + 1],
      data.xpos[pelvisId * 3 + 2],
    ];
    // MuJoCo xquat is (w,x,y,z) — convert to (x,y,z,w).
    const qw = data.xquat[pelvisId * 4 + 0];
    const qx = data.xquat[pelvisId * 4 + 1];
    const qy = data.xquat[pelvisId * 4 + 2];
    const qz = data.xquat[pelvisId * 4 + 3];
    const rootQuatXyzwWorld = [qx, qy, qz, qw];
    startBenchmarkCase(rootPosWorld, rootQuatXyzwWorld);

    // 3. Build full 1422D obs.
    //    If an object is selected (via mouse picker), read its world pose
    //    from MuJoCo and feed the cached local point cloud. Otherwise pass
    //    null and the obs builder zeros out task / obj_target / mask bits.
    let objPosWorld = null, objQuatXyzwWorld = null, objPointsObjLocal = null;
    if (activeObjBodyId >= 0 && activeObjPointsFlat !== null) {
      objPosWorld = [
        data.xpos[activeObjBodyId * 3 + 0],
        data.xpos[activeObjBodyId * 3 + 1],
        data.xpos[activeObjBodyId * 3 + 2],
      ];
      const ow = data.xquat[activeObjBodyId * 4 + 0];
      const ox = data.xquat[activeObjBodyId * 4 + 1];
      const oy = data.xquat[activeObjBodyId * 4 + 2];
      const oz = data.xquat[activeObjBodyId * 4 + 3];
      objQuatXyzwWorld = [ox, oy, oz, ow];
      objPointsObjLocal = activeObjPointsFlat;
    }

    // 3a. Build proprio struct for the goal translator's KNN query +
    //     FSM. All in body / world frame, matched to
    //     scripts/build_browser_clip_db.py:
    //     - upright_score = world_up · R(quat)·world_up  ∈ [-1, +1].
    //       For unit quat (qx,qy,qz,qw), body_up = R · (0,0,1) is the
    //       third column of R; its z-component is R[2,2] = 1 - 2(qx²+qy²).
    //       Identity quat → score = +1; horizontal → 0; inverted → -1.
    //     - foot contact = min(body_z over the side's ankle_pitch +
    //       ankle_roll bodies) < 0.10 m  (same threshold as DB build).
    const uprightScore = 1 - 2 * (qx*qx + qy*qy);
    let leftFootMinZ = Infinity, rightFootMinZ = Infinity;
    for (const bid of leftFootIds)  leftFootMinZ  = Math.min(leftFootMinZ,  data.xpos[bid * 3 + 2]);
    for (const bid of rightFootIds) rightFootMinZ = Math.min(rightFootMinZ, data.xpos[bid * 3 + 2]);
    const FOOT_CONTACT_HEIGHT_M = 0.10;
    const footContactL = leftFootMinZ  < FOOT_CONTACT_HEIGHT_M ? 1 : 0;
    const footContactR = rightFootMinZ < FOOT_CONTACT_HEIGHT_M ? 1 : 0;
    const proprio = {
      rootPosWorld,
      rootQuatXyzwWorld,
      rootVelWorld: Array.from(data.qvel.slice(0, 3)),
      pelvisZ:      rootPosWorld[2],
      uprightScore,
      footContactL,
      footContactR,
      rootHeight:   rootPosWorld[2],
      objPosWorld,
    };

    // 3b. Call the translator. Returns { goalSpec, mask, longTermT,
    //     fsmState, debug }. goalSpec is in body frame matching the
    //     V2-interactive task's NEW_CMD layout exactly.
    const previousSkillPhase = skillController?.phase;

    if (pendingNormalGroundPush?.readyAfterControl < episodeControlStep) {
      const pending=pendingNormalGroundPush;
      try {
        if(pending.episodeVersion!==episodeVersion||pending.requestId!==activeBoxTaskRequestId
            ||pending.requestId!==latestBoxTaskRequestId||user.activeObjName!==PUSH_LANE_BODY)
          throw new Error('push_request_changed');
        const controller=new GroundPushGoalSequenceController(pending.skill,pending.goal,
          {initialStanceFrames:0,maxSegments:1,settlingSteps:180,maxCorrection:.05,warpStartFrame:81,
            warpEndFrame:110,maxFacingError:Math.PI,maxReferenceStartDistance:Infinity,requireSegmentExit:boxExitEnabled});
        controller.startFromMeasuredEndpoint(skillProprio(controller));
        if(controller.phase!=='teacher')throw new Error(controller.completionReason??'push_start_refused');
        const nextTeacherObs=new TeacherObsBuilder(mujoco,model,controller.skill);
        nextTeacherObs.reset({lastDofPos:previousDofPos,lastDofVel:previousDofVel});
        skillController=activeCarryController=controller;pendingCarryController=null;teacherObs=nextTeacherObs;
        boxExitController?.reset();preserveCompletionCommand=false;translator.reset();
        user.humanGoalWorld=user.objGoalWorld=null;user.releaseKeys();user.deterministic=true;user.vaeNoise.fill(0);smoothingAlpha=1;
        lastLargeboxPushLiveDiagnostic={requestId:pending.requestId,normalPage:true,
          token:{episodeVersion,control:episodeControlStep},sourceIdentity:pending.sourceIdentity,
          approachTarget:structuredClone(pending.approachTarget),liveEntryQualified:false,normalPageIntegrationQualified:false};
        pendingNormalGroundPush=null;setTaskStatus('Pushing / sliding the Largebox to the requested destination…');
      } catch(error) {
        pendingNormalGroundPush=null;skillController=activeCarryController=null;teacherObs?.dispose();teacherObs=null;
        if(activeBoxTaskRequestId!==null){boxTaskRequestLog.transition(activeBoxTaskRequestId,'outcome','push_handoff_failed',boxRequestClock(),{message:error.message});
          retireBoxTaskOwnership('pre_push_handoff_failed',null);}
        setTaskStatus('Push / Slide stopped safely before contact.');
      }
    }
    let skillStep = skillController?.step(skillProprio());
    if (pendingOwnedRetirement) {
      skillStep = { phase: 'complete', mode: 'student', justCompleted: true, referenceFrames: null, supported: false, ...pendingOwnedRetirement, ownedRetirement: true };
      pendingOwnedRetirement = null;
    }
    if(firstPickupStanding?.active){
      const context=firstPickupStandingContext();
      if(!firstPickupStanding.isOwnedBy(context)){
        firstPickupStanding.abort('owner_changed_before_parent_sample');restoreControlHistory();return;
      }
      const transition=firstPickupStanding.observeParent(skillStep,context,skillProprio());
      if(transition.fallback){boxApproachPlanner?.reset();lastApproachRoute=null;}
    }
    if (pendingSegmentCarryController && skillController === boxExitController && skillStep?.justCompleted) {
      const carry = pendingSegmentCarryController;
      boxExitResults.push({ segmentIndex: carry.segmentIndex, betweenSegments: true,
        completionReason: skillStep.completionReason, recordClock: skillStep.recordClock, outcome: skillStep.outcome });
      carry.resumeAfterSegmentExit(skillProprio(carry), skillStep);
      stagedStudentTransport = null; lastStudentTransportEntry = null; transportTeacherResumePending = false;
      pendingSegmentCarryController = null;
      skillController = carry;
      translator.reset(); translator.setClickPositionSource('stable_receding');
      recordedApproachHold = null; executedApproachTerminal = null;
      boxApproachPlanner?.reset(); lastApproachRoute = null; headingPreparations = [];
      if (restrictedStudentApproachEnabled && carry.phase === 'approach') {
        const live = skillProprio(carry);
        const route = boxApproachPlanner.step(live.rootPosWorld, carry.approachGoalWorld, boxCollisionBounds.read(data));
        const attempt = StagedStudentApproachController.tryStart({ parent: carry, live, route, exit: skillStep,
          episode: episodeVersion, physicalControl: episodeControlStep });
        lastStudentApproachEntry = { episode: episodeVersion, physicalControl: episodeControlStep,
          segmentIndex: carry.segmentIndex, supported: attempt.supported, reason: attempt.reason,
          distanceM: attempt.entry?.distanceM ?? null };
        if (attempt.supported && carry.prepareStudentHandoff(live)) {
          stagedStudentApproach = attempt.controller;
          studentApproaches.push(stagedStudentApproach);
          if (studentApproaches.length > 32) studentApproaches.shift();
          setTaskStatus('Approaching the next grasp before continuing to your destination…');
        }
      }
      teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, carry.skill);
      skillStep = carry.step(skillProprio());
    }
    if (skillStep?.awaitingSegmentExit) {
      const carry = skillController, exitProprio = skillProprio(carry);
      boxExitController.start(exitProprio, {
        terminalFrame: carry.worldFrames[carry.skill.sourceFrames - 1],
        objectBodyName: carry.skill.objectBodyName, objectPointsLocal: carry.skill.objectPointsLocal,
        quietHolds: false,   // another pickup follows: both exit holds run their full fixed length (WS-G H083 evidence: int6 retest still refused with only the final hold fixed)
        ...objectRouter.exitStandOff(carry.skill.objectBodyName, exitProprio),   // v6e: OFF => {} (v5 option object); ON => {standOffWorld} for objects standing taller than the largebox
        ...objectRouter.exitRelease(carry.skill.objectBodyName, exitProprio, { measure: () => quietEndingSample(carry.skill.objectBodyName) }),   // v6g: OFF => {}; ON+declared => {releaseLift}
        ...objectRouter.exitHandClamp(carry.skill.objectBodyName, exitProprio),   // v6h: OFF => {}; ON+declared => {holdHandClamp} (initial hold hands clamped above the top face)
        ...objectRouter.exitHoldSource(carry.skill.objectBodyName),   // v6j: OFF => {}; ON+declared => {holdSource: 'retreat_row0'} (initial hold = the retreat clip's row 0 at the live root)
      });
      pendingSegmentCarryController = carry;
      skillController = boxExitController;
      recordedApproachHold = null; executedApproachTerminal = null; teacherStandingPlan = null;
      skillStep = boxExitController.step(skillProprio());
    }
    if (replanRemainingEnabled && skillStep?.justCompleted && skillController === activeCarryController
        && skillStep.completionReason === 'unsupported_live_distance') {
      // The next fixed source cannot start from the measured box. Instead of
      // ending the task, replan the remaining goal from the live box (WS-B #1).
      const replan = tryReplanRemainingGoal(skillController, skillStep);
      if (replan.started) skillStep = skillController.step(skillProprio());
      else skillStep = { ...skillStep, replanRefusal: { reason: replan.reason, remainingControls: replan.remainingControls,
        estimatedControls: replan.estimatedControls, plannedCandidateIds: replan.plannedCandidateIds, pickupsUsed: replan.pickupsUsed } };
    }
    skillStep = finishWaypointApproach(skillStep);
    if (pendingCarryController && skillStep?.justCompleted) {
      const carry = pendingCarryController;
      pendingCarryController = null;
      headingPreparations.push({ teacherFrames: skillController.skill.sourceFrames,
        completionReason: skillStep.completionReason, outcome: skillStep.outcome });
      if (skillStep.completionReason === 'finished') {
        carry.retryCurrentSegment(skillProprio(carry));
        skillController = carry;
        teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, skillController.skill);
        translator.reset(); user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
        skillStep = skillController.step(skillProprio());
      } else {
        skillStep = { ...skillStep, requestedGoalWorld: carry.requestedGoalWorld };
      }
    }
    if (pendingRecoveryParent && skillController instanceof ApproachRecoveryCycle && skillStep?.justCompleted) {
      const carry = pendingRecoveryParent, cycle = skillController; pendingRecoveryParent = null;
      let finished;
      try { finished = approachRecovery.finish(cycle, skillStep, skillProprio(carry), readSweepObstacles()); }
      catch (error) { console.warn('[approach recovery]', error); finished = { retry: false, reason: 'recovery_retry_unavailable' }; }
      let resumed = false;
      if (finished.retry) {
        try { resumed = carry.retryCurrentSegment(skillProprio(carry)); }
        catch (error) { console.warn('[approach recovery retry]', error); resumed = false; }
      }
      if (resumed) {
        skillController = carry;
        teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, carry.skill);
        translator.reset(); translator.setClickPositionSource('stable_receding');
        user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
        boxApproachPlanner?.reset(); lastApproachRoute = null;
        setTaskStatus(`Retrying the approach to the ${carriedObjectLabel()}…`);
        skillStep = carry.step(skillProprio());
      } else {
        skillController = null;
        skillStep = { ...skillStep, phase: 'complete', mode: 'student', justCompleted: true, supported: false, referenceFrames: null,
          completionReason: finished.retry ? 'recovery_retry_unavailable' : skillStep.completionReason,
          requestedGoalWorld: Array.from(carry.requestedGoalWorld) };
      }
    }
    if (skillController === activeCarryController && skillStep?.justCompleted
        && skillStep.completionReason === 'needs_facing' && headingPreparations.length < 2) {
      const preparation = prepareRecoveredFacingTurn(turnSkills, skillProprio(), skillStep.facingErrorRad, {
        approveReference:restrictedMode?approveRestrictedReference:null,admission:recoveredFacingAdmission,
        context:{owner:skillController,parent:skillController,activeParent:activeCarryController,
          episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep,
          completionReason:skillStep.completionReason},
      });
      if(preparation.recoveredFacingPending){
        terminalFacingPending=preparation.recoveredFacingPending;
        terminalFacingEvents.push({event:'recovered_turn_preview_admitted',
          ...recoveredFacingAdmission.summary(terminalFacingPending.recoveryTicket)});
        if(terminalFacingEvents.length>64)terminalFacingEvents.shift();
      }
      if (preparation.supported) {
        const turn = preparation.controller;
        if(terminalFacingPending){
          const pending=terminalFacingPending;
          if(pending.parent!==skillController||pending.episode!==episodeVersion
              ||pending.requestId!==activeBoxTaskRequestId||pending.physicalControl!==episodeControlStep)
            throw Error('The admitted physical turn lost its original owner');
          terminalFacingProbe=new RecoveredFacingTurnProbe({...pending,owner:turn});terminalFacingPending=null;
        }
        pendingCarryController = skillController; skillController = turn;
        teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, turn.skill);
        user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
        skillStep = skillController.step(skillProprio());
      } else {
        const recovered = recoveryFlags.approachRecoveryLoop
          ? startApproachRecovery('facing', skillController, { reason: preparation.reason, facingErrorRad: skillStep.facingErrorRad }, skillStep) : null;
        // The task reason stays needs_facing (the owner's refusal). Why the
        // recovering turn was not admitted is reported beside it, not instead;
        // facingTurnRefused lets the status say that a turn was attempted.
        skillStep = recovered ?? { ...skillStep, completionReason: preparation.turnRefusalReason ? 'needs_facing' : preparation.reason,
          facingTurnRefused: true,
          facingErrorDeg: Number.isFinite(skillStep.facingErrorRad) ? skillStep.facingErrorRad * 180 / Math.PI : null,
          turnAdmissible: false, turnRefusalReason: preparation.turnRefusalReason ?? null,
          turnRefusedCandidates: preparation.refusedCandidates ?? null };
      }
    }
    if (skillStep?.referenceChanged) {
      teacherObs.dispose();
      teacherObs = new TeacherObsBuilder(mujoco, model, skillController.skill);
      translator.reset();
      user.humanGoalWorld = user.objGoalWorld = null;
      user.releaseKeys();
      setTaskStatus(`Moving to another side of the ${carriedObjectLabel()} for this carry…`);
    }
    if (skillStep?.stepFinished || skillStep?.turnFinished) {
      translator.reset(); translator.setClickPositionSource('matched');
      user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
      setTaskStatus('Settling before the next step…');
    }
    if (skillStep?.segmentFinished && !skillStep.justCompleted) {
      translator.reset();
      user.humanGoalWorld = user.objGoalWorld = null;
      user.releaseKeys();
      setTaskStatus(skillStep.completedSegments < skillStep.segmentCount
        ? `${capitalize(carriedObjectLabel())} set down. Preparing carry ${skillStep.completedSegments + 1} of ${skillStep.segmentCount}…`
        : `${capitalize(carriedObjectLabel())} set down. Returning to standing…`);
    }
    if (stagedStudentApproach?.active) {
      const transition = stagedStudentApproach.observeParent(skillStep, firstPickupStudentContext());
      if (transition.fallback) {
        boxApproachPlanner?.reset(); lastApproachRoute = null;
        setTaskStatus('Preparing a measured approach from here…');
        if(debugControls&&urlParams.get('firstPickupStandingRecovery')==='1'
            &&stagedStudentApproach.firstPickupOrigin
            &&transition.reason==='stage_goal_window_elapsed'
            &&!firstPickupStandingUsed.has(stagedStudentApproach)){
          firstPickupStanding=new FirstPickupStandingRecovery({parent:skillController,
            student:stagedStudentApproach,readContext:firstPickupStandingContext});
          firstPickupStandingUsed.add(stagedStudentApproach);firstPickupStandings.push(firstPickupStanding);
          firstPickupStanding.observeParent(skillStep,firstPickupStandingContext(),skillProprio());
        }
      }
    }

    if(stagedStudentApproach?.firstPickupOrigin&&skillStep?.justEnteredTeacher
        &&stagedStudentApproach.isOwnedBy(firstPickupStudentContext())
        &&!firstPickupStudentHandoffs.some(r=>r.episode===episodeVersion&&r.requestId===activeBoxTaskRequestId))
      firstPickupStudentHandoffs.push(captureFirstPickupTeacherHandoff({controller:stagedStudentApproach,
        parent:skillController,step:skillStep,live:skillProprio(),episode:episodeVersion,requestId:activeBoxTaskRequestId,
        physicalControl:episodeControlStep}));

    const standingHandoff=firstPickupStudentHandoffs.at(-1);
    if(standingHandoff?.physicalControl===episodeControlStep
        &&firstPickupStanding?.ended?.arrived===true
        &&firstPickupStanding.ended.atControl===episodeControlStep){
      standingHandoff.arrivalOwner='restricted_standing_after_student_window';
      standingHandoff.actualStandingControlsBeforeHandoff=firstPickupStanding.controls;
      if(debugControls&&urlParams.get('firstPickupStandingReanchor')==='1'
          &&skillStep.justEnteredTeacher&&skillController===activeCarryController
          &&standingHandoff.episode===episodeVersion&&standingHandoff.requestId===activeBoxTaskRequestId
          &&firstPickupStanding.isOwnedBy(firstPickupStandingContext())){
        // The pre-pickup standing interval consumed this flag. Preserve the
        // ordinary post-carry reanchor after the source and exit change pose.
        restrictedAfterBox=true;
        standingHandoff.postCarryStandingReanchor={armed:true,episode:episodeVersion,
          requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep};
      }
    }
    const useFirstPickupStanding=Boolean(firstPickupStanding?.active
      &&firstPickupStanding.isOwnedBy(firstPickupStandingContext())&&skillStep?.phase==='approach');
    const useStagedStudentApproach = Boolean(stagedStudentApproach?.active
      && stagedStudentApproach.isOwnedBy(firstPickupStudentContext())
      && skillStep?.phase === 'approach');
    if (recordedApproachHold && (recordedApproachHold.episode !== episodeVersion || skillController !== recordedApproachHold.parent
        || skillStep?.phase !== 'approach' || Math.hypot(
          skillStep.approachGoalWorld[0] - recordedApproachHold.goalWorld[0],
          skillStep.approachGoalWorld[1] - recordedApproachHold.goalWorld[1]) > 1e-3)) {
      recordedApproachHold = null;
    }
    if (skillStep?.phase === 'approach') {
      user.humanGoalWorld = skillStep.approachGoalWorld;
      if (boxApproachPlanner && !recordedApproachHold && !useStagedStudentApproach && !useFirstPickupStanding) {
        if (previousSkillPhase !== 'approach') boxApproachPlanner.reset();
        const route = boxApproachPlanner.step(rootPosWorld, skillStep.approachGoalWorld, boxCollisionBounds.read(data));
        lastApproachRoute = { ...route, goal: Array.from(route.goal) };
        user.humanGoalWorld = Float32Array.from(route.goal);
        if (recordedApproachEnabled && !route.supported) {
          const reason = route.reason || 'route_unavailable';
          if (neverSuspendEligible(reason) && typeof skillController.refuseApproach === 'function' && skillController.referenceIndex === 0) {
            // Owned refusal in this control instead of a suspension; standing follows.
            skillController.refuseApproach(reason); skillStep = skillController.step(skillProprio());
          } else skillStep = { ...skillStep, mode: 'none', supported: false, completionReason: reason, referenceFrames: null };
        } else if (route.supported && (recordedApproachEnabled
            || (teacherApproachEnabled && skillController === activeCarryController && route.routed))) {
          let approach = null;
          if (pickupFacingApproachEnabled && carryReferenceSelection?.selectedId === 'medium_1224') {
            const plan = (pickupFacingEntryRegionEnabled ? planPickupFacingEntryRegion : planPickupFacingApproach)({ parent: skillController, stepSkills: restrictedWalkSkills,
              sweeps: pickupFacingSweeps, obstacles: boxCollisionBounds.read(data), live: skillProprio(),
              episode: episodeVersion, requestId: activeBoxTaskRequestId, physicalControl: episodeControlStep });
            const supported = plan.supported && plan.selected.sourceFrames === 254;
            pickupFacingPlans.push({ episode: episodeVersion, requestId: activeBoxTaskRequestId,
              physicalControl: episodeControlStep, supported,
              reason: supported ? null : plan.reason ?? 'final_source_not_in_preview',
              originalGoalWorld: Array.from(skillController.requestedGoalWorld),
              pregraspGoalWorld: Array.from(skillController.approachGoalWorld),
              entryRegionChanged: plan.entryRegionChanged ?? false,
              entryRegionSelectedOffsetM: plan.entryRegionSelectedOffsetM ?? null,
              entryRegionAttempts: plan.entryRegionAttempts ?? null,
              terminalGoalWorld: plan.terminalGoalWorld ? Array.from(plan.terminalGoalWorld) : null,
              selectedSourceFrames: plan.selected?.sourceFrames ?? null,
              requiredEntryPose: plan.selected ? Array.from(plan.selected.requiredEntryPose) : null });
            if (pickupFacingPlans.length > 32) pickupFacingPlans.shift();
            if (supported) {
              approach = new PickupFacingApproachController({ plan, stepSkills: restrictedWalkSkills,
                turnSkills: restrictedTurnSkills, approveReference: approveRestrictedReference,
                readObstacles: () => boxCollisionBounds.read(data) });
              pickupFacingOwners.push(approach);
              if (pickupFacingOwners.length > 32) pickupFacingOwners.shift();
              approach.start(skillProprio());
            }
          }
          if (!approach) {
            // No new approach action has executed. Keep the ordinary route if
            // the opt-in planner has no supported complete final source.
            approach = recordedApproachEnabled
            ? new TeacherRecordedApproachController(restrictedWalkSkills, {
              turnSkills: restrictedTurnSkills, approveReference: approveRestrictedReference,
              handoffRadius: skillController.arrivalRadius,
              retainTerminalInHandoffRegion: restrictedApproachTerminalEnabled,
              // H043: after an admitted finite student approach expires inside
              // the unchanged carry handoff region, settle at the measured
              // collision-clear root instead of translating the stance up to
              // 10 cm toward the pickup goal.
              preserveLiveRootInHandoffRegion: stagedStudentApproach?.ended?.fallback === true
                && stagedStudentApproach.isOwnedBy(firstPickupStudentContext()) })
            : new TeacherWaypointController(stepSkills, { turnSkills });
          approach.start(skillProprio(), {
            waypoints: route.waypoints.map(point => [point[0], point[1], skillStep.approachGoalWorld[2]]),
            finalGoalWorld: skillStep.approachGoalWorld,
          });
          }
          pendingWaypointCarryController = skillController; skillController = approach;
          teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, approach.skill);
          translator.reset(); translator.setClickPositionSource('matched');
          user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
          skillStep = approach.step(skillProprio());
          // Initial preflight refusal is also a one-time completion.
          skillStep = finishWaypointApproach(skillStep);
        }
      }
      if (previousSkillPhase !== 'approach') setTaskStatus(skillStep.segmentCount
        ? `Walking to the ${carriedObjectLabel()} for carry ${skillStep.segmentIndex + 1} of ${skillStep.segmentCount}…`
        : 'Walking to the box…');
      if (pickupFacingApproachEnabled && previousSkillPhase !== 'approach') {
        if (skillController instanceof PickupFacingApproachController)
          setTaskStatus(`Walking around the ${carriedObjectLabel()} to face the pickup…`);
        else if (pickupFacingPlans.at(-1)?.requestId === activeBoxTaskRequestId
            && pickupFacingPlans.at(-1)?.supported === false)
          setTaskStatus('Using the ordinary approach for this destination…');
      }
    }
    if (noResetApproachOwner && skillController === noResetApproachOwner && skillStep?.justCompleted) {
      const completedApproach = noResetApproachOwner;
      lastNoResetApproachReview = completedApproach.review();
      const pending = pendingNormalGroundPush;
      const validPending = pending && pending.requestId === activeBoxTaskRequestId
        && pending.episodeVersion === episodeVersion && latestBoxTaskRequestId === pending.requestId
        && user.activeObjName === PUSH_LANE_BODY;
      if (validPending && lastNoResetApproachReview.completionReason === 'finished') {
        pending.readyAfterControl=episodeControlStep;
        noResetApproachOwner=null;skillController=null;teacherObs?.dispose();teacherObs=null;
        restrictedController.reanchor();restrictedObs.reset({lastDofPos:previousDofPos,lastDofVel:previousDofVel});
        user.humanGoalWorld=user.objGoalWorld=null;user.releaseKeys();
        skillStep={...skillStep,mode:'student',justCompleted:false,referenceFrames:null,
          normalGroundPushHandoffPending:true};
        setTaskStatus('Approach complete. Starting Push / Slide on the next physical control…');
      } else {
        const reason=validPending ? lastNoResetApproachReview.completionReason : 'push_request_changed';
        pendingNormalGroundPush=null;noResetApproachOwner=null;skillController=null;
        teacherObs?.dispose();teacherObs=null;restrictedController.reanchor();
        restrictedObs.reset({lastDofPos:previousDofPos,lastDofVel:previousDofVel});
        if(activeBoxTaskRequestId!==null){const terminalReason='pre_grasp_terminal:'+reason;
          boxTaskRequestLog.transition(activeBoxTaskRequestId,'outcome',terminalReason,boxRequestClock());
          retireBoxTaskOwnership(terminalReason,completedApproach);}
        user.humanGoalWorld=user.objGoalWorld=null;user.releaseKeys();
        skillStep={...skillStep,mode:'student',justCompleted:false,referenceFrames:null,
          noResetApproachStoppedBeforeManipulation:true};
        setTaskStatus('Push / Slide approach stopped safely before contact.');
      }
    }
    const taskCompletedThisControl = Boolean(skillStep?.justCompleted);
    if (skillStep?.justCompleted) {
      const completedController = skillController;
      if (completedController !== boxExitController) boxTaskResults.push({
        requestId: activeBoxTaskRequestId,
        episodeVersion, episodeControlStep, task: activeBoxTask,
        goalWorld: skillStep.requestedGoalWorld ? Array.from(skillStep.requestedGoalWorld) : null,
        completionReason: skillStep.completionReason, outcome: skillStep.outcome ?? null,
        segments: activeCarryController?.segmentResults ? structuredClone(activeCarryController.segmentResults) : null,
        segmentExits: activeCarryController?.segmentExitResults ? structuredClone(activeCarryController.segmentExitResults) : null,
      });
      if (completedController !== boxExitController) boxTaskRequestLog.transition(activeBoxTaskRequestId,
        'outcome', skillStep.completionReason, boxRequestClock(), { completionReason: skillStep.completionReason,
          replanRefusal: skillStep.replanRefusal ?? null, taskPickupCount: carryTaskPickupCount(activeCarryController),
          ...(skillStep.turnAdmissible === false ? { facingErrorDeg: skillStep.facingErrorDeg ?? null, turnAdmissible: false,
            turnRefusalReason: skillStep.turnRefusalReason ?? null } : {}) });
      restoreSkillCommandStyle();
      translator.reset();
      if (skillStep.completionReason === 'cancelled') taskDestinationWorld = queuedBoxTask?.requestedGoal
        ? Array.from(queuedBoxTask.requestedGoal) : null;
      if (skillStep.completionReason !== 'cancelled' && !preserveCompletionCommand) {
        user.humanGoalWorld = user.objGoalWorld = null;
        user.releaseKeys();
      }
      if (teacherStandingAlignment && skillController === activeCarryController
          && skillStep.completionReason === 'finished' && !preserveCompletionCommand) {
        const live = skillProprio();
        const terminal = teacherStandingMode === 'neutral'
          ? stepSkills[0].frames[stepSkills[0].sourceFrames - 1]
          : skillController.worldFrames[skillController.skill.sourceFrames - 1];
        teacherStandingPlan = planTeacherStandingReference(
          terminal, {
            alignment: teacherStandingAlignment, rootPosition: live.rootPosWorld,
            rootQuaternion: live.rootQuatXyzwWorld, objectPosition: live.objPosWorld,
            objectQuaternion: live.objQuatXyzwWorld, objectPointsLocal: skillController.skill.objectPointsLocal,
          });
        if (teacherStandingMode === 'neutral') {
          teacherObs.dispose();
          teacherObs = new TeacherObsBuilder(mujoco, model, { ...skillController.skill, locomotionOnly: true });
        }
        teacherObs.reset({ lastDofPos: previousDofPos, lastDofVel: previousDofVel });
      }
      preserveCompletionCommand = false;
      const measuredObject = skillProprio().objPosWorld;
      const remaining = skillStep.requestedGoalWorld ? Math.hypot(
        skillStep.requestedGoalWorld[0] - measuredObject[0], skillStep.requestedGoalWorld[1] - measuredObject[1]) : null;
      setTaskStatus(skillStep.facingTurnRefused
          ? 'Could not turn to face the box from here. Choose another destination or approach the box from a different side.'
        : skillStep.completionReason === 'cancelled' ? 'Box task cancelled.'
        : skillStep.completionReason === 'placement_missed' ? `${capitalize(carriedObjectLabel())} set down. ${Math.round(remaining * 100)} cm remain; stepping clear before another request.`
        : skillStep.completionReason === 'finished' ? 'Box placed. Stepping clear…'
        : skillStep.completionReason === 'lost_balance' ? 'The robot lost balance. Reset the scene to try again.'
        : skillStep.completionReason === 'failed_lift' ? 'The box was not lifted. Move closer and try again.'
        : skillStep.completionReason === 'failed_setdown' ? 'The box did not settle on the floor. Reset the scene to try again.'
        : skillStep.completionReason === 'needs_facing' ? 'Approach the box from the other side, then try again.'
        : skillStep.completionReason === 'unsupported_live_distance' ? (skillStep.replanRefusal?.reason === 'not_enough_time_left'
          ? `${capitalize(carriedObjectLabel())} set down. Not enough time left to finish the carry to your destination.`
          : `${capitalize(carriedObjectLabel())} set down. Choose a new destination to continue.`)
        : skillStep.completionReason === 'occupied_carry_destination' ? 'There is not enough room for the box at that destination. Choose a clearer spot.'
        : skillStep.completionReason === 'carry_reference_clearance' ? 'The carry would pass too close to another box. Choose a different direction.'
        : skillStep.completionReason === 'reference_sweep_clearance' ? 'The approach needs more clearance. Choose another box destination or walking direction.'
        : skillStep.completionReason === 'unsupported_distance' ? 'That destination is outside the supported carry distances. Choose a closer point.'
        : RECOVERY_REFUSAL_MESSAGES[skillStep.completionReason] ?? 'Could not settle near the box. Move closer and face it, then try again.');
      updateTaskControls();
      if (completedController === boxExitController) {
        boxExitResults.push({ completionReason: skillStep.completionReason,
          recordClock: skillStep.recordClock, outcome: skillStep.outcome });
        const placementDistanceM = taskDestinationWorld && ['finished', 'placement_missed'].includes(activeCarryController?.completionReason)
          ? Math.hypot(taskDestinationWorld[0] - measuredObject[0], taskDestinationWorld[1] - measuredObject[1]) : null;
        if (placementDistanceM !== null && skillStep.completionReason === 'finished') {
          finalCarryPlacement = { requestId: activeBoxTaskRequestId, checkedAfterExit: true,
            originalGoalWorld: Array.from(taskDestinationWorld), measuredObjectPositionWorld: Array.from(measuredObject),
            remainingDistanceM: placementDistanceM, toleranceM: CARRY_PLACEMENT_TOLERANCE_M,
            goalReached: placementDistanceM <= CARRY_PLACEMENT_TOLERANCE_M + 1e-12 && measuredObject[2] <= .25 };
        }
        const correction = placementDistanceM !== null && skillStep.completionReason === 'finished'
          && !finalCarryPlacement?.goalReached ? maybeQueuePlacementCorrection({ measuredObject, placementDistanceM }) : null;
        const placedLabel = carriedObjectLabel();
        setTaskStatus(correction?.queued
          ? `${placedLabel[0].toUpperCase()}${placedLabel.slice(1)} set down ${Math.round(placementDistanceM * 100)} cm from your destination. Correcting the placement…`
          : queuedBoxTask || queuedRestrictedIntent()
          ? `Clear of the ${placedLabel}. Resuming the queued command.`
          : finalCarryPlacement?.goalReached ? `Complete. The ${placedLabel} is ${placementDistanceM < .01
            ? 'less than 1' : Math.round(placementDistanceM * 100)} cm from your destination.`
          : placementDistanceM !== null ? `Clear of the ${placedLabel}. ${Math.round(placementDistanceM * 100)} cm remain. Choose a destination to continue.`
          : `Clear of the ${placedLabel}. Choose your next movement.`);
        // Post-exit point: the terminal receipt was published at task completion and the
        // box exit has finished. Retire ownership unless a placement correction re-owns it.
        if (!correction?.queued) {
          const requestId = activeBoxTaskRequestId, task = activeBoxTask;
          const requestRecord = boxTaskRequestLog.records.get(requestId);
          const targetBodyName = activeCarryController?.skill?.objectBodyName ?? boxExitController.objectBodyName;
          const targetBodyId = targetBodyName ? findBodyIdByName(model, targetBodyName) : -1;
          const retirement = retireBoxTaskOwnership('box_exit_complete', completedController);
          postTaskTargetContactAdmission.issue({ retirement, requestId, task,
            episodeVersion, issuedAtControl: episodeControlStep,
            latestRequestId: latestBoxTaskRequestId,
            queuedRequestId: queuedBoxTask?.requestId ?? null,
            taskCompletionReason: activeCarryController?.completionReason,
            exitCompletionReason: skillStep.completionReason,
            placementGoalReached: finalCarryPlacement?.goalReached === true,
            requestDisposition: requestRecord?.disposition,
            requestOutcomeReason: requestRecord?.reason,
            targetBodyName, targetBodyId });
        }
      } else if (boxExitEnabled && ['finished', 'placement_missed', 'cancelled'].includes(skillStep.completionReason)
          && !(Number.isInteger(completedController.exitedSegmentIndex)
            && completedController.exitedSegmentIndex === completedController.segmentIndex)
          && completedController.worldFrames
          && completedController.referenceIndex >= completedController.skill.sourceFrames) {
        // Begin only after the complete setdown reference has actually run.
        // The original executed terminal pose provides the first fixed hold;
        // incoming user intentions remain queued through the complete exit.
        const exitProprio = skillProprio();
        boxExitController.start(exitProprio, {
          terminalFrame: completedController.worldFrames[completedController.skill.sourceFrames - 1],
          objectBodyName: completedController.skill.objectBodyName,
          objectPointsLocal: completedController.skill.objectPointsLocal,
          ...objectRouter.exitStandOff(completedController.skill.objectBodyName, exitProprio),   // v6e: OFF => {} (v5 option object); ON => {standOffWorld}
          ...objectRouter.exitRelease(completedController.skill.objectBodyName, exitProprio, { measure: () => quietEndingSample(completedController.skill.objectBodyName) }),   // v6g: OFF => {}; ON+declared => {releaseLift}
          ...objectRouter.exitHandClamp(completedController.skill.objectBodyName, exitProprio),   // v6h: OFF => {}; ON+declared => {holdHandClamp}
          ...objectRouter.exitHoldSource(completedController.skill.objectBodyName),   // v6j: OFF => {}; ON+declared => {holdSource: 'retreat_row0'}
        });
        skillController = boxExitController;
        recordedApproachHold = null; teacherStandingPlan = null;
        skillStep = boxExitController.step(skillProprio());
      } else if (completedController !== boxExitController && activeBoxTask !== null
          && (skillController === null || skillController === completedController)
          && (pendingCarryController === null || pendingCarryController === completedController)
          && recordedApproachHold === null) {
        // Ordinary terminal (carry finished/placement_missed/cancelled, or a pre-grasp
        // approach terminal such as step_limit / owned refusal): the outcome receipt was
        // published above, no box exit started, and no new parent/pending controller
        // took ownership in this control. Retire the task so the next ordinary
        // movement or request can be accepted. Failure reason is preserved in the log.
        retireBoxTaskOwnership(completedController === activeCarryController
          ? 'terminal_without_box_exit' : `pre_grasp_terminal:${skillStep.completionReason}`, completedController);
      }
    }
    const useApproachHold = recordedApproachHold !== null;
    if (useApproachHold) {
      // The parent observes arrival/facing once per actual control. Keep the
      // approved teacher stance while those samples accrue; advance only the
      // parent's approach clock after physics, never by synthetic samples.
      skillStep = { ...skillStep, mode: 'teacher', justEnteredTeacher: false,
        referenceFrames: recordedApproachHold.referenceFrames };
      user.humanGoalWorld = user.objGoalWorld = null;
      setTaskStatus(`Settling at the ${carriedObjectLabel()} before lifting…`);
    }
    if (teacherStandingPlan && (user.wasdActive || user.humanGoalWorld !== null || user.objGoalWorld !== null)) {
      teacherStandingPlan = null; translator.reset();
    }
    if (teacherStandingPlan) skillStep = { ...skillStep, mode: 'teacher', phase: 'teacher_standing',
      justEnteredTeacher: false, referenceFrames: [teacherStandingPlan.frame, teacherStandingPlan.frame] };
    let useTeacherDescentHold = false;
    if (teacherDescentHoldPending) {
      const pending = teacherDescentHoldPending; teacherDescentHoldPending = null;
      if (pending.atControl === episodeControlStep && teacherDescentHold === null && skillStep?.mode === 'teacher' && skillStep.phase === 'teacher'
          && skillController === activeCarryController && skillController.phase === 'teacher' && !skillController.finishRequested) {
        try {
          const live = skillProprio();
          const frame = planTeacherStandingReference(teacherObs.liveReferenceFrame(data), { alignment: 'original', rootPosition: live.rootPosWorld,
            rootQuaternion: live.rootQuatXyzwWorld, objectPosition: live.objPosWorld, objectQuaternion: live.objQuatXyzwWorld,
            objectPointsLocal: skillController.skill.objectPointsLocal }).frame;
          teacherDescentHold = new TeacherDescentContactHold({ parent: skillController, episode: episodeVersion, requestId: activeBoxTaskRequestId,
            physicalControl: episodeControlStep, segmentIndex: skillController.segmentIndex, referenceIndex: skillController.referenceIndex, frame, trigger: pending.trigger });
          console.log('[teacherDescentHold] start ' + JSON.stringify({ control: episodeControlStep, referenceIndex: skillController.referenceIndex, trigger: pending.trigger }));
        } catch (error) {
          teacherDescentHolds.push({ physicalControl: episodeControlStep, error: String(error?.message ?? error), trigger: pending.trigger });
          console.log('[teacherDescentHold] unavailable ' + JSON.stringify({ control: episodeControlStep, error: String(error?.message ?? error) }));
        }
      } else teacherDescentHolds.push({ physicalControl: episodeControlStep, skipped: 'owner_or_phase_changed_before_hold', trigger: pending.trigger });
    }
    if (teacherDescentHold) {
      const observed = teacherDescentHold.observe(teacherDescentHoldContext());
      if (observed.active && skillStep?.mode === 'teacher' && skillStep.phase === 'teacher' && skillController === activeCarryController) {
        useTeacherDescentHold = true; skillStep = { ...skillStep, referenceFrames: observed.referenceFrames };
      } else {
        const review = { ...teacherDescentHold.review(), endedAtControl: episodeControlStep, heldClock: skillController?.externalHoldSnapshot ?? null };
        teacherDescentHolds.push(review); if (teacherDescentHolds.length > 32) teacherDescentHolds.shift();
        console.log('[teacherDescentHold] end ' + JSON.stringify({ control: episodeControlStep, ended: review.ended, controls: review.controls }));
        teacherDescentHold = null;
      }
    }

    if(useFirstPickupStanding){
      user.humanGoalWorld=user.objGoalWorld=null;
      setTaskStatus('Settling at the grasp before continuing to your destination…');
    }
    // Hybrid keyboard mode: decide who owns a held key command before any reference is built.
    // Box tasks, floor clicks and suspensions always stay with the recorded supervisor.
    let hybridDecision = null;
    if (hybridArbiter && !restrictedSuspended) {
      const active = skillActive() || useFirstPickupStanding;
      hybridDecision = hybridArbiter.decide({ control: episodeControlStep, held: heldKeysFromUser(user),
        root: { pos: rootPosWorld, yaw: headingYawFromQuatXyzw(rootQuatXyzwWorld), velWorld: proprio.rootVelWorld },
        rects: boxCollisionBounds.read(data),
        restricted: { phase: restrictedController.phase, skillActive: active,
          initialStandingPending: lastRestrictedStep ? Boolean(lastRestrictedStep.initialStandingPending) : restrictedStartupStandingEnabled && episodeControlStep === 0,
          intentType: restrictedController.requestedIntent.type, suspended: restrictedSuspended } });
      if (hybridDecision.reanchor && !active) { restrictedController.reanchor(); restrictedPlan = null; }
      if (hybridDecision.enteredStudent) translator.reset();
      const fallbackView = fallbackKeyView(user, hybridDecision, hybridFallbackStepEnabled);
      if (fallbackView && !active) { restrictedController.requestKeys(fallbackView); restrictedPlan = null; }
      if (hybridDecision.owner !== 'recorded') lastRestrictedStep = null;
      if (hybridDecision.status && hybridDecision.status !== lastHybridStatus && !active) setTaskStatus(hybridDecision.status);
      lastHybridStatus = hybridDecision.status;
      lastHybridDecision = hybridDecision;
    }
    const hybridStudentOwns = hybridDecision !== null && hybridDecision.owner !== 'recorded' && !skillActive() && !useFirstPickupStanding;
    const useRestricted = restrictedMode && (!skillActive() || useFirstPickupStanding) && !hybridStudentOwns;
    if (useRestricted) {
      const wasInitialStandingPending = Boolean(lastRestrictedStep?.initialStandingPending);
      if (restrictedAfterBox) {
        restrictedController.reanchor(); restrictedAfterBox = false;
        teacherStandingPlan = null;
        const intent = restrictedController.requestedIntent;
        if (intent.type === 'floor') {
          // The completed carry changed both robot and obstacle locations.
          // Retain the requested destination and recompute its route here.
          restrictedPlan = planRestrictedFloorGoal(rootPosWorld, intent.goalWorld, boxCollisionBounds.read(data));
          if (restrictedPlan.supported) restrictedController.requestFloorGoal(restrictedPlan.finalGoalWorld,
            { waypoints: restrictedPlan.waypoints });
          else restrictedController.requestCancel();
        }
      }
      const q = data.xquat.slice(restrictedObs.objectId * 4, restrictedObs.objectId * 4 + 4);
      const priorRestrictedStep = lastRestrictedStep;
      lastRestrictedStep = restrictedController.step({ ...proprio,
        objectBodyName: restrictedController.skill.objectBodyName,
        objPosWorld: Array.from(data.xpos.slice(restrictedObs.objectId * 3, restrictedObs.objectId * 3 + 3)),
        objQuatXyzwWorld: [q[1], q[2], q[3], q[0]],
      });
      skillStep = lastRestrictedStep;
      if(useFirstPickupStanding&&skillStep.phase!=='teacher_standing'){
        firstPickupStanding.abort('restricted_standing_unavailable');restoreControlHistory();return;
      }
      const persistentContinuation = preservesPeriodicObservationHistory(priorRestrictedStep, skillStep);
      // A continuous periodic owner must retain the measured q(t-1)/qdot(t-1)
      // observation history across its verified successor seam. All other
      // entries keep the existing reset path.
      if (skillStep.justEnteredTeacher && !persistentContinuation) {
        restrictedObs.reset({ lastDofPos: previousDofPos, lastDofVel: previousDofVel });
      }
      if (restrictedPlan && !restrictedPlan.supported) setTaskStatus(restrictedPlan.reason === 'goal_distance'
        ? 'Choose a walking destination within 2 metres.' : 'That destination needs more clearance or a shorter route. Choose another point.');
      else if (skillStep.supported === false) setTaskStatus('That movement is not supported from here. Choose another direction or destination.');
      else if (skillStep.initialStandingPending) setTaskStatus('Settling into standing before starting…');
      else if (skillStep.pendingIntent) setTaskStatus('Finishing the current movement before changing direction…');
      else if (!taskCompletedThisControl && (skillStep.justEnteredTeacher || skillStep.justCompleted || wasInitialStandingPending)) setTaskStatus(
        skillStep.phase === 'teacher_standing' ? 'Standing. Choose a direction or destination.'
          : skillStep.phase === 'teacher_settling' ? 'Settling before the next movement…'
          : skillStep.phase === 'teacher_turn' ? 'Turning… Release to stop after this turn.'
          : 'Stepping… Release to stop after this step.');
    }
    if (stagedStudentTransport?.active) {
      const transition = stagedStudentTransport.observeParent(skillStep, { parent: skillController,
        episode: episodeVersion, physicalControl: episodeControlStep });
      if (!transition.active && !transition.requiresTeacherResume) transportTeacherResumePending = false;
    }
    if (skillStep?.mode === 'none') {
      restoreControlHistory();
      restrictedSuspended = skillStep.completionReason;
      lastRestrictedStep = skillStep;
      cancelPendingBoxTask('unsupported_motion'); discardQueuedBoxTask('unsupported_motion'); updateTaskControls();
      if (beginTransientSuspension(skillStep.completionReason)) return;
      setTaskStatus(skillStep.completionReason === 'lost_balance'
        ? 'The robot lost balance. Reset the scene to continue.'
        : skillStep.completionReason === 'unsettled'
          ? 'The robot could not settle into a supported stance. Reset the scene to continue.'
          : 'There is not enough room for a supported movement here. Reset the scene to continue.');
      setMode('PAUSED'); paused = true; return;
    }
    if (setdownClearanceDiagEnabled && skillController === activeCarryController && skillStep?.phase === 'teacher'
        && skillController.referenceIndex >= 330 && teacherObs?.objectId >= 0) {
      const objectId = teacherObs.objectId, box = Array.from(data.xpos.slice(objectId * 3, objectId * 3 + 3));
      const half = objectRouter.setdownClearanceExtents(skillController.skill.objectBodyName);   // OFF: v5 literal [.377, .367, .326]
      const clearance = name => {
        const id = findBodyIdByName(model, name); if (id < 0) return null;
        const p = Array.from(data.xpos.slice(id * 3, id * 3 + 3));
        const dx = Math.max(Math.abs(p[0] - box[0]) - half[0], 0), dy = Math.max(Math.abs(p[1] - box[1]) - half[1], 0);
        const dz = Math.max(Math.abs(p[2] - box[2]) - half[2], 0);
        return { pos: p, planarM: Math.hypot(dx, dy), gapM: Math.hypot(dx, dy, dz) };
      };
      const loads = contactDiagnostics ? contactDiagnostics.read(data, objectId) : null;
      setdownClearanceTrace.push({ physicalControl: episodeControlStep, referenceIndex: skillController.referenceIndex,
        segmentIndex: skillController.segmentIndex, rootZ: data.xpos[pelvisId * 3 + 2], boxPosWorld: box,
        rightKnee: clearance('right_knee_link'), leftKnee: clearance('left_knee_link'),
        legObjectNormalForceN: loads?.legObjectNormalForceN ?? null,
        handNormalForceN: loads ? [loads.leftHandObjectNormalForceN, loads.rightHandObjectNormalForceN] : null });
      if (setdownClearanceTrace.length > 400) setdownClearanceTrace.shift();
    }
    if (restrictedStudentTransportEnabled && skillController === activeCarryController
        && skillStep?.phase === 'teacher' && skillController.skill.studentTransportInterval) {
      const [startReferenceIndex, endReferenceIndex] = skillController.skill.studentTransportInterval;
      const atEntry = !stagedStudentTransport && !lastStudentTransportEntry && skillController.referenceIndex === startReferenceIndex;
      const windowActive = Boolean(stagedStudentTransport?.active
        && stagedStudentTransport.isOwnedBy({ parent: skillController, episode: episodeVersion }));
      // Solver hand loads at this control boundary, read only when a rule needs
      // them: the windowed admission history, the entry instant, or the
      // divergence exit. transportAdmission=instant reads only at the entry.
      const loads = transportAdmissionMode === 'window' || atEntry || (transportDivergenceLimits && windowActive)
        ? contactDiagnostics.read(data, teacherObs.objectId) : null;
      const handNormalForceN = loads ? [loads.leftHandObjectNormalForceN, loads.rightHandObjectNormalForceN] : null;
      if (transportAdmissionMode === 'window') {
        if (transportHandForceHistory?.parent !== skillController || transportHandForceHistory.episode !== episodeVersion
            || transportHandForceHistory.segmentIndex !== skillController.segmentIndex
            || transportHandForceHistory.referenceIndex >= skillController.referenceIndex) {
          transportHandForceHistory = { parent: skillController, episode: episodeVersion,
            segmentIndex: skillController.segmentIndex, referenceIndex: -1, samples: [] };
        }
        transportHandForceHistory.referenceIndex = skillController.referenceIndex;
        transportHandForceHistory.samples.push(handNormalForceN);
        if (transportHandForceHistory.samples.length > transportAdmission.windowControls) transportHandForceHistory.samples.shift();
      }
      if (atEntry) {
        const live = skillProprio();
        const attempt = StagedStudentTransportController.tryStart({ parent: skillController,
          live: { ...live, handNormalForceN, ...(transportAdmissionMode === 'window'
            ? { handNormalForceHistoryN: transportHandForceHistory.samples.map(pair => Array.from(pair)) } : {}) },
          episode: episodeVersion, physicalControl: episodeControlStep,
          startReferenceIndex, endReferenceIndex,
          expectedSourceControls: hasPreparedStudentLiftProfile(skillController.skill)
            ? STUDENT_LIFT_PROFILE.preparedSourceControls : 456,
          admission: transportAdmission, divergenceLimits: transportDivergenceLimits });
        lastStudentTransportEntry = { episode: episodeVersion, physicalControl: episodeControlStep,
          segmentIndex: skillController.segmentIndex, supported: attempt.supported, reason: attempt.reason,
          admission: { ...transportAdmission }, measured: structuredClone(attempt.measured ?? attempt.entry?.measured ?? null) };
        if (attempt.supported) {
          stagedStudentTransport = attempt.controller; studentTransports.push(stagedStudentTransport);
          if (studentTransports.length > 32) studentTransports.shift();
          setTaskStatus(`Carrying the ${carriedObjectLabel()} toward your destination…`);
        }
      }
      if (transportDivergenceLimits && windowActive) {
        // Ends the window at this unchanged reference index; the teacher resumes
        // from here with a fresh observation history, exactly as at 330.
        const check = stagedStudentTransport.checkDivergence({ ...skillProprio(), handNormalForceN },
          { parent: skillController, episode: episodeVersion, physicalControl: episodeControlStep });
        if (!check.active) {
          transportTeacherResumePending = true;
          setTaskStatus(`Preparing to complete the carry and set the ${carriedObjectLabel()} down…`);
        }
      }
      if (transportTeacherResumePending && stagedStudentTransport?.isOwnedBy({ parent: skillController, episode: episodeVersion })) {
        teacherObs.reset({ lastDofPos: previousDofPos, lastDofVel: previousDofVel });
        transportTeacherResumePending = false;
      }
    }
    const useStagedStudentTransport = Boolean(stagedStudentTransport?.active
      && stagedStudentTransport.isOwnedBy({ parent: skillController, episode: episodeVersion })
      && skillStep?.phase === 'teacher');

    const referenceApproachContext={owner:skillController,parent:pendingWaypointCarryController,
      episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep};
    if(referenceStudentTurn?.active){
      const transition=referenceStudentTurn.observe(skillStep,referenceApproachContext);
      if(transition.requiresTeacherResume)referenceTeacherResumePending=true;
    }
    if(heightAwareApproach?.active)heightAwareApproach.observe(skillStep,referenceApproachContext);
    if(referenceStudentTurnsEnabled && !referenceStudentTurn?.active && approvedReferenceTurn
        && approvedReferenceTurn.owner===skillController && approvedReferenceTurn.parent===pendingWaypointCarryController
        && approvedReferenceTurn.physicalControl===episodeControlStep){
      const parent=pendingWaypointCarryController;
      const used=parent?usedReferenceStudentTurns.get(parent):null;
      const attempt=ReferenceStudentTurnController.tryStart({...referenceApproachContext,...approvedReferenceTurn,
        live:skillProprio(),alreadyUsed:Boolean(used?.has(parent?.segmentIndex))});
      if(attempt.supported){
        referenceStudentTurn=attempt.controller;referenceStudentTurns.push(referenceStudentTurn);
        if(referenceStudentTurns.length>32)referenceStudentTurns.shift();
        const segments=used??new Set();segments.add(parent.segmentIndex);usedReferenceStudentTurns.set(parent,segments);
      }
    }
    if(referenceTeacherResumePending && referenceStudentTurn?.isOwnedBy(referenceApproachContext)){
      teacherObs.reset({lastDofPos:previousDofPos,lastDofVel:previousDofVel});referenceTeacherResumePending=false;
    }
    const useReferenceStudentTurn=Boolean(referenceStudentTurn?.active);
    const useHeightAwareApproach=Boolean(heightAwareApproach?.active);
    const privateTurn=terminalFacingProbe?.active?terminalFacingProbe:null;
    const privateTurnBodyHistory=privateTurn?Float32Array.from(bodyObsBuilder.historyBuf):null;
    const privateTurnCurrent=()=>!privateTurn||(!privateTurn.invalidReason(terminalFacingContext())
      &&(!privateTurn.builder||privateTurn.builder===teacherObs));
    const abortPrivateTurn=(reason,row=null)=>{
      if(privateTurn?.active){if(row)privateTurn.reject(row,reason);else privateTurn.finish(reason);}
      // Never roll back across Reset, an advanced physical clock or a partial real control.
      if(stepEpisodeVersion!==episodeVersion||stepPhysicalControl!==episodeControlStep||(row?.actualSubsteps??0)>0)return;
      // Preserve a replacement owner's same-clock buffers if they changed after sampling.
      if(privateTurnBodyHistory?.every((v,i)=>v===bodyObsBuilder.historyBuf[i]))restoreControlHistory();
      const context=terminalFacingContext();
      if(context.owner===privateTurn.owner&&context.parent===privateTurn.parent
          &&context.requestId===privateTurn.requestId){restrictedSuspended=reason;if(!beginTransientSuspension(reason)){paused=true;setMode('PAUSED');}}
    };
    if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_before_inference');return;}
    const pickupFacingOwner = skillController instanceof PickupFacingApproachController ? skillController : null;
    if (pickupFacingOwner && !pickupFacingOwner.isOwnedBy(pickupFacingContext())) {
      restoreControlHistory(); restrictedSuspended = 'pickup_approach_owner_changed';
      paused = true; setMode('PAUSED'); return;
    }
    const noResetPoseOwner = skillController instanceof NoResetApproachPoseController ? skillController : null;
    if (noResetPoseOwner && !noResetPoseOwner.isOwnedBy(noResetApproachContext())) {
      restoreControlHistory(); restrictedSuspended = 'no_reset_approach_owner_or_clock_changed';
      paused = true; setMode('PAUSED'); return;
    }
    const useTeacher = skillStep?.mode === 'teacher' && !useStagedStudentTransport && !useReferenceStudentTurn;
    const useRecoveryCycle = useTeacher && skillController instanceof ApproachRecoveryCycle;
    const usePreviewOwnedStanding = Boolean(useTeacher && controlPreview && standingPreviewOwned && standingPreviewOwned.episode === episodeVersion
      && ['teacher_standing', 'teacher_settling'].includes(skillStep?.phase));
    const useBoxExit = boxExitController !== null && skillController === boxExitController && !useRestricted && useTeacher;
    const useTeacherDescentCarry = Boolean(teacherDescentHoldEnabled && useTeacher && !useTeacherDescentHold && teacherDescentHold === null
      && skillController === activeCarryController && skillStep?.phase === 'teacher' && !useBoxExit && !useRecoveryCycle && !usePreviewOwnedStanding
      && !useApproachHold && !teacherStandingPlan);
    if (useBoxExit && skillStep.justEnteredTeacher) {
      teacherObs.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, boxExitController.skill);
    }
    if (skillStep?.justEnteredTeacher && !useRestricted) {
      teacherObs.reset({ lastDofPos: previousDofPos, lastDofVel: previousDofVel });
      user.humanGoalWorld = user.objGoalWorld = null;
      user.releaseKeys();
      setTaskStatus(useRecoveryCycle ? skillController.statusMessage : useBoxExit ? (pendingSegmentCarryController
        ? `Carry ${pendingSegmentCarryController.segmentIndex + 1} of ${pendingSegmentCarryController.plan.goals.length} placed. Stepping clear before continuing to your destination…`
        : skillStep.phase === 'teacher_exit_release' ? `${capitalize(carriedObjectLabel())} set down. Releasing the hands before stepping clear…`
        : skillStep.phase === 'teacher_exit_hold' ? `${capitalize(carriedObjectLabel())} set down. Preparing to step clear…`
        : skillStep.phase === 'teacher_exit_retreat' ? `Stepping clear of the ${carriedObjectLabel()}…` : 'Settling after stepping clear…')
        : skillStep.phase === 'teacher_settling' ? 'Settling before the next movement…'
        : skillStep.phase === 'teacher_step' ? `Stepping around the ${carriedObjectLabel()}…`
        : skillStep.phase === 'teacher_turn' ? `Turning to face the ${carriedObjectLabel()}…` : skillStep.segmentCount > 1
        ? `Carrying the box: ${skillStep.segmentIndex + 1} of ${skillStep.segmentCount}…`
        : skillController === activeCarryController ? 'Carrying the box toward your destination…'
        : BOX_TASKS[activeBoxTask].executionMessage);
    }
    const tr = useTeacher ? null : useReferenceStudentTurn
      ? {fsmState:FsmState.LOCO,goalSpec:null,mask:null,debug:{referenceStudentTurn:true}}
      : useStagedStudentTransport
      ? { fsmState: FsmState.HOI_FULL, goalSpec: null, mask: null, debug: { stagedTransport: true } }
      : translator.step({ user: hybridStudentOwns ? studentUserView(user, hybridDecision) : user, proprio });
    if (skillStep?.phase === 'approach' && tr?.fsmState === FsmState.LOCO && lastApproachRoute?.phase !== 'staging') {
      // Walk toward the pregrasp point while retaining the live heading.
      // A matched clip's bending/turning target can stall this short approach.
      tr.goalSpec.humanTargetRot = new Float32Array([1, 0, 0, 0, 1, 0]);
    }
    let stagedStudentObservation = null;
    if(useReferenceStudentTurn){
      const encoded=referenceStudentTurn.sample(skillProprio(),referenceApproachContext);
      tr.goalSpec=encoded.goalSpec;tr.mask={keepHumanPos:1,keepHumanRot:1,keepObjPos:0,keepObjPoints:0,keepTime:1};
      stagedStudentObservation=referenceStudentTurn.buildObservation(encoded,bodyObs);
    }
    if (useStagedStudentApproach) {
      const encoded = stagedStudentApproach.sample(skillProprio(), firstPickupStudentContext());
      tr.goalSpec = encoded.goalSpec;
      tr.mask = { keepHumanPos: encoded.mask.keepHuman, keepHumanRot: encoded.mask.keepHuman,
        keepObjPos: encoded.mask.keepObj, keepObjPoints: encoded.mask.keepObjPoints, keepTime: 1 };
      stagedStudentObservation = stagedStudentApproach.buildObservation(encoded, bodyObs);
    }
    if (useStagedStudentTransport) {
      const live = skillProprio();
      const encoded = stagedStudentTransport.sample(live, { parent: skillController,
        episode: episodeVersion, physicalControl: episodeControlStep });
      tr.goalSpec = encoded.goalSpec;
      tr.mask = { keepHumanPos: encoded.mask.keepHuman, keepHumanRot: encoded.mask.keepHuman,
        keepObjPos: encoded.mask.keepObj, keepObjPoints: encoded.mask.keepObjPoints, keepTime: 1 };
      // A queued cancel may clear the mouse selection while the owned carry
      // must still finish placement. Perception follows that physical box.
      const name = skillController.skill.objectBodyName;
      if (!ownedStudentObjectSelections.has(name)) ownedStudentObjectSelections.set(name,
        createObjectSelection(name, findBodyIdByName(model, name), pointCloudDb));
      const pointsHeading = computeObjPointsHeadingFrame(ownedStudentObjectSelections.get(name).pointsLocal,
        live.objPosWorld, live.objQuatXyzwWorld,
        rootPosWorld, rootQuatXyzwWorld);
      stagedStudentObservation = stagedStudentTransport.buildObservation(encoded, bodyObs, pointsHeading);
    }

    // 3c. Update the goal-viz markers (green sphere + heading arrow at
    //     the human target; orange sphere + connector at the object
    //     target). Opacities track the per-channel mask so an inactive
    //     goal field fades out smoothly. Toggle with G.
    if (restrictedMode && useTeacher) {
      const requested = restrictedController.requestedIntent;
      goalViz.updateRecorded({ humanGoal: requested.type === 'floor' ? requested.goalWorld : null,
        referenceFrame: skillStep.referenceFrames[1],
        objectGoal: queuedBoxTask?.requestedGoal ?? (skillActive() ? taskDestinationWorld : null),
        objectPosition: skillProprio().objPosWorld });
    } else {
      goalViz.group.visible = goalViz.visible && !useTeacher;
      goalViz.update(tr, proprio, user);
    }

    updateMatchedCarryMarker();
    const inferenceTeacherBuilder = useTeacher ? (useRestricted || useApproachHold ? restrictedObs : teacherObs) : null;
    // Phase transitions may create/reset a builder after the body-history
    // snapshot. Roll back only the actual observation build on its live owner.
    if (controlHistory && inferenceTeacherBuilder && !stagedStudentObservation) {
      queriedTeacherHistory = { builder: inferenceTeacherBuilder, saved: teacherHistory(inferenceTeacherBuilder),
        locomotion: useRestricted || useApproachHold };
    }
    let obs = stagedStudentObservation ?? (useTeacher ? inferenceTeacherBuilder.build(data, skillStep.referenceFrames, lastAction, lastTorque) : buildObs(
      user,
      rootPosWorld, rootQuatXyzwWorld,
      objPosWorld, objQuatXyzwWorld,
      objPointsObjLocal,
      bodyObs,
      tr,
    ));

    // 4. Policy inference. Use zeros for deterministic mode (F1), otherwise
    // an episode-held latent sampled by Space. No per-step latent sampling.
    const noise = (user.deterministic || user.vaeNoise === null)
      ? zeroNoise : user.vaeNoise;
    const inferenceStudentApproach = useStagedStudentApproach ? stagedStudentApproach : null;
    const inferenceStudentTransport = useStagedStudentTransport ? stagedStudentTransport : null;

    const inferenceReferenceStudentTurn=useReferenceStudentTurn?referenceStudentTurn:null;
    const inferenceHeightAwareApproach=useHeightAwareApproach?heightAwareApproach:null;
    const referenceApproachInput=(inferenceReferenceStudentTurn||inferenceHeightAwareApproach)&&recordReferenceApproachInputs
      ?{observation:Array.from(obs),observationFloat32Bytes:Array.from(new Uint8Array(obs.buffer,obs.byteOffset,obs.byteLength))}:{};
    const pickupFacingInference = pickupFacingOwner ? { controls: pickupFacingOwner.controls,
      referenceIndex: pickupFacingOwner.referenceIndex, physicalControl: episodeControlStep } : null;
    const noResetPoseInference = noResetPoseOwner ? { controls: noResetPoseOwner.controls,
      referenceIndex: noResetPoseOwner.referenceIndex, physicalControl: episodeControlStep } : null;
    if (pickupFacingOwner && (!useTeacher || useReferenceStudentTurn || useStagedStudentTransport)) {
      restoreControlHistory();
      throw new Error('Pickup-facing approach must retain its teacher action owner');
    }
    if (noResetPoseOwner && (!useTeacher || useReferenceStudentTurn || useStagedStudentTransport
        || useStagedStudentApproach || inferenceTeacherBuilder !== teacherObs)) {
      restoreControlHistory();
      throw new Error('No-reset approach must retain its recorded teacher action owner');
    }
    let privateTurnCandidate=null;
    if(privateTurn){
      if(!useTeacher||useReferenceStudentTurn||useStagedStudentTransport||useStagedStudentApproach
          ||inferenceTeacherBuilder!==teacherObs){abortPrivateTurn('turn_actor_owner_changed');return;}
      try{privateTurnCandidate=privateTurn.begin(terminalFacingContext(),{
        teacherObservation:Array.from(obs),previousAction:Array.from(lastAction),previousTorque:Array.from(lastTorque),
        previousDofPos:Array.from(previousDofPos),previousDofVel:Array.from(previousDofVel),
        cachedBodyObservation:Array.from(bodyObs),rootPositionWorld:Array.from(rootPosWorld),
        rootQuaternionWorld:Array.from(rootQuatXyzwWorld),rawAction:null,preview:null});
      }catch(error){abortPrivateTurn('turn_input_ownership_error');throw error;}
    }

    const inferenceFirstPickupStanding=useFirstPickupStanding?firstPickupStanding:null;
    let firstPickupStandingCandidate=null;
    if(inferenceFirstPickupStanding){
      if(!useTeacher||!useRestricted||inferenceTeacherBuilder!==restrictedObs)
        throw Error('Owned settling must use the existing restricted teacher-standing builder');
      firstPickupStandingCandidate=inferenceFirstPickupStanding.begin(firstPickupStandingContext(),{
        referenceFrames:skillStep.referenceFrames,observation:obs,rootPositionWorld:rootPosWorld});
    }
    let mu;
    try {
      mu = useTeacher ? await teacherPolicy.infer(obs) : await policy.infer(obs, noise);
    } catch (error) {

      if(inferenceFirstPickupStanding){inferenceFirstPickupStanding.abort('standing_inference_error',firstPickupStandingCandidate);restoreControlHistory();}
      if(privateTurn){abortPrivateTurn('turn_inference_error',privateTurnCandidate);throw error;}
      if (pickupFacingOwner) {
        if (stepEpisodeVersion !== episodeVersion || stepPhysicalControl !== episodeControlStep) return;
        restoreControlHistory();
      }
      if (noResetPoseOwner) {
        if (stepEpisodeVersion !== episodeVersion || stepPhysicalControl !== episodeControlStep) return;
        restoreControlHistory();
      }
      throw error;
    }
    // Keyboard reset can occur while inference yields to the event loop.
    // Never apply an action computed from the previous episode's state.
    if(privateTurnCandidate)privateTurnCandidate.rawAction=mu?Array.from(mu):null;

    if(firstPickupStandingCandidate)firstPickupStandingCandidate.rawAction=mu?Array.from(mu):null;
    if(inferenceFirstPickupStanding&&(!inferenceFirstPickupStanding.isOwnedBy(firstPickupStandingContext())
        ||inferenceFirstPickupStanding!==firstPickupStanding||stepPhysicalControl!==episodeControlStep)){
      inferenceFirstPickupStanding.abort('standing_owner_changed_during_inference',firstPickupStandingCandidate);
      restoreControlHistory();return;
    }
    if (stepEpisodeVersion !== episodeVersion) {
      if(privateTurnCandidate)privateTurnCandidate.discardedReason='episode_reset_during_inference';return;
    }
    if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_during_inference',privateTurnCandidate);return;}
    if (pickupFacingOwner && (!pickupFacingOwner.isOwnedBy(pickupFacingContext())
        || inferenceTeacherBuilder !== (useRestricted || useApproachHold ? restrictedObs : teacherObs)
        || pickupFacingOwner.controls !== pickupFacingInference.controls
        || pickupFacingOwner.referenceIndex !== pickupFacingInference.referenceIndex
        || episodeControlStep !== pickupFacingInference.physicalControl)) {
      restoreControlHistory(); restrictedSuspended = 'pickup_approach_owner_or_clock_changed';
      paused = true; setMode('PAUSED'); return;
    }
    if (noResetPoseOwner && (!noResetPoseOwner.isOwnedBy(noResetApproachContext())
        || noResetPoseOwner.controls !== noResetPoseInference.controls
        || noResetPoseOwner.referenceIndex !== noResetPoseInference.referenceIndex
        || episodeControlStep !== noResetPoseInference.physicalControl)) {
      restoreControlHistory(); restrictedSuspended = 'no_reset_approach_owner_or_clock_changed';
      paused = true; setMode('PAUSED'); return;
    }
    const currentReferenceContext={owner:skillController,parent:pendingWaypointCarryController,
      episode:episodeVersion,requestId:activeBoxTaskRequestId,physicalControl:episodeControlStep};
    if(inferenceReferenceStudentTurn && (inferenceReferenceStudentTurn!==referenceStudentTurn
        ||!inferenceReferenceStudentTurn.isOwnedBy(currentReferenceContext)||skillController.cancelRequested
        ||pendingWaypointCarryController?.finishRequested)){
      const ended=inferenceReferenceStudentTurn.cancel('command_changed_during_inference',currentReferenceContext);
      referenceTeacherResumePending=ended.requiresTeacherResume;restoreControlHistory();return;
    }
    if(inferenceHeightAwareApproach && (inferenceHeightAwareApproach!==heightAwareApproach
        ||!inferenceHeightAwareApproach.isOwnedBy(currentReferenceContext))){
      inferenceHeightAwareApproach.cancel('owner_changed_during_inference');restoreControlHistory();return;
    }
    if (inferenceStudentTransport && (inferenceStudentTransport !== stagedStudentTransport
        || !inferenceStudentTransport.isOwnedBy({ parent: skillController, episode: episodeVersion })
        || skillController.phase !== 'teacher')) {
      inferenceStudentTransport.cancel('transport_owner_changed_during_inference', { parent: skillController,
        episode: episodeVersion, physicalControl: episodeControlStep });
      restoreControlHistory(); return;
    }
    // A new command can cancel this finite approach while student inference
    // yields. Its unexecuted action must not consume history or physical time.
    if (inferenceStudentApproach && (inferenceStudentApproach !== stagedStudentApproach
        || !inferenceStudentApproach.isOwnedBy(firstPickupStudentContext())
        || skillController.phase !== 'approach' || skillController.finishRequested)) {
      inferenceStudentApproach.cancel('command_changed_during_inference', firstPickupStudentContext());
      restoreControlHistory();
      return;
    }
    if (!mu || mu.length !== ACTION_DIM || typeof mu.every !== 'function' || !mu.every(Number.isFinite)) {

      if(inferenceFirstPickupStanding){
        inferenceFirstPickupStanding.abort('standing_invalid_action',firstPickupStandingCandidate);
        restoreControlHistory();
      }
      if (pickupFacingOwner) restoreControlHistory();
      if(privateTurn)abortPrivateTurn('turn_invalid_action',privateTurnCandidate);
      throw new Error('Policy returned invalid actions; simulation paused');
    }
    // 5. Compute PD target: raw target = ACTION_SCALE * clamp(mu, -1, 1).
    //    NO default-pose offset; matches training (humanoid.py:118
    //    `_initial_dof_pos = 0`) and the proven sim2sim deploy
    //    (intermimic/sim2sim_vae.py:1272).
    //    Optional EMA smoothing: target_q[i] = α·raw + (1-α)·target_q[i].
    //    α=1.0 → pass-through (no smoothing, matches training exactly).
    const oneMinusAlpha = 1.0 - smoothingAlpha;
    const candidateTarget = new Float32Array(ACTION_DIM), candidateAction = new Float32Array(ACTION_DIM);
    for (let i = 0; i < ACTION_DIM; i++) {
      let m = mu[i];
      if (m > 1.0) m = 1.0;
      else if (m < -1.0) m = -1.0;
      const raw = ACTION_SCALE * m;
      candidateTarget[i] = smoothingAlpha * raw + oneMinusAlpha * targetQ[i];
      candidateAction[i] = m;
    }
    const useApproachRecovery = skillStep?.approachRecovery === true
      && skillController instanceof TeacherApproachRecoveryController;
    const useRecoveredApproachHold = useApproachHold && recordedApproachHold.requiresPreview
      && recordedApproachHold.episode === episodeVersion && recordedApproachHold.parent === skillController;
    if (shouldRunOutcomeLaneControlPreview(useBoxExit || useApproachRecovery || useRecoveredApproachHold || useStagedStudentApproach || useReferenceStudentTurn || useHeightAwareApproach || pickupFacingOwner || privateTurn || useFirstPickupStanding || useRecoveryCycle || usePreviewOwnedStanding || useTeacherDescentCarry || useTeacherDescentHold, noResetPoseOwner, lastLargeboxPushLiveDiagnostic)) {
      // Preview this already-inferred action using exactly the same PD targets
      // and physics substeps. No policy history or real action is committed
      // when the predicted movement contacts a box or loses balance.
      if(privateTurn){
        try{lastControlPreview=controlPreview.evaluate(data,candidateTarget);}
        catch(error){abortPrivateTurn('turn_preview_error',privateTurnCandidate);throw error;}
        privateTurnCandidate.preview=structuredClone(lastControlPreview);
        if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_during_preview',privateTurnCandidate);return;}
        if(!lastControlPreview.supported)privateTurn.reject(privateTurnCandidate,'physical_preview_refused');
      }
      else if (useTeacherDescentCarry || useTeacherDescentHold) {
        const objectBodyName = skillController.skill.objectBodyName;
        const allowSupport = item => item.objectName === objectBodyName && STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.includes(item.bodyName);
        const samples = [];
        const geometry = useTeacherDescentCarry ? teacherDescentBoxGeometry(objectBodyName) : null;
        lastControlPreview = controlPreview.evaluate(data, candidateTarget, { allowContact: allowSupport, onSubstepData: geometry ? scratch => {
          const g = geometry.geom, gp = [scratch.geom_xpos[g * 3], scratch.geom_xpos[g * 3 + 1], scratch.geom_xpos[g * 3 + 2]], gm = Array.from(scratch.geom_xmat.slice(g * 9, g * 9 + 9));
          for (const t of geometry.triggers) samples.push({ body: t.name, metricM: originToBoxSurfaceM([scratch.xpos[t.id * 3], scratch.xpos[t.id * 3 + 1], scratch.xpos[t.id * 3 + 2]], gp, gm, geometry.center, geometry.half) });
        } : null });
        previewControls++;
        if (useTeacherDescentHold && !lastControlPreview.supported) {
          // The hold action itself previews hard-unsafe: end the hold (ownership stays with the carry parent); this control
          // takes the unchanged preview-refusal path below because no safe action is available in it.
          teacherDescentHold.refuseUnsafeHold(lastControlPreview, teacherDescentHoldContext());
        }
        if (useTeacherDescentCarry && lastControlPreview.supported) {
          const decision = holdOnsetDecision(samples);
          if (decision.trigger) {
            const trigger = { ...decision, previewedControl: episodeControlStep, referenceIndex: skillController.referenceIndex };
            const used = teacherDescentHoldUsed.get(skillController) ?? new Set();
            if (used.has(skillController.segmentIndex)) {
              teacherDescentHolds.push({ physicalControl: episodeControlStep, skipped: 'one_hold_per_segment', trigger });
              console.log('[teacherDescentHold] skipped one_hold_per_segment ' + JSON.stringify(trigger));
            } else {
              used.add(skillController.segmentIndex); teacherDescentHoldUsed.set(skillController, used);
              // Same-control replacement, as in the offline study: the hard-safe clip action is NOT executed; roll back this
              // control's observation histories, build the live-FK hold frame from the actual state, re-query the teacher for the
              // held reference (tagged for the evaluator as a replacement query) and hard-preview the hold action.
              restoreControlHistory();
              const live = skillProprio();
              const frame = planTeacherStandingReference(teacherObs.liveReferenceFrame(data), { alignment: 'original', rootPosition: live.rootPosWorld,
                rootQuaternion: live.rootQuatXyzwWorld, objectPosition: live.objPosWorld, objectQuaternion: live.objQuatXyzwWorld,
                objectPointsLocal: skillController.skill.objectPointsLocal }).frame;
              const hold = new TeacherDescentContactHold({ parent: skillController, episode: episodeVersion, requestId: activeBoxTaskRequestId,
                physicalControl: episodeControlStep, segmentIndex: skillController.segmentIndex, referenceIndex: skillController.referenceIndex, frame, trigger });
              let holdMu;
              globalThis.__teacherDescentHoldReplacement = true;
              try { obs = inferenceTeacherBuilder.build(data, hold.referenceFrames, lastAction, lastTorque); holdMu = await teacherPolicy.infer(obs); }
              finally { globalThis.__teacherDescentHoldReplacement = false; }
              if (stepEpisodeVersion !== episodeVersion || stepPhysicalControl !== episodeControlStep || skillController !== activeCarryController
                  || skillController.phase !== 'teacher' || skillController.finishRequested) { restoreControlHistory(); return; }
              if (!holdMu || holdMu.length !== ACTION_DIM || !Array.from(holdMu).every(Number.isFinite)) throw new Error('Policy returned invalid actions; simulation paused');
              for (let i = 0; i < ACTION_DIM; i++) {
                let m = holdMu[i]; if (m > 1.0) m = 1.0; else if (m < -1.0) m = -1.0;
                candidateTarget[i] = smoothingAlpha * (ACTION_SCALE * m) + oneMinusAlpha * targetQ[i]; candidateAction[i] = m;
              }
              mu = holdMu;
              lastControlPreview = controlPreview.evaluate(data, candidateTarget, { allowContact: allowSupport });
              if (lastControlPreview.supported) {
                teacherDescentHold = hold; useTeacherDescentHold = true; skillStep = { ...skillStep, referenceFrames: hold.referenceFrames };
                console.log('[teacherDescentHold] start ' + JSON.stringify({ control: episodeControlStep, referenceIndex: skillController.referenceIndex, trigger }));
              } else {
                hold.refuseUnsafeHold(lastControlPreview, teacherDescentHoldContext());
                teacherDescentHolds.push({ ...hold.review(), endedAtControl: episodeControlStep });
                console.log('[teacherDescentHold] hold_action_previewed_unsafe ' + JSON.stringify({ control: episodeControlStep, reason: lastControlPreview.reason,
                  contacts: (lastControlPreview.unwantedContacts ?? []).slice(0, 3).map(c => [c.bodyName, +Number(c.normalForceN ?? 0).toFixed(2)]) }));
              }
            }
          }
        }
      }
      else {
        try{lastControlPreview = controlPreview.evaluate(data, candidateTarget);}
        catch(error){
          if(inferenceFirstPickupStanding){
            inferenceFirstPickupStanding.abort('standing_preview_error',firstPickupStandingCandidate);
            restoreControlHistory();
          }
          throw error;
        }
      }
      previewControls++;
      if(firstPickupStandingCandidate){
        firstPickupStandingCandidate.preview=structuredClone(lastControlPreview);
        if(!inferenceFirstPickupStanding.isOwnedBy(firstPickupStandingContext())){
          inferenceFirstPickupStanding.abort('standing_owner_changed_during_preview',firstPickupStandingCandidate);
          restoreControlHistory();return;
        }
        if(!lastControlPreview.supported)
          inferenceFirstPickupStanding.abort('standing_preview_refused',firstPickupStandingCandidate);
      }
      if (useStagedStudentApproach && studentClosedLoopPreview && lastControlPreview.supported) {
        const liveUpright = 1 - 2 * (rootQuatXyzwWorld[0] ** 2 + rootQuatXyzwWorld[1] ** 2);
        const trigger = { rootHeightM: rootPosWorld[2], upright: liveUpright,
          crossed: rootPosWorld[2] < STAGED_STUDENT_APPROACH_LIMITS.minRootHeightM || liveUpright < STAGED_STUDENT_APPROACH_LIMITS.minUpright };
        if (trigger.crossed) {
          const context = firstPickupStudentContext();
          const report = await studentClosedLoopPreview.run({ liveData: data, liveBodyObsBuilder: bodyObsBuilder, window: stagedStudentApproach,
            buildObservation: (encoded, body) => stagedStudentApproach.buildObservation(encoded, body),
            candidateTarget, candidateAction, lastAction, targetQ, smoothingAlpha, noise, episode: episodeVersion,
            physicalControl: episodeControlStep, objectBodyId: findBodyIdByName(model, skillController.skill.objectBodyName), trigger });
          studentClosedLoopPreviews.push({ episode: episodeVersion, physicalControl: episodeControlStep, ...report });
          if (studentClosedLoopPreviews.length > 64) studentClosedLoopPreviews.shift();
          // The lookahead awaited policy queries: the owned command must be unchanged before it may act.
          const ownerCurrent = inferenceStudentApproach === stagedStudentApproach && inferenceStudentApproach.isOwnedBy(firstPickupStudentContext())
            && skillController.phase === 'approach' && !skillController.finishRequested
            && stepEpisodeVersion === episodeVersion && stepPhysicalControl === episodeControlStep;
          const decision = decideClosedLoopCommit({ report, oneStepPreview: lastControlPreview, ownerCurrent });
          console.log('[studentClosedLoopPreview] ' + JSON.stringify({ control: episodeControlStep, status: report.status, coverage: report.coverage,
            decision: decision.action, decisionReason: decision.reason ?? decision.coverage ?? null, refused: report.refused, reason: report.reason,
            at: report.refusedAtControl, expiredAt: report.expiredAtControl, evaluated: report.evaluatedControls, minZ: report.minRootHeightM,
            minUp: report.minUpright, queries: report.policyQueries, ms: Math.round(report.elapsedMs), error: report.error ?? null }));
          if (decision.action === 'cancel') {
            inferenceStudentApproach.cancel(decision.reason, firstPickupStudentContext());
            restoreControlHistory();
            return;
          }
          // 'refuse' (physical prediction OR unavailable/incomplete forecast) takes the existing preview-refusal fallback below;
          // an unknown forecast never commits the candidate. 'commit' keeps the one-step result unchanged.
          if (decision.action === 'refuse') lastControlPreview = decision.preview;
        }
      }
      if (!lastControlPreview.supported) {

        restoreControlHistory();
        if(useReferenceStudentTurn){
          referenceStudentTurn.requestPreviewFallback({...referenceApproachContext,
            record:{...referenceApproachInput,rawAction:Array.from(mu),preview:lastControlPreview}});
          referenceTeacherResumePending=true;return;
        }
        if(useHeightAwareApproach)heightAwareApproach.refuse({...referenceApproachContext,
          record:{...referenceApproachInput,rawAction:Array.from(mu),preview:lastControlPreview}});
        if (useStagedStudentApproach) {
          stagedStudentApproach.requestPreviewFallback({ ...firstPickupStudentContext(),
            record: { command: Array.from(obs.slice(0, 13)), rawAction: Array.from(mu),
              rootPositionWorld: Array.from(rootPosWorld), preview: lastControlPreview } });
          boxApproachPlanner?.reset(); lastApproachRoute = null;
          setTaskStatus('Adjusting the approach before continuing to your destination…');
          return;
        }
        if (useRecoveryCycle) skillController.refusePreview(lastControlPreview, episodeControlStep);
        restrictedSuspended = lastControlPreview.reason;
        lastRestrictedStep = { ...skillStep, mode: 'none', supported: false,
          completionReason: restrictedSuspended, referenceFrames: null };
        cancelPendingBoxTask('control_preview_refused'); discardQueuedBoxTask('control_preview_refused'); updateTaskControls();
        if (beginTransientSuspension(useRecoveryCycle ? 'recovery_preview_refused' : lastControlPreview.reason)) return;
        setTaskStatus(useBoxExit ? 'The robot cannot step clear safely from here. Reset the scene to continue.'
          : 'The robot could not settle safely near the box. Reset the scene to continue.');
        setMode('PAUSED'); paused = true; return;
      }
    }
    if (useStagedStudentTransport) {
      // The reviewed loaded window supports the selected box with both hands
      // and distal wrist pitch/yaw links. Other robot/object contacts are refused.
      lastControlPreview = controlPreview.evaluate(data, candidateTarget, {
        allowContact: item => item.objectName === skillController.skill.objectBodyName
          && STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.includes(item.bodyName),
      });
      previewControls++;
      if(firstPickupStandingCandidate){
        firstPickupStandingCandidate.preview=structuredClone(lastControlPreview);
        if(!inferenceFirstPickupStanding.isOwnedBy(firstPickupStandingContext())){
          inferenceFirstPickupStanding.abort('standing_owner_changed_during_preview',firstPickupStandingCandidate);
          restoreControlHistory();return;
        }
        if(!lastControlPreview.supported)
          inferenceFirstPickupStanding.abort('standing_preview_refused',firstPickupStandingCandidate);
      }
      if (!lastControlPreview.supported) {
        restoreControlHistory();
        stagedStudentTransport.requestPreviewFallback({ parent: skillController,
          episode: episodeVersion, physicalControl: episodeControlStep,
          record: { command: Array.from(obs.slice(0, 13)), rawAction: Array.from(mu),
            rootPositionWorld: Array.from(rootPosWorld), preview: lastControlPreview } });
        transportTeacherResumePending = true;
        setTaskStatus(`Preparing to complete the carry and set the ${carriedObjectLabel()} down…`);
        return;
      }
    }
    if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_before_physics',privateTurnCandidate);return;}

    if(inferenceFirstPickupStanding&&!inferenceFirstPickupStanding.isOwnedBy(firstPickupStandingContext())){
      inferenceFirstPickupStanding.abort('standing_owner_changed_before_physics',firstPickupStandingCandidate);
      restoreControlHistory();return;
    }
    lastTranslatorResult = tr;
    lastObservation = obs;
    lastRawAction = new Float32Array(mu);
    lastControlPhase = useReferenceStudentTurn ? 'student_reference_turn' : useStagedStudentTransport ? 'student_transport'
      : hybridStudentOwns ? `hybrid_${hybridDecision.owner}` : skillStep?.phase || 'student';
    targetQ.set(candidateTarget); lastAction.set(candidateAction);

    // 6. Step physics SIM_DECIMATION times with **inner-loop PD at the sim
    //    rate** (matches training in humanoid.py:574-578 and the proven
    //    deploy path in sim2sim_vae.py:1395). Torque must be recomputed
    //    every substep against fresh dof_pos/dof_vel — holding it constant
    //    for the whole control step makes the damping term `-kv*qd` stale
    //    and lets MuJoCo's Euler integrator run open-loop, which with stiff
    //    PD (kp~85, kd~0.4) explodes within a few control steps (observed:
    //    ||qvel|| 17→100 rad/s over 10 control steps).
    for (let j = 0; j < ACTION_DIM; j++) {
      previousDofPos[j] = data.qpos[addresses.qposAddr[j]];
      previousDofVel[j] = data.qvel[addresses.qvelAddr[j]];
    }
    for (let i = 0; i < SIM_DECIMATION; i++) {
      if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_before_substep',privateTurnCandidate);return;}
      applyPDTorques(model, data, targetQ, addresses);
      mujoco.mj_step(model, data);
      if(firstPickupStandingCandidate)
        inferenceFirstPickupStanding.observeActualSubstep(firstPickupStandingCandidate,firstPickupStandingContext());
      if(privateTurnCandidate)privateTurn.observeActualSubstep(privateTurnCandidate);
      if(useApproachRecovery)recoveredFacingAdmission.observeActualSubstep({
        ...recoveredFacingRecoveryContext(skillController),physicalControl:episodeControlStep+1,substep:i+1});
      if(!privateTurnCurrent()){abortPrivateTurn('turn_owner_changed_after_substep',privateTurnCandidate);return;}
      matchedCarryHost?.observeActualSubstep(data, {episode:stepEpisodeVersion,
        physicalControl:episodeControlStep + 1, substep:i + 1, phase:lastControlPhase});
    }
    for (let j = 0; j < ACTION_DIM; j++) lastTorque[j] = data.ctrl[addresses.actuatorOrder[j]];
    if (useRecoveryCycle) skillController.commit(lastControlPreview, episodeControlStep + 1);
    if (restrictedApproachRecoveryEnabled && ['teacher_step', 'teacher_turn'].includes(skillStep?.phase)) {
      // A later moving action invalidates the old terminal, including turns.
      executedApproachTerminal = null;
      if (skillStep.phase === 'teacher_step' && skillController instanceof TeacherRecordedApproachController
          && pendingWaypointCarryController && !skillController.cancelRequested
          && !pendingWaypointCarryController.finishRequested) {
        const terminal = skillController.terminalFrameForCurrentStep();
        if (terminal) {
          const live = skillProprio(), parent = pendingWaypointCarryController, goal = parent.approachGoalWorld;
          const endpointDistanceM = Math.hypot(live.rootPosWorld[0] - goal[0], live.rootPosWorld[1] - goal[1]);
          if (endpointDistanceM <= parent.arrivalRadius) {
            const plan = planTeacherStandingReference(terminal, { alignment: 'original',
              rootPosition: live.rootPosWorld, rootQuaternion: live.rootQuatXyzwWorld,
              objectPosition: live.objPosWorld, objectQuaternion: live.objQuatXyzwWorld,
              objectPointsLocal: skillController.skill.objectPointsLocal });
            executedApproachTerminal = { episode: episodeVersion, owner: skillController, parent, terminal, plan,
              skill: skillController.skill, executedAtControl: episodeControlStep + 1, endpointDistanceM };
          }
        }
      }
    }

    if (useRestricted){
      restrictedController.advance();
      if(useFirstPickupStanding){
        skillController.advance({ teacherOwned: false, proprio: skillProprio(skillController) });
        inferenceFirstPickupStanding.commit(firstPickupStandingCandidate,{
          ...firstPickupStandingContext(),physicalControl:episodeControlStep+1});
      }
    }
    else if (skillStep?.phase === 'teacher_standing') teacherStandingSteps++;
    else if (['approach', 'settling', 'settling_quiet'].includes(skillStep?.phase) || useTeacher || useStagedStudentTransport || useReferenceStudentTurn)
      { if (useTeacherDescentHold && teacherDescentHold?.active) skillController.holdReferenceClock('predicted_knee_hip_box_proximity', { physicalControl: episodeControlStep + 1 });
        else skillController.advance({ teacherOwned: useTeacher, proprio: skillProprio(skillController) }); }
    if(useApproachRecovery)recoveredFacingAdmission.commitRecovery({
      ...recoveredFacingRecoveryContext(skillController),physicalControl:episodeControlStep+1,preview:lastControlPreview});
    if (useTeacherDescentHold && teacherDescentHold?.active) teacherDescentHold.commit({ ...teacherDescentHoldContext(), physicalControl: episodeControlStep + 1,
      physicsSubsteps: SIM_DECIMATION, preview: lastControlPreview });
    if (useStagedStudentApproach) stagedStudentApproach.commit({ ...firstPickupStudentContext(),physicsSubsteps:SIM_DECIMATION,
      physicalControl: episodeControlStep + 1,
      record: { command: Array.from(obs.slice(0, 13)), rawAction: Array.from(mu),
        rootPositionWorld: Array.from(rootPosWorld), preview: lastControlPreview } });
    if (useStagedStudentTransport) {
      stagedStudentTransport.commit({ parent: skillController, episode: episodeVersion,
        physicalControl: episodeControlStep + 1, physicsSubsteps: SIM_DECIMATION,
        record: { command: Array.from(obs.slice(0, 13)), rawAction: Array.from(mu),
          rootPositionWorld: Array.from(rootPosWorld), preview: lastControlPreview } });
      if (!stagedStudentTransport.active) transportTeacherResumePending = true;
    }

    if(useReferenceStudentTurn)referenceStudentTurn.commit({...referenceApproachContext,physicalControl:episodeControlStep+1,
      record:{...referenceApproachInput,rawAction:Array.from(mu),preview:lastControlPreview}});
    if(useHeightAwareApproach)heightAwareApproach.commit({...referenceApproachContext,physicalControl:episodeControlStep+1,
      record:{...referenceApproachInput,rawAction:Array.from(mu),preview:lastControlPreview}});
    if (pickupFacingOwner && (pickupFacingOwner.controls !== pickupFacingInference.controls + 1
        || !lastControlPreview?.supported || lastControlPreview.completedSubsteps !== SIM_DECIMATION
        || lastControlPreview.allowedContactCount !== 0 || lastControlPreview.unwantedContactCount !== 0))
      throw new Error('Pickup-facing approach action or contact preview did not complete');
    if (noResetPoseOwner && !noResetActionAdvancedExactlyOnce(noResetPoseOwner.controls,noResetPoseInference.controls))
      throw new Error('No-reset approach action did not advance exactly once');
    if(privateTurnCandidate)privateTurn.commit({...terminalFacingContext(),physicalControl:episodeControlStep+1},privateTurnCandidate);
    const firstPickupHandoff=firstPickupStudentHandoffs.at(-1);
    if(firstPickupHandoff&&!firstPickupHandoff.firstTeacherControlCommitted&&useTeacher
        &&skillController===activeCarryController&&firstPickupHandoff.episode===episodeVersion
        &&firstPickupHandoff.requestId===activeBoxTaskRequestId&&firstPickupHandoff.physicalControl===episodeControlStep){
      firstPickupHandoff.firstTeacherControlCommitted=true;firstPickupHandoff.actualTeacherSubsteps=SIM_DECIMATION;
      firstPickupHandoff.physicalControlAfter=episodeControlStep+1;
    }
    controlStep += 1;
    episodeControlStep += 1;

    const rootAfterWorld = [
      data.xpos[pelvisId * 3 + 0],
      data.xpos[pelvisId * 3 + 1],
      data.xpos[pelvisId * 3 + 2],
    ];
    updateBenchmark(rootAfterWorld, rootAfterWorld[2]);

    // 7. Sync three.js mesh transforms from updated body poses.
    syncBodyTransforms(data, bodyGroups);

    // 7b. RPG follow camera — smoothly anchor `controls.target` to the
    // robot's pelvis world-position. Horizontal lerp is fast (keeps the
    // robot centered while walking); vertical lerp is slow and floor-
    // clamped at FOLLOW_TARGET_Y_FLOOR so a fall doesn't drag the camera
    // downward with the body.
    if (followMode) {
      const pelvisGroup = bodyGroups[pelvisId];
      if (pelvisGroup) {
        const pelvisWorld = new THREE.Vector3();
        pelvisGroup.getWorldPosition(pelvisWorld);
        const aFast = 0.10;
        followTarget.x += (pelvisWorld.x - followTarget.x) * aFast;
        followTarget.z += (pelvisWorld.z - followTarget.z) * aFast;
        const targetY = Math.max(FOLLOW_TARGET_Y_FLOOR, pelvisWorld.y);
        const aSlow = 0.04;
        followTarget.y += (targetY - followTarget.y) * aSlow;
        // Translate target AND camera by the same delta → orbit angle and
        // zoom distance from OrbitControls are preserved as the view
        // trails the robot.
        const dx = followTarget.x - controls.target.x;
        const dy = followTarget.y - controls.target.y;
        const dz = followTarget.z - controls.target.z;
        controls.target.set(followTarget.x, followTarget.y, followTarget.z);
        camera.position.x += dx;
        camera.position.y += dy;
        camera.position.z += dz;
      }
    }

    // 8. FPS counter + status line + info-panel update (every 10th step).
    frameCount += 1;
    if (frameCount % 10 === 0) {
      // Maximum |mu| across joints — early warning that the policy is
      // saturating the action range.
      let actAbsMax = 0;
      for (let i = 0; i < ACTION_DIM; i++) {
        const a = Math.abs(mu[i]);
        if (a > actAbsMax) actAbsMax = a;
      }
      const pz = data.xpos[pelvisId * 3 + 2];
      // FPS averaged over the last 30 control steps.
      let fps;
      if (frameCount % 30 === 0) {
        const now = performance.now();
        fps = 30000 / (now - lastFpsT);
        lastFpsT = now;
        setStatus(`Running at ~${fps.toFixed(0)} Hz · pelvis height ${pz.toFixed(2)} m`);
      }
      // FSM state from the translator drives the mode badge (replaces
      // the legacy user.mode derivation, which didn't know about RECOVER).
      const displayMode = useTeacher ? (['teacher_standing', 'teacher_settling', 'teacher_exit_hold', 'teacher_exit_release', 'teacher_exit_settling', 'teacher_exit_quiet_hold', 'teacher_exit_quiet_settling', 'recovery_stance', 'recovery_stance2', 'recovery_stance3'].includes(skillStep.phase) ? 'IDLE'
        : ['teacher_turn', 'recovery_turn'].includes(skillStep.phase) ? 'TURN' : ['teacher_step', 'teacher_exit_retreat', 'recovery_retreat', 'recovery_retreat2'].includes(skillStep.phase) ? 'LOCO'
        : activeBoxTask === 'carry' ? 'BOX_CARRY' : 'BOX_TASK') : (tr ? tr.fsmState : user.mode);
      setMode(displayMode);

      // User-friendly labels — no internal jargon.
      const walkTargetStr = user.humanGoalWorld
        ? `${user.humanGoalWorld[0].toFixed(1)}, ${user.humanGoalWorld[1].toFixed(1)}, ${user.humanGoalWorld[2].toFixed(1)}`
        : '—';
      const objTargetStr = user.objGoalWorld
        ? `${user.objGoalWorld[0].toFixed(1)}, ${user.objGoalWorld[1].toFixed(1)}, ${user.objGoalWorld[2].toFixed(1)}`
        : '—';
      const objLabel = user.activeObjName
        ? user.activeObjName.replace(/^active_/, '').replace(/_\d{3}_\d{3}_\d{3}$/, '')
        : '—';
      const styleLabel = user.deterministic ? 'fixed' : 'held';
      const smoothLabel = smoothingAlpha >= 0.99 ? 'off'
        : (smoothingAlpha >= 0.45 ? 'mild' : 'strong');
      const lines = [
        `Mode       ${MODE_LABELS[displayMode] || displayMode}`,
        `Object     ${objLabel}`,
        `Walk goal  ${walkTargetStr}`,
        `Obj goal   ${objTargetStr}`,
        `Pelvis z   ${pz.toFixed(2)} m`,
        `Max action ${actAbsMax.toFixed(2)}${actAbsMax > 1.0 ? '  (clipped)' : ''}`,
        `Smoothing  ${smoothLabel}`,
        `Style      ${styleLabel}`,
        `Camera     ${followMode ? 'follow (V)' : 'free'}`,
      ];
      if (useStagedStudentTransport) {
        lines.push('Carry phase Student transport',
          `Time left  ${((stagedStudentTransport.horizonControls - stagedStudentTransport.controls) / CONTROL_HZ).toFixed(2)} s`);
      } else if (useReferenceStudentTurn) {
        lines.push('Approach: student turn',
          'Time left '+((referenceStudentTurn.horizonControls-referenceStudentTurn.controls)/CONTROL_HZ).toFixed(2)+' s');
      } else if (tr) {
        const reachH = isFinite(tr.debug.reachDistHuman) ? tr.debug.reachDistHuman.toFixed(2) + ' m' : '—';
        const reachO = isFinite(tr.debug.reachDistObj)   ? tr.debug.reachDistObj.toFixed(2) + ' m'   : '—';
        const clickDist = isFinite(tr.debug.humanGoalDist) ? tr.debug.humanGoalDist.toFixed(2) + ' m' : '—';
        lines.push(
          `Goal row   #${tr.debug.selectedRow}  dist ${tr.debug.matchDistance.toFixed(3)}`,
          `Hold       ${tr.debug.holdFrames} / ${Math.round(tr.debug.heldKFrames)} frames`,
          `Reach      body ${reachH}   obj ${reachO}`,
          `Click dist ${clickDist}${tr.debug.arrivalOverride ? '  arrival' : ''}`,
          `Goal src   ${tr.debug.clickPositionSource}${tr.debug.clickPositionOverride ? '  stable' : ''}`,
        );
      }
      setInfoPanel(lines);
    }
  }

  function runStep() {
    return runStepWithRetargetBarrier({
      readActiveStep: () => activeStepPromise,
      // R1b (v14a): a pending push admission is a step barrier exactly like a loaded retarget — no physics between the lane's
      // boundary capture and its admission, whoever drives the steps (browser loop or an external api.step()).
      readActiveRetarget: () => activeLoadedRetargetPromise ?? pendingPushAdmission,
      startStep: () => {
        picker.syncFromUserState();
        activeStepPromise = step().finally(() => {
          activeStepPromise = null;
          if (restrictedMode && queuedBoxTask && !skillActive() && restrictedController.isSettled) {
            const request = queuedBoxTask; queuedBoxTask = null;
            void startBoxTask(request.kind, request.requestedGoal, request.requestId);
          }
        });
        return activeStepPromise;
      },
    });
  }

  // --- B9: push lane entry chain lifted to closure level ----------------- //
  // Bodies are the v5 __interactiveDemo methods verbatim, dedented, with the self-references
  // `this.getPeriodicBoundarySnapshot()`, `this.requestNoResetApproachPose(` and
  // `window.__interactiveDemo.getPeriodicBoundarySnapshot()` replaced by direct calls. The debug API
  // methods below delegate here, so the evaluator contract is unchanged and the unified click path
  // (routeFloorGoalThroughSkillArbiter) does not depend on the debug object existing.
  const pause = async () => {
    paused = true;
    if (activeStepPromise) await activeStepPromise;
  };
  async function requestNoResetApproachPose(args = {}) {
    const started = await startNoResetApproachPoseRequest(args, {
      pause,
      available: () => Boolean(restrictedController && restrictedObs
        && approveRestrictedReference && restrictedWalkSkills.length && restrictedTurnSkills.length
        && contactDiagnostics),
      busy: () => Boolean(skillActive() || restrictedSuspended || noResetApproachOwner
        || activeLoadedRetargetPromise || pendingLoadedRetarget),
      readBoundary: () => getPeriodicBoundarySnapshot(),
      readSelection: () => ({ objectBody: user.activeObjName, objectBodyId: activeObjBodyId }),
      readBounds: () => boxCollisionBounds.read(data),
      readLive: () => skillProprio(restrictedController),
      stepSkills: restrictedWalkSkills, turnSkills: restrictedTurnSkills,
      approveReference: approveRestrictedReference,
      readEndpointBoundary: () => getPeriodicBoundarySnapshot(),
      installOwner: owner => {
        restrictedController.requestCancel(); restrictedController.reanchor();
        skillController = noResetApproachOwner = owner;
        if (!owner.isOwnedBy(noResetApproachContext())) {
          throw new Error('No-reset approach owner was not installed atomically');
        }
        teacherObs?.dispose(); teacherObs = new TeacherObsBuilder(mujoco, model, owner.skill);
        translator.reset(); user.humanGoalWorld = user.objGoalWorld = null; user.releaseKeys();
        lastNoResetApproachReview = null;
      },
    });
    if (!started.supported) return started;
    setTaskStatus('Following the measured route to the requested pose; manipulation remains disabled.');
    return { supported: true, reason: null, diagnosticOnly: true, promotionQualified: false,
      token: { generation: episodeVersion, physicalControl: episodeControlStep },
      route: structuredClone(started.route), savedPlan: structuredClone(started.savedPlan),
      capability: 'recorded planar XY/yaw approach only; source69 posture, Z, roll/pitch, velocity and history remain unresolved' };
  }
  async function startGroundPushToGoal(value = null) {
    const selectedBody = PUSH_LANE_BODY;
    const explicitGoal = value === null ? null : Array.from(value);
    if (explicitGoal && (explicitGoal.length !== 3 || !explicitGoal.every(Number.isFinite)))
      throw new Error('Push destination must contain finite XYZ');
    const wasPaused = paused; await pause();
    if (skillLoading || skillActive() || restrictedSuspended || pendingNormalGroundPush || activeLoadedRetargetPromise || pendingLoadedRetarget) {
      if(!wasPaused)paused=false; return { supported:false, reason:'task_busy' };
    }
    if (user.activeObjName !== selectedBody || activeObjBodyId < 0 || !activeObjPointsFlat) {
      const selection=createObjectSelection(selectedBody,findBodyIdByName(model,selectedBody),pointCloudDb);
      user.activeObjName=selectedBody;user.humanGoalWorld=user.objGoalWorld=null;
      activeObjPointsFlat=selection.pointsLocal;activeObjBodyId=selection.bodyId;picker.syncFromUserState();
    }
    const requestId = beginBoxTaskRequest('push', explicitGoal, 'ground_push_destination');
    const requestEpisode = episodeVersion, requestVersion = ++skillRequestVersion;
    boxTaskRequestLog.transition(requestId, 'loading', 'ground_push_reference_load', boxRequestClock());
    skillLoading = true; updateTaskControls();
    try {
      // R1 (v14a): the prefetched, already-parsed assets when the arbiter is ON; otherwise the v5 on-demand fetch (byte-for-byte below).
      const prefetched = pushAssetPrefetch ? await pushAssetPrefetch : null;
      const [raw, evidence] = prefetched ?? await (async () => {
        const [assetResponse,evidenceResponse] = await Promise.all([
          fetch('public/task-assets/hand002_predecessor69_phase_reference.json'),
          fetch('public/task-assets/native-summary.json')]);
        if (!assetResponse.ok || !evidenceResponse.ok) throw new Error('Ground-push task assets are unavailable');
        return [await assetResponse.json(), await evidenceResponse.json()];
      })();
      const skill = bindHand002Predecessor69(raw,evidence);
      if (episodeVersion !== requestEpisode || skillRequestVersion !== requestVersion)
        return { supported:false, reason:'request_changed_during_load', requestId };
      if (user.activeObjName !== selectedBody || activeObjBodyId < 0)
        return { supported:false, reason:'selected_largebox_changed', requestId };
      const boundary = getPeriodicBoundarySnapshot();
      const sourceFirst=raw.reference_frames747[0],sourceLast=raw.reference_frames747[skill.sourceFrames-1];
      const distance=Math.hypot(sourceLast[71]-sourceFirst[71],sourceLast[72]-sourceFirst[72]);
      const goal=explicitGoal ?? [boundary.state.objectPosition[0]+distance,boundary.state.objectPosition[1],0];
      const aligned=prepareGroundPushGoalWarpEntry({raw,evidence,objectPosition:boundary.state.objectPosition,
        goalWorld:goal,planCarryToGoal});
      const prepared=prepareOutcomeBasedNoResetEntry({entryPacket:boundary,alignedTarget:aligned,
        compiledObstacles:boxCollisionBounds.read(data),planBoxApproach});
      const approach=await requestNoResetApproachPose({expectedGeneration:boundary.episodeVersion,
        expectedPhysicalControl:boundary.control,requestedTarget:boundary.requestedNextSource,
        objectBinding:boundary.objectBinding,compiledPlan:prepared.plan,
        targetRootPosition:prepared.targetRootPosition,targetRootQuaternionWxyz:prepared.targetRootQuaternionWxyz,
        boundaryPacket:boundary});
      if (!approach.supported) {
        boxTaskRequestLog.transition(requestId,'refused',approach.reason??'approach_refused',boxRequestClock());
        return { ...approach, requestId };
      }
      pendingNormalGroundPush={requestId,episodeVersion:requestEpisode,goal:Array.from(goal),skill,
        sourceIdentity:skill.sourceIdentity,approachTarget:structuredClone(aligned)};
      activeBoxTaskRequestId=requestId;activeBoxTask='push';taskDestinationWorld=Array.from(goal);
      boxTaskRequestLog.transition(requestId,'started','ground_push_approach_started',boxRequestClock(),
        {sourceIdentity:skill.sourceIdentity,sourcePhaseInclusive:[69,293]});
      setTaskStatus('Walking to the Largebox for Push / Slide…');
      return {supported:true,reason:null,requestId,goalWorld:Array.from(goal),phase:'approach'};
    } catch(error) {
      if (episodeVersion===requestEpisode && latestBoxTaskRequestId===requestId)
        boxTaskRequestLog.transition(requestId,'refused','task_load_error',boxRequestClock(),{message:error.message});
      setTaskStatus('Push / Slide unavailable: '+error.message);
      return {supported:false,reason:'task_load_error',requestId,message:error.message};
    } finally { if(skillRequestVersion===requestVersion && episodeVersion===requestEpisode){
      skillLoading=false;updateTaskControls();if(!wasPaused)paused=false;
    } }
  }
  function getPeriodicBoundarySnapshot() {
    if (!restrictedObs) {
      throw new Error('Measured boundary capture requires the live locomotion teacher state');
    }
    const objectBodyName = NO_RESET_APPROACH_SOURCE.objectBody;
    const objectBodyId = findBodyIdByName(model, objectBodyName);
    const bodies = restrictedObs.bodyIds.map(id => restrictedObs._bodyState(data, id));
    const root = bodies[0], object = objectBodyId >= 0 ? restrictedObs._bodyState(data, objectBodyId) : null;
    const objectGeometry = [];
    if (objectBodyId >= 0) for (let geomId = 0; geomId < model.ngeom; geomId++) {
      if (model.geom_bodyid[geomId] === objectBodyId) objectGeometry.push({ geomId,
        type: model.geom_type[geomId], dataId: model.geom_dataid[geomId],
        size: Array.from(model.geom_size.slice(geomId * 3, geomId * 3 + 3)) });
    }
    const missingFields = [];
    const finiteOrMissing = (values, length, name) => {
      if (!values || values.length !== length || !Array.from(values).every(Number.isFinite)) {
        missingFields.push(name); return null;
      }
      return Array.from(values);
    };
    const current = skillController instanceof NoResetApproachPoseController ? {
      phase: skillController.phase, skill: skillController.skill,
      referenceIndex: skillController.referenceIndex,
      periodicTeacher: skillController.skill?.periodicTeacher ?? null,
    } : lastRestrictedStep;
    const boundaryHistoryBuilder = skillController instanceof NoResetApproachPoseController
      ? teacherObs : restrictedObs;
    return {
      control: episodeControlStep, simTime: data.time, episodeVersion,
      provenance: { continuousControls: episodeControlStep,
        initializedFromTargetState: false, captureMode: 'read_only_periodic_boundary',
        control: episodeControlStep,
        evidence: 'episodeControlStep resets only in doReset and increments only after a physical control; this route never imports a target state' },
      controllerIdentity: current?.phase ?? lastControlPhase,
      coordinateFrame: 'mujoco_world:g1_scene.xml', quaternionOrder: 'wxyz',
      currentSource: { kind: 'BONES_AMASS_locomotion', skillName: current?.skill?.name ?? null,
        sourceName: current?.periodicTeacher?.sourceName ?? null,
        sourcePhase: current?.periodicTeacher?.sourcePhase ?? null,
        referenceIndex: current?.referenceIndex ?? null },
      locomotionCandidate: periodicTeacherSkills?.forward ? {
        sourceName: periodicTeacherSkills.forward.periodicTeacher?.sourceName ?? periodicTeacherSkills.forward.name,
        sourcePhase: periodicTeacherSkills.forward.periodicTeacher?.sourcePhase ?? null,
        sourceFrames: periodicTeacherSkills.forward.sourceFrames,
      } : null,
      requestedNextSource: { ...NO_RESET_APPROACH_SOURCE,
        historyAvailable: false, status: 'requested_only_not_current' },
      objectBinding: { objectBody: objectBodyName, objectBodyId,
        geometryIdentity: NO_RESET_APPROACH_SOURCE.geometryIdentity, geometries: objectGeometry },
      state: {
        rootPosition: finiteOrMissing(root?.position, 3, 'state.rootPosition'),
        rootQuaternion: root?.rotation ? [root.rotation[3], root.rotation[0], root.rotation[1], root.rotation[2]] : null,
        rootVelocity: root ? [...root.velocity, ...root.angularVelocity] : null,
        jointPosition: finiteOrMissing(Array.from(addresses.qposAddr, id => data.qpos[id]), 29, 'state.jointPosition'),
        jointVelocity: finiteOrMissing(Array.from(addresses.qvelAddr, id => data.qvel[id]), 29, 'state.jointVelocity'),
        objectPosition: finiteOrMissing(object?.position, 3, 'state.objectPosition'),
        objectQuaternion: object?.rotation ? [object.rotation[3], object.rotation[0], object.rotation[1], object.rotation[2]] : null,
        objectVelocity: object ? [...object.velocity, ...object.angularVelocity] : null,
        bodyPositions: bodies.map(body => Array.from(body.position)),
        bodyRotationsXyzw: bodies.map(body => Array.from(body.rotation)),
        bodyLinearVelocities: bodies.map(body => Array.from(body.velocity)),
        bodyAngularVelocities: bodies.map(body => Array.from(body.angularVelocity)),
      },
      history: {
        previousAction: finiteOrMissing(lastAction, 29, 'history.previousAction'),
        appliedTorque: finiteOrMissing(lastTorque, 29, 'history.appliedTorque'),
        previousDofPosition: finiteOrMissing(boundaryHistoryBuilder?.lastDofPos, 29, 'history.previousDofPosition'),
        previousDofVelocity: finiteOrMissing(boundaryHistoryBuilder?.lastDofVel, 29, 'history.previousDofVelocity'),
      },
      contactState: contactDiagnostics.read(data, objectBodyId), missingFields,
    };
  }
  // Opt-in observability for browser automation. Reset can vary the starting
  // heading; subsequent motion uses the normal control loop, without overlapping inference.
  if (urlParams.get('debug') === '1' || benchmark) {
    const vector3 = (value) => {
      const v = [value[0], value[1], value[2] ?? 0];
      if (!v.every(Number.isFinite)) throw new Error('Goal must contain finite coordinates');
      return new Float32Array(v);
    };
    window.__interactiveDemo = {
      getTaskCoverageCaptures() { return taskCoverageCapture?.snapshot() ?? null; },
      getMatchedCarryReview() { return matchedCarryHost?.review() ?? null; },
      getTeacherDescentReview() { return {enabled:teacherDescentRematchEnabled,
        decisions:structuredClone(teacherDescentDecisions)}; },
      getMatchedCarryLastObservation() { return lastObservation ? Array.from(lastObservation) : null; },
      getMatchedCarryMarker() { return {visible:goalViz.group.visible && goalViz.objSphere.visible,
        position:goalViz.objSphere.position.toArray(), originalGoalWorld:taskDestinationWorld}; },
      getMatchedCarryExecutionReference() {
        const owner = matchedCarryHost?.owner;
        return owner?.bank ? {skill:owner.skill, bank:owner.bank, transform:owner.transform,
          originalGoalWorld:owner.request.originalGoalWorld, role:owner.role, sourceIndex:owner.sourceIndex} : null;
      },
      requestLoadedCarryRetarget,
      cancelLoadedCarryRetarget,
      getLoadedCarryRetargetReview() { return {
        enabled: midCarryRetargetEnabled,
        active: Boolean(activeLoadedRetargetPromise),
        pending: pendingLoadedRetarget ? structuredClone(pendingLoadedRetarget.preview) : null,
        requests: loadedRetargetRequestLog.snapshot(), reviews: structuredClone(loadedRetargetReviews),
      }; },
      async requestNoResetApproachPose(args = {}) { return requestNoResetApproachPose(args); },
      getNoResetApproachPoseReview() {
        return noResetApproachOwner?.review() ?? structuredClone(lastNoResetApproachReview);
      },
      getSkillDecisionReview() { return { enabled: skillArbiterEnabled, version: SKILL_ARBITER_VERSION, thresholds: SKILL_ARBITER_THRESHOLDS,
        pushAssetPrefetch: pushAssetPrefetchState, pushRefusalFallbackToCarry: PUSH_REFUSAL_FALLBACK_TO_CARRY, pendingPushAdmission: Boolean(pendingPushAdmission),
        pushLane: SKILL_ARBITER_PUSH_LANE, selectedSkill: selectedSkill ? structuredClone(selectedSkill) : null,
        decisions: structuredClone(skillDecisions) }; },
      previewSkillDecision(value) {
        // Dry run: the click path's goal normalization + the arbiter, without routing or state change.
        const goal = objectGoalFromGround(vector3(value));
        const { decision, decisionMs } = measureChooseSkill(readSkillArbiterRequest(Array.from(goal), null));
        return { ...structuredClone(decision), decisionMs, goalWorld: Array.from(goal) };
      },
      async startGroundPushToGoal(value = null) { return startGroundPushToGoal(value); },
      async requestLargeboxPushLiveDiagnostic(args = {}) {
        await pause();
        const result = await startLargeboxPushLiveDiagnostic(args, {
          readBoundary: () => this.getPeriodicBoundarySnapshot(),
          readApproachReview: () => structuredClone(lastNoResetApproachReview),
          busy: () => Boolean(skillActive() || restrictedSuspended || activeLoadedRetargetPromise || pendingLoadedRetarget || noResetApproachOwner),
          readSelection: () => ({ objectBody: user.activeObjName, objectBodyId: activeObjBodyId }),
          loadAsset: async url => { const r=await fetch(url); if(!r.ok)throw new Error(`Could not load hand002 source (${r.status})`); return r.json(); },
          loadEvidence: async url => { const r=await fetch(url); if(!r.ok)throw new Error(`Could not load hand002 evidence (${r.status})`); return r.json(); },
          createController: (skill, goal) => new GroundPushGoalSequenceController(skill, goal, {initialStanceFrames:0,maxSegments:1,settlingSteps:180,maxCorrection:.05,warpStartFrame:81,warpEndFrame:110,maxFacingError:Math.PI,maxReferenceStartDistance:Infinity,requireSegmentExit:boxExitEnabled}),
          readLive: controller => skillProprio(controller),
          install: (controller, diagnostic) => {
            const actionBefore=Array.from(lastAction),torqueBefore=Array.from(lastTorque);
            const nextTeacherObs=new TeacherObsBuilder(mujoco,model,controller.skill);
            try{nextTeacherObs.reset({lastDofPos:Float32Array.from(diagnostic.endpointHistory.previousDofPosition),lastDofVel:Float32Array.from(diagnostic.endpointHistory.previousDofVelocity)});
              if(!exactArray(actionBefore,lastAction)||!exactArray(torqueBefore,lastTorque)||!exactArray(diagnostic.endpointHistory.previousDofPosition,nextTeacherObs.lastDofPos)||!exactArray(diagnostic.endpointHistory.previousDofVelocity,nextTeacherObs.lastDofVel))throw new Error('Largebox push handoff changed measured policy history');
            }catch(error){nextTeacherObs.dispose();throw error;}
            const requestId=beginBoxTaskRequest('push',args.goalWorld,'ground_push_destination');
            boxTaskRequestLog.transition(requestId,'started','unqualified_live_handoff_diagnostic',boxRequestClock(),{sourceIdentity:diagnostic.skill.sourceIdentity,sourcePhaseInclusive:[69,293]});
            activeBoxTaskRequestId=requestId; activeBoxTask='push'; taskDestinationWorld=Array.from(args.goalWorld);
            restrictedController.requestCancel(); restrictedController.reanchor(); skillController=activeCarryController=controller; pendingCarryController=null;
            teacherObs?.dispose(); teacherObs=nextTeacherObs;
            boxExitController?.reset(); preserveCompletionCommand=false;
            translator.reset(); user.releaseKeys(); user.objGoalWorld=null; user.activeObjName=controller.skill.objectBodyName; user.deterministic=true; user.vaeNoise.fill(0); smoothingAlpha=1;
            lastLargeboxPushLiveDiagnostic={requestId,token:{episodeVersion,control:episodeControlStep},measurement:structuredClone(diagnostic.measurement),sourceIdentity:diagnostic.skill.sourceIdentity,liveEntryQualified:false};
          },
        });
        if(result.supported)setTaskStatus('Running one unqualified Largebox hand-push live-handoff diagnostic.'); return result;
      },
      getLargeboxPushLiveDiagnosticReview() { return lastLargeboxPushLiveDiagnostic ? {...structuredClone(lastLargeboxPushLiveDiagnostic),controller:activeCarryController?.manipulationKind==='ground_push'?{phase:activeCarryController.phase,completionReason:activeCarryController.completionReason,referenceIndex:activeCarryController.referenceIndex,sourceFrames:activeCarryController.skill.sourceFrames,outcome:activeCarryController.outcome}:null}:null; },
      getLargeboxPushCompiledBounds() { const before={episodeVersion,episodeControlStep,simTime:data.time},bounds=structuredClone(boxCollisionBounds.read(data)); if(episodeVersion!==before.episodeVersion||episodeControlStep!==before.episodeControlStep||data.time!==before.simTime)throw new Error('Compiled-bounds read advanced physics'); return bounds; },
      getPeriodicBoundarySnapshot() { return getPeriodicBoundarySnapshot(); },
      getState() {
        const objectPoses = {};
        for (const name of selectableNames) {
          const id = findBodyIdByName(model, name);
          if (id < 0) continue;
          objectPoses[name] = Array.from(data.xpos.slice(id * 3, id * 3 + 3));
        }
        const q = data.xquat.slice(pelvisId * 4, pelvisId * 4 + 4);
        const contactObjectName = skillActive() ? (skillController.skill?.objectBodyName || activeCarryController?.skill?.objectBodyName)
          : user.activeObjName || skillController?.skill?.objectBodyName || null;
        const contactObjectId = contactObjectName ? findBodyIdByName(model, contactObjectName) : -1;
        return {
          paused, controlStep, episodeControlStep, episodeVersion, skillLoading,
          matchedCarryPreviewEnabled,
          teacherDescentRematchEnabled,
          skillRequestVersion, latestBoxTaskRequestId, activeBoxTaskRequestId,
          boxTaskRequests: boxTaskRequestLog.snapshot(),
          midCarryRetarget: { enabled: midCarryRetargetEnabled,
            active: Boolean(activeLoadedRetargetPromise),
            pending: pendingLoadedRetarget ? structuredClone(pendingLoadedRetarget.preview) : null,
            requests: loadedRetargetRequestLog.snapshot(), reviews: structuredClone(loadedRetargetReviews) },
          quietEndings: { enabled: quietEndingsEnabled, ...quietEndingConfig,
            placementCorrection: placementCorrectionEnabled, ...placementCorrectionConfig },
          quietExitSettling: { mode: quietExitSettlingMode, exitClock: boxExitController?.recordClock?.quietExitPolicy ?? null },
          placementCorrections: structuredClone(placementCorrections),
          simTime: data.time,
          qpos: Array.from(data.qpos), qvel: Array.from(data.qvel), ctrl: Array.from(data.ctrl),
          skillPhase: skillController?.phase || 'inactive', skillFrame: skillController?.referenceIndex || 0,
          skillSourceFrames: skillController?.sourceFrames || 0,
          activeCarryTiming: activeCarryTimingSnapshot(),
          boxTask: activeBoxTask,
          selectedSkill: selectedSkill ? structuredClone(selectedSkill) : null, skillDecisions: structuredClone(skillDecisions),
          skillArbiter: { enabled: skillArbiterEnabled, version: SKILL_ARBITER_VERSION, thresholds: SKILL_ARBITER_THRESHOLDS, pushLane: SKILL_ARBITER_PUSH_LANE,
            pushAssetPrefetch: pushAssetPrefetchState, pushRefusalFallbackToCarry: PUSH_REFUSAL_FALLBACK_TO_CARRY },
          boxTaskRetirement: lastBoxTaskRetirement,
          postTaskTargetContact: postTaskTargetContactAdmission.review(),
          controlPhase: lastControlPhase,
          referenceStudentTurnsEnabled,heightAwareApproachEnabled,
          laterSegmentPartAdmissionEnabled, laterSegmentPartAdmissions: structuredClone(laterSegmentPartAdmissions), midClipLibraryEnabled, midClipFinalOnlyEnabled,
          referenceStudentTurnActive:Boolean(referenceStudentTurn?.active),
          referenceStudentTurnControls:referenceStudentTurn?.controls??null,referenceStudentTurnEnded:referenceStudentTurn?.ended??null,
          heightAwareApproachActive:Boolean(heightAwareApproach?.active),heightAwareApproachControls:heightAwareApproach?.controls??null,
          heightAwareApproachEnded:heightAwareApproach?.ended??null,
          restrictedMode,
          keyboardMode,
          teacherPeriodicPersistentEnabled,
          recordedLateralEnabled,
          hybridKeyboard: hybridArbiter ? hybridArbiter.snapshot() : null,
          restrictedBackwardEnabled,
          restrictedKeyTerminalEnabled,
          restrictedStartupStandingEnabled,
          restrictedApproachTerminalEnabled,
          restrictedApproachRoutingEnabled,
          restrictedApproachRecoveryEnabled,
          recoveryFlags: { ...recoveryFlags }, approachRecoveryLoop: approachRecovery?.snapshot() ?? null,
          transientSuspension: structuredClone(transientSuspension), standingPreviewOwned: Boolean(standingPreviewOwned),
          recoveryCycleActive: skillController instanceof ApproachRecoveryCycle && skillController.active,
          carryRequestClearance: lastCarryRequestClearance ? {
            requestId: lastCarryRequestClearance.requestId, supported: lastCarryRequestClearance.supported,
            reason: lastCarryRequestClearance.reason, segmentIndex: lastCarryRequestClearance.segmentIndex ?? null,
            checks: lastCarryRequestClearance.checks.map(check => ({ segmentIndex: check.segmentIndex,
              destinationSupported: check.destination.supported, collisions: check.destination.collisions ?? [],
              path: check.path })),
          } : null,
          carryEntryClearance: lastCarryEntryClearance ? {
            requestId: lastCarryEntryClearance.requestId, segmentIndex: lastCarryEntryClearance.segmentIndex,
            episodeControlStep: lastCarryEntryClearance.episodeControlStep,
            supported: lastCarryEntryClearance.supported, reason: lastCarryEntryClearance.reason,
            collisions: lastCarryEntryClearance.destination.collisions ?? [], path: lastCarryEntryClearance.path,
          } : null,
          effectiveCommand: restrictedMode ? (skillActive() ? { type: 'box', task: activeBoxTask,
            requestId: activeBoxTaskRequestId,
            goalWorld: matchedCarryHost?.originalGoalWorld ?? (activeCarryController?.requestedGoalWorld ? Array.from(activeCarryController.requestedGoalWorld) : null) }
            : lastHybridDecision && lastHybridDecision.owner !== 'recorded' ? { type: `hybrid_${lastHybridDecision.owner}`, keys: heldKeysFromUser(user), reason: lastHybridDecision.reason }
            : lastRestrictedStep?.activeIntent ?? null) : null,
          queuedCommand: restrictedMode ? queuedBoxTask ?? (lastHybridDecision && lastHybridDecision.owner !== 'recorded' ? null : queuedRestrictedIntent()) : null,
          queueReason: restrictedMode ? lastRestrictedStep?.initialStandingPending
              && (queuedBoxTask || lastRestrictedStep.pendingIntent) ? 'initial_standing'
            : queuedBoxTask ? 'finish_current_motion'
            : skillActive() && restrictedController.requestedIntent.revision > 0 ? 'finish_box_setdown'
              : lastRestrictedStep?.pendingIntent ? 'finish_record_and_settle' : null : null,
          restrictedControl: restrictedMode ? {
            phase: restrictedController?.phase, supported: lastRestrictedStep?.supported ?? null,
            suspended: restrictedSuspended,
            geometry: lastRestrictedGeometry,
            geometryAttempts: structuredClone(restrictedReferenceAttempts),
            geometryCheckStep: restrictedGeometryCheckStep,
            referenceIndex: lastRestrictedStep?.referenceIndex ?? null,
            sourceFrames: lastRestrictedStep?.sourceFrames ?? null,
            skillName: lastRestrictedStep?.skill?.name ?? null,
            keyDirectionHeadingRad: lastRestrictedStep?.keyDirectionHeadingRad ?? null,
            keyDirectionRevision: (lastRestrictedStep?.pendingIntent ?? lastRestrictedStep?.activeIntent)?.revision ?? null,
            standingControlsBeforeAction: lastRestrictedStep?.standingControls ?? null,
            standingStableControls: lastRestrictedStep?.standingStableControls ?? null,
            isSettled: restrictedController?.isSettled ?? false,
            initialStandingPending: lastRestrictedStep?.initialStandingPending ?? false,
            retainedAdditionalKeyTerminal: lastRestrictedStep?.retainedAdditionalKeyTerminal ?? false,
            retainedKeyTerminal: lastRestrictedStep?.retainedKeyTerminal ?? false,
            persistentPeriodic: structuredClone(lastRestrictedStep?.persistentPeriodic ?? null),
            reason: lastRestrictedStep?.completionReason ?? null, plan: restrictedPlan,
            effectiveGoalWorld: lastRestrictedStep?.effectiveGoalWorld ?? null,
            outcome: lastRestrictedStep?.outcome ?? null,
          } : null,
          headingPreparations: structuredClone(headingPreparations),
          teacherApproachEnabled, recordedApproachEnabled, carryInitialStanceFrames,
          boxExitEnabled, boxExitMode, boxExitActive: Boolean(boxExitController?.isActive()),
          boxExitQuietWaiting: Boolean(boxExitController?.quietWaiting),
          carrySegmentExitActive: pendingSegmentCarryController !== null,
          carrySegmentExitResults: activeCarryController?.segmentExitResults ? structuredClone(activeCarryController.segmentExitResults) : null,
          restrictedStudentApproachEnabled,
          restrictedStudentTransportEnabled,
          transportAdmission: { ...transportAdmission },
          transportDivergenceExitEnabled,
          setdownClearanceDiagEnabled,
          setdownClearanceTrace: setdownClearanceDiagEnabled ? structuredClone(setdownClearanceTrace) : null,
          carryDescentSagHoldEnabled,
          carryDescentSagHold: activeCarryController?.descentSagHoldSnapshot
            ? structuredClone(activeCarryController.descentSagHoldSnapshot) : null,
          transportDivergenceLimits: transportDivergenceLimits ? { ...transportDivergenceLimits } : null,
          refObjYawSnapEnabled,
          studentLiftPreviewEnabled,
          pickupFacingApproachEnabled,
          pickupFacingEntryRegionEnabled,
          pickupPoseSearchEnabled,
          pickupPoseSearchTiers: [...pickupPoseSearchTiers],
          alternateCarryEnabled,
          pickupPoseSearch: pickupPoseSearchReviews.at(-1) ? { requestId: pickupPoseSearchReviews.at(-1).requestId,
            selectedTier: pickupPoseSearchReviews.at(-1).selectedTier, pickupPoseReachable: pickupPoseSearchReviews.at(-1).pickupPoseReachable,
            selectedCandidateIds: [...pickupPoseSearchReviews.at(-1).selectedCandidateIds], planningMs: pickupPoseSearchReviews.at(-1).planningMs } : null,
          carryStylePreview: carryStylePreview ? { ...carryStylePreview } : null,
          carryStyleSampling: carryStyleRequestSampler?.snapshot() ?? null,
          restrictedLongCarryEnabled,
          restrictedCarryLibraryEnabled,
          restrictedCarryGoalRegionEnabled,
          carryPlacementToleranceM: CARRY_PLACEMENT_TOLERANCE_M,
          carryPlacementStatus: finalCarryPlacement ?? activeCarryController?.placementStatus ?? null,
          carryReferenceSelection: carryReferenceSelection ? structuredClone(carryReferenceSelection) : null,
          carryRanking, replanRemainingEnabled, carryBudgetGuardEnabled, carryControlBudgetControls,
          carryControlBudget: carryRequestControlBudget(),
          carryRefusal: lastCarryRefusal ? structuredClone(lastCarryRefusal) : null,
          carryReplans: structuredClone(carryReplans),
          carryTaskLineage: structuredClone(carryTaskLineage),
          carryTaskPickupCount: activeCarryController ? carryTaskPickupCount(activeCarryController) : null,
          studentTransportActive: stagedStudentTransport?.active ?? false,
          studentTransportControls: stagedStudentTransport?.controls ?? null,
          studentTransportEnded: stagedStudentTransport?.ended ?? null,
          studentTransportEntry: lastStudentTransportEntry ? { ...lastStudentTransportEntry } : null,
          studentTransportInterval: skillController?.skill?.studentTransportInterval ?? null,
          studentApproachActive: stagedStudentApproach?.active ?? false,
          studentApproachControls: stagedStudentApproach?.controls ?? null,
          studentApproachEnded: stagedStudentApproach?.ended ?? null,
          studentApproachEntry: lastStudentApproachEntry ? { ...lastStudentApproachEntry } : null,
          carrySegmentPreparations: activeCarryController?.segmentPreparations ? structuredClone(activeCarryController.segmentPreparations) : null,
          boxExitClock: boxExitController?.recordClock ?? null,
          boxExitResults: structuredClone(boxExitResults),
          boxTaskResults: structuredClone(boxTaskResults),
          controlPreview: lastControlPreview ? { ...structuredClone(lastControlPreview), evaluatedControls: previewControls } : null,
          studentClosedLoopPreview: studentClosedLoopPreviewEnabled ? { enabled: true, calls: studentClosedLoopPreview?.calls ?? 0,
            policyQueries: studentClosedLoopPreview?.policyQueries ?? 0, totalElapsedMs: studentClosedLoopPreview?.totalElapsedMs ?? 0,
            recent: structuredClone(studentClosedLoopPreviews) } : { enabled: false },
          teacherDescentHold: teacherDescentHoldEnabled ? { enabled: true, active: Boolean(teacherDescentHold?.active), current: teacherDescentHold ? teacherDescentHold.review() : null,
            pending: structuredClone(teacherDescentHoldPending), history: structuredClone(teacherDescentHolds),
            heldClock: activeCarryController?.externalHoldSnapshot ?? null } : { enabled: false },
          recordedApproachHoldActive: recordedApproachHold !== null,
          approachRecoveryActive: skillController instanceof TeacherApproachRecoveryController
            && skillController.phase === 'teacher_settling',
          approachRecoveryControls: skillController instanceof TeacherApproachRecoveryController ? skillController.controls : null,
          waypointApproachActive: pendingWaypointCarryController !== null,
          teacherStandingMode, teacherStandingAlignment, teacherStandingActive: teacherStandingPlan !== null, teacherStandingSteps,
          teacherStandingTarget: teacherStandingPlan ? {
            root: [...teacherStandingPlan.bodyRootTargetPosition], object: [...teacherStandingPlan.objectTargetPosition],
            objectQuaternion: [...teacherStandingPlan.objectTargetQuaternion],
          } : null,
          waypointApproaches: structuredClone(waypointApproaches),
          boxApproachRouting: boxApproachPlanner !== null,
          approachRoute: lastApproachRoute ? structuredClone(lastApproachRoute) : null,
          skillOutcome: skillController?.outcome ? { ...skillController.outcome } : null,
          skillCompletionReason: skillController?.completionReason || null,
          carrySegmentIndex: activeCarryController?.segmentIndex ?? null,
          carrySegmentCount: activeCarryController?.plan?.goals.length ?? null,
          carrySegmentResults: activeCarryController?.segmentResults ? structuredClone(activeCarryController.segmentResults) : null,
          carryReferenceAttempts: activeCarryController?.referenceAttempts ? structuredClone(activeCarryController.referenceAttempts) : null,
          requestedCarryGoal: matchedCarryHost?.originalGoalWorld ?? (activeCarryController?.requestedGoalWorld ? Array.from(activeCarryController.requestedGoalWorld) : null),
          referenceCarryGoal: activeCarryController?.referencePlan?.referenceGoalWorld || null,
          policyKind: lastObservation?.length === 4052 ? 'teacher' : 'student',
          teacherObjectFeatureMaxAbs: lastObservation?.length === 4052 ? Math.max(
            ...lastObservation.slice(1771, 2026).map(Math.abs), ...lastObservation.slice(3797, 4052).map(Math.abs)) : null,
          rootPosWorld: Array.from(data.xpos.slice(pelvisId * 3, pelvisId * 3 + 3)),
          rootQuatXyzwWorld: [q[1], q[2], q[3], q[0]],
          rootGeneralizedVelocity: Array.from(data.qvel.slice(0, 6)),
          uprightScore: 1 - 2 * (q[1] * q[1] + q[2] * q[2]),
          humanGoalWorld: user.humanGoalWorld ? Array.from(user.humanGoalWorld) : null,
          objGoalWorld: user.objGoalWorld ? Array.from(user.objGoalWorld) : null,
          activeObjName: user.activeObjName, objectPoses,
          contactObjectName,
          contacts: contactDiagnostics.read(data, contactObjectId),
          command: lastObservation?.length === 1422 ? Array.from(lastObservation.slice(0, 13)) : null,
          mask: lastObservation?.length === 1422 ? Array.from(lastObservation.slice(-205)) : null,
          rawAction: lastRawAction ? Array.from(lastRawAction) : null,
          fsmState: lastTranslatorResult?.fsmState || translator.fsmState,
          translatorDebug: lastTranslatorResult?.debug || null,
        };
      },
      reset: doReset,
      getRecoveredFacingTurnReview(){return {enabled:urlParams.get('recoveredFacingTurn')==='1',recoveryAdmission:recoveredFacingAdmission.review(),
        probe:terminalFacingProbe?.review()??null,events:structuredClone(terminalFacingEvents),
        skill:terminalFacingProbe?{name:terminalFacingProbe.owner.skill.name,
          sourceFrames:terminalFacingProbe.owner.skill.sourceFrames,locomotionOnly:true,
          objectBodyName:terminalFacingProbe.owner.skill.objectBodyName,
          objectPointsLocal:structuredClone(terminalFacingProbe.owner.skill.objectPointsLocal)}:null};},
      getApproachRecoveryLoopReview(){return {flags:{...recoveryFlags},coordinator:approachRecovery?.review()??null,
        decisions:structuredClone(approachRecoveryDecisions),transientSuspension:structuredClone(transientSuspension),
        standingPreviewOwned:structuredClone(standingPreviewOwned)};},
      getTerminalRefusalRecoveryReview() {
        const recovery = skillController instanceof TeacherApproachRecoveryController ? skillController : null;
        return {
          enabled:restrictedApproachRecoveryEnabled && urlParams.get('terminalRefusalRecovery')==='1',
          episode:episodeVersion, physicalControl:episodeControlStep,
          saved:readExecutedApproachTerminalReview(),
          lastAttempt:structuredClone(terminalRefusalRecoveryAttempt),
          currentRecovery:recovery ? {
            controls:recovery.controls,quiet:recovery.quiet,regionEntry:recovery.regionEntry,
            phase:recovery.phase,completionReason:recovery.completionReason,
            sourceFrames:recovery.skill.sourceFrames,planFrame:Array.from(recovery.plan.frame),
            goalWorld:Array.from(recovery.goalWorld),radius:recovery.radius,
            geometry:structuredClone(recovery.geometry)
          } : null
        };
      },
      getApproachRecoveryReview() {
        return approachRecoveries.map(({ controller, ...review }) => ({ ...structuredClone(review),
          outcome: controller.outcome, records: structuredClone(controller.records),
          geometry: structuredClone(controller.geometry) }));
      },
      getCarryRequestClearance() { return structuredClone(lastCarryRequestClearance); },
      getCarryEntryClearance() { return structuredClone(lastCarryEntryClearance); },



      getFirstPickupRecoveryProgress(){return {
        episode:episodeVersion,physicalControl:episodeControlStep,requestId:activeBoxTaskRequestId,
        decisionCount:firstPickupStudentDecisions.length,admissionCount:firstPickupStudentRegistry.admissionCount,
        controllers:studentApproaches.filter(c=>c.firstPickupOrigin).map(c=>({episode:c.episode,
          startControl:c.startControl,controls:c.controls,horizonControls:c.horizonControls,active:c.active,ended:c.ended})),
        handoffs:firstPickupStudentHandoffs.map(h=>({episode:h.episode,requestId:h.requestId,
          physicalControl:h.physicalControl,arrivalOwner:h.arrivalOwner??null,
          actualStandingControlsBeforeHandoff:h.actualStandingControlsBeforeHandoff??null,
          rawSourceFrames:h.rawSourceFrames,preparedSourceFrames:h.preparedSourceFrames,
          firstTeacherControlCommitted:h.firstTeacherControlCommitted===true,
          actualTeacherSubsteps:h.actualTeacherSubsteps??null,physicalControlAfter:h.physicalControlAfter??null,
          postCarryStandingReanchor:h.postCarryStandingReanchor?{...h.postCarryStandingReanchor}:null})),
        standings:firstPickupStandings.map(c=>c.progress())};},
      getFirstPickupStandingRecoveryReview(){return firstPickupStandings.map(controller=>controller.review());},
      getFirstPickupStudentRecoveryReview() {return {decisions:structuredClone(firstPickupStudentDecisions),
        admissions:firstPickupStudentRegistry.review(),handoffs:structuredClone(firstPickupStudentHandoffs),controllers:studentApproaches
          .filter(c=>c.firstPickupOrigin).map(c=>c.review())};},
      getStudentApproachReview() { return studentApproaches.map(controller => controller.review()); },
      getStudentTransportReview() { return studentTransports.map(controller => controller.review()); },
      getPickupFacingApproachReview() { return { enabled: pickupFacingApproachEnabled, entryRegionEnabled: pickupFacingEntryRegionEnabled,
        plans: structuredClone(pickupFacingPlans), owners: pickupFacingOwners.map(owner => owner.review()) }; },
      getPickupPoseSearchReview() { return { enabled: pickupPoseSearchEnabled, tiers: [...pickupPoseSearchTiers], alternateCarryEnabled,
        reviews: structuredClone(pickupPoseSearchReviews) }; },
      getReferenceApproachReviews(){return{
        studentTurns:referenceStudentTurns.map(c=>c.review()),heightAwareApproaches:heightAwareApproaches.map(c=>c.review())};},
      getHistoryState() {
        return readPolicyHistoryState();
      },
      getTeacherTerminalReference() {
        const carry = activeCarryController;
        const frame = carry?.worldFrames?.[carry.skill.sourceFrames - 1];
        if (!frame) return null;
        return { frame: Array.from(frame), skill: {
          name: carry.skill.name, sourceFrames: carry.skill.sourceFrames,
          objectBodyName: carry.skill.objectBodyName,
          objectPointsLocal: structuredClone(carry.skill.objectPointsLocal),
          locomotionOnly: Boolean(carry.skill.locomotionOnly),
        }, segmentIndex: carry.segmentIndex, completionReason: carry.completionReason };
      },
      getTeacherExecutionReference() {
        const carry = activeCarryController;
        if (carry?.phase !== 'teacher' || !carry.worldFrames) return null;
        return { episodeVersion, episodeControlStep, segmentIndex: carry.segmentIndex,
          referenceIndex: carry.referenceIndex, requestedGoalWorld: Array.from(carry.requestedGoalWorld),
          skill: { name: carry.skill.name, sourceFrames: carry.skill.sourceFrames,
            objectBodyName: carry.skill.objectBodyName, objectPointsLocal: structuredClone(carry.skill.objectPointsLocal) },
          plan: { transform: structuredClone(carry.referencePlan.transform),
            correctionWorld: carry.referencePlan.correctionWorld ? Array.from(carry.referencePlan.correctionWorld) : null,
            approachGoalWorld: carry.referencePlan.approachGoalWorld ? Array.from(carry.referencePlan.approachGoalWorld) : null },
          referenceFrames: carry.worldFrames.map(frame => Array.from(frame)) };
      },
      getIntegrationState() {
        const specification = mujoco.mjtState.mjSTATE_INTEGRATION.value;
        const buffer = new mujoco.DoubleBuffer(mujoco.mj_stateSize(model, specification));
        let builder = null;
        try {
          mujoco.mj_getState(model, data, buffer, specification);
          const skill = [activeCarryController?.skill, skillController?.skill, restrictedController?.skill]
            .find(candidate => typeof candidate?.objectBodyName === 'string' && candidate.objectBodyName.length > 0);
          builder = skill ? new TeacherObsBuilder(mujoco, model, skill) : null;
          return { specification, state: Array.from(buffer.GetView()),
            qaccWarmstart: Array.from(data.qacc_warmstart),
            fixedStandingReference: lastControlPhase === 'teacher_standing'
              && lastRestrictedStep?.referenceFrames ? {
                frame: Array.from(lastRestrictedStep.referenceFrames[0]),
                skill: { name: restrictedController.skill.name,
                  objectBodyName: restrictedController.skill.objectBodyName,
                  objectPointsLocal: structuredClone(restrictedController.skill.objectPointsLocal),
                  locomotionOnly: true },
              } : null,
            // mj_step caches transforms before its final integration. Preserve
            // the teacher's actual observed body states for replay comparisons.
            teacherObserved: builder ? {
              bodies: builder.bodyIds.map(id => builder._bodyState(data, id)),
              object: builder._bodyState(data, builder.objectId), contacts: builder._contacts(data),
            } : null };
        } finally { builder?.dispose(); buffer.delete(); }
      },
      getContactsForObject(name) {
        if (!selectableNames.has(name)) throw new Error(`Unknown scene object: ${name}`);
        return contactDiagnostics.read(data, findBodyIdByName(model, name));
      },
      startPickup,
      startCarry,
      startCarryToGoal,
      projectWorldPoint(value) {
        root.updateMatrixWorld(true);
        camera.updateMatrixWorld(true);
        const point = root.localToWorld(new THREE.Vector3(...vector3(value))).project(camera);
        const rect = canvas.getBoundingClientRect();
        return { clientX: rect.left + (point.x + 1) * rect.width / 2,
          clientY: rect.top + (1 - point.y) * rect.height / 2 };
      },
      setWalkGoal(value) {
        user.releaseKeys();
        user.activeObjName = user.objGoalWorld = null;
        user.humanGoalWorld = value === null ? null : vector3(value);
        if (restrictedMode) submitRestrictedFloorGoal(user.humanGoalWorld);
        picker.syncFromUserState();
      },
      getReleasePresentationState() {
        return releasePresentationContract({ enabled: presentationEnabled, cameraMode,
          interactiveNames: interactiveSelectableNames, hiddenNames: presentationHiddenObjects });
      },
      selectObject(name) {
        if (name !== null && (!interactiveSelectableNames.has(name) || findBodyIdByName(model, name) < 0)) {
          throw new Error(`Unknown scene object: ${name}`);
        }
        user.activeObjName = name;
        user.humanGoalWorld = user.objGoalWorld = null;
        picker.syncFromUserState();
      },
      setObjectGoal(value) {
        if (!user.activeObjName) throw new Error('Select an object first');
        user.objGoalWorld = value === null ? null : vector3(value);
        user.humanGoalWorld = null;
      },
      setObjectFloorGoal(value) {
        user.objGoalWorld = objectGoalFromGround(vector3(value));
        user.humanGoalWorld = null;
      },
      requestCarryDestination(value) {
        // Picker-free evaluation arm: exactly the click path's normalization
        // (objectGoalFromGround → onObjectGoal) for a MuJoCo floor point, with
        // no raycast or camera. Ordinary tasks return their registered request;
        // a loaded retarget returns a distinct pending receipt whose completion
        // resolves after asynchronous preview/publication or concrete refusal.
        const floor = vector3(value);
        const serialBefore = boxTaskRequestSerial;
        const retargetReceiptBefore = loadedRetargetReceiptSerial;
        const decisionSerialBefore = skillDecisionSerial;
        const delivery = picker.requestDestination(floor);
        if (delivery.delivered && loadedRetargetReceiptSerial > retargetReceiptBefore) {
          return latestLoadedRetargetReceipt;
        }
        const requestId = boxTaskRequestSerial > serialBefore ? latestBoxTaskRequestId : null;
        const record = requestId === null ? null : boxTaskRequestLog.records.get(requestId);
        // B9: a push-routed click registers its request only after the lane's `await pause()` (contract delta b), so this
        // synchronous receipt has no record yet. Report that transparently instead of the v5 fallback text.
        const pushPending = record === null && delivery.delivered && skillDecisionSerial > decisionSerialBefore
          && selectedSkill?.decisionId === skillDecisionSerial && selectedSkill.skill === 'push';
        return { requestId, delivered: delivery.delivered,
          disposition: record?.disposition ?? (pushPending ? 'pending' : 'refused'),
          reason: record?.reason ?? (pushPending ? 'push_request_pending' : (delivery.delivered ? 'unsupported_object' : delivery.reason)),
          goalWorld: record?.goalWorld ? Array.from(record.goalWorld) : null,
          selectedObject: user.activeObjName, ...(pushPending ? { selectedSkill: 'push', decisionId: selectedSkill.decisionId } : {}) };
      },
      setKeys(keys) {
        user.releaseKeys();
        for (const key of ['w', 's', 'a', 'd', 'q', 'e']) user[key] = Boolean(keys[key]);
        if (user.wasdActive) user.humanGoalWorld = null;
        if (restrictedMode) restrictedController.requestKeys(user);
      },
      pause,
      resume() { paused = false; },
      async step(count = 1) {
        if (!Number.isInteger(count) || count < 1 || count > 6000) throw new Error('Step count must be between 1 and 6000');
        await pause();
        for (let i = 0; i < count; i++) await runStep();
        renderer.render(scene, camera);
        return this.getState();
      },
    };
  }

  function loop() {
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
    // Sync picker each frame — handles Esc / Reset clearing selection
    // through the keyboard, so highlight/cache stay coherent.
    picker.syncFromUserState();

    if (paused || activeStepPromise || activeLoadedRetargetPromise || pendingPushAdmission) { lastLoopMs = null; return; }
    if (realtimePacing) {
      const now = performance.now();
      stepDebtMs = Math.min(2 * controlPeriodMs, stepDebtMs + (lastLoopMs === null ? controlPeriodMs : now - lastLoopMs));
      lastLoopMs = now;
      if (stepDebtMs < controlPeriodMs - 1e-6) return;
      stepDebtMs -= controlPeriodMs;
    }
    runStep().catch((e) => {
      paused = true;
      console.error('[loop] step error', e);
      setStatus(`ERROR in step(): ${e.message}`);
    });
  }
  if (restrictedMode) warmCarryReferences();
  loop();
}

window.addEventListener('DOMContentLoaded', () => {
  main().catch((e) => {
    console.error(e);
    setStatus(`ERROR: ${e.message}\nSee console for details.`);
  });
});
