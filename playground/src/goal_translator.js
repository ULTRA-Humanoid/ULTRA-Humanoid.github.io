// goal_translator.js — motion-matching goal selector + state machine.
//
// Replaces the keyboard's hand-coded NEW_CMD synthesis with KNN over a
// pre-built database of (match_features, goal_features) sampled from
// real AMASS+BONES+OMOMO clips. Retrieval keeps the command close to reference
// examples; compatibility with the live pose and command history still needs
// rollout validation, especially during abrupt retargeting and interaction.
//
// THE POLICY IS UNTOUCHED. We only change what `new_cmd` and the mask
// block contain at deploy. The matched clip is read only to fetch
// `(root_pos, root_rot, obj_pos)` at t+K — never to drive the robot's
// body, which remains policy-controlled.
//
// See PLAN_GOAL_TRANSLATOR.md for design rationale, including the
// per-frame KNN + hold-window approach and why HOI's object trajectory
// is procedural rather than KNN'd.
//
// Public API:
//
//   const t = new GoalTranslator();
//   await t.load('clip_db.bin', 'clip_db.json');
//   ...
//   each frame: const { goalSpec, mask, longTermT, fsmState } = t.step(input);
//   where input is:
//     {
//       user:   UserState (keyboard + object selection state),
//       proprio: {
//         rootPosWorld:     [x, y, z],
//         rootQuatXyzwWorld:[x, y, z, w],
//         pelvisZ:          number,
//         uprightScore:     number,    // world_up · body_z_axis(R(quat))
//         footContactL:     0|1,
//         footContactR:     0|1,
//         rootHeight:       number,
//         objPosWorld:      [x, y, z] | null,
//         objClass:         string | null,  // for HOI bucket filtering
//       },
//     }
//
// goalSpec is in body (heading-aligned) frame, matching the training
// task's _build_interactive_command layout exactly:
//   { humanTargetPos:Float32Array(3),
//     humanTargetRot:Float32Array(6),  // rot6d
//     objTargetPos:  Float32Array(3),
//     timeToTarget:  number in [0,1] }
//
// mask is in [0, 1] per channel:
//   { keepHumanPos, keepHumanRot, keepObjPos, keepTime }
//
// State machine: IDLE / LOCO / HOI_FULL / HOI_OBJ_ONLY / RECOVER.
// Transitions on WASD/click input + proprio-derived fall detection.

import { Mode } from './state.js';
import { calcHeadingQuatInv, quatRotateOne, quatMulXyzw, quatToRot6d, yawQuat } from './math.js';

// -------------------- Constants -------------------- //

export const FsmState = Object.freeze({
  IDLE:         'IDLE',
  LOCO:         'LOCO',
  HOI_FULL:     'HOI_FULL',
  HOI_OBJ_ONLY: 'HOI_OBJ_ONLY',
  RECOVER:      'RECOVER',
});

const MATCH_DIM = 13;
const GOAL_DIM = 11;
const FPS = 30.0;
const G_MAX_NORMALIZER = 240.0;   // matches V2's long_term_t upper bound

// Match-feature weights — used in the weighted-L2 distance. Tuned by
// inspection (not by gradient): trajectory + height most informative,
// foot contact noisiest.
const MATCH_WEIGHTS = new Float32Array([
  1.0, 1.0,   // traj_vxy_05s_x, _y
  1.0, 1.0,   // traj_vxy_10s_x, _y
  1.0, 1.0,   // traj_vxy_15s_x, _y
  0.5, 0.5,   // facing_cos_05s, sin_05s
  0.5, 0.5,   // facing_cos_10s, sin_10s
  0.2, 0.2,   // foot_contact_L, _R
  1.0,        // root_height
]);

// Top-K buffer size. Deploy row selection is deterministic top-1 to match the
// trusted Python/MuJoCo gate. The softmax sampler is kept only for future
// explicitly-enabled style variation experiments.
const K_TOP = 8;
// Achievement-gated re-query for object rows. Locomotion commands are compiled
// live from the current root pose every frame, matching the MuJoCo metric gate's
// default clip_match path. The older browser-style held-row mode was useful as
// a diagnostic, but MuJoCo rejected it for behind goals and it makes held WASD
// commands feel brittle.
//
// Reach thresholds: 0.30 m for the human-target body position (~ one
// stride), 0.15 m for the object target (placement precision).
const REACHED_THRESHOLD_HUMAN = 0.30;
const REACHED_THRESHOLD_OBJ   = 0.15;
// Safety floor on the hold time: don't re-query within this many frames
// no matter what (besides intent edges). Avoids re-querying in the same
// frame we match if proprio happens to land near the goal.
const MIN_HOLD_FRAMES = 5;
// Safety ceiling: never hold longer than this even if K_frames is huge.
// (IDLE K is 180 frames = 6 s, which is fine; the ceiling exists for
// future buckets with very long K.)
const MAX_HOLD_FRAMES = 240;
// Re-query when the live query has moved farther than this from the
// held row's match features (weighted-L2 in same units as KNN distance).
const REQUERY_DIST_THRESH = 1.0;

// WASD → trajectory mapping (matches training-time interpretation).
// W/S move along body-frame +x (forward) at WALK_SPEED; A/D along +y
// (left). Q/E rotate yaw at YAW_RATE rad/s.
const WALK_SPEED_M_PER_S = 1.0;
const CLICK_STRAFE_FACTOR = 0.5;      // validated click-goal lateral query scale
const KEYBOARD_STRAFE_FACTOR = 0.25;  // held D at 0.5 collapsed; 0.25 keeps keyboard conservative
const BACK_FACTOR = 0.5;       // backward slower than forward
const YAW_RATE_RAD_PER_S = 0.5;
const GOAL_ALIGN_WEIGHT = 2.0;  // match Python ClipLocoCommandCompiler
// Legacy behind-goal turn-in-place query. The validated MuJoCo/browser deploy
// path uses route-behind instead; keeping this disabled preserves parity after
// the route stops applying near the final goal.
const TURN_BEHIND_ENABLED = false;
const TURN_BEHIND_ANGLE_RAD = 135.0 * Math.PI / 180.0;
const TURN_FORWARD_SPEED_M_PER_S = 0.25;
const TURN_YAW_RATE_RAD_PER_S = 0.5;
const ROUTE_BEHIND_ENABLED = true;
const ROUTE_BEHIND_ANGLE_RAD = 135.0 * Math.PI / 180.0;
const ROUTE_BEHIND_FORWARD_M = -0.5;
const ROUTE_BEHIND_LATERAL_M = 0.35;
const ROUTE_BEHIND_MIN_DIST_M = 0.35;
const ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT = 2.0;
const ROUTE_BEHIND_RELEASE_ON_SIGN_FLIP = false;
const ROUTE_BEHIND_RELEASE_LATERAL_PROGRESS_M = 0.0;
const ARRIVAL_RECEDING_RADIUS_M = 0.45;
const ARRIVAL_RECEDING_MAX_ABS_ANGLE_RAD = 135.0 * Math.PI / 180.0;
const ARRIVAL_SEGMENT_DISTANCE_M = 0.25;
const ARRIVAL_HORIZON_FRAMES = 60;
const HOLD_ON_SUCCESS_ENABLED = true;
const HOLD_REACH_RADIUS_M = 0.25;
const HOLD_SEGMENT_DISTANCE_M = 0.15;
const HOLD_TIME_TO_TARGET = 0.25;
// Validated against the MuJoCo goal-locomotion gate on 2026-05-31:
// low-pass only the live clicked-goal waypoint in world space. This preserves
// deterministic top-1 clip matching while removing frame-to-frame command
// jitter before the waypoint is converted into the policy's body-frame command.
const LIVE_GOAL_WORLD_TARGET_SMOOTH_ALPHA = 0.75;
const LIVE_GOAL_WORLD_TARGET_MAX_STEP_M = 0.0;
const CLICK_POSITION_SOURCE = 'matched';
const CLICK_POSITION_SOURCES = new Set(['matched', 'stable_receding']);
const STABLE_RECEDING_SEGMENT_DISTANCE_M = 0.5;
const STAND_ANCHOR_ENABLED = true;
const STAND_ANCHOR_SEGMENT_DISTANCE_M = 0.15;
const STAND_ANCHOR_TIME_TO_TARGET = 0.25;

// Fall detection thresholds
const FALL_PELVIS_Z = 0.5;
const FALL_UPRIGHT = 0.4;
const FALL_SUSTAINED_FRAMES = 8;
const RECOVER_PELVIS_Z = 0.7;
const RECOVER_UPRIGHT = 0.85;
const RECOVER_SUSTAINED_FRAMES = 8;

// Keep deploy masks binary. The MuJoCo gate and training obs use hard mask
// bits; fractional mask ramping was a browser-only nicety that can push the
// goal encoder off distribution during quick key transitions.
const MASK_LERP_RATE = 1.0;

// Procedural object trajectory (HOI). T = clamp(distance / speed, ...).
const PROC_OBJ_SPEED_M_PER_S = 0.5;
const PROC_OBJ_MIN_T_SEC = 1.5;
const PROC_OBJ_MAX_T_SEC = 4.0;


// -------------------- Helpers -------------------- //

function clamp(x, lo, hi) {
  return x < lo ? lo : x > hi ? hi : x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(x) {
  const s = clamp(x, 0, 1);
  return s * s * (3 - 2 * s);
}

// Heading-only forward yaw quat (inverse of math.js calcHeadingQuatInv).
// Rotates body-frame deltas into world-frame deltas.
function headingQuatForward(rootQuatXyzw) {
  const [x, y, z, w] = rootQuatXyzw;
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const half = 0.5 * yaw;
  return [0, 0, Math.sin(half), Math.cos(half)];
}

function headingYaw(rootQuatXyzw) {
  const [x, y, z, w] = rootQuatXyzw;
  const n = Math.max(Math.sqrt(x*x + y*y + z*z + w*w), 1e-9);
  const qx = x / n, qy = y / n, qz = z / n, qw = w / n;
  const fx = 1.0 - 2.0 * (qy * qy + qz * qz);
  const fy = 2.0 * (qx * qy + qw * qz);
  return Math.atan2(fy, fx);
}

function softmaxSample(distances, tau, rng) {
  // Sample one index in [0, K) with weights = softmax(-distances / tau).
  const K = distances.length;
  if (K === 0) return 0;
  // Numerically stable softmax
  let mn = Infinity;
  for (let i = 0; i < K; i++) if (-distances[i] / tau > -Infinity) mn = Math.min(mn, distances[i]);
  let sum = 0;
  const w = new Float32Array(K);
  for (let i = 0; i < K; i++) {
    w[i] = Math.exp(-(distances[i] - mn) / tau);
    sum += w[i];
  }
  let r = (rng() * sum);
  for (let i = 0; i < K; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return K - 1;
}


// -------------------- DB loader -------------------- //

async function fetchBinary(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url} failed: ${resp.status}`);
  return resp.json();
}


// -------------------- GoalTranslator -------------------- //

export class GoalTranslator {
  constructor(options = {}) {
    this.clickPositionSource = CLICK_POSITION_SOURCES.has(options.clickPositionSource)
      ? options.clickPositionSource
      : CLICK_POSITION_SOURCE;
    this.liveGoalWorldTargetMaxStepM = Number.isFinite(options.liveGoalWorldTargetMaxStepM)
      ? Math.max(0.0, options.liveGoalWorldTargetMaxStepM)
      : LIVE_GOAL_WORLD_TARGET_MAX_STEP_M;

    // Loaded DB
    this.nRows = 0;
    this.matchData = null;     // Float32Array(N * MATCH_DIM)
    this.goalData  = null;     // Float32Array(N * GOAL_DIM)
    this.meta = null;          // Array<{bucket, source, clip_id, frame, obj_class?, mirror?}>
    this.bucketIndices = null; // {IDLE: Int32Array, LOCO: ..., HOI: ..., GETUP: ...}
    this.hoiByClass = null;    // Map<string, Int32Array>
    this.manifest = null;

    this.rng = Math.random;
    this.reset();
  }

  // Clear episode state while keeping the loaded clip database and options.
  // Resetting only MuJoCo leaves goals anchored to the previous episode and
  // can even leave a freshly standing robot stuck in RECOVER.
  reset() {

    // FSM state
    this.fsmState = FsmState.IDLE;
    this.fallSustained = 0;
    this.recoverSustained = 0;

    // Cursor / hold-window state
    this.selectedRow = -1;
    this.holdFrames = 1e9;
    this.intentFpPrev = null;
    this.framesSinceIntentEdge = 0;
    this.lastMatchDistance = 0;

    // Held world-frame target — set at match time, ONLY changes on
    // re-query. The body-frame goal fed to the policy each frame is
    // (heldWorldTarget - current_root) rotated by heading_inv(current),
    // matching V2 training's `_build_interactive_command` exactly.
    // This way the policy sees the body-frame delta SHRINK as it walks
    // toward the target — the "you've arrived" signal — and ROTATE as
    // it turns. Storing the snapshot body-frame delta directly (the
    // earlier bug) yielded a goal that drifted with the robot.
    this.heldHumanTargetWorld = null;        // [x, y, z]   or null
    this.heldHumanTargetRotWorld = null;     // [x, y, z, w] (full quat) or null
    this.heldObjTargetWorld = null;          // [x, y, z]   or null (OMOMO HOI only)
    this.heldKFrames = MAX_HOLD_FRAMES;
    this.lastReachDistHuman = Infinity;
    this.lastReachDistObj = Infinity;
    this.humanGoalHoldActive = false;
    this.humanGoalHoldFp = null;

    // Mask state (exponential lerp toward `target`)
    this.maskCurrent = { keepHumanPos: 0, keepHumanRot: 0, keepObjPos: 0, keepObjPoints: 0, keepTime: 1 };

    // Reusable scratch buffers (no per-frame allocation)
    this._queryBuf = new Float32Array(MATCH_DIM);
    this._topKDist = new Float32Array(K_TOP);
    this._topKIdx  = new Int32Array(K_TOP);
    this._queryGoalDir = null;       // body-frame xy unit direction for click-goals
    this._usingHumanGoal = false;
    this._turnBehindActive = false;
    this._routeBehindActive = false;
    this._routeBehindSign = 0.0;
    this._routeGoalDist = 0.0;
    this._routeTargetBody = null;
    this._routeBehindOriginWorld = null;
    this._routeBehindLateralAxisWorld = [0.0, 0.0];
    this._standAnchorWorld = null;
    this._prevCommandActive = false;
    this._smoothHumanTargetWorld = null;

  }

  async load(binUrl, jsonUrl) {
    const [binBytes, manifest] = await Promise.all([
      fetchBinary(binUrl),
      fetchJson(jsonUrl),
    ]);
    this.manifest = manifest;
    this.nRows = manifest.n_rows;
    if (manifest.match_dim !== MATCH_DIM || manifest.goal_dim !== GOAL_DIM) {
      throw new Error(`DB schema mismatch: expected match=${MATCH_DIM} goal=${GOAL_DIM}, got match=${manifest.match_dim} goal=${manifest.goal_dim}`);
    }
    if (manifest.binary_row_stride_floats !== MATCH_DIM + GOAL_DIM) {
      throw new Error(`unexpected binary stride ${manifest.binary_row_stride_floats}`);
    }
    if (binBytes.byteLength !== this.nRows * (MATCH_DIM + GOAL_DIM) * 4) {
      throw new Error(`bin size mismatch: ${binBytes.byteLength} vs ${this.nRows * (MATCH_DIM + GOAL_DIM) * 4}`);
    }

    // Split flat row-major float32 into match + goal halves. We do this
    // because the KNN inner loop is faster on contiguous match-only data.
    const flatF32 = new Float32Array(binBytes.buffer, binBytes.byteOffset, this.nRows * (MATCH_DIM + GOAL_DIM));
    this.matchData = new Float32Array(this.nRows * MATCH_DIM);
    this.goalData  = new Float32Array(this.nRows * GOAL_DIM);
    const stride = MATCH_DIM + GOAL_DIM;
    for (let i = 0; i < this.nRows; i++) {
      const srcOff = i * stride;
      const mDst = i * MATCH_DIM;
      const gDst = i * GOAL_DIM;
      for (let j = 0; j < MATCH_DIM; j++) this.matchData[mDst + j] = flatF32[srcOff + j];
      for (let j = 0; j < GOAL_DIM; j++)  this.goalData[gDst + j]  = flatF32[srcOff + MATCH_DIM + j];
    }

    // Build per-bucket index arrays
    this.meta = manifest.rows_meta;
    const buckets = { IDLE: [], LOCO: [], HOI: [], GETUP: [] };
    const hoiByClass = new Map();
    for (let i = 0; i < this.nRows; i++) {
      const m = this.meta[i];
      buckets[m.bucket].push(i);
      if (m.bucket === 'HOI') {
        const c = m.obj_class || '';
        if (!hoiByClass.has(c)) hoiByClass.set(c, []);
        hoiByClass.get(c).push(i);
      }
    }
    this.bucketIndices = {};
    for (const k of Object.keys(buckets)) this.bucketIndices[k] = new Int32Array(buckets[k]);
    this.hoiByClass = new Map();
    for (const [k, v] of hoiByClass.entries()) this.hoiByClass.set(k, new Int32Array(v));

    console.log(`[GoalTranslator] loaded ${this.nRows} rows  ` +
      Object.entries(this.bucketIndices).map(([k, v]) => `${k}=${v.length}`).join(' '));
    console.log(`[GoalTranslator] HOI obj classes: ${[...this.hoiByClass.keys()].map(k => `${k}(${this.hoiByClass.get(k).length})`).join(', ')}`);
  }

  setClickPositionSource(source) {
    if (!CLICK_POSITION_SOURCES.has(source)) {
      throw new Error(`Unknown click position source: ${source}`);
    }
    this.clickPositionSource = source;
    this._smoothHumanTargetWorld = null;
  }

  setLiveGoalWorldTargetMaxStepM(maxStepM) {
    this.liveGoalWorldTargetMaxStepM = Number.isFinite(maxStepM)
      ? Math.max(0.0, maxStepM)
      : LIVE_GOAL_WORLD_TARGET_MAX_STEP_M;
  }

  // -------------------- FSM transitions -------------------- //

  _updateFsm(user, proprio) {
    // Recover detection — only relevant when we WEREN'T recovering before.
    if (this.fsmState !== FsmState.RECOVER) {
      const isFallenNow = proprio.pelvisZ < FALL_PELVIS_Z && proprio.uprightScore < FALL_UPRIGHT;
      this.fallSustained = isFallenNow ? this.fallSustained + 1 : 0;
      if (this.fallSustained >= FALL_SUSTAINED_FRAMES) {
        this.fsmState = FsmState.RECOVER;
        this.fallSustained = 0;
        this.recoverSustained = 0;
        return;
      }
    } else {
      // In RECOVER, watch for the exit condition
      const isUprightNow = proprio.pelvisZ > RECOVER_PELVIS_Z && proprio.uprightScore > RECOVER_UPRIGHT;
      this.recoverSustained = isUprightNow ? this.recoverSustained + 1 : 0;
      if (this.recoverSustained >= RECOVER_SUSTAINED_FRAMES) {
        this.fsmState = FsmState.IDLE;
        this.recoverSustained = 0;
      }
      return;   // While recovering, ignore keyboard
    }

    // Not recovering: derive FSM state from user input
    const hasHumanGoal = user.humanGoalWorld !== null;
    const hasObjGoal = user.objGoalWorld !== null && user.activeObjName !== null;
    const wasd = user.wasdActive;
    if ((wasd || hasHumanGoal) && hasObjGoal) this.fsmState = FsmState.HOI_FULL;
    else if (wasd || hasHumanGoal)             this.fsmState = FsmState.LOCO;
    else if (hasObjGoal)     this.fsmState = FsmState.HOI_OBJ_ONLY;
    else                     this.fsmState = FsmState.IDLE;
  }

  // -------------------- Query construction -------------------- //

  _buildQuery(user, proprio) {
    const q = this._queryBuf;
    q.fill(0);
    this._queryGoalDir = null;
    this._usingHumanGoal = false;
    this._turnBehindActive = false;
    this._routeBehindActive = false;
    this._routeGoalDist = 0.0;
    this._routeTargetBody = null;
    // facing identity by default
    q[6] = 1.0;  // cos +0.5s
    q[8] = 1.0;  // cos +1.0s
    q[7] = 0.0;
    q[9] = 0.0;
    // Match scripts/goal_clip_command.py's accepted MuJoCo gate: it compiles
    // deploy commands with both contact bits on instead of live foot contacts.
    // Live contacts are noisy across MuJoCo/WASM and are low-weight in the DB;
    // keeping them fixed avoids gait-phase row flicker in the browser.
    q[10] = 1.0;
    q[11] = 1.0;
    q[12] = proprio.rootHeight;

    if (this.fsmState === FsmState.IDLE
        || this.fsmState === FsmState.HOI_OBJ_ONLY
        || this.fsmState === FsmState.RECOVER) {
      // traj = 0, facing identity, foot+height from proprio. Done.
      return q;
    }

    // LOCO or HOI_FULL — integrate WASD over the lookahead windows
    let fwd = 0, str = 0;
    if (user.humanGoalWorld !== null) {
      const dWorld = [
        user.humanGoalWorld[0] - proprio.rootPosWorld[0],
        user.humanGoalWorld[1] - proprio.rootPosWorld[1],
        0.0, // A floor click specifies navigation XY, never pelvis height.
      ];
      const headingInv = calcHeadingQuatInv(proprio.rootQuatXyzwWorld);
      const dBody = quatRotateOne(headingInv, dWorld);
      const dist = Math.sqrt(dBody[0] * dBody[0] + dBody[1] * dBody[1]);
      if (dist > 1e-6) {
        const dirX = dBody[0] / dist;
        const dirY = dBody[1] / dist;
        const targetAngle = Math.atan2(dirY, dirX);
        const rawRouteBehind = ROUTE_BEHIND_ENABLED
            && dBody[0] < 0.0
            && dist > ROUTE_BEHIND_MIN_DIST_M
            && Math.abs(targetAngle) >= ROUTE_BEHIND_ANGLE_RAD;
        // A straight-behind target has two equivalent routes. Rotation into
        // the heading frame can leave a tiny signed lateral residual; using
        // its sign made the same goal choose opposite rows at different world
        // headings. Resolve only that numerical tie consistently to the left.
        const lateralTie = Math.abs(dirY) <= 1e-6;
        const targetSign = lateralTie ? 1.0 : (targetAngle >= 0.0 ? 1.0 : -1.0);
        const routeSignMismatch = ROUTE_BEHIND_RELEASE_ON_SIGN_FLIP
            && this._routeBehindSign !== 0.0
            && targetSign !== this._routeBehindSign;
        let routeProgressReleased = false;
        if (ROUTE_BEHIND_RELEASE_LATERAL_PROGRESS_M > 0.0
            && this._routeBehindOriginWorld !== null) {
          const pX = proprio.rootPosWorld[0] - this._routeBehindOriginWorld[0];
          const pY = proprio.rootPosWorld[1] - this._routeBehindOriginWorld[1];
          const routeProgress = pX * this._routeBehindLateralAxisWorld[0]
              + pY * this._routeBehindLateralAxisWorld[1];
          routeProgressReleased = routeProgress >= ROUTE_BEHIND_RELEASE_LATERAL_PROGRESS_M;
        }
        const routeBehind = rawRouteBehind && !routeSignMismatch && !routeProgressReleased;
        const turnBehind = !routeBehind
            && TURN_BEHIND_ENABLED
            && dirX < 0.0
            && Math.abs(targetAngle) >= TURN_BEHIND_ANGLE_RAD;
        if (routeBehind) {
          if (this._routeBehindSign === 0.0) {
            this._routeBehindSign = targetSign;
            const yaw0 = headingYaw(proprio.rootQuatXyzwWorld);
            this._routeBehindOriginWorld = [
              proprio.rootPosWorld[0],
              proprio.rootPosWorld[1],
            ];
            this._routeBehindLateralAxisWorld = [
              -Math.sin(yaw0) * this._routeBehindSign,
              Math.cos(yaw0) * this._routeBehindSign,
            ];
          }
          const routeSign = this._routeBehindSign;
          const routeX = Math.min(ROUTE_BEHIND_FORWARD_M, Math.max(0.05, 0.5 * dist));
          const routeY = routeSign * Math.min(ROUTE_BEHIND_LATERAL_M, Math.max(0.05, 0.75 * dist));
          const routeNorm = Math.max(Math.sqrt(routeX * routeX + routeY * routeY), 1e-6);
          fwd = routeX / routeNorm;
          str = routeY / routeNorm;
          this._queryGoalDir = [fwd, str];
          this._routeBehindActive = true;
          this._routeGoalDist = routeNorm;
          this._routeTargetBody = [routeX, routeY, 0.0];
          q[6] = Math.cos(routeSign * TURN_YAW_RATE_RAD_PER_S * 0.5);
          q[7] = Math.sin(routeSign * TURN_YAW_RATE_RAD_PER_S * 0.5);
          q[8] = Math.cos(routeSign * TURN_YAW_RATE_RAD_PER_S);
          q[9] = Math.sin(routeSign * TURN_YAW_RATE_RAD_PER_S);
        } else if (turnBehind) {
          const turnSign = targetAngle >= 0.0 ? 1.0 : -1.0;
          fwd = TURN_FORWARD_SPEED_M_PER_S / WALK_SPEED_M_PER_S;
          str = 0.0;
          this._queryGoalDir = [1.0, 0.0];
          this._turnBehindActive = true;
          q[6] = Math.cos(turnSign * TURN_YAW_RATE_RAD_PER_S * 0.5);
          q[7] = Math.sin(turnSign * TURN_YAW_RATE_RAD_PER_S * 0.5);
          q[8] = Math.cos(turnSign * TURN_YAW_RATE_RAD_PER_S);
          q[9] = Math.sin(turnSign * TURN_YAW_RATE_RAD_PER_S);
        } else {
          fwd = dirX;
          str = dirY;
          this._queryGoalDir = [dirX, dirY];
        }
        this._usingHumanGoal = true;
      }
    } else {
      if (user.w) fwd += 1.0;
      if (user.s) fwd -= BACK_FACTOR;
      if (user.a) str += KEYBOARD_STRAFE_FACTOR;
      if (user.d) str -= KEYBOARD_STRAFE_FACTOR;
      // Hybrid keyboard mode only (hybrid_keyboard_arbiter.js attaches `keyboardShaping` to a
      // UserState view): uniform scale of the encoded translation. Ordinary UserState objects
      // carry no shaping, so the query is byte-identical to the evaluated default.
      const shaping = user.keyboardShaping ?? null;
      if (shaping && Number.isFinite(shaping.translationScale)) { fwd *= shaping.translationScale; str *= shaping.translationScale; }
      const dirNorm = Math.sqrt(fwd * fwd + str * str);
      if (dirNorm > 1e-6) {
        this._queryGoalDir = [fwd / dirNorm, str / dirNorm];
      }
    }
    const vx = fwd * WALK_SPEED_M_PER_S;
    const vy = str * WALK_SPEED_M_PER_S;
    const scaledVx = this._usingHumanGoal && vx < 0 ? vx * BACK_FACTOR : vx;
    const scaledVy = this._usingHumanGoal ? vy * CLICK_STRAFE_FACTOR : vy;
    q[0] = scaledVx; q[1] = scaledVy;
    q[2] = scaledVx; q[3] = scaledVy;
    q[4] = scaledVx; q[5] = scaledVy;

    // Yaw integration: Q = +yaw (left), E = -yaw (right)
    if (!this._turnBehindActive && !this._routeBehindActive) {
      let yawRate = 0;
      if (user.q) yawRate += 1.0;
      if (user.e) yawRate -= 1.0;
      // Hybrid keyboard mode only: the fixed student turns 3-6x faster than the encoded 0.5 rad/s.
      if (user.humanGoalWorld === null && Number.isFinite(user.keyboardShaping?.yawScale)) yawRate *= user.keyboardShaping.yawScale;
      const yaw05 = yawRate * YAW_RATE_RAD_PER_S * 0.5;
      const yaw10 = yawRate * YAW_RATE_RAD_PER_S * 1.0;
      q[6] = Math.cos(yaw05);
      q[7] = Math.sin(yaw05);
      q[8] = Math.cos(yaw10);
      q[9] = Math.sin(yaw10);
    }
    return q;
  }

  _intentFingerprint(user) {
    // Hash the user input + FSM into a small integer. On change → re-query.
    let h = 0;
    h = h * 2 + (user.w ? 1 : 0);
    h = h * 2 + (user.s ? 1 : 0);
    h = h * 2 + (user.a ? 1 : 0);
    h = h * 2 + (user.d ? 1 : 0);
    h = h * 2 + (user.q ? 1 : 0);
    h = h * 2 + (user.e ? 1 : 0);
    h = h * 2 + (user.humanGoalWorld ? 1 : 0);
    if (user.humanGoalWorld) {
      h = ((h * 31) + Math.round(user.humanGoalWorld[0] * 20)) | 0;
      h = ((h * 31) + Math.round(user.humanGoalWorld[1] * 20)) | 0;
      h = ((h * 31) + Math.round(user.humanGoalWorld[2] * 20)) | 0;
    }
    h = h * 8 + (user.activeObjName ? 1 : 0);
    // FSM state — bake in
    const fsmCode = { IDLE: 0, LOCO: 1, HOI_FULL: 2, HOI_OBJ_ONLY: 3, RECOVER: 4 }[this.fsmState];
    h = h * 8 + fsmCode;
    // Object class (only matters for HOI)
    if (this.fsmState === FsmState.HOI_FULL || this.fsmState === FsmState.HOI_OBJ_ONLY) {
      const cls = user.activeObjName || '';
      for (let i = 0; i < cls.length; i++) h = ((h * 31) + cls.charCodeAt(i)) | 0;
    }
    return h;
  }

  // -------------------- Candidate subset -------------------- //

  _candidateSubset(user) {
    switch (this.fsmState) {
      case FsmState.IDLE:         return this.bucketIndices.IDLE;
      case FsmState.LOCO:         return this.bucketIndices.LOCO;
      case FsmState.RECOVER:      return this.bucketIndices.GETUP;
      case FsmState.HOI_FULL:
      case FsmState.HOI_OBJ_ONLY: {
        // Filter HOI by object class if known; fall back to all HOI.
        // For HOI_FULL, also include LOCO so the body channel can produce
        // walking goals when the user is walking toward a clicked object.
        const cls = user.activeObjName ? this._objClassFromName(user.activeObjName) : null;
        const hoi = cls && this.hoiByClass.has(cls) ? this.hoiByClass.get(cls) : this.bucketIndices.HOI;
        if (this.fsmState === FsmState.HOI_FULL) {
          // Concatenate hoi + loco indices
          const out = new Int32Array(hoi.length + this.bucketIndices.LOCO.length);
          out.set(hoi, 0);
          out.set(this.bucketIndices.LOCO, hoi.length);
          return out;
        }
        return hoi;
      }
    }
    return this.bucketIndices.IDLE;
  }

  _objClassFromName(activeObjName) {
    // Browser uses 'active_largebox_080_080_080'. The Python side encodes
    // object class as the second underscore-token, but here the prefix is
    // 'active_' so we strip it and re-split.
    if (!activeObjName) return null;
    let s = activeObjName.startsWith('active_') ? activeObjName.slice('active_'.length) : activeObjName;
    return s.split('_')[0];
  }

  // -------------------- KNN -------------------- //

  _knnTopK(candidateIdx, query) {
    // Compute weighted-L2 squared distances over candidateIdx and
    // maintain a top-K min-heap. Returns nothing; writes _topKDist /
    // _topKIdx in ascending-distance order.
    const w = MATCH_WEIGHTS;
    const M = this.matchData;
    const G = this.goalData;
    const D = MATCH_DIM;
    const K = K_TOP;
    const N = candidateIdx.length;
    const goalDir = this._queryGoalDir;

    // Initialize topK with first K
    const topD = this._topKDist;
    const topI = this._topKIdx;
    const realK = Math.min(K, N);
    for (let k = 0; k < realK; k++) {
      const ri = candidateIdx[k];
      let d2 = 0;
      const off = ri * D;
      for (let j = 0; j < D; j++) {
        const e = (M[off + j] - query[j]);
        d2 += w[j] * e * e;
      }
      if (goalDir !== null) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const gn = Math.sqrt(gx * gx + gy * gy);
        if (gn > 1e-6) {
          const align = (gx / gn) * goalDir[0] + (gy / gn) * goalDir[1];
          const miss = 1.0 - align;
          d2 += GOAL_ALIGN_WEIGHT * miss * miss;
        }
      }
      if (this._routeBehindActive && ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT > 0) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const miss = Math.sqrt(gx * gx + gy * gy) - this._routeGoalDist;
        d2 += ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT * miss * miss;
      }
      topD[k] = d2;
      topI[k] = ri;
    }
    // Heapify max-heap on topD/topI (so we can swap the max out)
    function siftDown(start, end) {
      let root = start;
      while ((root * 2 + 1) < end) {
        let child = root * 2 + 1;
        if (child + 1 < end && topD[child] < topD[child + 1]) child += 1;
        if (topD[root] < topD[child]) {
          const td = topD[root]; topD[root] = topD[child]; topD[child] = td;
          const ti = topI[root]; topI[root] = topI[child]; topI[child] = ti;
          root = child;
        } else return;
      }
    }
    for (let s = Math.floor(realK / 2) - 1; s >= 0; s--) siftDown(s, realK);

    // Scan the rest, replacing max when smaller distance found
    let maxd = topD[0];
    for (let i = realK; i < N; i++) {
      const ri = candidateIdx[i];
      let d2 = 0;
      const off = ri * D;
      for (let j = 0; j < D; j++) {
        const e = (M[off + j] - query[j]);
        d2 += w[j] * e * e;
      }
      if (goalDir !== null) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const gn = Math.sqrt(gx * gx + gy * gy);
        if (gn > 1e-6) {
          const align = (gx / gn) * goalDir[0] + (gy / gn) * goalDir[1];
          const miss = 1.0 - align;
          d2 += GOAL_ALIGN_WEIGHT * miss * miss;
        }
      }
      if (this._routeBehindActive && ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT > 0) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const miss = Math.sqrt(gx * gx + gy * gy) - this._routeGoalDist;
        d2 += ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT * miss * miss;
      }
      if (d2 < maxd) {
        topD[0] = d2;
        topI[0] = ri;
        siftDown(0, realK);
        maxd = topD[0];
      }
    }
    // Sort topK ascending (insertion sort, K is tiny)
    for (let i = 1; i < realK; i++) {
      const td = topD[i], ti = topI[i];
      let j = i - 1;
      while (j >= 0 && topD[j] > td) {
        topD[j + 1] = topD[j];
        topI[j + 1] = topI[j];
        j -= 1;
      }
      topD[j + 1] = td;
      topI[j + 1] = ti;
    }
    return realK;
  }

  _argminRow(candidateIdx, query) {
    // Deterministic deploy path: scan candidates in DB order and return the
    // exact argmin, matching scripts/goal_clip_command.py's np/torch argmin.
    const w = MATCH_WEIGHTS;
    const M = this.matchData;
    const G = this.goalData;
    const D = MATCH_DIM;
    const goalDir = this._queryGoalDir;
    let bestIdx = -1;
    let bestD2 = Infinity;

    for (let i = 0; i < candidateIdx.length; i++) {
      const ri = candidateIdx[i];
      let d2 = 0;
      const off = ri * D;
      for (let j = 0; j < D; j++) {
        const e = (M[off + j] - query[j]);
        d2 += w[j] * e * e;
      }
      if (goalDir !== null) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const gn = Math.sqrt(gx * gx + gy * gy);
        if (gn > 1e-6) {
          const align = (gx / gn) * goalDir[0] + (gy / gn) * goalDir[1];
          const miss = 1.0 - align;
          d2 += GOAL_ALIGN_WEIGHT * miss * miss;
        }
      }
      if (this._routeBehindActive && ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT > 0) {
        const goff = ri * GOAL_DIM;
        const gx = G[goff + 0];
        const gy = G[goff + 1];
        const miss = Math.sqrt(gx * gx + gy * gy) - this._routeGoalDist;
        d2 += ROUTE_BEHIND_GOAL_DISTANCE_WEIGHT * miss * miss;
      }
      if (d2 < bestD2) {
        bestD2 = d2;
        bestIdx = ri;
      }
    }

    this.lastMatchDistance = Math.sqrt(bestD2);
    return bestIdx;
  }

  _selectRow(candidateIdx, query, stochastic = true) {
    // stochastic=true: softmax-sample within top-K (use on intent edges
    //                  to add variety on each new behavior).
    // stochastic=false: deterministic top-1 (use for forced HOLD_WINDOW
    //                   re-queries and drift triggers — we want the
    //                   best match here, no random switching).
    if (!stochastic) return this._argminRow(candidateIdx, query);
    const realK = this._knnTopK(candidateIdx, query);
    if (realK === 0) return -1;
    const dists = new Float32Array(realK);
    for (let k = 0; k < realK; k++) dists[k] = Math.sqrt(this._topKDist[k]);
    // Tau scales with the median of the top-K distances. Floor to avoid
    // divide-by-zero when the top-K all match exactly.
    const sorted = [...dists].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const tau = Math.max(median / 3.0, 0.05);
    const sampledK = softmaxSample(dists, tau, this.rng);
    this.lastMatchDistance = dists[sampledK];
    return this._topKIdx[sampledK];
  }

  // -------------------- Goal + mask -------------------- //

  _readGoalRow(rowIdx, outHumanPos, outHumanRotQuat, outObjPos) {
    const off = rowIdx * GOAL_DIM;
    outHumanPos[0] = this.goalData[off + 0];
    outHumanPos[1] = this.goalData[off + 1];
    outHumanPos[2] = this.goalData[off + 2];
    outHumanRotQuat[0] = this.goalData[off + 3];
    outHumanRotQuat[1] = this.goalData[off + 4];
    outHumanRotQuat[2] = this.goalData[off + 5];
    outHumanRotQuat[3] = this.goalData[off + 6];
    outObjPos[0] = this.goalData[off + 7];
    outObjPos[1] = this.goalData[off + 8];
    outObjPos[2] = this.goalData[off + 9];
    return this.goalData[off + 10];  // K_frames
  }

  _maskTarget() {
    // Base mask per FSM state
    switch (this.fsmState) {
      case FsmState.IDLE:
        if (STAND_ANCHOR_ENABLED) {
          return { keepHumanPos: 1, keepHumanRot: 1, keepObjPos: 0, keepTime: 1 };
        }
        return { keepHumanPos: 0, keepHumanRot: 0, keepObjPos: 0, keepTime: 1 };
      case FsmState.LOCO:
        return { keepHumanPos: 1, keepHumanRot: 1, keepObjPos: 0, keepTime: 1 };
      case FsmState.HOI_FULL:
        return { keepHumanPos: 1, keepHumanRot: 1, keepObjPos: 1, keepTime: 1 };
      case FsmState.HOI_OBJ_ONLY:
        return { keepHumanPos: 0, keepHumanRot: 0, keepObjPos: 1, keepTime: 1 };
      case FsmState.RECOVER:
        // Tell policy to "stand here." Don't constrain facing direction
        // (mask out humanRot). No object goal.
        return { keepHumanPos: 1, keepHumanRot: 0, keepObjPos: 0, keepTime: 1 };
    }
    return { keepHumanPos: 0, keepHumanRot: 0, keepObjPos: 0, keepTime: 1 };
  }

  _stepMaskLerp(target) {
    const m = this.maskCurrent;
    const r = MASK_LERP_RATE;
    m.keepHumanPos += (target.keepHumanPos - m.keepHumanPos) * r;
    m.keepHumanRot += (target.keepHumanRot - m.keepHumanRot) * r;
    m.keepObjPos   += (target.keepObjPos   - m.keepObjPos)   * r;
    m.keepTime     += (target.keepTime     - m.keepTime)     * r;
    m.keepObjPoints = target.keepObjPoints ?? 0;
    return m;
  }

  _filterLiveHumanTargetWorld(rawWorld) {
    const raw = [rawWorld[0], rawWorld[1], rawWorld[2]];
    const prev = this._smoothHumanTargetWorld;
    const alpha = clamp(LIVE_GOAL_WORLD_TARGET_SMOOTH_ALPHA, 0.0, 1.0);
    let next = raw;
    if (prev !== null && alpha < 1.0) {
      next = [
        alpha * raw[0] + (1.0 - alpha) * prev[0],
        alpha * raw[1] + (1.0 - alpha) * prev[1],
        alpha * raw[2] + (1.0 - alpha) * prev[2],
      ];
    }
    const maxStep = this.liveGoalWorldTargetMaxStepM;
    if (prev !== null && maxStep > 0.0) {
      const dx = next[0] - prev[0];
      const dy = next[1] - prev[1];
      const dz = next[2] - prev[2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (dist > maxStep) {
        const s = maxStep / Math.max(dist, 1e-6);
        next = [
          prev[0] + dx * s,
          prev[1] + dy * s,
          prev[2] + dz * s,
        ];
      }
    }
    this._smoothHumanTargetWorld = [next[0], next[1], next[2]];
    return this._smoothHumanTargetWorld;
  }

  _stableRecedingTargetWorld(proprio, dBody) {
    let body = null;
    if (this._routeBehindActive && this._routeTargetBody !== null) {
      body = this._routeTargetBody;
    } else {
      const xy = Math.sqrt(dBody[0]*dBody[0] + dBody[1]*dBody[1]);
      const scale = xy > 1e-6
        ? Math.min(xy, STABLE_RECEDING_SEGMENT_DISTANCE_M) / xy
        : 0.0;
      body = [dBody[0] * scale, dBody[1] * scale, 0.0];
    }
    const headingFwd = headingQuatForward(proprio.rootQuatXyzwWorld);
    const deltaWorld = quatRotateOne(headingFwd, body);
    return [
      proprio.rootPosWorld[0] + deltaWorld[0],
      proprio.rootPosWorld[1] + deltaWorld[1],
      proprio.rootPosWorld[2] + deltaWorld[2],
    ];
  }

  // -------------------- Procedural HOI object goal -------------------- //

  _proceduralObjGoal(objPosWorld, objGoalWorld, rootQuatXyzwWorld, K_frames) {
    // Eased trajectory from current obj position toward user's click
    // target. Time-to-target = clamp(distance / speed, 1.5s, 4s);
    // smoothstep parametrization.
    let dx = objGoalWorld[0] - objPosWorld[0];
    let dy = objGoalWorld[1] - objPosWorld[1];
    let dz = objGoalWorld[2] - objPosWorld[2];
    let dist = Math.hypot(dx, dy, dz);
    // Limit the spatial segment as well as its duration. Clamping only T
    // made a remote floor click (e.g. 20 m) produce a multi-metre command
    // inside a three-second horizon. Preserve the final user goal and advance
    // toward it through the existing bounded planning horizon.
    const maxSegment = PROC_OBJ_SPEED_M_PER_S * PROC_OBJ_MAX_T_SEC;
    if (dist > maxSegment) {
      const scale = maxSegment / dist;
      dx *= scale; dy *= scale; dz *= scale;
      dist = maxSegment;
    }
    const T = clamp(dist / PROC_OBJ_SPEED_M_PER_S, PROC_OBJ_MIN_T_SEC, PROC_OBJ_MAX_T_SEC);
    const tFrac = (K_frames / FPS) / Math.max(T, 1e-3);
    const s = smoothstep(tFrac);
    const futureWorld = [
      objPosWorld[0] + s * dx,
      objPosWorld[1] + s * dy,
      objPosWorld[2] + s * dz,
    ];
    const deltaWorld = [
      futureWorld[0] - objPosWorld[0],
      futureWorld[1] - objPosWorld[1],
      futureWorld[2] - objPosWorld[2],
    ];
    const headingInv = calcHeadingQuatInv(rootQuatXyzwWorld);
    return new Float32Array(quatRotateOne(headingInv, deltaWorld));
  }

  // -------------------- Public: per-frame step -------------------- //

  step(input) {
    const { user, proprio } = input;
    if (this.matchData === null) {
      throw new Error('GoalTranslator.step() called before load()');
    }

    // 1) FSM
    this._updateFsm(user, proprio);
    const commandActive = (
      user.wasdActive
      || user.humanGoalWorld !== null
      || user.objGoalWorld !== null
    );
    if (
        STAND_ANCHOR_ENABLED
        && this.fsmState === FsmState.IDLE
        && (!commandActive)
        && (this._standAnchorWorld === null || this._prevCommandActive)
    ) {
      this._standAnchorWorld = [
        proprio.rootPosWorld[0],
        proprio.rootPosWorld[1],
        proprio.rootPosWorld[2],
      ];
    }
    if (commandActive) {
      this._standAnchorWorld = null;
    }
    this._prevCommandActive = commandActive;
    const fp = this._intentFingerprint(user);
    const liveHumanGoalSmoothingActive = (
      (this.fsmState === FsmState.LOCO || this.fsmState === FsmState.HOI_FULL)
      && user.humanGoalWorld !== null
    );
    if (fp !== this.intentFpPrev) {
      this.humanGoalHoldActive = false;
      this.humanGoalHoldFp = null;
      this._routeBehindSign = 0.0;
      this._routeBehindOriginWorld = null;
      this._routeBehindLateralAxisWorld = [0.0, 0.0];
      // Keep the live click waypoint filter warm across click-to-click
      // retargets. Resetting it on every intent edge makes the green target
      // jump to a fresh matched-row waypoint, which the sequence MuJoCo gate
      // showed is enough to break rapid retargeting. Non-click transitions
      // still clear the filter below.
      if (!liveHumanGoalSmoothingActive) {
        this._smoothHumanTargetWorld = null;
      }
    }
    if (!liveHumanGoalSmoothingActive) this._smoothHumanTargetWorld = null;

    // Live click commands re-query every frame, so the held-row achievement
    // check below cannot activate their arrival hold. Check the final planar
    // goal directly and keep the hold until user intent changes. This also
    // prevents a small overshoot from selecting a full backward-walking row.
    if (HOLD_ON_SUCCESS_ENABLED
        && this.fsmState === FsmState.LOCO
        && user.humanGoalWorld !== null
        && Math.hypot(user.humanGoalWorld[0] - proprio.rootPosWorld[0],
          user.humanGoalWorld[1] - proprio.rootPosWorld[1]) <= HOLD_REACH_RADIUS_M) {
      this.humanGoalHoldActive = true;
      this.humanGoalHoldFp = fp;
    }

    // 2) Query
    const query = this._buildQuery(user, proprio);

    // 3) Re-query decision — achievement-gated.
    //    Re-query when ANY of:
    //      (a) no row held yet (first frame / after reset)
    //      (b) intent edge (user input or FSM transition)
    //      (c) we've REACHED the matched target in world frame
    //          (only meaningful in LOCO/HOI states; IDLE/RECOVER
    //          fall through to (d)/(e))
    //      (d) safety timeout: held for K_frames (the matched clip's
    //          own lookahead horizon) → goal would be "now"
    //      (e) drift: the live query has moved far from the held row's
    //          match features (proprio diverged from the matched state)
    let needRequery = (this.selectedRow < 0)
        || (fp !== this.intentFpPrev)
        || (this.holdFrames >= MAX_HOLD_FRAMES);
    const liveHumanGoalRequery = (this.fsmState === FsmState.LOCO || this.fsmState === FsmState.HOI_FULL)
        && user.humanGoalWorld !== null
        && !(this.humanGoalHoldActive && this.humanGoalHoldFp === fp);
    const liveKeyboardLocoRequery = (this.fsmState === FsmState.LOCO || this.fsmState === FsmState.HOI_FULL)
        && user.humanGoalWorld === null
        && user.wasdActive;
    if (liveHumanGoalRequery) needRequery = true;
    if (liveKeyboardLocoRequery) needRequery = true;

    if (!needRequery && this.selectedRow >= 0 && this.holdFrames >= MIN_HOLD_FRAMES) {
      // Achievement check (LOCO / HOI states only)
      const fsm = this.fsmState;
      const isLocoLike = (fsm === FsmState.LOCO || fsm === FsmState.HOI_FULL);
      const isObjOnly  = (fsm === FsmState.HOI_OBJ_ONLY);

      if ((isLocoLike || isObjOnly) && this.heldHumanTargetWorld !== null) {
        const dx = proprio.rootPosWorld[0] - this.heldHumanTargetWorld[0];
        const dy = proprio.rootPosWorld[1] - this.heldHumanTargetWorld[1];
        const dz = proprio.rootPosWorld[2] - this.heldHumanTargetWorld[2];
        this.lastReachDistHuman = Math.sqrt(dx*dx + dy*dy + dz*dz);
      } else {
        this.lastReachDistHuman = Infinity;
      }
      if ((isLocoLike || isObjOnly) && this.heldObjTargetWorld !== null
          && proprio.objPosWorld !== null) {
        const dx = proprio.objPosWorld[0] - this.heldObjTargetWorld[0];
        const dy = proprio.objPosWorld[1] - this.heldObjTargetWorld[1];
        const dz = proprio.objPosWorld[2] - this.heldObjTargetWorld[2];
        this.lastReachDistObj = Math.sqrt(dx*dx + dy*dy + dz*dz);
      } else {
        this.lastReachDistObj = Infinity;
      }

      // "Reached": body close AND (no obj target OR obj close)
      const humanReached = isLocoLike && this.lastReachDistHuman < REACHED_THRESHOLD_HUMAN;
      const objReached = (isObjOnly || (isLocoLike && this.heldObjTargetWorld !== null))
          && this.lastReachDistObj < REACHED_THRESHOLD_OBJ;
      let reached = false;
      if (isObjOnly)        reached = objReached;
      else if (isLocoLike)  reached = humanReached
            && (this.heldObjTargetWorld === null || objReached);
      const canHoldHumanGoal = HOLD_ON_SUCCESS_ENABLED
          && isLocoLike
          && user.humanGoalWorld !== null
          && this.heldObjTargetWorld === null;
      if (reached && canHoldHumanGoal) {
        this.humanGoalHoldActive = true;
        this.humanGoalHoldFp = fp;
      } else if (reached) {
        needRequery = true;
      }

      // K-frames safety timeout
      if (this.holdFrames >= this.heldKFrames) needRequery = true;

      // Drift safety net
      if (!needRequery) {
        let d2 = 0;
        const off = this.selectedRow * MATCH_DIM;
        for (let j = 0; j < MATCH_DIM; j++) {
          const e = (this.matchData[off + j] - query[j]);
          d2 += MATCH_WEIGHTS[j] * e * e;
        }
        if (Math.sqrt(d2) > REQUERY_DIST_THRESH) needRequery = true;
      }
    }

    if (needRequery) {
      const candidates = this._candidateSubset(user);
      if (candidates.length === 0) {
        // Empty bucket — fall back to IDLE bucket
        this.selectedRow = this.bucketIndices.IDLE[0] || -1;
      } else {
        // Deploy uses deterministic top-1 selection, matching
        // scripts/goal_clip_command.py and the MuJoCo quantitative gate. Even
        // intent-edge top-K randomness can create a row mismatch between the
        // browser demo and the metric gate, so keep stochasticity disabled here.
        this.selectedRow = this._selectRow(candidates, query, false);
      }

      // Cache the world-frame target at match time. The body-frame goal
      // fed to the policy each frame will be derived from this FIXED
      // world target via heading_inv(current_quat) · (target - current).
      // — matches V2 training's `_build_interactive_command` exactly.
      if (this.selectedRow >= 0) {
        const off = this.selectedRow * GOAL_DIM;
        const headingFwd = headingQuatForward(proprio.rootQuatXyzwWorld);
        // Body-frame deltas stored in the DB:
        //   goalData[off + 0..2]   human target Δpos (body frame at match's anchor)
        //   goalData[off + 3..6]   human target rel-quat (body frame at match's anchor, full quat)
        //   goalData[off + 7..9]   object target Δpos (body frame; OMOMO only)
        const bodyHuman = [
          this.goalData[off + 0],
          this.goalData[off + 1],
          this.goalData[off + 2],
        ];
        const worldHumanDelta = quatRotateOne(headingFwd, bodyHuman);
        this.heldHumanTargetWorld = [
          proprio.rootPosWorld[0] + worldHumanDelta[0],
          proprio.rootPosWorld[1] + worldHumanDelta[1],
          proprio.rootPosWorld[2] + worldHumanDelta[2],
        ];
        if (
            liveHumanGoalSmoothingActive
            && this.clickPositionSource === 'matched'
            && (
              LIVE_GOAL_WORLD_TARGET_SMOOTH_ALPHA < 1.0
              || this.liveGoalWorldTargetMaxStepM > 0.0
            )
        ) {
          this._filterLiveHumanTargetWorld(this.heldHumanTargetWorld);
          this.heldHumanTargetWorld = [
            this._smoothHumanTargetWorld[0],
            this._smoothHumanTargetWorld[1],
            this._smoothHumanTargetWorld[2],
          ];
        }
        // World-frame target FULL quaternion. The stored body quat is
        // head_inv(match_anchor) · root_quat(match+K); multiplying by
        // headingFwd(current) gives the absolute target orientation in
        // world. (At match time current_heading == match_anchor_heading,
        // so this is the matched clip's t+K root_quat.)
        const bodyRelQuat = [
          this.goalData[off + 3],
          this.goalData[off + 4],
          this.goalData[off + 5],
          this.goalData[off + 6],
        ];
        const wq = quatMulXyzw(headingFwd, bodyRelQuat);
        const wqn = Math.sqrt(wq[0]*wq[0] + wq[1]*wq[1] + wq[2]*wq[2] + wq[3]*wq[3]) || 1.0;
        this.heldHumanTargetRotWorld = [wq[0]/wqn, wq[1]/wqn, wq[2]/wqn, wq[3]/wqn];

        if (proprio.objPosWorld !== null) {
          const bodyObj = [
            this.goalData[off + 7],
            this.goalData[off + 8],
            this.goalData[off + 9],
          ];
          const worldObjDelta = quatRotateOne(headingFwd, bodyObj);
          this.heldObjTargetWorld = [
            proprio.objPosWorld[0] + worldObjDelta[0],
            proprio.objPosWorld[1] + worldObjDelta[1],
            proprio.objPosWorld[2] + worldObjDelta[2],
          ];
        } else {
          this.heldObjTargetWorld = null;
        }
        this.heldKFrames = Math.min(Math.max(this.goalData[off + 10], MIN_HOLD_FRAMES + 1),
                                    MAX_HOLD_FRAMES);
      }
      this.holdFrames = 0;
      if (fp !== this.intentFpPrev) {
        this.intentFpPrev = fp;
        this.framesSinceIntentEdge = 0;
      }
    }
    this.holdFrames += 1;
    this.framesSinceIntentEdge += 1;

    // 4) Derive body-frame NEW_CMD fields FROM THE FIXED WORLD TARGET
    //    each frame — matches V2 training's _build_interactive_command
    //    which recomputes the goal delta from current root pose every
    //    step. As the robot walks toward the target, body-frame Δpos
    //    shrinks ("you've arrived" signal). As it turns, Δquat rotates.
    //    Storing the snapshot body-frame delta directly was the prior
    //    bug — produced a goal that translated/rotated with the robot.
    const headingInv = calcHeadingQuatInv(proprio.rootQuatXyzwWorld);

    const humanTargetPosBody = new Float32Array(3);
    const humanRelQuatBody   = new Float32Array(4);
    if (this.heldHumanTargetWorld !== null) {
      const dWorld = [
        this.heldHumanTargetWorld[0] - proprio.rootPosWorld[0],
        this.heldHumanTargetWorld[1] - proprio.rootPosWorld[1],
        this.heldHumanTargetWorld[2] - proprio.rootPosWorld[2],
      ];
      const dBody = quatRotateOne(headingInv, dWorld);
      humanTargetPosBody[0] = dBody[0];
      humanTargetPosBody[1] = dBody[1];
      humanTargetPosBody[2] = dBody[2];
    }
    if (this.heldHumanTargetRotWorld !== null) {
      const rel = quatMulXyzw(headingInv, this.heldHumanTargetRotWorld);
      const n = Math.sqrt(rel[0]*rel[0] + rel[1]*rel[1] + rel[2]*rel[2] + rel[3]*rel[3]) || 1.0;
      humanRelQuatBody[0] = rel[0] / n;
      humanRelQuatBody[1] = rel[1] / n;
      humanRelQuatBody[2] = rel[2] / n;
      humanRelQuatBody[3] = rel[3] / n;
    } else {
      humanRelQuatBody[3] = 1.0;   // identity
    }

    // Object target: same world→body re-derivation. If the user clicked
    // a world target, step (6) below will override with the procedural
    // eased trajectory (which already does live world→body each frame).
    const objTargetPosBody = new Float32Array(3);
    if (this.heldObjTargetWorld !== null && proprio.objPosWorld !== null) {
      const dWorld = [
        this.heldObjTargetWorld[0] - proprio.objPosWorld[0],
        this.heldObjTargetWorld[1] - proprio.objPosWorld[1],
        this.heldObjTargetWorld[2] - proprio.objPosWorld[2],
      ];
      const dBody = quatRotateOne(headingInv, dWorld);
      objTargetPosBody[0] = dBody[0];
      objTargetPosBody[1] = dBody[1];
      objTargetPosBody[2] = dBody[2];
    }
    const K_frames = this.heldKFrames;

    // 5) Build NEW_CMD fields (in body frame)
    const goalSpec = {
      humanTargetPos: humanTargetPosBody,
      humanTargetRot: new Float32Array(quatToRot6d(humanRelQuatBody)),
      objTargetPos:   objTargetPosBody,
      timeToTarget:   clamp(K_frames / G_MAX_NORMALIZER, 0, 1),
    };

    let humanGoalDist = Infinity;
    let arrivalOverride = false;
    let clickPositionOverride = false;
    if (
        STAND_ANCHOR_ENABLED
        && this.fsmState === FsmState.IDLE
        && this._standAnchorWorld !== null
    ) {
      const dWorld = [
        this._standAnchorWorld[0] - proprio.rootPosWorld[0],
        this._standAnchorWorld[1] - proprio.rootPosWorld[1],
        0.0,
      ];
      const dBody = quatRotateOne(headingInv, dWorld);
      const xy = Math.sqrt(dBody[0]*dBody[0] + dBody[1]*dBody[1]);
      const scale = xy > 1e-6 ? Math.min(xy, STAND_ANCHOR_SEGMENT_DISTANCE_M) / xy : 0.0;
      goalSpec.humanTargetPos = new Float32Array([
        dBody[0] * scale,
        dBody[1] * scale,
        0.0,
      ]);
      goalSpec.humanTargetRot = new Float32Array(quatToRot6d(yawQuat(0.0)));
      goalSpec.timeToTarget = STAND_ANCHOR_TIME_TO_TARGET;
      arrivalOverride = true;
    }
    if ((this.fsmState === FsmState.LOCO || this.fsmState === FsmState.HOI_FULL)
        && user.humanGoalWorld !== null) {
      const dWorld = [
        user.humanGoalWorld[0] - proprio.rootPosWorld[0],
        user.humanGoalWorld[1] - proprio.rootPosWorld[1],
        0.0,
      ];
      // Ground raycasts have z=0; including that coordinate made arrival
      // impossible for a standing pelvis and requested a downward root goal.
      humanGoalDist = Math.hypot(dWorld[0], dWorld[1]);
      const dBody = quatRotateOne(headingInv, dWorld);
      const xy = Math.sqrt(dBody[0]*dBody[0] + dBody[1]*dBody[1]);
      const goalAngle = Math.atan2(dBody[1], dBody[0]);
      const holdOverride = this.humanGoalHoldActive && this.humanGoalHoldFp === fp;
      if (holdOverride) {
        const scale = xy > 1e-6 ? Math.min(xy, HOLD_SEGMENT_DISTANCE_M) / xy : 0.0;
        goalSpec.humanTargetPos = new Float32Array([
          dBody[0] * scale,
          dBody[1] * scale,
          dBody[2],
        ]);
        goalSpec.humanTargetRot = new Float32Array(quatToRot6d(yawQuat(0.0)));
        goalSpec.timeToTarget = HOLD_TIME_TO_TARGET;
        arrivalOverride = true;
      } else if (humanGoalDist <= ARRIVAL_RECEDING_RADIUS_M
          && Math.abs(goalAngle) <= ARRIVAL_RECEDING_MAX_ABS_ANGLE_RAD) {
        const scale = xy > 1e-6 ? Math.min(xy, ARRIVAL_SEGMENT_DISTANCE_M) / xy : 0.0;
        goalSpec.humanTargetPos = new Float32Array([
          dBody[0] * scale,
          dBody[1] * scale,
          dBody[2],
        ]);
        goalSpec.humanTargetRot = new Float32Array(quatToRot6d(yawQuat(0.0)));
        goalSpec.timeToTarget = clamp(ARRIVAL_HORIZON_FRAMES / G_MAX_NORMALIZER, 0, 1);
        arrivalOverride = true;
      } else if (this.clickPositionSource === 'stable_receding') {
        const rawTargetWorld = this._stableRecedingTargetWorld(proprio, dBody);
        const targetWorld = this._filterLiveHumanTargetWorld(rawTargetWorld);
        const stableWorldDelta = [
          targetWorld[0] - proprio.rootPosWorld[0],
          targetWorld[1] - proprio.rootPosWorld[1],
          targetWorld[2] - proprio.rootPosWorld[2],
        ];
        const stableBody = quatRotateOne(headingInv, stableWorldDelta);
        goalSpec.humanTargetPos = new Float32Array([
          stableBody[0],
          stableBody[1],
          0.0,
        ]);
        clickPositionOverride = true;
      }
    }

    // 6) HOI: overlay procedural object goal — only when user has
    //    clicked a target. Already recomputes from current obj_pos
    //    each frame, so this path was correct.
    if ((this.fsmState === FsmState.HOI_FULL || this.fsmState === FsmState.HOI_OBJ_ONLY)
        && user.objGoalWorld !== null
        && proprio.objPosWorld !== null) {
      goalSpec.objTargetPos = this._proceduralObjGoal(
          proprio.objPosWorld, user.objGoalWorld,
          proprio.rootQuatXyzwWorld, K_frames);
    }

    // 7) Mask schedule
    const target = this._maskTarget();
    // Selecting a UI object is not itself an interaction command. Distant
    // object perception perturbs the locomotion prior even with object goals
    // masked; keep the trained LOCO observation until an HOI task is active.
    target.keepObjPoints = (this.fsmState === FsmState.HOI_FULL
      || this.fsmState === FsmState.HOI_OBJ_ONLY) ? 1 : 0;
    const mask = this._stepMaskLerp(target);

    return {
      goalSpec,
      mask: { ...mask },     // shallow copy so callers can't mutate
      longTermT: K_frames,
      fsmState: this.fsmState,
      debug: {
        selectedRow: this.selectedRow,
        matchDistance: this.lastMatchDistance,
        holdFrames: this.holdFrames,
        heldKFrames: this.heldKFrames,
        reachDistHuman: this.lastReachDistHuman,
        reachDistObj: this.lastReachDistObj,
        humanGoalDist,
        arrivalOverride,
        clickPositionSource: this.clickPositionSource,
        clickPositionOverride,
        liveGoalWorldTargetMaxStepM: this.liveGoalWorldTargetMaxStepM,
        holdOnSuccess: this.humanGoalHoldActive && this.humanGoalHoldFp === fp,
        turnBehind: this._turnBehindActive,
        routeBehind: this._routeBehindActive,
        routeBehindSign: this._routeBehindSign,
        smoothHumanTargetWorld: this._smoothHumanTargetWorld ? [...this._smoothHumanTargetWorld] : null,
      },
    };
  }
}
