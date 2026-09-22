// Experimental goal-directed carry. Keep separate from generic pickup: its
// standing reference has a wider measured initial-heading tolerance.
import { TeacherSkillController } from './teacher_controller.js';
import { withInitialStance } from './teacher_skill.js';
import { planCarryToGoal } from './teacher_goal_warp.js';

// One frozen engineering candidate: half the existing 5 cm live-goal tracking
// tolerance, with at most 0.5 s of extra reference time at 60 Hz. These are not
// task success/contact thresholds and are intentionally not URL-tunable.
export const CARRY_DESCENT_SAG_MARGIN_M = 0.025;
export const CARRY_DESCENT_SAG_MAX_HOLD_CONTROLS = 30;

// Frozen largebox outcome requirements (unchanged since v5). An object profile
// may pass `outcomeRequirements` overrides (B5 suitcase: maxFinalObjectHeightM
// 0.30 for its 0.222 m standing rest, plus tippedObjectTiltDeg/objectUpAxisLocal
// for the fail-closed 'object_tipped' outcome); only these keys are accepted.
export const CARRY_OUTCOME_REQUIREMENTS = Object.freeze({ minLiftM: 0.35, maxFinalObjectHeightM: 0.25, minRootHeightM: 0.45, minUpright: 0.5 });
export const CARRY_OUTCOME_REQUIREMENT_KEYS = Object.freeze([...Object.keys(CARRY_OUTCOME_REQUIREMENTS), 'tippedObjectTiltDeg', 'objectUpAxisLocal']);

export class CarryGoalController extends TeacherSkillController {
  /** Supply an unmodified carry skill and warp indices in that source skill.
   * The controller adds its stance and shifts the carry interval internally.
   */
  constructor(skill, goalWorld, { warpStartFrame, warpEndFrame, maxCorrection = 0.25,
    initialStanceFrames = 30, settleFrames = 12, settleSpeed = 0.25, maxApproachSteps = 360,
    arrivalRadius = 0.25, maxReferenceStartDistance = 0.4, maxFacingError = 1.3, snapObjectYaw = false,
    carryDescentSagHold = false, outcomeRequirements = null, yawSymmetry = null } = {}) {
    if (!goalWorld || goalWorld.length !== 3 || !Array.from(goalWorld).every(Number.isFinite)) throw new Error('Finite carry goal XYZ is required');
    if (!Number.isInteger(warpStartFrame) || !Number.isInteger(warpEndFrame)
        || warpStartFrame < 0 || warpStartFrame >= warpEndFrame || warpEndFrame >= skill.sourceFrames) {
      throw new Error('Carry warp interval must lie within the original skill');
    }
    if (!Number.isFinite(maxCorrection) || maxCorrection <= 0 || maxCorrection > 0.25) {
      throw new Error('Carry correction must be positive and at most 0.25m');
    }
    if (outcomeRequirements !== null) {
      if (!outcomeRequirements || typeof outcomeRequirements !== 'object' || Array.isArray(outcomeRequirements)) throw new Error('Carry outcome requirement overrides must be an object');
      for (const key of Object.keys(outcomeRequirements))
        if (!CARRY_OUTCOME_REQUIREMENT_KEYS.includes(key)) throw new Error(`Unknown carry outcome requirement ${key}`);
    }
    super(withInitialStance(skill, initialStanceFrames), {
      settleFrames, settleSpeed, maxApproachSteps, arrivalRadius, maxReferenceStartDistance, maxFacingError,
      outcomeRequirements: outcomeRequirements === null ? { ...CARRY_OUTCOME_REQUIREMENTS } : { ...CARRY_OUTCOME_REQUIREMENTS, ...outcomeRequirements },
    });
    this.alignmentMode = 'object-goal';
    this.requestedGoalWorld = Object.freeze(Array.from(goalWorld));
    this.warpStartFrame = warpStartFrame + initialStanceFrames;
    this.warpEndFrame = warpEndFrame + initialStanceFrames;
    this.maxCorrection = maxCorrection;
    if (typeof snapObjectYaw !== 'boolean') throw new Error('Carry object yaw snap must be boolean');
    this.snapObjectYaw = snapObjectYaw;
    // Yaw symmetry of the selected object (B4/B5 shared option: null => teacher_goal_warp's v5 quarter-turn default; standing
    // suitcase {turns: 2, stepRad: PI}). Only read when snapObjectYaw. B5 extra: the turn count is bounded to 1..8.
    if (yawSymmetry !== null && !(Number.isInteger(yawSymmetry?.turns) && yawSymmetry.turns >= 1 && yawSymmetry.turns <= 8 && yawSymmetry.stepRad > 0))
      throw new Error('Carry object yaw symmetry must be null or {turns 1..8, stepRad > 0}');
    this.yawSymmetry = yawSymmetry;
    if (typeof carryDescentSagHold !== 'boolean') throw new Error('Carry descent sag hold option must be boolean');
    this.carryDescentSagHold = carryDescentSagHold;
    this._resetDescentSagHold();
  }

  reset() {
    super.reset();
    this._resetDescentSagHold();
  }

  _resetDescentSagHold() {
    this.descentSagHoldActive = false;
    this.descentSagHoldExhausted = false;
    this.descentSagHoldHeldControls = 0;
    this.descentSagHoldTeacherControls = 0;
    this.descentSagHoldEvents = [];
    this.descentSagHoldLastEvent = null;
    this.descentSagHoldLastSagM = null;
  }

  _recordDescentSagHold(type, fields = {}) {
    const event = Object.freeze({ type, teacherControl: this.descentSagHoldTeacherControls,
      referenceIndex: this.referenceIndex, heldControls: this.descentSagHoldHeldControls, ...fields });
    this.descentSagHoldEvents.push(event);
    this.descentSagHoldLastEvent = event;
    return event;
  }

  get descentSagHoldSnapshot() {
    return Object.freeze({ enabled: this.carryDescentSagHold === true,
      active: this.descentSagHoldActive, exhausted: this.descentSagHoldExhausted,
      marginM: CARRY_DESCENT_SAG_MARGIN_M, maxHoldControls: CARRY_DESCENT_SAG_MAX_HOLD_CONTROLS,
      heldControls: this.descentSagHoldHeldControls, teacherControls: this.descentSagHoldTeacherControls,
      referenceClockLagControls: this.descentSagHoldHeldControls,
      referenceIndex: this.referenceIndex, sagM: this.descentSagHoldLastSagM,
      lastEvent: this.descentSagHoldLastEvent,
      events: Object.freeze(this.descentSagHoldEvents.map(event => ({ ...event }))) });
  }

  step(proprio) {
    return { ...super.step(proprio), carryDescentSagHold: this.descentSagHoldSnapshot };
  }

  /** Advance after one real physics control. The opt-in intervention owns only
   * the nominal teacher reference clock: it never writes simulator state. */
  /** Externally owned reference-clock hold (WS-C descent contact hold): one teacher control is executed against a caller-supplied
   *  reference while this source clock does not advance. The caller owns the trigger/target/budget; this only records the event
   *  and keeps the clock. Physics time still advances in the caller. */
  holdReferenceClock(reason, fields = {}) {
    if (this.phase !== 'teacher') throw new Error('Reference clock can only be held while the teacher owns the carry');
    if (typeof reason !== 'string' || !reason) throw new Error('An explicit hold reason is required');
    this.externalHoldEvents ??= [];
    this.externalHeldControls = (this.externalHeldControls ?? 0) + 1;
    const event = Object.freeze({ type: 'external_hold', reason, referenceIndex: this.referenceIndex,
      heldControls: this.externalHeldControls, ...fields });
    this.externalHoldEvents.push(event);
    return event;
  }

  get externalHoldSnapshot() {
    return Object.freeze({ heldControls: this.externalHeldControls ?? 0, referenceIndex: this.referenceIndex,
      events: Object.freeze((this.externalHoldEvents ?? []).map(event => ({ ...event }))) });
  }

  advance({ teacherOwned = true, proprio = null } = {}) {
    if (this.phase !== 'teacher') { super.advance(); return; }
    this.descentSagHoldTeacherControls++;
    this.descentSagHoldLastEvent = null;
    if (!this.carryDescentSagHold || teacherOwned !== true || !proprio?.objPosWorld) {
      if (this.descentSagHoldActive) {
        this.descentSagHoldActive = false;
        this._recordDescentSagHold('release', { reason: teacherOwned === true ? 'live_state_unavailable' : 'teacher_not_owner',
          sagM: this.descentSagHoldLastSagM });
      }
      super.advance();
      return;
    }
    const liveBoxZ = proprio.objPosWorld[2];
    const reference = this.worldFrames?.[this.referenceIndex];
    const previous = this.worldFrames?.[this.referenceIndex - 1];
    if (!Number.isFinite(liveBoxZ) || !reference || !Number.isFinite(reference[73])
        || (previous && !Number.isFinite(previous[73]))) throw new Error('Finite live and reference box heights are required for carry descent sag hold');
    const referenceBoxZ = reference[73], sagM = referenceBoxZ - liveBoxZ;
    this.descentSagHoldLastSagM = sagM;
    const loadedDescent = liveBoxZ > this.outcomeRequirements.maxFinalObjectHeightM
      && previous && referenceBoxZ < previous[73];
    const overMargin = sagM > CARRY_DESCENT_SAG_MARGIN_M;
    if (this.descentSagHoldActive && this.descentSagHoldHeldControls >= CARRY_DESCENT_SAG_MAX_HOLD_CONTROLS) {
      this.descentSagHoldActive = false; this.descentSagHoldExhausted = true;
      this._recordDescentSagHold('release', { reason: 'budget_exhausted', sagM, liveBoxZ, referenceBoxZ });
      super.advance();
      return;
    }
    if (this.descentSagHoldActive && (!loadedDescent || !overMargin)) {
      this.descentSagHoldActive = false;
      this._recordDescentSagHold('release', { reason: overMargin ? 'not_loaded_descent' : 'sag_recovered',
        sagM, liveBoxZ, referenceBoxZ });
      super.advance();
      return;
    }
    if (!this.descentSagHoldExhausted && loadedDescent && overMargin) {
      if (!this.descentSagHoldActive) {
        this.descentSagHoldActive = true;
        this._recordDescentSagHold('activate', { sagM, liveBoxZ, referenceBoxZ });
      }
      this.descentSagHoldHeldControls++;
      if (this.descentSagHoldHeldControls >= CARRY_DESCENT_SAG_MAX_HOLD_CONTROLS)
        this.descentSagHoldExhausted = true;
      this._recordDescentSagHold('hold', { sagM, liveBoxZ, referenceBoxZ });
      return;
    }
    super.advance();
  }

  _referencePlan(proprio) {
    // refObjYawSnap=1: align the clip's object quaternion to the live object by
    // one of its yaw-symmetry turns about the vertical axis (largebox: quarter
    // turn, square footprint; suitcase: half turn). Off by default.
    const plan = planCarryToGoal(this.skill.frames, this.skill.sourceFrames, proprio.objPosWorld, this.requestedGoalWorld,
      { startFrame: this.warpStartFrame, endFrame: this.warpEndFrame, maxCorrection: this.maxCorrection,
        snapObjectYaw: this.snapObjectYaw, objectQuaternion: this.snapObjectYaw ? proprio.objQuatXyzwWorld : null,
        yawSymmetry: this.yawSymmetry });
    return { ...plan, first: plan.frames[0] };
  }

  _observeOutcome(proprio) {
    super._observeOutcome(proprio);
    const position = proprio.objPosWorld, endpoint = this.referencePlan.referenceGoalWorld;
    this.outcome.finalObjectPositionWorld = Array.from(position);
    this.outcome.requestedGoalErrorM = Math.hypot(position[0] - this.requestedGoalWorld[0], position[1] - this.requestedGoalWorld[1]);
    this.outcome.referenceGoalErrorM = Math.hypot(position[0] - endpoint[0], position[1] - endpoint[1]);
  }
}
