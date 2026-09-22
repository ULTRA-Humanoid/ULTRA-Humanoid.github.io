// Restricted command-to-reference orchestration. Owns reference clocks only;
// the caller owns physics, policy inference, actions and observation history.
import { TeacherWaypointController } from './teacher_waypoint_controller.js';
import { TeacherTurnController } from './teacher_turn_controller.js';
import { planTeacherStandingReference } from './teacher_standing_reference.js';

const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const planarDistance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const copy = value => value == null ? null : structuredClone(value);
function finite(values, length, label) {
  if (!values || values.length !== length || !Array.from(values).every(Number.isFinite)) throw new Error(`${label} requires ${length} finite values`);
}
function point(value) { finite(value, 3, 'World goal'); return Array.from(value); }
const KEY_ALIASES = { forward: ['forward', 'w', 'KeyW'], backward: ['backward', 's', 'KeyS'],
  left: ['left', 'a', 'KeyA'], right: ['right', 'd', 'KeyD'], turnLeft: ['turnLeft', 'q', 'KeyQ'], turnRight: ['turnRight', 'e', 'KeyE'] };
function keys(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('A key snapshot is required');
  return Object.fromEntries(Object.entries(KEY_ALIASES).map(([name, aliases]) => {
    const selected = aliases.find(alias => value[alias] !== undefined), held = selected ? value[selected] : false;
    if (typeof held !== 'boolean') throw new Error('Key states must be booleans');
    return [name, held];
  }));
}
function periodicKeyName(held) {
  const turnSign = Number(held.turnLeft) - Number(held.turnRight);
  const forward = Number(held.forward) - Number(held.backward);
  const left = Number(held.left) - Number(held.right);
  return forward === 1 && left === 0
    ? turnSign > 0 ? 'turnLeft' : turnSign < 0 ? 'turnRight' : 'forward' : null;
}

export function preservesPeriodicObservationHistory(previous, current) {
  return Boolean(current?.justEnteredTeacher
    && previous?.phase === 'teacher_step'
    && previous.referenceIndex === previous.sourceFrames - 1
    && previous.persistentPeriodic?.active
    && current.referenceIndex === 0
    && current.persistentPeriodic?.segmentRole === 'persistent_loop');
}

export class RestrictedLocomotionController {
  constructor(stepSkills, { turnSkills = [], neutralSkill = stepSkills?.[0], arrivalRadius = .1,
    settlingSteps = 60, stableSteps = 12, maxSettlingSteps = 180, maxSettlingSpeedMps = .25,
    waypointRadius = .28, maxHeadingOffsetRad = Math.PI / 3, maxSteps = 8, maxTurns = 4,
    minRootHeightM = .45, minUpright = .5, planKeyboardGoal = null, approveKeyboardTurn = null,
    approveReference = null, maxStandingSpeedMps = .05, minStandingUpright = .95, minStandingHeightM = .7,
    keySettlingGoalAlignment = false, keyDirectionLine = false, maxKeyLateralCorrectionM = .1,
    keyLineCorrectionLimitM = .1,
    preferAlignedSteps = false, retainStandingOnRefusal = false, directionalKeySteps = false,
    additionalKeySkills = [], maxDirectionalKeyHeadingOffsetRad = maxHeadingOffsetRad,
    keyDirectionLineAdditionalOnly = false, retainAdditionalKeyTerminal = false, retainKeyTerminal = false,
    settleOnStart = false, approachTerminalRadius = null, liveRootHandoffRadius = null, periodicKeySkills = null,
    persistentPeriodicKeySkills = null } = {}) {
    if (arrivalRadius > .1) throw new Error('Restricted floor arrival requires a radius of at most 10cm');
    if ([planKeyboardGoal, approveKeyboardTurn, approveReference].some(callback => callback !== null && typeof callback !== 'function')) {
      throw new Error('Geometry approval callbacks must be functions');
    }
    this.planKeyboardGoal = planKeyboardGoal; this.approveKeyboardTurn = approveKeyboardTurn;
    this.approveReference = approveReference;
    if (typeof retainStandingOnRefusal !== 'boolean') throw new Error('Standing retention must be explicitly enabled or disabled');
    this.retainStandingOnRefusal = retainStandingOnRefusal;
    if (typeof directionalKeySteps !== 'boolean') throw new Error('Directional key records must be explicitly enabled or disabled');
    this.directionalKeySteps = directionalKeySteps;
    if (!Array.isArray(additionalKeySkills) || new Set(additionalKeySkills).size !== additionalKeySkills.length
        || additionalKeySkills.some(skill => stepSkills?.includes(skill))) {
      throw new Error('Additional key skills must be a separate array of unique complete recordings');
    }
    if (!Number.isFinite(maxDirectionalKeyHeadingOffsetRad) || maxDirectionalKeyHeadingOffsetRad <= 0
        || maxDirectionalKeyHeadingOffsetRad > maxHeadingOffsetRad) {
      throw new Error('Directional key heading limit must remain within the existing positive heading bound');
    }
    if (typeof keyDirectionLineAdditionalOnly !== 'boolean') throw new Error('Additional-only key line correction must be explicitly enabled or disabled');
    if (typeof retainAdditionalKeyTerminal !== 'boolean') throw new Error('Additional key terminal retention must be explicitly enabled or disabled');
    if (typeof retainKeyTerminal !== 'boolean') throw new Error('Key terminal retention must be explicitly enabled or disabled');
    this.additionalKeySkills = [...additionalKeySkills];
    if (periodicKeySkills !== null && (!periodicKeySkills || typeof periodicKeySkills !== 'object'
        || !['forward', 'turnLeft', 'turnRight'].every(name => periodicKeySkills[name]))) {
      throw new Error('Periodic forward and forward+yaw teacher skills must be supplied together');
    }
    if (persistentPeriodicKeySkills !== null && (!persistentPeriodicKeySkills?.entry
        || !Array.isArray(persistentPeriodicKeySkills.loops) || !persistentPeriodicKeySkills.loops.length
        || ![persistentPeriodicKeySkills.entry, ...persistentPeriodicKeySkills.loops]
          .every(group => ['forward', 'turnLeft', 'turnRight'].every(name => group?.[name])))) {
      throw new Error('Persistent periodic entry and loop skills must be supplied together');
    }
    this.maxDirectionalKeyHeadingOffsetRad = maxDirectionalKeyHeadingOffsetRad;
    this.keyDirectionLineAdditionalOnly = keyDirectionLineAdditionalOnly;
    this.retainAdditionalKeyTerminal = retainAdditionalKeyTerminal;
    this.retainKeyTerminal = retainKeyTerminal;
    if (approachTerminalRadius !== null && (!Number.isFinite(approachTerminalRadius) || approachTerminalRadius <= 0)) {
      throw new Error('Approach terminal retention requires the positive parent handoff radius');
    }
    this.approachTerminalRadius = approachTerminalRadius;
    if (liveRootHandoffRadius !== null && (!Number.isFinite(liveRootHandoffRadius) || liveRootHandoffRadius <= 0)) {
      throw new Error('Live-root handoff standing requires the positive parent handoff radius');
    }
    this.liveRootHandoffRadius = liveRootHandoffRadius;
    if (typeof settleOnStart !== 'boolean') throw new Error('Initial standing must be explicitly enabled or disabled');
    this.settleOnStart = settleOnStart;
    if (![maxStandingSpeedMps, minStandingUpright, minStandingHeightM].every(Number.isFinite)
        || maxStandingSpeedMps <= 0 || minStandingUpright < minUpright || minStandingUpright > 1 || minStandingHeightM < minRootHeightM) {
      throw new Error('Valid measured standing requirements are required');
    }
    Object.assign(this, { maxStandingSpeedMps, minStandingUpright, minStandingHeightM });
    if (typeof keySettlingGoalAlignment !== 'boolean') throw new Error('Keyboard goal settling must be explicitly enabled or disabled');
    this.keySettlingGoalAlignment = keySettlingGoalAlignment;
    if (typeof keyDirectionLine !== 'boolean' || !Number.isFinite(maxKeyLateralCorrectionM)
        || maxKeyLateralCorrectionM <= 0 || maxKeyLateralCorrectionM > .1) throw new Error('Bounded keyboard line correction required');
    this.keyDirectionLine = keyDirectionLine; this.maxKeyLateralCorrectionM = maxKeyLateralCorrectionM;
    if (!Number.isFinite(keyLineCorrectionLimitM) || keyLineCorrectionLimitM < 0 || keyLineCorrectionLimitM > .1) {
      throw new Error('Keyboard line correction per record must be between zero and 10cm');
    }
    this.keyLineCorrectionLimitM = keyLineCorrectionLimitM;
    this.options = { arrivalRadius, settlingSteps, stableSteps, maxSettlingSteps, maxSettlingSpeedMps,
      waypointRadius, maxHeadingOffsetRad, maxSteps, maxTurns, minRootHeightM, minUpright, settlingPolicy: 'teacher', preferAlignedSteps };
    // Reuse the complete-record validation and all existing geometric bounds.
    const validated = new TeacherWaypointController(stepSkills, { ...this.options, turnSkills });
    new TeacherWaypointController([neutralSkill], this.options);
    this.stepSkills = stepSkills; this.turnSkills = turnSkills; this.neutralSkill = neutralSkill;
    this.keyStep = [...validated.steps].sort((a, b) => a.travelM - b.travelM)[0];
    // Additional keyboard records never enter floor planning or choose its
    // neutral source. The ordinary shortest key record also stays unchanged.
    const additional = additionalKeySkills.length ? new TeacherWaypointController(additionalKeySkills, this.options).steps : [];
    this.keyStepCandidates = [...validated.steps, ...additional];
    this.periodicKeySteps = periodicKeySkills === null ? null : Object.fromEntries(
      Object.entries(periodicKeySkills).map(([name, skill]) => [name,
        new TeacherWaypointController([skill], this.options).steps[0]]));
    this.persistentPeriodicSteps = persistentPeriodicKeySkills === null ? null : {
      entry: Object.fromEntries(Object.entries(persistentPeriodicKeySkills.entry).map(([name, skill]) =>
        [name, new TeacherWaypointController([skill], this.options).steps[0]])),
      loops: persistentPeriodicKeySkills.loops.map(group => Object.fromEntries(Object.entries(group).map(([name, skill]) =>
        [name, new TeacherWaypointController([skill], this.options).steps[0]]))),
      metadata: copy(persistentPeriodicKeySkills.metadata),
    };
    this.turns = validated.turns;
    this.reset();
  }

  get standingSkill() { return this._standing ? this._standingSkill : null; }
  get skill() { return this._standing ? this.standingSkill : this._route?.skill ?? this._turn?.skill ?? this.neutralSkill; }
  get sourceFrames() { return this._standing ? 0 : this._route?.sourceFrames ?? this._turn?.sourceFrames ?? 0; }
  get locomotionOnly() { return true; }
  get requestedGoalWorld() { return this._intent.type === 'floor' ? [...this._intent.goalWorld] : null; }
  get requestedIntent() { return copy(this._intent); }
  get isSettled() { return this.phase === 'teacher_standing' && this._activeIntent?.revision === this._intent.revision
    && this._blockedRevision !== this._intent.revision
    && this._holdCount >= this.options.settlingSteps && this._standingStableCount >= this.options.stableSteps
    && ['stopped', 'finished', 'opposed_keys'].includes(this.completionReason); }

  reset() {
    this.phase = 'inactive'; this.completionReason = null; this.referenceIndex = 0;
    this._revision = 0; this._intent = { type: 'stop', revision: 0 };
    this._activeIntent = null; this._completedFloorRevision = -1; this._blockedRevision = -1;
    this._route = null; this._turn = null; this._routeKind = null; this._routeRequestRevision = -1;
    this._turnSettling = null; this._standing = null; this._standingKind = null; this._holdCount = 0;
    this._keyHeadingRevision = -1; this._keyHeading = null; this._keyOrigin = null; this._delegate = null; this._awaitingAdvance = false;
    this._justCompleted = false; this._motionFinished = false; this._newStanding = false;
    this._standingPending = false; this._standingStableCount = 0;
    this._suspensionReason = null;
    this._retainedStandingOnRefusal = false;
    this._keyUsesAdditionalSkill = false;
    this._keyUsesPeriodicSkill = false;
    this._keyUsesPersistentPeriodic = false; this._persistentPeriodicActive = false; this._persistentLoopIndex = 0;
    this._persistentSuccessorRootPose = null;
    this._standingSkill = null; this._keyTerminal = null; this._pendingKeyTerminal = null;
    this._executingMovingReference = false;
    this._initialStandingPending = this.settleOnStart;
    this.effectiveGoalWorld = null; this.motionResults = []; this.controls = 0;
  }

  requestFloorGoal(goalWorld, { waypoints = [] } = {}) {
    const goal = point(goalWorld);
    if (!Array.isArray(waypoints)) throw new Error('World waypoints must be an array');
    const route = waypoints.map(point);
    this._intent = { type: 'floor', revision: ++this._revision, goalWorld: goal, waypoints: route };
    return this._revision;
  }

  /** Call for an explicit key event, including release/blur. Identical held-key
   * snapshots do not create new requests or restart an ongoing record. */
  requestKeys(snapshot, { expectedRevision = null } = {}) {
    if (expectedRevision !== null && (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0)) {
      throw new Error('Expected keyboard revision must be a non-negative safe integer');
    }
    // Optional owner token for delayed/out-of-order input delivery. Ordinary
    // synchronous browser edges omit it. A stale producer observes the current
    // revision but cannot replace the latest intent.
    if (expectedRevision !== null && expectedRevision !== this._intent.revision) return this._intent.revision;
    const held = keys(snapshot), type = Object.values(held).some(Boolean) ? 'keys' : 'stop';
    if (this._intent.type === type && (type === 'stop'
        || Object.keys(held).every(name => held[name] === this._intent.keys[name]))) return this._intent.revision;
    this._intent = { type, revision: ++this._revision, ...(type === 'keys' ? { keys: held } : {}) };
    return this._revision;
  }

  requestCancel() { return this.requestKeys({}); }

  /** The caller may use this after another complete skill changes the physical
   * state. Preserve latest intent, discard stale reference execution, and read
   * the next live state before creating any new reference. */
  reanchor() {
    this.phase = 'inactive'; this.completionReason = null; this.referenceIndex = 0;
    this._route = this._turn = this._turnSettling = this._standing = null;
    this._standingKind = null; this._activeIntent = null; this._completedFloorRevision = -1;
    this._blockedRevision = -1; this._routeRequestRevision = -1; this._keyHeadingRevision = -1;
    this._delegate = null; this._awaitingAdvance = false; this._holdCount = 0;
    this._standingPending = false; this._standingStableCount = 0;
    this._suspensionReason = null;
    this._retainedStandingOnRefusal = false;
    this._keyUsesAdditionalSkill = false;
    this._keyUsesPeriodicSkill = false;
    this._keyUsesPersistentPeriodic = false; this._persistentPeriodicActive = false; this._persistentLoopIndex = 0;
    this._persistentSuccessorRootPose = null;
    this._standingSkill = null; this._keyTerminal = null; this._pendingKeyTerminal = null;
    this._executingMovingReference = false;
    // A caller-validated complete external skill has its own physical settling.
    // This option concerns initial/reset stance, not a new hold after every skill.
    this._initialStandingPending = false;
    this.effectiveGoalWorld = null;
  }

  _neutral(proprio, goalWorld = null, kind = 'standing') {
    // Only the parent box approach opts in. Ordinary floor/key requests keep
    // their existing stance behavior. The measured endpoint must already be
    // inside the parent's unchanged arrival region before retaining this pose.
    const approachTerminal = this.approachTerminalRadius !== null && kind === 'route_settling'
      && this._routeKind === 'floor' && this._route
      && planarDistance(proprio.rootPosWorld, this._route.requestedGoalWorld) <= this.approachTerminalRadius;
    const retained = !goalWorld && this._keyTerminal
      && (approachTerminal || (this.retainKeyTerminal || this.retainAdditionalKeyTerminal)
      && (kind === 'route_settling' && this._routeKind === 'keys'
        || kind === 'turn_settling' && this.retainKeyTerminal
        || kind === 'standing' && this._intent.type === 'stop' || kind === 'unsupported'))
      ? this._keyTerminal : null;
    if (retained) {
      // This frame was committed by advance() only after its complete source's
      // final action executed. It keeps its original world pose through settle
      // and stop; another live-root anchor would compound tracking bias.
      retained.plan ??= planTeacherStandingReference(retained.frame, { alignment: 'original',
        rootPosition: proprio.rootPosWorld, rootQuaternion: proprio.rootQuatXyzwWorld,
        objectPosition: proprio.objPosWorld ?? Array.from(retained.frame.slice(71, 74)),
        objectQuaternion: proprio.objQuatXyzwWorld ?? Array.from(retained.frame.slice(74, 78)),
        objectPointsLocal: retained.skill.objectPointsLocal });
      this._standing = retained.plan; this._standingSkill = retained.skill;
      this._standing.requestedGoalWorld = null; this._standing.observationMask = 'locomotion';
      this._standingKind = kind; this._holdCount = retained.holdCount; this._standingStableCount = retained.stableCount;
      this._newStanding = true; this._standingPending = false; this._retainedStandingOnRefusal = false;
      return;
    }
    const terminal = this.neutralSkill.frames[this.neutralSkill.sourceFrames - 1];
    // A box-approach parent already accepts every measured root inside its
    // handoff radius.  Preserve that physically reached root instead of
    // translating the standing reference to the exact floor goal.  The latter
    // can move a safe stance toward the box by the floor controller's full
    // 10 cm arrival allowance and manufacture a clearance refusal.  This does
    // not enlarge either arrival region, and the resulting live-root stance is
    // still subject to the unchanged whole-reference geometry approval and
    // per-control physical preview.
    const liveRootHandoff = goalWorld && this._routeKind === 'floor'
      && ['standing', 'route_settling'].includes(kind) && this.liveRootHandoffRadius !== null
      && planarDistance(proprio.rootPosWorld, goalWorld) <= this.liveRootHandoffRadius;
    const root = goalWorld && !liveRootHandoff
      ? [goalWorld[0], goalWorld[1], proprio.rootPosWorld[2]] : proprio.rootPosWorld;
    if (goalWorld && planarDistance(root, proprio.rootPosWorld) > this.options.arrivalRadius + 1e-6) {
      throw new Error('Goal-aligned stance must remain within the measured arrival radius');
    }
    this._standing = planTeacherStandingReference(terminal, { alignment: 'live-root', rootPosition: root,
      rootQuaternion: proprio.rootQuatXyzwWorld,
      objectPosition: proprio.objPosWorld ?? Array.from(terminal.slice(71, 74)),
      objectQuaternion: proprio.objQuatXyzwWorld ?? Array.from(terminal.slice(74, 78)),
      objectPointsLocal: this.neutralSkill.objectPointsLocal });
    this._standingSkill = this.neutralSkill;
    this._standing.requestedGoalWorld = goalWorld ? [...goalWorld] : null;
    this._standing.observationMask = 'locomotion';
    this._standingKind = kind; this._holdCount = 0; this._newStanding = true;
    this._standingPending = false; this._standingStableCount = 0;
    this._retainedStandingOnRefusal = false;
  }

  _finishUnsupported(proprio, reason, retainedStanding = null) {
    if (this._route?.outcome?.teacherSteps > 0) {
      this.motionResults.push({ kind: this._routeKind, completionReason: reason, outcome: copy(this._route.outcome) });
    }
    if (this._turn?.referenceIndex > 0) {
      this.motionResults.push({ kind: 'keys_turn', completionReason: reason, outcome: copy(this._turn.outcome) });
    }
    this.completionReason = reason; this._blockedRevision = this._intent.revision;
    this._persistentPeriodicActive = false; this._persistentLoopIndex = 0; this._keyUsesPersistentPeriodic = false;
    this._persistentSuccessorRootPose = null;
    this._activeIntent = copy(this._intent); this._route = this._turn = this._turnSettling = null;
    if (retainedStanding) {
      this._standing = retainedStanding.plan; this._holdCount = retainedStanding.controls;
      this._standingSkill = retainedStanding.skill;
      this._standingStableCount = retainedStanding.stableControls;
      this._standingKind = 'unsupported'; this._standingPending = false; this._newStanding = false;
      this._retainedStandingOnRefusal = true;
    } else this._neutral(proprio, null, 'unsupported');
    this._justCompleted = true;
  }

  _approve(proprio, control, moving) {
    return this.approveReference?.(copy(proprio), { phase: this.phase, skill: moving ? this.skill : this.standingSkill,
      sourceFrames: moving ? this.sourceFrames : 1,
      alignedReferenceFrames: moving ? (this._route?.worldFrames ?? this._turn.worldFrames) : [this._standing.frame],
      referencePlan: moving ? (this._route?.referencePlan ?? this._turn.referencePlan) : this._standing,
      activeIntent: copy(this._activeIntent), requestedGoalWorld: this.requestedGoalWorld });
  }

  _noReference(reason) {
    // A rejected stance cannot become valid merely because its one-time
    // approval flag was consumed, or because another input arrived. Only an
    // explicit reset/reanchor may build and approve a fresh reference.
    this._suspensionReason ??= reason;
    this.phase = 'unsupported'; this.completionReason = this._suspensionReason;
    this._blockedRevision = this._intent.revision; this._awaitingAdvance = false;
    this._standing = null; this._standingKind = null; this._delegate = null;
    this._standingSkill = null; this._keyTerminal = null; this._pendingKeyTerminal = null;
    this._executingMovingReference = false;
    this._standingPending = false; this._standingStableCount = 0;
    return { phase: this.phase, mode: 'none', supported: false, completionReason: this.completionReason,
      activeIntent: copy(this._activeIntent), pendingIntent: copy(this._intent),
      requestedGoalWorld: this.requestedGoalWorld, effectiveGoalWorld: copy(this.effectiveGoalWorld),
      isSettled: false, referenceFrames: null };
  }

  _beginIntent(proprio) {
    this._activeIntent = copy(this._intent); this.completionReason = null;
    this._standing = null; this._standingKind = null;
    this._retainedStandingOnRefusal = false;
    this._keyUsesAdditionalSkill = false;
    this._keyUsesPeriodicSkill = false;
    this._keyUsesPersistentPeriodic = false;
    if (this._intent.type === 'stop') {
      this._persistentPeriodicActive = false; this._persistentLoopIndex = 0;
      this._persistentSuccessorRootPose = null;
      this.effectiveGoalWorld = [...proprio.rootPosWorld];
      this._neutral(proprio); this.completionReason = 'stopped'; return;
    }
    if (this._intent.type === 'floor') {
      this._persistentPeriodicActive = false; this._persistentLoopIndex = 0;
      this._persistentSuccessorRootPose = null;
      this._route = new TeacherWaypointController(this.stepSkills, { ...this.options, turnSkills: this.turnSkills });
      this._routeKind = 'floor'; this._routeRequestRevision = this._intent.revision;
      this.effectiveGoalWorld = [...this._intent.goalWorld];
      this._route.start(proprio, { finalGoalWorld: this._intent.goalWorld, waypoints: this._intent.waypoints });
      return;
    }
    const held = this._intent.keys;
    const turnSign = Number(held.turnLeft) - Number(held.turnRight);
    const forward = Number(held.forward) - Number(held.backward), left = Number(held.left) - Number(held.right);
    // Opt-in teacher-first composition only: forward+yaw selects a curved
    // locomotion reference instead of discarding W for a pure turn.  Lateral
    // A/D commands keep the existing, separately reported semantics.
    const periodicName = periodicKeyName(held);
    const persistentGroup = this.persistentPeriodicSteps && periodicName
      ? this._persistentPeriodicActive
        ? this.persistentPeriodicSteps.loops[this._persistentLoopIndex]
        : this.persistentPeriodicSteps.entry : null;
    const periodicKeyStep = persistentGroup?.[periodicName]
      ?? (this.periodicKeySteps && periodicName ? this.periodicKeySteps[periodicName] : null);
    if (turnSign && !periodicKeyStep) {
      const turn = this.turns.filter(item => item.yawChangeRad * turnSign > 0)
        .sort((a, b) => Math.abs(a.yawChangeRad) - Math.abs(b.yawChangeRad))[0];
      if (!turn) return this._finishUnsupported(proprio, 'unsupported_turn');
      const approval = this.approveKeyboardTurn?.(copy(proprio), { skill: turn.skill,
        yawChangeRad: turn.yawChangeRad, intent: copy(this._intent) });
      if (approval === false || approval?.supported === false) return this._finishUnsupported(proprio, approval?.reason ?? 'turn_geometry_unsupported');
      this._turn = new TeacherTurnController(turn.skill, this.options); this._turn.start(proprio);
      this.effectiveGoalWorld = [...proprio.rootPosWorld]; return;
    }
    if (!forward && !left) {
      this.effectiveGoalWorld = [...proprio.rootPosWorld];
      this._neutral(proprio); this.completionReason = 'opposed_keys'; return;
    }
    // Hold the direction resolved when this combination was pressed. Recomputing
    // S/A/D from the body after each recorded turn would reverse held intent.
    const direction = this._keyHeading + Math.atan2(left, forward);
    let keyStep = periodicKeyStep ?? this.keyStep;
    const headingOffset = item => Math.abs(wrap(direction + item.sourceYawRad - item.travelDirectionRad
      - yaw(proprio.rootQuatXyzwWorld)));
    if (!periodicKeyStep && this.directionalKeySteps && headingOffset(keyStep) > this.options.maxHeadingOffsetRad) {
      // Keyboard input specifies a direction. When the normal shortest record
      // needs a turn, prefer a complete record supported by the current heading,
      // with travel closest to the normal key step. Geometry still checks the
      // whole selected record and the resulting unprojected goal below.
      const aligned = this.keyStepCandidates.filter(item => headingOffset(item) <= this.maxDirectionalKeyHeadingOffsetRad);
      aligned.sort((a, b) => Math.abs(a.travelM - this.keyStep.travelM) - Math.abs(b.travelM - this.keyStep.travelM)
        || a.skill.sourceFrames - b.skill.sourceFrames);
      if (aligned.length) keyStep = aligned[0];
    }
    this._keyUsesAdditionalSkill = this.additionalKeySkills.includes(keyStep.skill);
    this._keyUsesPeriodicSkill = Boolean(periodicKeyStep);
    this._keyUsesPersistentPeriodic = Boolean(persistentGroup);
    this.effectiveGoalWorld = [proprio.rootPosWorld[0] + keyStep.travelM * Math.cos(direction),
      proprio.rootPosWorld[1] + keyStep.travelM * Math.sin(direction), proprio.rootPosWorld[2]];
    if (!periodicKeyStep && this.keyDirectionLine && (!this.keyDirectionLineAdditionalOnly || this._keyUsesAdditionalSkill)) {
      const dx = proprio.rootPosWorld[0] - this._keyOrigin[0], dy = proprio.rootPosWorld[1] - this._keyOrigin[1];
      const c = Math.cos(direction), s = Math.sin(direction), lateral = -dx * s + dy * c;
      if (Math.abs(lateral) > this.maxKeyLateralCorrectionM) return this._finishUnsupported(proprio, 'keyboard_lateral_correction_unsupported');
      const along = dx * c + dy * s + keyStep.travelM;
      // Limit the directional change requested of one complete record without
      // moving the original key ray or enlarging its supported lateral region.
      const residual = lateral - Math.max(-this.keyLineCorrectionLimitM, Math.min(this.keyLineCorrectionLimitM, lateral));
      this.effectiveGoalWorld = [this._keyOrigin[0] + along * c - residual * s,
        this._keyOrigin[1] + along * s + residual * c, proprio.rootPosWorld[2]];
    }
    // A held periodic reference is admitted by its exact complete-body sweep in
    // approveReference.  It is not a finite click waypoint, so the 2 m click
    // planner is intentionally inapplicable; no obstacle or tracking reserve is
    // changed.
    if (!periodicKeyStep && this.planKeyboardGoal) {
      const planned = this.planKeyboardGoal([...proprio.rootPosWorld], [...this.effectiveGoalWorld], { intent: copy(this._intent) });
      if (!planned || planned.supported !== true) return this._finishUnsupported(proprio, planned?.reason ?? 'step_geometry_unsupported');
      const approvedGoal = point(planned.finalGoalWorld ?? planned.requestedGoalWorld);
      if (approvedGoal.some((value, axis) => value !== this.effectiveGoalWorld[axis])) {
        return this._finishUnsupported(proprio, 'projected_keyboard_goal_unsupported');
      }
      if (planned.waypoints?.length) return this._finishUnsupported(proprio, 'keyboard_detour_unsupported');
    }
    const entryReferenceRootPose = this._keyUsesPersistentPeriodic && this._persistentPeriodicActive
      ? this._persistentSuccessorRootPose : null;
    this._route = new TeacherWaypointController([keyStep.skill], { ...this.options,
      maxHeadingOffsetRad: keyStep === this.keyStep ? this.options.maxHeadingOffsetRad : this.maxDirectionalKeyHeadingOffsetRad,
      maxSteps: 1, maxTurns: 2, turnSkills: this.turnSkills,
      continuousCompletion: this._keyUsesPersistentPeriodic, entryReferenceRootPose });
    this._routeKind = 'keys'; this._routeRequestRevision = this._intent.revision;
    this._route.start(proprio, { finalGoalWorld: this.effectiveGoalWorld });
    this._persistentSuccessorRootPose = null;
    if (this._keyUsesPersistentPeriodic) {
      if (this._persistentPeriodicActive) {
        this._persistentLoopIndex = (this._persistentLoopIndex + 1) % this.persistentPeriodicSteps.loops.length;
      } else {
        this._persistentPeriodicActive = true; this._persistentLoopIndex = 0;
      }
    } else {
      this._persistentPeriodicActive = false; this._persistentLoopIndex = 0;
      this._persistentSuccessorRootPose = null;
    }
  }

  step(proprio) {
    finite(proprio.rootPosWorld, 3, 'Live root position'); finite(proprio.rootQuatXyzwWorld, 4, 'Live root quaternion');
    finite(proprio.rootVelWorld, 3, 'Live root velocity');
    if (this._suspensionReason !== null) return this._noReference(this._suspensionReason);
    const q = proprio.rootQuatXyzwWorld;
    const upright = Number.isFinite(proprio.uprightScore) ? proprio.uprightScore : 1 - 2 * (q[0] ** 2 + q[1] ** 2);
    if (proprio.rootPosWorld[2] < this.options.minRootHeightM || upright < this.options.minUpright) {
      return this._noReference('lost_balance');
    }
    if (this._intent.type === 'keys' && this._keyHeadingRevision !== this._intent.revision) {
      this._keyHeading = yaw(q); this._keyHeadingRevision = this._intent.revision;
      this._keyOrigin = [...proprio.rootPosWorld];
    }
    if (this._standing && this._standingPending) {
      this._standingPending = false;
      const ready = Math.hypot(...proprio.rootVelWorld.slice(0, 2)) <= this.maxStandingSpeedMps
        && upright >= this.minStandingUpright && proprio.rootPosWorld[2] >= this.minStandingHeightM;
      this._standingStableCount = ready ? this._standingStableCount + 1 : 0;
      if (this._keyTerminal?.plan === this._standing) {
        this._keyTerminal.stableCount = this._standingStableCount;
      }
    }
    // This local snapshot cannot survive an executed moving control, reset or
    // reanchor. The prior quiet sample above is actual physical feedback.
    const priorStanding = this.retainStandingOnRefusal && this._standing && this._holdCount > 0
      ? { plan: this._standing, skill: this.standingSkill, controls: this._holdCount, stableControls: this._standingStableCount } : null;
    this._justCompleted = false; this._motionFinished = false; this._newStanding = false;
    let control = null;
    // Bounded loop handles a completed record/route and the next intent in the
    // same control tick, without inserting a student action or resetting state.
    for (let transition = 0; transition < 5; transition++) {
      if (this._initialStandingPending) {
        if (!this._standing) {
          this._activeIntent = { type: 'stop', revision: 0 };
          this._neutral(proprio); this.completionReason = 'stopped';
          this.effectiveGoalWorld = [...proprio.rootPosWorld];
        }
        // Input updates replace the pending request without changing the fixed
        // initial reference. Releasing before readiness must not execute an old
        // key press or rebuild the same stance from a drifted root pose.
        if (this._intent.type === 'stop') this._activeIntent = copy(this._intent);
        if (this._holdCount >= this.options.settlingSteps
            && this._standingStableCount >= this.options.stableSteps) {
          this._initialStandingPending = false;
        } else { this._delegate = 'standing'; break; }
      }
      if (this._route) {
        if (this._intent.revision !== this._routeRequestRevision) {
          if (!this._route.finishRequested && this._routeKind === 'floor' && this._intent.type === 'floor') {
            if (this._route.requestRetarget({ finalGoalWorld: this._intent.goalWorld, waypoints: this._intent.waypoints })) {
              this._routeRequestRevision = this._intent.revision;
            }
          } else {
            const replaceWithoutStanding = this._keyUsesPersistentPeriodic && this._intent.type === 'keys'
              && periodicKeyName(this._intent.keys) !== null;
            // Re-evaluate an already pending boundary cancellation. A newer
            // release must restore measured settling; a newer eligible hold
            // can conversely resume directly at the same safe boundary.
            this._route.requestCancel({ afterSettling: !replaceWithoutStanding });
          }
        }
        control = this._route.step(proprio);
        this._motionFinished ||= control.stepFinished || control.turnFinished;
        if (!control.pendingGoalWorld && !this._route.finishRequested && this._routeRequestRevision === this._intent.revision) {
          this._activeIntent = copy(this._intent); this.effectiveGoalWorld = [...this._route.requestedGoalWorld];
        }
        if (control.mode === 'teacher') {
          if (this._routeKind === 'keys' && control.phase === 'teacher_turn' && control.justEnteredTeacher && this.approveKeyboardTurn) {
            const skill = control.skill;
            const change = yaw(skill.frames[skill.sourceFrames - 1].slice(3, 7)) - yaw(skill.frames[0].slice(3, 7));
            const approval = this.approveKeyboardTurn(copy(proprio), { skill,
              yawChangeRad: Math.atan2(Math.sin(change), Math.cos(change)), intent: copy(this._intent) });
            if (approval === false || approval?.supported === false) {
              this._finishUnsupported(proprio, approval?.reason ?? 'turn_geometry_unsupported'); control = null; continue;
            }
          }
          this._standing = null; this._standingKind = null; this._delegate = 'route'; break;
        }
        if (control.phase === 'settling') {
          if (this._standingKind !== 'route_settling') {
            const goal = this.keySettlingGoalAlignment && this._routeKind === 'keys' && control.stepFinished
              && this._intent.type === 'keys' && this._routeRequestRevision === this._intent.revision && !this._route.finishRequested
              && planarDistance(proprio.rootPosWorld, this._route.requestedGoalWorld) <= this.options.arrivalRadius
              ? this._route.requestedGoalWorld : null;
            this._neutral(proprio, goal, 'route_settling');
          }
          this._delegate = 'route'; break;
        }
        if (control.phase !== 'complete') throw new Error('Unexpected waypoint controller phase');
        const reason = control.completionReason, kind = this._routeKind;
        const successorRootPose = this._keyUsesPersistentPeriodic
          && this._route.worldFrames?.[this._route.sourceFrames]
          ? Array.from(this._route.worldFrames[this._route.sourceFrames].slice(0, 7)) : null;
        const continuesPersistent = successorRootPose && this._intent.type === 'keys'
          && periodicKeyName(this._intent.keys) !== null && ['finished', 'cancelled'].includes(reason);
        this.motionResults.push({ kind, completionReason: reason, outcome: copy(control.outcome) });
        this._route = null; this._standing = null; this._standingKind = null; control = null;
        this._persistentSuccessorRootPose = continuesPersistent ? successorRootPose : null;
        this._justCompleted = true;
        if (!['finished', 'cancelled'].includes(reason)) { this._finishUnsupported(proprio, reason); continue; }
        if (reason === 'finished' && kind === 'floor' && this._activeIntent.revision === this._intent.revision) {
          this._completedFloorRevision = this._intent.revision; this.completionReason = 'finished';
          this._neutral(proprio, this._intent.goalWorld); this.effectiveGoalWorld = [...this._intent.goalWorld];
        }
        continue;
      }
      if (this._turn) {
        control = this._turn.step(proprio);
        if (control.mode === 'teacher') { this._delegate = 'turn'; break; }
        this.motionResults.push({ kind: 'keys_turn', completionReason: control.completionReason, outcome: copy(control.outcome) });
        const reason = control.completionReason;
        this._turn = null; control = null; this._motionFinished = true;
        if (reason !== 'finished') { this._finishUnsupported(proprio, reason); continue; }
        this._turnSettling = { count: 0, stable: 0, pending: false };
        this._neutral(proprio, null, 'turn_settling');
      }
      if (this._turnSettling) {
        const settling = this._turnSettling;
        if (settling.pending) {
          settling.pending = false;
          settling.stable = Math.hypot(...proprio.rootVelWorld.slice(0, 2)) <= this.options.maxSettlingSpeedMps ? settling.stable + 1 : 0;
        }
        if (settling.count >= this.options.settlingSteps && settling.stable >= this.options.stableSteps) {
          this._turnSettling = null; this._standing = null; this._standingKind = null; this._justCompleted = true; continue;
        }
        if (settling.count >= this.options.maxSettlingSteps) { this._finishUnsupported(proprio, 'unsettled'); continue; }
        this._delegate = 'turn_settling'; break;
      }
      const sameIntent = this._activeIntent?.revision === this._intent.revision;
      const heldFloor = this._intent.type === 'floor' && this._completedFloorRevision === this._intent.revision;
      const blocked = this._blockedRevision === this._intent.revision;
      if (this._standing && sameIntent && (heldFloor || this._intent.type === 'stop' || blocked || this.completionReason === 'opposed_keys')) {
        this._delegate = 'standing'; break;
      }
      this._beginIntent(proprio);
    }
    let moving = this._delegate === 'route' && control?.mode === 'teacher' || this._delegate === 'turn';
    if (!moving && !this._standing) throw new Error('Restricted controller did not produce a complete reference');
    this.phase = moving ? control.phase : this._delegate === 'standing' ? 'teacher_standing' : 'teacher_settling';
    if (this.approveReference && (moving ? control.justEnteredTeacher : this._newStanding)) {
      const approval = this._approve(proprio, control, moving);
      if (approval === false || approval?.supported === false) {
        const reason = approval?.reason ?? 'reference_geometry_unsupported';
        if (!moving && !priorStanding) return this._noReference(reason);
        const alternative = moving && control.phase === 'teacher_step' && this._route
          ? this._route.trySmallerStep(proprio, candidate => this.approveReference(copy(proprio), {
            ...candidate, activeIntent: copy(this._activeIntent), requestedGoalWorld: this.requestedGoalWorld,
          })) : null;
        if (alternative) control = { ...control, ...alternative };
        else {
          this._finishUnsupported(proprio, reason, priorStanding); this._delegate = 'standing'; this.phase = 'teacher_standing';
          control = null; moving = false;
          const fallback = this._approve(proprio, null, false);
          if (fallback === false || fallback?.supported === false) return this._noReference(fallback?.reason ?? 'standing_geometry_unsupported');
        }
      }
    }
    this.referenceIndex = moving ? control.referenceIndex : this._holdCount;
    this._executingMovingReference = moving;
    const movingSkill = moving ? this.skill : null;
    const keyOwnedMotion = moving && (this._delegate === 'turn' || this._delegate === 'route' && this._routeKind === 'keys');
    const approachOwnedStep = this.approachTerminalRadius !== null && moving && this._delegate === 'route'
      && this._routeKind === 'floor' && control.phase === 'teacher_step';
    this._pendingKeyTerminal = (keyOwnedMotion || approachOwnedStep)
      && (approachOwnedStep || this.retainKeyTerminal || this.retainAdditionalKeyTerminal && control.phase === 'teacher_step'
        && this.additionalKeySkills.includes(movingSkill))
      && control.referenceIndex === movingSkill.sourceFrames - 1
      ? { skill: movingSkill, frame: Float32Array.from((this.retainKeyTerminal || approachOwnedStep)
        ? (this._delegate === 'turn' ? this._turn.worldFrames : this._route.worldFrames)[movingSkill.sourceFrames - 1]
        : control.referenceFrames[0]), plan: null,
        holdCount: 0, stableCount: 0 } : null;
    this._awaitingAdvance = true;
    return { phase: this.phase, mode: 'teacher', locomotionOnly: true, skill: this.skill, sourceFrames: this.sourceFrames,
      referenceIndex: this.referenceIndex, referenceFrames: moving ? control.referenceFrames : [this._standing.frame, this._standing.frame],
      justEnteredTeacher: Boolean(moving ? control.justEnteredTeacher : this._newStanding),
      stepFinished: Boolean(control?.stepFinished), turnFinished: Boolean(control?.turnFinished),
      motionFinished: this._motionFinished, justCompleted: this._justCompleted,
      completionReason: this.completionReason, supported: this._blockedRevision !== this._intent.revision,
      activeIntent: copy(this._activeIntent), pendingIntent: this._activeIntent?.revision !== this._intent.revision ? copy(this._intent) : null,
      requestedGoalWorld: this.requestedGoalWorld, effectiveGoalWorld: copy(this.effectiveGoalWorld),
      keyDirectionHeadingRad: this._intent.type === 'keys' ? this._keyHeading : null,
      keyLineOriginWorld: this._intent.type === 'keys' && this.keyDirectionLine
        && (!this.keyDirectionLineAdditionalOnly || this._keyUsesAdditionalSkill) ? copy(this._keyOrigin) : null,
      settlingCount: this._route?.settlingCount ?? this._turnSettling?.count ?? null,
      standingControls: this._holdCount, standingStableControls: this._standingStableCount, isSettled: this.isSettled,
      initialStandingPending: this._initialStandingPending,
      retainedStandingOnRefusal: this._retainedStandingOnRefusal,
      retainedAdditionalKeyTerminal: !moving && this._keyTerminal?.plan === this._standing
        && this.additionalKeySkills.includes(this._keyTerminal.skill),
      periodicTeacher: this._keyUsesPeriodicSkill ? copy(this.skill?.periodicTeacher ?? null) : null,
      persistentPeriodic: this._keyUsesPersistentPeriodic ? {
        active: this._persistentPeriodicActive, nextLoopIndex: this._persistentLoopIndex,
        segmentRole: this.skill?.periodicTeacher?.role ?? null,
        segmentIndex: this.skill?.periodicTeacher?.segmentIndex ?? null,
        pendingEligibleIntent: this._intent.type === 'keys' && periodicKeyName(this._intent.keys) !== null,
      } : null,
      retainedKeyTerminal: !moving && this._keyTerminal?.plan === this._standing,
      standingPlan: !moving ? { ...this._standing, frame: undefined } : null,
      outcome: { controls: this.controls, motions: copy(this.motionResults) } };
  }

  advance() {
    if (!this._awaitingAdvance) return;
    this._awaitingAdvance = false; this.controls++;
    if (this._executingMovingReference) {
      // Clear old holds after real movement, or commit this action's final
      // keyboard step/turn target. An unexecuted proposal is ineligible.
      this._keyTerminal = this._pendingKeyTerminal;
      this._standingSkill = null;
    }
    this._pendingKeyTerminal = null; this._executingMovingReference = false;
    if (this._delegate === 'route') this._route.advance();
    else if (this._delegate === 'turn') this._turn.advance();
    else if (this._delegate === 'turn_settling') { this._turnSettling.count++; this._turnSettling.pending = true; }
    if (this._standing) {
      this._holdCount++; this._standingPending = true;
      if (this._keyTerminal?.plan === this._standing) this._keyTerminal.holdCount = this._holdCount;
      else this._keyTerminal = null;
    }
  }
}
