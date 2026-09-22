/**
 * Ownership for an opt-in goal update while one selected object is carried.
 *
 * This module deliberately does not rewrite a carry reference or step physics.
 * It gives the controller integration one latest, generation-bound planner
 * input, rejects stale asynchronous output, and only commits a separately
 * validated synchronous no-release splice.
 */
const terminal = new Set(['refused', 'superseded', 'reset', 'outcome']);
const finiteGoal = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
const sameGoal = (a, b, tolerance) => finiteGoal(a) && finiteGoal(b)
  && Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= tolerance;
const count = value => Number.isSafeInteger(value) && value >= 0;
const freeze = value => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
};

function requireRequest(log, requestId, episodeVersion) {
  const record = log?.records?.get(requestId);
  if (!record || record.episodeVersion !== episodeVersion || terminal.has(record.disposition)) {
    throw new Error('A current nonterminal request-log record is required');
  }
  return record;
}

/**
 * A small adapter around the existing BoxTaskRequestLog/request-generation
 * semantics.  Object/controller/history identity cannot change during a
 * grasp.  Different-object/skill commands remain explicit queued work for the
 * ordinary post-release planner instead of being swallowed or applied to the
 * wrong object.
 */
export class StreamingCarryGoalUpdateOwner {
  constructor({ requestLog, readClock, duplicateToleranceM = 1e-9 } = {}) {
    if (!requestLog?.records || typeof requestLog.transition !== 'function'
        || typeof readClock !== 'function' || !Number.isFinite(duplicateToleranceM)
        || duplicateToleranceM < 0 || duplicateToleranceM > 1e-3) {
      throw new Error('Existing request log, clock, and a bounded duplicate tolerance are required');
    }
    this.requestLog = requestLog;
    this.readClock = readClock;
    this.duplicateToleranceM = duplicateToleranceM;
    this.resetLocal();
  }

  resetLocal() {
    this.active = null;
    this.pending = null;
    this.queuedAfterRelease = null;
    this.revision = 0;
    this.sealed = false;
    this.events = [];
  }

  attach({ requestId, episodeVersion, objectBodyName, skillName, goalWorld,
    controllerOwner, teacherBuilder, commandGeneration, historyGeneration }) {
    if (this.active) throw new Error('Finish or reset the existing carry owner first');
    requireRequest(this.requestLog, requestId, episodeVersion);
    if (!objectBodyName || !skillName || !finiteGoal(goalWorld) || !controllerOwner || !teacherBuilder
        || !count(commandGeneration) || !count(historyGeneration)) {
      throw new Error('A live object, skill, goal, controller, builder, and generations are required');
    }
    // Shallow-freeze the binding only.  Controller and builder are live mutable
    // runtime owners; freezing through those identity references would disable
    // their control clocks/history.
    this.active = Object.freeze({ requestId, episodeVersion, objectBodyName, skillName,
      goalWorld: Object.freeze(Array.from(goalWorld)), controllerOwner, teacherBuilder,
      commandGeneration, historyGeneration });
    this.sealed = false;
    this.#event('attached', requestId, { goalWorld: Array.from(goalWorld) });
    return this.snapshot();
  }

  /** Register a request that was already begun in BoxTaskRequestLog. */
  submit({ requestId, episodeVersion, objectBodyName, skillName, goalWorld }) {
    const clock = this.readClock();
    const record = requireRequest(this.requestLog, requestId, episodeVersion);
    if (!this.active || episodeVersion !== this.active.episodeVersion) {
      this.requestLog.transition(requestId, 'refused', 'streaming_owner_unavailable', clock);
      return freeze({ accepted: false, reason: 'streaming_owner_unavailable', requestId });
    }
    if (!finiteGoal(goalWorld)) {
      this.requestLog.transition(requestId, 'refused', 'invalid_goal', clock);
      return freeze({ accepted: false, reason: 'invalid_goal', requestId });
    }
    if (this.sealed || objectBodyName !== this.active.objectBodyName || skillName !== this.active.skillName) {
      if (this.queuedAfterRelease) this.requestLog.transition(this.queuedAfterRelease.requestId,
        'superseded', 'newer_post_release_command', clock);
      this.queuedAfterRelease = freeze({ requestId, episodeVersion, objectBodyName, skillName,
        goalWorld: Array.from(goalWorld), reason: this.sealed ? 'release_boundary' : 'object_or_skill_switch' });
      this.requestLog.transition(requestId, 'queued', 'after_current_release', clock,
        { ownedObjectBodyName: this.active.objectBodyName, requestedObjectBodyName: objectBodyName });
      this.#event('queued_after_release', requestId, { objectBodyName, skillName });
      return freeze({ accepted: true, behavior: 'queue_after_release', requestId });
    }
    const newestGoal = this.pending?.goalWorld ?? this.active.goalWorld;
    if (sameGoal(goalWorld, newestGoal, this.duplicateToleranceM)) {
      const ownerRequestId = this.pending?.requestId ?? this.active.requestId;
      this.requestLog.transition(requestId, 'superseded', 'duplicate_goal', clock,
        { coalescedIntoRequestId: ownerRequestId });
      this.#event('duplicate', requestId, { coalescedIntoRequestId: ownerRequestId });
      return freeze({ accepted: true, behavior: 'duplicate_coalesced', requestId, ownerRequestId });
    }
    if (this.pending) this.requestLog.transition(this.pending.requestId,
      'superseded', 'newer_mid_carry_goal', clock, { supersededByRequestId: requestId });
    this.revision++;
    const token = freeze({ episodeVersion, requestId, revision: this.revision,
      objectBodyName, skillName, commandGeneration: this.active.commandGeneration,
      historyGeneration: this.active.historyGeneration });
    this.pending = freeze({ requestId, episodeVersion, objectBodyName, skillName,
      goalWorld: Array.from(goalWorld), token });
    this.requestLog.transition(requestId, 'queued', 'mid_carry_goal_update_pending', clock,
      { revision: this.revision, preservesRequestId: this.active.requestId });
    this.#event('pending', requestId, { revision: this.revision, goalWorld: Array.from(goalWorld) });
    return freeze({ accepted: true, behavior: 'latest_mid_carry_goal', requestId,
      revision: this.revision, token });
  }

  /**
   * Return the one current planner input.  The exact controller/builder object
   * identities and their generations prove that command generation and the
   * 4052-D observation history owner have not changed.
   */
  claim({ episodeVersion, controllerOwner, teacherBuilder, commandGeneration,
    historyGeneration, phase }) {
    if (!this.pending || this.sealed) return null;
    const current = episodeVersion === this.active.episodeVersion
      && controllerOwner === this.active.controllerOwner
      && teacherBuilder === this.active.teacherBuilder
      && commandGeneration === this.active.commandGeneration
      && historyGeneration === this.active.historyGeneration;
    if (!current) return null;
    // No release/descent splice is implied.  The integration must explicitly
    // select a loaded transport phase for which it can preserve continuity.
    if (!['teacher_loaded_transport', 'student_transport'].includes(phase)) return null;
    this.#event('claimed', this.pending.requestId, { revision: this.pending.token.revision, phase });
    return this.pending;
  }

  /** Accept only freshness/ownership here; this is not physical activation. */
  acceptPlannerResult({ token, episodeVersion, controllerOwner, teacherBuilder,
    commandGeneration, historyGeneration, phase, supported }) {
    const current = this.claim({ episodeVersion, controllerOwner, teacherBuilder,
      commandGeneration, historyGeneration, phase });
    if (!current || token !== current.token) return freeze({ accepted: false, reason: 'stale_result' });
    if (supported !== true) {
      this.requestLog.transition(current.requestId, 'refused', 'mid_carry_splice_unsupported', this.readClock(),
        { revision: current.token.revision });
      this.pending = null;
      this.#event('planner_refused', current.requestId, { revision: current.token.revision });
      return freeze({ accepted: false, reason: 'mid_carry_splice_unsupported' });
    }
    // The result remains pending until a real controller splice commits it.
    // Keeping this separate prevents a static planner pass being reported as a
    // no-reset physical goal update.
    this.#event('planner_supported', current.requestId, { revision: current.token.revision });
    return freeze({ accepted: true, behavior: 'await_controller_splice', request: current });
  }

  /**
   * Publish one synchronous, zero-control splice produced by the active
   * controller integration.  The owner does not know reference geometry; the
   * splice callback must enforce that contract and return its immutable record.
   */
  commitControllerSplice({ token, episodeVersion, controllerOwner, teacherBuilder,
    commandGeneration, historyGeneration, phase, splice }) {
    const current = this.claim({ episodeVersion, controllerOwner, teacherBuilder,
      commandGeneration, historyGeneration, phase });
    if (!current || token !== current.token) return freeze({ accepted: false, reason: 'stale_result' });
    if (typeof splice !== 'function') throw new Error('A synchronous loaded-reference splice is required');
    const result = splice(current);
    if (!result || typeof result.then === 'function' || result.record?.physicsControlsConsumed !== 0
        || result.record?.controllerIdentityPreserved !== true
        || result.record?.teacherBuilderIdentityPreserved !== true
        || result.record?.published !== true
        || result.record?.physicalGoalUpdateQualified !== false) {
      throw new Error('Splice must preserve controller/history ownership and consume zero physics');
    }
    this.active = Object.freeze({ ...this.active, goalWorld: Object.freeze(Array.from(current.goalWorld)) });
    this.pending = null;
    this.requestLog.transition(current.requestId, 'started', 'mid_carry_goal_update_spliced', this.readClock(),
      { revision: current.token.revision, physicsControlsConsumed: 0 });
    this.#event('controller_spliced', current.requestId,
      { revision: current.token.revision, physicsControlsConsumed: 0 });
    // Do not recursively freeze the result: it intentionally references the
    // live controller's Float32 frame bank.
    return Object.freeze({ accepted: true, behavior: 'controller_spliced', requestId: current.requestId,
      revision: current.token.revision, result });
  }

  cancelPending(reason = 'user_cancel') {
    if (!this.pending) return false;
    this.requestLog.transition(this.pending.requestId, 'superseded', reason, this.readClock(),
      { revision: this.pending.token.revision });
    this.#event('pending_cancelled', this.pending.requestId, { reason });
    this.pending = null;
    return true;
  }

  markRelease({ episodeVersion, controllerOwner, teacherBuilder }) {
    if (!this.active || episodeVersion !== this.active.episodeVersion
        || controllerOwner !== this.active.controllerOwner || teacherBuilder !== this.active.teacherBuilder) {
      throw new Error('Only the current physical owner can seal release');
    }
    if (this.pending) {
      this.requestLog.transition(this.pending.requestId, 'superseded', 'release_boundary', this.readClock(),
        { revision: this.pending.token.revision });
      this.#event('release_superseded_pending', this.pending.requestId,
        { revision: this.pending.token.revision });
      this.pending = null;
    }
    this.sealed = true;
    this.#event('release_sealed', this.active.requestId, {});
    return this.queuedAfterRelease;
  }

  takeQueuedAfterRelease() {
    if (!this.sealed) throw new Error('Post-release command cannot run before release');
    const value = this.queuedAfterRelease;
    this.queuedAfterRelease = null;
    return value;
  }

  #event(type, requestId, detail) {
    this.events.push(freeze({ type, requestId, clock: freeze({ ...this.readClock() }), ...detail }));
    if (this.events.length > 64) this.events.shift();
  }

  snapshot() {
    return {
      active: this.active ? { ...this.active, controllerOwner: '[identity]', teacherBuilder: '[identity]' } : null,
      pending: this.pending ? structuredClone(this.pending) : null,
      queuedAfterRelease: this.queuedAfterRelease ? structuredClone(this.queuedAfterRelease) : null,
      revision: this.revision, sealed: this.sealed,
      events: this.events.map(event => structuredClone(event)),
      physicalGoalUpdateQualified: false,
      integrationQualification: 'static_contract_only',
      controllerSpliceRequired: true,
    };
  }
}
