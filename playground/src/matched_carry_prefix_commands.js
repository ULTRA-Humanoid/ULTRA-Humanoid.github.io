/** Command ownership before the matched programme's second student window.
 * No physics, reference advancement, observation history or contact permission.
 * The host records accepted latest/queued IDs before handle(), then applies the
 * returned effect synchronously. In particular, finish_existing_programme must
 * NOT also call the ordinary carry parent's requestCancel().
 * Rejected diagnostic request IDs are not control ownership: continuous owner
 * contexts and non-box events use owner.permittedLatestRequestId. A validated
 * box event supplies its actual new request ID and the same accepted queue ID.
 */
import {makeTaskRelativeCarryRequest} from './matched_carry_request.js';
const require = (yes, message) => { if (!yes) throw Error(message); };
const count = n => Number.isSafeInteger(n) && n >= 0;
const vector = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);
const same = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const freeze = value => {
  if (value && typeof value === 'object') { Object.values(value).forEach(freeze); Object.freeze(value); }
  return value;
};

export class MatchedCarryPrefixCommands {
  constructor({owner}) {
    require(owner?.role === 'prefix' && !owner.ended && owner.prefixParent && Object.isFrozen(owner.request),
      'A current prefix owner and immutable request are required');
    this.owner = owner; this.request = owner.request; this.prefixParent = owner.prefixParent;
    this.lastEventControl = this.request.issuedAtPhysicalControl;
    this.committed = null; this.state = 'prefix'; this.queued = null;
    this.pendingSelection = undefined; this.eventCount = 0; this.events = [];
  }
  #context(c) {
    require(this.state === 'prefix' && this.owner.role === 'prefix' && !this.owner.ended && !this.owner.cancelRequested,
      'A live prefix command owner is required');
    require(c?.episode === this.request.episode && c.requestId === this.request.requestId
      && c.parent === this.prefixParent && this.owner.prefixParent === this.prefixParent
      && c.activeCarryController === this.prefixParent && this.owner.request === this.request
      && same(c.originalGoalWorld, this.request.originalGoalWorld), 'Prefix request, parent or original destination changed');
    require(count(c.physicalControl) && c.physicalControl >= this.lastEventControl,
      'Commands must use the current actual physical clock');
    require(c.controller === this.prefixParent || (this.prefixParent.phase === 'approach'
      && c.approachParent === this.prefixParent && c.controller != null), 'Unrelated approach controller');
    require(['approach', 'teacher'].includes(this.prefixParent.phase), 'Prefix no longer at approach or pickup');
    if (this.committed) require(this.prefixParent.phase === 'teacher' && c.controller === this.prefixParent
      && this.prefixParent.skill === this.committed.skill && this.prefixParent.worldFrames === this.committed.frames,
    'Committed pickup source or controller changed');
  }
  /** Call at committed-control boundaries or immediately before a UI effect.
   * Teacher preparation is conservatively committed from its first admitted
   * row, even before measured hand loading. This does not assert a loaded box.
   */
  observe(c) {
    this.#context(c);
    if (this.prefixParent.phase === 'teacher') {
      require(c.controller === this.prefixParent && this.prefixParent.skill && this.prefixParent.worldFrames,
        'The actual pickup parent and its current reference are required');
      this.committed ??= {skill:this.prefixParent.skill, frames:this.prefixParent.worldFrames,
        physicalControl:c.physicalControl};
    }
    this.lastEventControl = c.physicalControl;
    return Boolean(this.committed);
  }
  /** command.kind is box, selection, movement or escape. A selection/movement
   * replaces the queued box intent; it does not change owned carry perception.
   * Box commands contain no initial box pose: that is measured at execution.
   * Prescribed toolbar pickup/carry requests retain a null original goal and
   * their taskKind; they are handed back to the ordinary task planner.
   */
  handle(command, c) {
    this.#context(c);
    require(['box','selection','movement','escape'].includes(command?.kind), 'Unknown prefix command');
    let queued = null, selection = this.pendingSelection;
    if (command.kind === 'box') {
      const taskKind = command.taskKind ?? 'carry';
      require(Number.isSafeInteger(command.requestId) && command.requestId > this.request.requestId
        && c.latestRequestId === command.requestId && c.queuedRequestId === command.requestId
        && command.requestId > Math.max(this.queued?.requestId ?? this.request.requestId,
          this.owner.permittedLatestRequestId ?? this.request.requestId), 'Only the latest explicitly queued box request may replace intent');
      require(['pickup','carry'].includes(taskKind)
        && (command.originalGoalWorld === null || (taskKind === 'carry' && vector(command.originalGoalWorld, 3)))
        && typeof command.objectBodyName === 'string'
        && command.objectBodyName.length > 0 && (command.rawClickWorld == null || vector(command.rawClickWorld,3)),
      'A box intent requires its task, selected object and finite destination or explicit prescribed null goal');
      queued = freeze({kind:taskKind, taskKind, requestId:command.requestId, episode:this.request.episode,
        commandIssuedAtPhysicalControl:c.physicalControl,
        originalGoalWorld:command.originalGoalWorld === null ? null : Array.from(command.originalGoalWorld),
        objectBodyName:command.objectBodyName,
        ...(command.rawClickWorld ? {rawClickWorld:Array.from(command.rawClickWorld)} : {})});
      selection = command.objectBodyName;
    } else {
      require(c.queuedRequestId == null, 'Selection, movement and Escape must discard the queued box request first');
      if (command.kind === 'selection') {
        require(command.objectBodyName === null || (typeof command.objectBodyName === 'string' && command.objectBodyName.length),
          'Selection must name an object or explicitly deselect');
        selection = command.objectBodyName;
      }
    }
    const committed = this.observe(c);
    if (committed) {
      require(typeof this.owner.requestPrefixFinish === 'function', 'Prefix finish registration is required before integration');
      this.owner.requestPrefixFinish({...c, parent:this.prefixParent});
      require(!this.prefixParent.finishRequested, 'Legacy cancellation must not disable the first student window');
    } else {
      // Host invokes the ordinary approach/walking cancellation and waits for
      // its existing ending. This adapter never declares that ending complete.
      this.owner.cancel('unloaded_approach_command'); this.state = 'ordinary_approach_ending';
    }
    const supersededRequestId = this.queued?.requestId ?? null;
    this.queued = queued; this.pendingSelection = selection;
    const effect = freeze({behavior:committed ? 'finish_existing_programme' : 'ordinary_approach_cancel',
      preserveDestinationWorld:Array.from(this.request.originalGoalWorld),
      ownedObjectBodyName:this.prefixParent.skill.objectBodyName,
      pendingSelection:selection, queuedCommand:queued, supersededRequestId,
      suppressLegacyCarryCancel:committed, replaceProgrammeRequest:false,
      eventPhysicalControl:c.physicalControl, physicalControlsConsumed:0});
    this.events.push(effect); if(this.events.length > 32) this.events.shift(); this.eventCount++;
    return effect;
  }
  /** Completion is supplied by the existing physical lifecycle, never by a
   * timer, a new click, a nearby box, or an adapter event counter.
   */
  release(c) {
    require(this.state !== 'reset' && this.state !== 'released' && c.episode === this.request.episode
      && count(c.physicalControl) && c.physicalControl >= this.lastEventControl
      && c.skillActive === false && c.restrictedSettled === true,
    'Wait for current-episode ordinary control ownership and completed movement');
    require(this.state === 'ordinary_approach_ending' || (this.owner.returnedToMain === true
      && this.owner.ended?.goalReached === true), 'The loaded programme must complete and transfer ownership first');
    this.state = 'released'; this.lastEventControl = c.physicalControl;
    return {queuedCommand:this.queued, pendingSelection:this.pendingSelection};
  }
  /** Synchronous factory transaction. Read the actual box at execution start,
   * then recheck episode/clock/latest ownership after the factory returns.
   * latestRequestId here is the current accepted queued command identity, not
   * the last diagnostic/refused request record. Use this method only if the
   * host has selected a matched programme; release().queuedCommand also works
   * with the ordinary planner without invoking this factory. Asset loading
   * and physical admission remain the host's responsibility.
   */
  takeQueuedExecution(readExecutionContext, factory = makeTaskRelativeCarryRequest) {
    require(this.state === 'released' && this.queued, 'No released queued box request');
    const command = this.queued;
    require(command.taskKind === 'carry' && vector(command.originalGoalWorld,3),
      'Prescribed pickup/carry remains an ordinary task, without a manufactured destination');
    const c = readExecutionContext(command.objectBodyName);
    const valid = value => value?.episode === command.episode && value.latestRequestId === command.requestId
      && value.skillActive === false && value.restrictedSettled === true && count(value.physicalControl)
      && value.physicalControl >= this.lastEventControl && value.objectBodyName === command.objectBodyName
      && vector(value.objectPositionWorld,3);
    require(valid(c), 'Queued command needs current execution-start ownership and measured box');
    const object = Array.from(c.objectPositionWorld);
    const request = factory({episode:c.episode, requestId:command.requestId, physicalControl:c.physicalControl,
      commandIssuedAtPhysicalControl:command.commandIssuedAtPhysicalControl,
      originalGoalWorld:Array.from(command.originalGoalWorld), initialObjectPositionWorld:object,
      objectBodyName:command.objectBodyName});
    require(request && typeof request.then !== 'function', 'Execution request factory must be synchronous');
    const after = readExecutionContext(command.objectBodyName);
    require(this.state === 'released' && this.queued === command && valid(after)
      && after.physicalControl === c.physicalControl && same(after.objectPositionWorld,object),
    'Queued execution ownership changed while preparing its request');
    this.queued = null;
    return {command, request, executionStartedAtPhysicalControl:c.physicalControl,
      actualExecutionStartObjectPositionWorld:object};
  }
  reset() {
    if (this.state === 'reset') return;
    this.owner.cancel('episode_reset'); this.state = 'reset'; this.queued = null; this.pendingSelection = undefined;
  }
}
