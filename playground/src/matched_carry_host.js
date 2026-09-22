/** Browser integration of the optional finite teacher/student carry.
 * The ordinary task planner remains responsible for selecting a request.
 * No request is admitted from a diagnostic callback or a saved physics state.
 */
import {MatchedCarryProgrammeOwner} from './matched_carry_programme.js';
import {MatchedCarryRuntime} from './matched_carry_runtime.js';
import {MatchedCarryPrefixCommands} from './matched_carry_prefix_commands.js';
import {StreamingCarryPhysicsMonitor} from './streaming_carry_physics_monitor.js';
import {OBJECT_PROFILES} from './object_profiles.js';

const require = (condition, message) => { if (!condition) throw new Error(message); };
const pick = (value, keys) => Object.fromEntries(keys.filter(key => value[key] !== undefined)
  .map(key => [key, value[key]]));
function lastControlDiagnostic(runtime) {
  const record = runtime.records.at(-1);
  if (!record || !Number.isSafeInteger(record.physicalControlBefore)) return null;
  // Preserve the reason for a stopped programme after its runtime is disposed.
  // Explicit fields keep optional full-state recordings out of live downloads.
  const summary = pick(record, ['physicalControlBefore', 'physicalControlAfter', 'phase',
    'sourceIndex', 'executed', 'actualSubsteps', 'observationDimension', 'reason']);
  if (record.error) summary.error = {message:String(record.error.message).slice(0, 2000)};
  if (record.preview) {
    summary.preview = pick(record.preview, ['supported', 'reason', 'requestedSubsteps',
      'completedSubsteps', 'minRootHeightM', 'minUpright', 'unwantedContactCount',
      'endpointUnwantedContactCount', 'allowedContactCount', 'peakUnwantedForceN', 'boxBoxUnwantedCount']);
    summary.preview.unwantedContacts = (record.preview.unwantedContacts ?? []).slice(0, 32)
      .map(contact => pick(contact, ['substep', 'endpoint', 'bodyId', 'bodyName', 'objectId',
        'objectName', 'hand', 'leg', 'normalForceN', 'distanceM']));
    if (record.preview.error) summary.preview.error = String(record.preview.error).slice(0, 2000);
    summary.preview.scope = 'Predicted substeps only; actualSubsteps records executed physics.';
  }
  return structuredClone(summary);
}
const runtimeDiagnostics = runtime => ({captureDiagnostics:runtime.captureDiagnostics,
  retainedControls:runtime.records.length, denseStorageAllocated:runtime.dense != null,
  recordsContainFullState:runtime.records.some(record => record.observation || record.historyBefore
    || record.historyAfter || record.physical), observerErrorCount:runtime.observerErrorCount,
  disposed:runtime.disposed, lastControl:lastControlDiagnostic(runtime)});

export class MatchedCarryHost {
  constructor({mujoco, model, readContext, runtimeEnvironment,
    getTeacherBuilder, setTeacherBuilder, monitor = null, allowOrdinaryLoadedSupport = null,
    runtimeFactory = env => new MatchedCarryRuntime(env)}) {
    this.readContext = readContext;
    this.runtimeEnvironment = runtimeEnvironment;
    this.getTeacherBuilder = getTeacherBuilder;
    this.setTeacherBuilder = setTeacherBuilder;
    this.runtimeFactory = runtimeFactory;
    this.allowOrdinaryLoadedSupport = allowOrdinaryLoadedSupport;
    this.monitor = monitor ?? new StreamingCarryPhysicsMonitor(mujoco, model,
      {objectBodyName:OBJECT_PROFILES.largebox.bodyName});
    this.owner = null;
    this.prefixCommands = null;
    this.runtime = null;
    this.lastRuntimeDiagnostics = null;
    this.lastFailure = null;
  }

  get inProgress() { return Boolean(this.owner && !this.owner.ended); }
  get ownsDestination() { return this.inProgress && !this.owner.returnedToMain; }
  get originalGoalWorld() { return this.ownsDestination ? this.owner.request.originalGoalWorld : null; }

  context() {
    const actual = this.readContext();
    // A rejected request can advance the UI diagnostic serial. Only an accepted
    // queued command advances carry ownership. Parent, active request, episode
    // and physical clock still come from the actual simulator/controller.
    return {...actual, latestRequestId:this.owner?.permittedLatestRequestId ?? actual.latestRequestId};
  }

  attach({request, rawSkill, prefixParent}) {
    require(!this.inProgress, 'Finish the current carry before attaching another');
    this.disposeRuntime();
    this.owner = new MatchedCarryProgrammeOwner({request, rawSkill, prefixParent});
    this.prefixCommands = new MatchedCarryPrefixCommands({owner:this.owner});
    this.lastRuntimeDiagnostics = null;
    this.lastFailure = null;
    return this.owner;
  }

  /** Attach an already admitted live continuation without changing its parent
   * or physics. Publish ownership only after runtime construction succeeds. */
  attachContinuation(owner) {
    require(!this.inProgress, 'Finish the current carry before attaching another');
    const context = this.readContext();
    require(owner?.active && owner.role !== 'prefix' && owner.prefixParent === context.parent
      && owner.current({...context, parent:owner.parent}), 'Current live continuation required');
    const runtime = this.runtimeFactory({...this.runtimeEnvironment(), owner,
      context:() => this.context(), physicsMonitor:this.monitor,
      captureDiagnostics:false, diagnosticCapacity:32,
      getCurrentTeacherBuilder:this.getTeacherBuilder, setTeacherBuilder:this.setTeacherBuilder});
    this.disposeRuntime();
    this.owner = owner; this.runtime = runtime; this.prefixCommands = null;
    this.lastRuntimeDiagnostics = this.lastFailure = null;
    return owner.parent;
  }

  command(command) {
    if (!this.inProgress) return null;
    const actual = this.readContext();
    const context = command.kind === 'box' ? actual : this.context();
    if (this.owner.role === 'prefix') {
      return this.prefixCommands.handle(command, {...context,
        parent:this.owner.prefixParent, controller:actual.parent});
    }
    this.owner.requestFinish(context);
    return {behavior:'finish_existing_programme', suppressLegacyCarryCancel:true,
      preserveDestinationWorld:this.owner.request.originalGoalWorld,
      ownedObjectBodyName:this.owner.raw.objectBodyName};
  }

  observePrefixCommandState() {
    if (!this.inProgress || this.owner.role !== 'prefix') return;
    if (!['approach', 'teacher'].includes(this.owner.prefixParent.phase)) {
      // The ordinary source can finish, refuse or fall back before first90.
      // Retire only this optional handoff; its ordinary ending remains owned
      // by main and still measures the original destination after retreat.
      this.owner.cancel('prefix_ended_without_student_handoff');
      return;
    }
    const actual = this.context();
    this.prefixCommands.observe({...actual, parent:this.owner.prefixParent, controller:actual.parent});
  }

  observeActualSubstep(data, {episode, physicalControl, substep, phase}) {
    const context = this.context();
    const ordinarySupport = !this.inProgress && this.allowOrdinaryLoadedSupport?.(context, phase) === true;
    return this.monitor.observe(data, {episode, physicalControl, substep, phase,
      allowLoadedSupport:Boolean(ordinarySupport || (this.inProgress && this.owner.role === 'prefix'
        && this.owner.requestCurrent(context) && context.parent === this.owner.prefixParent
        && ['teacher', 'student_transport'].includes(phase)))});
  }

  beginAfter90(student) {
    require(this.inProgress && this.owner.role === 'prefix' && !this.runtime,
      'Only the current first student window can hand off');
    const owner = this.owner;
    try {
      const runtime = this.runtimeFactory({...this.runtimeEnvironment(), owner,
        context:() => this.context(), physicsMonitor:this.monitor,
        captureDiagnostics:false, diagnosticCapacity:32,
        getCurrentTeacherBuilder:this.getTeacherBuilder, setTeacherBuilder:this.setTeacherBuilder});
      this.runtime = runtime;
      return owner.beginAfter90(this.context(), {student, live:runtime.live(),
        prefixSafety:this.monitor.snapshot(), readCommandContext:() => this.context()});
    } catch (error) {
      this.lastFailure = {phase:'first_student_handoff', physicalControl:this.context().physicalControl,
        episode:owner.request.episode, message:String(error?.message ?? error)};
      owner.cancel('matched_carry_entry_refused');
      this.disposeRuntime();
      throw error;
    }
  }

  async step() {
    require(this.runtime && this.owner?.active, 'Only an active matched carry can step');
    return this.runtime.step();
  }

  returnToOrdinary(transfer) {
    require(this.runtime, 'A completed runtime is required');
    const result = this.owner.returnToMain(this.context(), transfer);
    this.disposeRuntime();
    return result;
  }

  disposeRuntime() {
    if (!this.runtime) return;
    const runtime = this.runtime;
    this.lastRuntimeDiagnostics = {...runtimeDiagnostics(runtime), disposed:true};
    // The normal restricted builder is independent. Revoke only the builder
    // this runtime owns; never clear or dispose a newer main-loop builder.
    if (runtime.teacher && this.getTeacherBuilder() === runtime.teacher) this.setTeacherBuilder(null);
    runtime.dispose();
    this.runtime = null;
  }

  reset({episode, simulationTime}) {
    this.prefixCommands?.reset();
    this.owner?.cancel('episode_reset');
    this.disposeRuntime();
    this.owner = this.prefixCommands = null;
    this.lastRuntimeDiagnostics = null;
    this.lastFailure = null;
    this.monitor.reset({episode, simulationTime});
  }

  review() {
    return {programme:this.owner?.review() ?? null, monitor:this.monitor.snapshot(), lastFailure:this.lastFailure,
      prefix:this.prefixCommands ? {state:this.prefixCommands.state,
        committedAtControl:this.prefixCommands.committed?.physicalControl ?? null,
        eventCount:this.prefixCommands.eventCount, events:this.prefixCommands.events} : null,
      runtime:this.runtime ? runtimeDiagnostics(this.runtime) : this.lastRuntimeDiagnostics};
  }
}
