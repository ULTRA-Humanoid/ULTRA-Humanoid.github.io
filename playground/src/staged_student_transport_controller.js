// A measured moving interval inside an explicitly declared carry reference.
// The caller owns inference, previews, physical integration and the carry clock.
import { BoundedStageGoalWindow, packLegacyStageStudentInput } from './stage_goal.js';
import { hasPreparedStudentLiftProfile, STUDENT_LIFT_PROFILE } from './student_lift_profile.js';
import { OBJECT_PROFILES } from './object_profiles.js';

export const STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES = Object.freeze([
  'left_rubber_hand', 'right_rubber_hand', 'left_wrist_yaw_link',
  'right_wrist_yaw_link', 'right_wrist_pitch_link', 'left_wrist_pitch_link',
]);
export const STAGED_STUDENT_TRANSPORT_LIMITS = Object.freeze({
  sourceControls: 456, horizonControls: 90, minRootHeightM: .65,
  minUpright: .8, minObjectHeightM: .5, minEachHandForceN: 1,
  maxEntryRootAndObjectErrorM: .15, physicsSubsteps: 17,
  objectBodyName: OBJECT_PROFILES.largebox.bodyName,
});
/** Hand-load admission rules. `instant` is the original one-state test (each
 * hand > 1 N at the entry instant). `window` requires at least
 * minLoadedControls of the last windowControls control boundaries (the entry
 * instant included) to have both hands > 1 N; the default 6 of 10 is the
 * windowed median, chosen from the saved combined cells (see
 * scripts/joint100_v2/transport_window_tables.py): a windowed minimum is
 * strictly stricter than the instant test and refused 7 of 9 clean carries.
 */
export const STAGED_STUDENT_TRANSPORT_ADMISSION = Object.freeze({
  instant: Object.freeze({ mode: 'instant' }),
  window: Object.freeze({ mode: 'window', windowControls: 10, minLoadedControls: 6 }),
});
/** Mid-window divergence exit against the interpolated world reference. The
 * root XY limit is 0.30 m, not 0.25 m: clean saved carries drift to 0.25-0.29 m
 * inside the window (H062 H083 H087 H088) while the two diverging carries were
 * separated by yaw (>25 deg at reference 299), then hands and root height.
 */
export const STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS = Object.freeze({
  maxRootXYErrorM: .30, maxYawErrorRad: 25 * Math.PI / 180, minRootHeightM: .65, maxUnloadedControls: 5,
});

const vector = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
const count = value => Number.isSafeInteger(value) && value >= 0;
const equal = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const distance = (a, b) => Math.hypot(...Array.from(a, (v, i) => v - b[i]));
const unit = q => vector(q, 4) && Math.abs(Math.hypot(...q) - 1) <= 1e-5;
const refusal = reason => ({ supported: false, reason });
const headingRad = q => Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));
const wrapRad = a => Math.atan2(Math.sin(a), Math.cos(a));
export function normalizeTransportAdmission(admission = STAGED_STUDENT_TRANSPORT_ADMISSION.instant) {
  if (admission?.mode === 'instant') return { mode: 'instant' };
  if (admission?.mode !== 'window') throw new Error('Transport admission mode must be instant or window');
  const { windowControls = 10, minLoadedControls = 6 } = admission;
  if (!Number.isSafeInteger(windowControls) || windowControls < 1 || windowControls > 90
      || !Number.isSafeInteger(minLoadedControls) || minLoadedControls < 1 || minLoadedControls > windowControls) {
    throw new Error('Windowed transport admission requires 1 <= minLoadedControls <= windowControls <= 90');
  }
  return { mode: 'window', windowControls, minLoadedControls };
}
export function normalizeTransportDivergenceLimits(limits) {
  if (limits === null || limits === undefined) return null;
  const out = { ...STAGED_STUDENT_TRANSPORT_DIVERGENCE_LIMITS, ...limits };
  for (const key of ['maxRootXYErrorM', 'maxYawErrorRad', 'minRootHeightM']) {
    if (!Number.isFinite(out[key]) || out[key] <= 0) throw new Error(`Transport divergence limit ${key} must be positive`);
  }
  if (!Number.isSafeInteger(out.maxUnloadedControls) || out.maxUnloadedControls < 1) throw new Error('maxUnloadedControls must be a positive integer');
  return Object.freeze(out);
}
function freeze(value) {
  if (ArrayBuffer.isView(value)) return Object.freeze(Array.from(value));
  if (value && typeof value === 'object') {
    for (const [key, child] of Object.entries(value)) value[key] = freeze(child);
    Object.freeze(value);
  }
  return value;
}
function supportBodies(value) {
  return Array.isArray(value) && value.length === STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.length
    && new Set(value).size === STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.length
    && value.every(name => STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES.includes(name));
}

/** Admission screens the actual moving predecessor. It does not claim arrival
 * or certify a future action. The skill must explicitly declare this interval;
 * source length or a skill name alone cannot enable student transport.
 */
export function checkStagedStudentTransportEntry({ parent, live, episode, physicalControl,
  startReferenceIndex, endReferenceIndex, expectedSourceControls,
  allowedSupportBodies = STAGED_STUDENT_TRANSPORT_SUPPORT_BODIES,
  admission = STAGED_STUDENT_TRANSPORT_ADMISSION.instant }) {
  const limits = STAGED_STUDENT_TRANSPORT_LIMITS, rule = normalizeTransportAdmission(admission);
  const studentLiftProfile = hasPreparedStudentLiftProfile(parent?.skill)
    && expectedSourceControls === STUDENT_LIFT_PROFILE.preparedSourceControls
    && startReferenceIndex === STUDENT_LIFT_PROFILE.preparedInterval[0]
    && endReferenceIndex === STUDENT_LIFT_PROFILE.preparedInterval[1];
  if (!count(episode) || !count(physicalControl)) return refusal('invalid_physical_clock');
  if (!count(startReferenceIndex) || !count(endReferenceIndex)
      || (!studentLiftProfile && (endReferenceIndex - startReferenceIndex !== limits.horizonControls
        || expectedSourceControls !== limits.sourceControls)) || endReferenceIndex >= expectedSourceControls) {
    return refusal('unsupported_transport_interval');
  }
  if (!supportBodies(allowedSupportBodies)) return refusal('unsupported_grip_contact_set');
  if (!parent || parent.phase !== 'teacher' || parent.finishRequested
      || !count(parent.segmentIndex) || parent.referenceIndex !== startReferenceIndex) {
    return refusal('carry_not_at_declared_transport_entry');
  }
  const skill = parent.skill, frames = parent.worldFrames;
  if (skill?.sourceFrames !== expectedSourceControls
      || !equal(skill.studentTransportInterval, [startReferenceIndex, endReferenceIndex])
      || skill.objectBodyName !== limits.objectBodyName || live?.objectBodyName !== skill.objectBodyName) {
    return refusal('carry_transport_interval_not_declared');
  }
  const entryFrame = frames?.[startReferenceIndex], targetFrame = frames?.[endReferenceIndex];
  if (!Array.isArray(frames) || frames.length < expectedSourceControls + 16
      || !vector(entryFrame, 747) || !vector(targetFrame, 747)
      || !unit(entryFrame.slice(3, 7)) || !unit(targetFrame.slice(3, 7))
      || !vector(parent.requestedGoalWorld, 3)
      || !vector(live.rootPosWorld, 3) || !unit(live.rootQuatXyzwWorld)
      || !vector(live.rootVelWorld, 3) || !vector(live.objPosWorld, 3)
      || !vector(live.handNormalForceN, 2) || live.handNormalForceN.some(v => v < 0)) {
    return refusal('invalid_transport_state_or_reference');
  }
  const q = live.rootQuatXyzwWorld, upright = 1 - 2 * (q[0] ** 2 + q[1] ** 2);
  const rootErrorM = distance(live.rootPosWorld, entryFrame.slice(0, 3));
  const objectErrorM = distance(live.objPosWorld, entryFrame.slice(71, 74));
  const measured = { rootErrorM, objectErrorM, upright, rootHeightM: live.rootPosWorld[2],
    objectHeightM: live.objPosWorld[2], handNormalForceN: Array.from(live.handNormalForceN), admission: { ...rule } };
  let loaded = !live.handNormalForceN.some(v => v <= limits.minEachHandForceN);
  if (rule.mode === 'window') {
    // The caller supplies the solver hand loads read at the last control
    // boundaries, oldest first; the final sample is this entry instant.
    const history = live.handNormalForceHistoryN;
    if (!Array.isArray(history) || history.length < rule.windowControls
        || !history.every(pair => vector(pair, 2) && pair.every(v => v >= 0))
        || !equal(history.at(-1), live.handNormalForceN)) {
      return { ...refusal('transport_entry_force_history_short'), measured };
    }
    const recent = history.slice(-rule.windowControls).map(pair => Array.from(pair));
    const loadedControls = recent.filter(pair => pair.every(v => v > limits.minEachHandForceN)).length;
    measured.handNormalForceHistoryN = recent; measured.admission.loadedControls = loadedControls;
    loaded = loadedControls >= rule.minLoadedControls;
  }
  if (live.rootPosWorld[2] < limits.minRootHeightM || upright < limits.minUpright
      || live.objPosWorld[2] <= limits.minObjectHeightM || !loaded) {
    return { ...refusal('transport_entry_not_loaded_and_balanced'), measured };
  }
  if (rootErrorM > limits.maxEntryRootAndObjectErrorM || objectErrorM > limits.maxEntryRootAndObjectErrorM) {
    return { ...refusal('transport_entry_reference_error'), measured };
  }
  return { supported: true, reason: null, measured, admission: rule, segmentIndex: parent.segmentIndex,
    objectBodyName: skill.objectBodyName, startReferenceIndex, endReferenceIndex,
    expectedSourceControls, allowedSupportBodies: Array.from(allowedSupportBodies),
    ...(studentLiftProfile ? { studentTransportProfile: STUDENT_LIFT_PROFILE.id } : {}),
    entryReferenceFrame: Array.from(entryFrame), targetReferenceFrame: Array.from(targetFrame),
    plan: { stage: 'transport', mode: 'HOI_FULL', humanGoalWorld: Array.from(targetFrame.slice(0, 3)),
      humanGoalRotationWorld: Array.from(targetFrame.slice(3, 7)), objectGoalWorld: Array.from(targetFrame.slice(71, 74)),
      finalDestinationWorld: Array.from(parent.requestedGoalWorld) } };
}

export class StagedStudentTransportController {
  #parent;
  #skill;
  #frames;
  #entry;
  #window;
  #lastControl;
  #records = [];
  #pending = null;
  #refusedPreview = null;
  #ended = null;
  #divergence = null;
  #unloadedControls = 0;
  #lastDivergence = null;

  static tryStart(options) {
    const entry = checkStagedStudentTransportEntry(options);
    return entry.supported ? { supported: true, reason: null, entry,
      controller: new StagedStudentTransportController(options) } : { ...entry, controller: null };
  }

  constructor(options) {
    const entry = checkStagedStudentTransportEntry(options);
    if (!entry.supported) throw new Error(`Student transport entry refused: ${entry.reason}`);
    this.#parent = options.parent; this.#skill = options.parent.skill; this.#frames = options.parent.worldFrames;
    this.#entry = freeze(structuredClone(entry));
    this.#window = new BoundedStageGoalWindow(entry.plan, { episode: options.episode,
      physicalControl: options.physicalControl, horizonControls: entry.endReferenceIndex - entry.startReferenceIndex });
    this.#lastControl = options.physicalControl;
    this.#divergence = normalizeTransportDivergenceLimits(options.divergenceLimits);
  }

  get active() { return this.#ended === null; }
  get episode() { return this.#window.episode; }
  get startControl() { return this.#window.startControl; }
  get controls() { return this.#lastControl - this.startControl; }
  get horizonControls() { return this.#window.horizonControls; }
  get startReferenceIndex() { return this.#entry.startReferenceIndex; }
  get endReferenceIndex() { return this.#entry.endReferenceIndex; }
  get plan() { return this.#window.plan; }
  get ended() { return this.#ended ? structuredClone(this.#ended) : null; }
  get nextReferenceIndex() { return this.#entry.startReferenceIndex + this.controls; }
  get divergenceLimits() { return this.#divergence ? { ...this.#divergence } : null; }

  isOwnedBy({ parent, episode }) {
    return parent === this.#parent && episode === this.episode && parent?.phase === 'teacher'
      && parent.segmentIndex === this.#entry.segmentIndex && parent.skill === this.#skill
      && parent.skill.objectBodyName === this.#entry.objectBodyName && parent.worldFrames === this.#frames
      && parent.skill.sourceFrames === this.#entry.expectedSourceControls
      && (!this.#entry.studentTransportProfile
        || parent.skill.studentTransportProfile === this.#entry.studentTransportProfile)
      && equal(parent.skill.studentTransportInterval, [this.#entry.startReferenceIndex, this.#entry.endReferenceIndex])
      && equal(parent.requestedGoalWorld, this.plan.finalDestinationWorld)
      && equal(parent.worldFrames[this.#entry.startReferenceIndex], this.#entry.entryReferenceFrame)
      && equal(parent.worldFrames[this.#entry.endReferenceIndex], this.#entry.targetReferenceFrame);
  }

  #clock(context) {
    if (!this.isOwnedBy(context)) throw new Error('Student transport episode, owner, reference or original goal changed');
    if (!count(context.physicalControl) || context.physicalControl !== this.#lastControl
        || context.parent.referenceIndex !== this.nextReferenceIndex) throw new Error('Student transport physical and source clocks must match committed controls');
  }

  #finish(reason, context, { requiresTeacherResume = false, fallback = false, divergence = null } = {}) {
    if (!this.#ended) this.#ended = freeze({ reason, atControl: this.#lastControl,
      observedEpisode: context.episode, observedPhysicalControl: context.physicalControl,
      controls: this.controls, arrived: false, requiresTeacherResume, fallback,
      resumeReferenceIndex: requiresTeacherResume ? this.nextReferenceIndex : null,
      ...(divergence ? { divergence: structuredClone(divergence) } : {}) });
    this.#pending = null;
    return { active: false, ...this.ended };
  }

  /** A queued user cancellation remains with the carry's existing safe-place
   * lifecycle. Reset or owner replacement ends this helper without a resume.
   */
  observeParent(skillStep, context) {
    if (!this.active) return { active: false, ...this.ended };
    if (!this.isOwnedBy(context)) return this.#finish(context.episode !== this.episode
      ? 'episode_changed' : 'transport_owner_changed', context);
    this.#clock(context);
    if (skillStep?.mode === 'none') return this.#finish('parent_transport_ended', context);
    return { active: true, remainingControls: this.horizonControls - this.controls, arrived: false };
  }

  /** Mid-window divergence exit. Call once per control before sample(), with
   * the live state at this control boundary. The live root is compared to the
   * world reference row the window is about to execute (nextReferenceIndex);
   * either hand below the entry load for maxUnloadedControls consecutive
   * boundaries also ends the window. Ending requests teacher resume at this
   * unchanged reference index, not at the declared end of the interval.
   * Repeated calls at the same control return the same result.
   */
  checkDivergence(live, context) {
    if (!this.#divergence) throw new Error('Divergence exit is not enabled for this student transport');
    this.#clock(context);
    if (!this.active) return { active: false, ...this.ended };
    if (live?.objectBodyName !== this.#entry.objectBodyName) throw new Error('Loaded object changed');
    if (!vector(live.rootPosWorld, 3) || !unit(live.rootQuatXyzwWorld)
        || !vector(live.handNormalForceN, 2) || live.handNormalForceN.some(v => v < 0)) {
      throw new Error('Divergence check requires a finite live root pose and nonnegative hand loads');
    }
    if (this.#lastDivergence?.physicalControl === this.#lastControl) return structuredClone(this.#lastDivergence.result);
    const limits = this.#divergence, reference = this.#frames[this.nextReferenceIndex];
    const rootXYErrorM = Math.hypot(live.rootPosWorld[0] - reference[0], live.rootPosWorld[1] - reference[1]);
    const yawErrorRad = wrapRad(headingRad(live.rootQuatXyzwWorld) - headingRad(reference.slice(3, 7)));
    const unloaded = live.handNormalForceN.some(v => v < STAGED_STUDENT_TRANSPORT_LIMITS.minEachHandForceN);
    this.#unloadedControls = unloaded ? this.#unloadedControls + 1 : 0;
    const measured = { referenceIndex: this.nextReferenceIndex, controls: this.controls, rootXYErrorM,
      yawErrorDeg: yawErrorRad * 180 / Math.PI, rootHeightM: live.rootPosWorld[2],
      handNormalForceN: Array.from(live.handNormalForceN), unloadedControls: this.#unloadedControls };
    const rule = rootXYErrorM > limits.maxRootXYErrorM ? 'root_xy_error'
      : Math.abs(yawErrorRad) > limits.maxYawErrorRad ? 'yaw_error'
      : live.rootPosWorld[2] < limits.minRootHeightM ? 'root_height'
      : this.#unloadedControls >= limits.maxUnloadedControls ? 'hands_unloaded' : null;
    const result = rule
      ? this.#finish('transport_divergence_exit', context, { requiresTeacherResume: true, divergence: { rule, measured, limits: { ...limits } } })
      : { active: true, diverged: false, measured };
    this.#lastDivergence = { physicalControl: this.#lastControl, result: structuredClone(result) };
    return result;
  }

  /** Repeated reads consume no action or source time. A private copy binds the
   * subsequent committed record to the command that this helper supplied.
   */
  sample(live, context) {
    this.#clock(context);
    if (!this.active) return { command: null, active: false, ...this.ended };
    if (live?.objectBodyName !== this.#entry.objectBodyName) throw new Error('Loaded object changed');
    const encoded = this.#window.sample({ rootPositionWorld: live.rootPosWorld,
      rootQuaternionWorld: live.rootQuatXyzwWorld, objectPositionWorld: live.objPosWorld }, context);
    if (!this.#pending || !equal(this.#pending.command, encoded.command)) {
      this.#pending = { physicalControl: this.#lastControl, command: Array.from(encoded.command) };
    }
    return encoded;
  }

  buildObservation(encoded, bodyObservation, objectPointsHeading) {
    if (!this.active || !this.#pending || !equal(encoded.command, this.#pending.command)) throw new Error('Use this transport window\'s current command');
    if (encoded.stage !== 'transport' || encoded.mode !== 'HOI_FULL'
        || encoded.remainingControls !== this.horizonControls - this.controls
        || encoded.mask?.keepHuman !== 1 || encoded.mask.keepObj !== 1 || encoded.mask.keepObjPoints !== 1) {
      throw new Error('Loaded transport retains the complete HOI_FULL mask and actual countdown');
    }
    const observation = packLegacyStageStudentInput(encoded, bodyObservation, objectPointsHeading);
    // Independent copy of the actual packed input; the returned Float32Array
    // and caller-owned history/point arrays are unchanged.
    this.#pending.studentObservation = Array.from(observation);
    return observation;
  }

  #record(record, supported) {
    if (!this.#pending || this.#pending.physicalControl !== this.#lastControl
        || this.#pending.studentObservation?.length !== 1422
        || !vector(record?.command, 13) || !equal(record.command, this.#pending.command)
        || !vector(record.rawAction, 29) || record.preview?.supported !== supported) {
      throw new Error('A packed sampled input, actual candidate action and its physical preview are required');
    }
  }

  /** Caller has applied physics and advanced the parent once before commit.
   * This helper never changes the parent, its references, history or simulator.
   */
  commit({ record, physicsSubsteps, ...context }) {
    if (!this.active || !this.isOwnedBy(context) || this.controls >= this.horizonControls) throw new Error('Cannot commit an inactive or changed student transport');
    if (physicsSubsteps !== STAGED_STUDENT_TRANSPORT_LIMITS.physicsSubsteps
        || context.physicalControl !== this.#lastControl + 1
        || context.parent.referenceIndex !== this.nextReferenceIndex + 1) throw new Error('Commit requires17 real substeps and one advanced physical/source control');
    this.#record(record, true);
    if (record.preview.completedSubsteps !== 17 || record.preview.requestedSubsteps !== 17) throw new Error('The complete candidate action must pass its physical preview');
    this.#records.push(freeze({ ...structuredClone(record), preControl: this.#lastControl,
      physicalControl: context.physicalControl, sourceIndexBefore: this.nextReferenceIndex,
      sourceIndexAfter: this.nextReferenceIndex + 1, physicsSubsteps,
      studentObservation: this.#pending.studentObservation }));
    this.#lastControl = context.physicalControl; this.#pending = null;
    if (this.controls === this.horizonControls) return this.#finish('moving_window_complete', context, { requiresTeacherResume: true });
    return { active: true, controls: this.controls, remainingControls: this.horizonControls - this.controls, arrived: false };
  }

  /** An unexecuted refusal requests teacher inference at this unchanged source
   * index. The caller restores speculative history before that inference.
   */
  requestPreviewFallback({ record, ...context }) {
    this.#clock(context);
    if (!this.active) throw new Error('Cannot refuse an inactive student transport');
    this.#record(record, false);
    if (typeof record.preview.reason !== 'string' || !record.preview.reason) throw new Error('Explicit preview refusal reason required');
    this.#refusedPreview = freeze({ ...structuredClone(record), physicalControl: this.#lastControl,
      sourceIndex: this.nextReferenceIndex, executed: false,
      studentObservation: this.#pending.studentObservation });
    return this.#finish('student_preview_refused', context, { requiresTeacherResume: true, fallback: true });
  }

  cancel(reason, context) {
    if (typeof reason !== 'string' || !reason) throw new Error('Explicit cancellation reason required');
    if (this.isOwnedBy(context)) this.#clock(context);
    return this.#finish(reason, context);
  }

  review() {
    return { episode: this.episode, startControl: this.startControl, controls: this.controls,
      horizonControls: this.horizonControls, nextReferenceIndex: this.nextReferenceIndex,
      plan: structuredClone(this.plan), entry: structuredClone(this.#entry),
      allowedSupportBodies: Array.from(this.#entry.allowedSupportBodies), ended: this.ended,
      divergenceLimits: this.divergenceLimits,
      records: structuredClone(this.#records), refusedPreview: structuredClone(this.#refusedPreview) };
  }
}
