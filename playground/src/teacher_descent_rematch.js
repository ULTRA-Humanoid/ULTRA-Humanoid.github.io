/** Optional source026 descent feedback. Reference phase changes never change physics time. */
import {transformTeacherReference} from './teacher_reference.js';
import {measurePlacementRelease} from './placement_feedback.js';
import {OBJECT_PROFILES} from './object_profiles.js';

export const TEACHER_DESCENT_PROFILE = Object.freeze({
  sourceName:'Full source026 carry about1.224 metres and set down',
  objectBodyName:OBJECT_PROFILES.largebox.bodyName, rawFrames:388,
  firstRaw:280, lastRaw:387, teacherControls:108,
  settlingControls:180, exitControls:439, standingControls:120, totalControls:847,
});
const require = (condition, message) => { if (!condition) throw new Error(message); };
const finite = (a, n) => a?.length === n && Array.from(a).every(Number.isFinite);
const same = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const integer = n => Number.isSafeInteger(n) && n >= 0;
const freeze = value => {
  if (ArrayBuffer.isView(value)) return Object.freeze(Array.from(value));
  if (value && typeof value === 'object') {
    for (const key of Object.keys(value)) value[key] = freeze(value[key]);
    Object.freeze(value);
  }
  return value;
};
const endingPhase = n => n < 180 ? 'settling' : n < 240 ? 'teacher_exit_hold'
  : n < 439 ? 'teacher_exit_retreat' : n < 619 ? 'teacher_exit_settling' : 'teacher_standing';

/** Pure admission/planning. Unsupported cases retain their ordinary controller. */
export function planTeacherDescentRematch({parent, live, originalGoalWorld}) {
  const p = TEACHER_DESCENT_PROFILE;
  const decline = reason => ({supported:false, reason});
  if (parent?.phase !== 'teacher' || parent.finishRequested || parent.cancelRequested)
    return decline('ordinary_teacher_not_current');
  const raw = parent.rawSkill, skill = parent.skill, source = parent.worldFrames;
  if (raw?.name !== p.sourceName || raw.sourceFrames !== p.rawFrames
      || skill?.name !== p.sourceName || skill.objectBodyName !== p.objectBodyName
      || raw.objectBodyName !== p.objectBodyName)
    return decline('unsupported_descent_source');
  const offset = skill.sourceFrames - raw.sourceFrames;
  if (!integer(offset) || parent.referenceIndex !== offset + p.firstRaw)
    return decline('not_original_source_phase280');
  const start = offset + p.firstRaw, last = offset + p.lastRaw;
  if (!Array.isArray(source) || source.length < last + 17 || !source.every(row => finite(row, 747)))
    return decline('incomplete_descent_reference');
  const segment = parent.segmentIndex, goals = parent.plan?.goals;
  if (!integer(segment) || !Array.isArray(goals) || segment !== goals.length - 1
      || !finite(originalGoalWorld, 3) || !same(goals[segment], originalGoalWorld)
      || !same(parent.requestedGoalWorld, originalGoalWorld))
    return decline('descent_requires_original_final_segment');
  if (!finite(live?.rootPositionWorld, 3) || !finite(live.rootQuaternionWorld, 4)
      || !finite(live.objectPositionWorld, 3) || !finite(live.rootLinearVelocityWorld, 3)
      || !finite(live.objectLinearVelocityWorldMps, 3))
    return decline('invalid_actual_descent_state');
  const q = live.rootQuaternionWorld;
  if (Math.abs(Math.hypot(...q) - 1) > 1e-5 || live.rootPositionWorld[2] < .65
      || 1 - 2 * (q[0] ** 2 + q[1] ** 2) < .8 || live.objectPositionWorld[2] <= .5
      || !finite(live.handNormalForceN, 2) || live.handNormalForceN.some(x => x <= 1)
      || !Number.isFinite(live.loadedFootNormalForceN) || live.loadedFootNormalForceN <= 5
      || !integer(live.loadedFootGroundContacts) || live.loadedFootGroundContacts === 0)
    return decline('actual_descent_entry_not_held_and_supported');
  const a = source[start], z = source[last], box = live.objectPositionWorld;
  const sourceRay = [z[71] - a[71], z[72] - a[72]];
  const goalRay = [originalGoalWorld[0] - box[0], originalGoalWorld[1] - box[1]];
  if (Math.hypot(...sourceRay) <= 1e-9 || Math.hypot(...goalRay) <= 1e-9)
    return decline('undefined_descent_goal_ray');
  const yaw = Math.atan2(goalRay[1], goalRay[0]) - Math.atan2(sourceRay[1], sourceRay[0]);
  const co = Math.cos(yaw), si = Math.sin(yaw);
  const transform = freeze({yawRadians:yaw,
    translation:[box[0] - co * a[71] + si * a[72], box[1] - si * a[71] - co * a[72], 0]});
  const bank = freeze(source.map(row => Array.from(transformTeacherReference(row, transform))));
  const entryMismatch = {
    rootM:distance(live.rootPositionWorld, bank[start].slice(0, 3)),
    boxM:distance(box, bank[start].slice(71, 74)), boxHeightM:box[2] - bank[start][73],
  };
  if (entryMismatch.rootM > .15 || entryMismatch.boxM > .15)
    return {...decline('descent_reference_entry_mismatch'), entryMismatch};
  const finalReferenceGoalResidualM = distance(bank[last].slice(71, 73), originalGoalWorld.slice(0, 2));
  if (finalReferenceGoalResidualM > .1)
    return {...decline('descent_reference_misses_original_goal'), entryMismatch, finalReferenceGoalResidualM};
  return {supported:true, reason:null, transform, bank, start, last, offset, segmentIndex:segment,
    entryMismatch, finalReferenceGoalResidualM};
}

/** Shared MatchedCarryRuntime owner. No saved observations or initial-pose whitelist. */
export class TeacherDescentRematchOwner {
  constructor({context:c, prefixParent, live, prefixSafety, readCommandContext = null}) {
    require(integer(c?.episode) && integer(c.physicalControl) && integer(c.requestId) && c.requestId > 0
      && c.latestRequestId === c.requestId && c.queuedRequestId == null && c.parent === prefixParent,
    'Current original request and actual clock required');
    require(prefixSafety?.episode === c.episode && prefixSafety.evaluatedSubsteps === c.physicalControl * 17
      && prefixSafety.missingEvaluation === false && prefixSafety.anyActualViolation === false,
    'Complete actual prefix must satisfy unchanged contact limits');
    const plan = planTeacherDescentRematch({parent:prefixParent, live, originalGoalWorld:c.originalGoalWorld});
    require(plan.supported, plan.reason);
    this.prefixParent = this.originalCarryController = prefixParent;
    this.originalPlan = prefixParent.plan;
    this.originalBank = prefixParent.worldFrames;
    this.originalSkill = prefixParent.skill;
    this.originalRawSkill = prefixParent.rawSkill;
    this.originalSegmentIndex = plan.segmentIndex;
    Object.assign(this, {bank:plan.bank, transform:plan.transform, entryMismatch:plan.entryMismatch,
      finalReferenceGoalResidualM:plan.finalReferenceGoalResidualM, start:plan.start, last:plan.last});
    this.skill = freeze({name:prefixParent.skill.name, sourceFrames:prefixParent.skill.sourceFrames,
      objectBodyName:prefixParent.skill.objectBodyName,
      objectPointsLocal:structuredClone(prefixParent.skill.objectPointsLocal), locomotionOnly:false});
    this.raw = this.skill;
    this.endingReferenceFrame = this.bank[this.last];
    this.request = freeze({episode:c.episode, requestId:c.requestId,
      issuedAtPhysicalControl:c.requestIssuedAtPhysicalControl ?? null,
      originalGoalWorld:Array.from(c.originalGoalWorld)});
    this.lastPhysicalControl = this.entryPhysicalControl = c.physicalControl;
    this.maximumPhysicalControl = c.physicalControl + TEACHER_DESCENT_PROFILE.totalControls;
    this.role = 'teacher_descent108'; this.counts = {teacher_descent108:0, postplacement:0};
    this.released = this.serial = this.finishRequestCount = 0;
    this.pending = this.ended = null; this.window = null;
    this.anyActualViolation = this.cancelRequested = this.finishRequested = this.returnedToMain = false;
    this.permittedLatestRequestId = c.requestId; this.finishRequests = [];
    this.prefixSafety = structuredClone(prefixSafety);
    this.parent = {phase:this.role, segmentIndex:plan.segmentIndex, referenceIndex:this.start,
      skill:this.skill, sourceFrames:this.skill.sourceFrames, worldFrames:this.bank,
      requestedGoalWorld:this.request.originalGoalWorld, plan:this.originalPlan,
      segmentResults:prefixParent.segmentResults, segmentExitResults:prefixParent.segmentExitResults,
      outcome:{}, reset:() => this.cancel('episode_reset'), isActive:() => this.active,
      requestCancel:() => this.requestFinish(readCommandContext ? readCommandContext()
        : {...c, parent:this.parent, physicalControl:this.lastPhysicalControl, latestRequestId:this.permittedLatestRequestId})};
    this.transitions = [{role:this.role, physicalControl:c.physicalControl, sourceIndex:this.start}];
  }
  get active() { return !this.ended && !this.cancelRequested; }
  get sourceIndex() { return this.role === 'teacher_descent108' ? this.start + this.counts.teacher_descent108 : this.last + 1; }
  originalCurrent() {
    const p = this.prefixParent;
    return p.plan === this.originalPlan && p.worldFrames === this.originalBank && p.skill === this.originalSkill
      && p.rawSkill === this.originalRawSkill && p.segmentIndex === this.originalSegmentIndex
      && p.referenceIndex === this.start && p.phase === 'teacher'
      && same(p.requestedGoalWorld, this.request.originalGoalWorld);
  }
  requestCurrent(c) { return c.episode === this.request.episode && c.requestId === this.request.requestId
    && c.latestRequestId === this.permittedLatestRequestId && !this.cancelRequested
    && same(c.originalGoalWorld, this.request.originalGoalWorld); }
  current(c) { return this.requestCurrent(c) && this.originalCurrent() && c.parent === this.parent
    && c.physicalControl === this.lastPhysicalControl && this.parent.worldFrames === this.bank
    && this.parent.skill === this.skill && this.parent.plan === this.originalPlan
    && this.parent.segmentIndex === this.originalSegmentIndex && this.parent.referenceIndex === this.sourceIndex
    && this.parent.phase === this.role && same(this.parent.requestedGoalWorld, this.request.originalGoalWorld); }
  cancel(reason = 'cancelled') {
    this.cancelRequested = true; this.pending = null;
    this.ended ??= {completionReason:reason, goalReached:false, physicalControl:this.lastPhysicalControl};
    return this.ended;
  }
  requestFinish(c) {
    require(this.active && this.current({...c, latestRequestId:this.permittedLatestRequestId}),
      'Only the current actual owner may request finite finish');
    require(integer(c.latestRequestId) && c.latestRequestId >= this.permittedLatestRequestId
      && (c.latestRequestId === this.permittedLatestRequestId || c.queuedRequestId === c.latestRequestId),
    'Only an accepted queued request may advance ownership');
    this.finishRequested = true; this.permittedLatestRequestId = c.latestRequestId;
    this.finishRequests.push({physicalControl:c.physicalControl, latestRequestId:c.latestRequestId,
      queuedRequestId:c.queuedRequestId ?? null, behavior:'complete original descent and full ending; retain original goal'});
    this.finishRequestCount++; if (this.finishRequests.length > 32) this.finishRequests.shift();
    return this.review();
  }
  sample(c) {
    require(this.active && this.current(c), 'Current actual descent owner required');
    const phase = this.role === 'postplacement' ? endingPhase(this.counts.postplacement) : this.role;
    const sample = {candidateId:++this.serial, phase, sourceIndex:this.sourceIndex,
      mode:phase === 'settling' ? 'student' : 'teacher'};
    if (this.role === 'teacher_descent108') sample.referenceFrames = [this.bank[this.sourceIndex + 1], this.bank[this.sourceIndex + 16]];
    this.pending = {candidateId:sample.candidateId, phase, sourceIndex:sample.sourceIndex};
    return sample;
  }
  canActuate(c, s) { return this.active && this.current(c) && this.pending?.candidateId === s.candidateId
    && this.pending.phase === s.phase && this.pending.sourceIndex === s.sourceIndex
    && s.mode === (s.phase === 'settling' ? 'student' : 'teacher')
    && (this.role !== 'teacher_descent108' || (s.referenceFrames?.length === 2
      && s.referenceFrames[0] === this.bank[this.sourceIndex + 1] && s.referenceFrames[1] === this.bank[this.sourceIndex + 16])); }
  commit(c, {sample, physicsSubsteps, preview, live, actualViolation}) {
    require(physicsSubsteps === 17 && this.canActuate({...c, physicalControl:c.physicalControl - 1}, sample)
      && c.physicalControl === this.lastPhysicalControl + 1 && c.physicalControl <= this.maximumPhysicalControl,
    'Only the next actual complete control may commit');
    if (sample.phase !== 'settling') require(preview?.supported && preview.requestedSubsteps === 17
      && preview.completedSubsteps === 17 && preview.unwantedContactCount === 0 && preview.boxBoxUnwantedCount === 0,
    'All17 preview substeps must retain the original contact limits');
    require(typeof actualViolation === 'boolean', 'Actual physics measurement is mandatory');
    this.lastPhysicalControl = c.physicalControl; this.counts[this.role]++; this.pending = null;
    this.anyActualViolation ||= actualViolation; this.parent.referenceIndex = this.sourceIndex;
    if (this.anyActualViolation) return this.cancel('actual_contact_or_balance_violation');
    const m = measurePlacementRelease(live, this.endingReferenceFrame);
    if (this.role === 'teacher_descent108') {
      this.released = m.released ? this.released + 1 : 0;
      if (!m.valid || !m.balanced) return this.cancel('invalid_teacher_descent_measurement');
      if (this.counts.teacher_descent108 === 108) {
        this.teacherEndpoint = {physicalControl:c.physicalControl, measured:m, releasedControls:this.released};
        if (this.released < 6 || !m.handoff) return this.cancel('teacher_descent_not_released');
        this.role = this.parent.phase = 'postplacement'; this.parent.referenceIndex = this.sourceIndex;
        this.transitions.push({role:this.role, physicalControl:c.physicalControl, sourceIndex:this.sourceIndex});
      }
    } else if (this.counts.postplacement === 739) {
      const error = distance(live.objectPositionWorld.slice(0, 2), this.request.originalGoalWorld.slice(0, 2));
      const complete = m.released && error <= .1 && !this.anyActualViolation;
      this.ended = {completionReason:complete ? 'finished' : 'ending_incomplete', goalReached:complete,
        physicalControl:c.physicalControl, remainingDistanceM:error, measuredRelease:m, originalGoalWorld:this.request.originalGoalWorld};
      this.parent.phase = 'complete'; this.parent.completionReason = this.ended.completionReason; this.parent.outcome = this.ended;
    }
    return this.review();
  }
  canReturnToMain(c) { return !this.returnedToMain && this.ended?.goalReached === true
    && this.requestCurrent(c) && this.originalCurrent() && c.parent === this.parent
    && c.physicalControl === this.lastPhysicalControl && c.physicalControl === this.maximumPhysicalControl
    && this.ended.physicalControl === c.physicalControl && this.parent.phase === 'complete'
    && this.parent.referenceIndex === this.last + 1 && this.parent.segmentIndex === this.originalSegmentIndex
    && this.parent.skill === this.skill && this.parent.worldFrames === this.bank && this.parent.plan === this.originalPlan
    && same(this.parent.requestedGoalWorld, this.request.originalGoalWorld)
    && this.counts.teacher_descent108 === 108 && this.counts.postplacement === 739; }
  returnToMain(c, transfer) {
    require(this.canReturnToMain(c) && typeof transfer === 'function', 'Only a completed current descent may return control');
    const value = transfer(); require(!value || typeof value.then !== 'function', 'Ownership transfer must be synchronous');
    this.returnedToMain = true; this.returnedAtPhysicalControl = c.physicalControl; return this.review();
  }
  review() { return {request:this.request, role:this.role, counts:{...this.counts}, sourceIndex:this.sourceIndex,
    entryPhysicalControl:this.entryPhysicalControl, lastPhysicalControl:this.lastPhysicalControl,
    maximumPhysicalControl:this.maximumPhysicalControl, transform:this.transform, entryMismatch:this.entryMismatch,
    finalReferenceGoalResidualM:this.finalReferenceGoalResidualM, originalSegmentIndex:this.originalSegmentIndex,
    transitions:structuredClone(this.transitions), teacherEndpoint:this.teacherEndpoint ?? null,
    ended:this.ended ? structuredClone(this.ended) : null, anyActualViolation:this.anyActualViolation,
    prefixSafety:this.prefixSafety, completeOriginalTeacherSuffix:[280, 387],
    executedTeacherRawSources:this.counts.teacher_descent108 ? [280, 279 + this.counts.teacher_descent108] : null,
    singleCoupledSE2:true, noGoalHeightOrTimingEdits:true, finishRequested:this.finishRequested,
    finishRequests:structuredClone(this.finishRequests), finishRequestCount:this.finishRequestCount,
    permittedLatestRequestId:this.permittedLatestRequestId, returnedToMain:this.returnedToMain,
    returnedAtPhysicalControl:this.returnedAtPhysicalControl ?? null}; }
}
