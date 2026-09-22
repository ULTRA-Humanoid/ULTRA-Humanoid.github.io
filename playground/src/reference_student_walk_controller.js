// Private experiment: a finite student window in a complete approved floor walk.
// The recorded controller remains the owner and executes the teacher suffix.
import {encodeStageGoal, packLegacyStageStudentInput} from './stage_goal.js';

const finite = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);
const equal = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const copy = v => v == null ? v : JSON.parse(JSON.stringify(v));
const freeze = v => { if (v && typeof v === 'object') { Object.values(v).forEach(freeze); Object.freeze(v); } return v; };
const count = v => Number.isSafeInteger(v) && v >= 0;
export const REFERENCE_STUDENT_WALK_HORIZON = 120;

export class ReferenceStudentWalkController {
  #owner; #skill; #last; #pending = null; #ended = null; #records = [];
  static tryStart(options) {
    const {owner, request, geometry, live, episode, physicalControl, alreadyUsed, useRestricted} = options;
    const sourceStartIndex = options.sourceStartIndex ?? 0;
    const intent = owner?.requestedIntent;
    const no = reason => ({supported: false, reason, controller: null});
    if (!useRestricted || alreadyUsed || !count(episode) || !count(physicalControl)) return no('invalid_owner_or_reused_request');
    if (intent?.type !== 'floor' || !count(intent.revision) || !finite(intent.goalWorld, 3)) return no('floor_request_required');
    if (![0, 90].includes(sourceStartIndex) || (sourceStartIndex && (!count(options.approvedPhysicalControl)
        || physicalControl !== options.approvedPhysicalControl + sourceStartIndex))) return no('actual_complete_teacher_prefix_required');
    if (owner.referenceIndex !== sourceStartIndex || owner.skill !== request?.skill || request.phase !== 'teacher_step'
        || request.skill?.locomotionOnly !== true || request.sourceFrames !== request.skill.sourceFrames
        || request.sourceFrames <= sourceStartIndex + REFERENCE_STUDENT_WALK_HORIZON
        || request.alignedReferenceFrames?.length < request.sourceFrames + 16
        || !request.alignedReferenceFrames.every(r => finite(r, 747))) return no('complete_unstarted_walk_required');
    if (geometry?.supported !== true) return no('full_reference_geometry_required');
    if (!finite(live?.rootPosWorld, 3) || !finite(live.rootQuatXyzwWorld, 4) || !finite(live.rootVelWorld, 3)
        || !finite(live.objPosWorld, 3) || Math.abs(Math.hypot(...live.rootQuatXyzwWorld) - 1) > 1e-5) return no('invalid_measured_state');
    const upright = 1 - 2 * (live.rootQuatXyzwWorld[0] ** 2 + live.rootQuatXyzwWorld[1] ** 2);
    const speed = Math.hypot(...live.rootVelWorld.slice(0, 2));
    if (live.rootPosWorld[2] < .7 || upright < .95) return no('balanced_entry_required');
    if (!sourceStartIndex && speed > .05) return no('quiet_standing_required');
    if (sourceStartIndex) {
      const [x,y,z,w] = live.rootQuatXyzwWorld, yaw = Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
      const forward = live.rootVelWorld[0]*Math.cos(yaw) + live.rootVelWorld[1]*Math.sin(yaw);
      if (speed < .1 || speed > 1.2 || forward < .05) return no('measured_forward_gait_required');
    }
    return {supported: true, reason: null, controller: new ReferenceStudentWalkController(options)};
  }
  constructor({owner, request, geometry, episode, physicalControl, sourceStartIndex = 0, live}) {
    this.#owner = owner; this.#skill = request.skill; this.#last = physicalControl;
    this.episode = episode; this.revision = owner.requestedIntent.revision; this.startControl = physicalControl;
    this.sourceStartIndex = sourceStartIndex;
    this.horizonControls = REFERENCE_STUDENT_WALK_HORIZON; this.sourceFrames = request.sourceFrames;
    this.plan = freeze({sourceName: request.skill.name, sourceFrames: request.sourceFrames,
      originalDestinationWorld: Array.from(owner.requestedIntent.goalWorld),
      sourceStartIndex, targetSourceIndex: sourceStartIndex + this.horizonControls,
      entryRootPositionWorld: copy(live.rootPosWorld), entryRootVelocityWorld: copy(live.rootVelWorld),
      alignedReferenceFrames: request.alignedReferenceFrames.map(r => Array.from(r)), geometry: copy(geometry)});
  }
  get active() { return this.#ended === null; }
  get controls() { return this.#last - this.startControl; }
  get ended() { return copy(this.#ended); }
  sameOwner({owner, episode, useRestricted}) { return useRestricted && owner === this.#owner && episode === this.episode && owner.skill === this.#skill; }
  isOwnedBy(context) {
    const intent = context.owner?.requestedIntent;
    return this.sameOwner(context) && intent?.type === 'floor' && intent.revision === this.revision
      && equal(intent.goalWorld, this.plan.originalDestinationWorld);
  }
  #finish(reason, requiresTeacherResume) {
    this.#ended ??= {reason, atControl: this.#last, controls: this.controls, requiresTeacherResume,
      resumeSourceIndex: this.sourceStartIndex + this.controls,
      remainingTeacherSourceControls: this.sourceFrames - this.sourceStartIndex - this.controls, arrived: false};
    this.#pending = null; return {active: false, ...this.ended};
  }
  observe(step, context) {
    if (!this.active) return {active: false, ...this.ended};
    if (!this.sameOwner(context)) return this.#finish('owner_changed', false);
    if (context.physicalControl !== this.#last) throw new Error('Walk must use its committed physical clock');
    if (!this.isOwnedBy(context)) return this.#finish('command_changed', true);
    if (context.owner.referenceIndex !== this.sourceStartIndex + this.controls) throw new Error('Complete walk source and physical clock diverged');
    if (this.controls === this.horizonControls) return this.#finish('student_walk_window_complete', true);
    if (step?.mode !== 'teacher' || step.phase !== 'teacher_step') return this.#finish('recorded_walk_ended', false);
    return {active: true, remainingControls: this.horizonControls - this.controls, arrived: false};
  }
  sample(live, context) {
    if (!this.active || !this.isOwnedBy(context) || context.physicalControl !== this.#last
        || this.controls >= this.horizonControls) throw new Error('Current walk ownership and remaining time required');
    const target = this.plan.alignedReferenceFrames[this.plan.targetSourceIndex];
    const encoded = encodeStageGoal({stage: 'approach', mode: 'LOCO', humanGoalWorld: target.slice(0, 3),
      humanGoalRotationWorld: target.slice(3, 7), objectGoalWorld: target.slice(71, 74),
      finalDestinationWorld: this.plan.originalDestinationWorld, remainingControls: this.horizonControls - this.controls},
    {rootPositionWorld: live.rootPosWorld, rootQuaternionWorld: live.rootQuatXyzwWorld, objectPositionWorld: live.objPosWorld});
    this.#pending = Array.from(encoded.command); return encoded;
  }
  buildObservation(encoded, body) { return packLegacyStageStudentInput(encoded, body, null); }
  commit({record, ...context}) {
    if (!this.active || !this.isOwnedBy(context) || context.physicalControl !== this.#last + 1 || !this.#pending
        || !finite(record?.rawAction, 29) || record.preview?.supported !== true || record.preview.completedSubsteps !== 17
        || record.preview.unwantedContactCount !== 0 || record.preview.allowedContactCount !== 0) throw new Error('One previewed physical walk control required');
    this.#records.push({...copy(record), command: this.#pending, sourceIndexBefore: this.sourceStartIndex + this.controls,
      preControl: this.#last, physicalControl: context.physicalControl});
    this.#last = context.physicalControl; this.#pending = null;
  }
  refuse({record, ...context}) {
    if (!this.active || !this.isOwnedBy(context) || context.physicalControl !== this.#last || !this.#pending
        || record?.preview?.supported !== false) throw new Error('Current unexecuted refused candidate required');
    this.refusedPreview = {...copy(record), command: this.#pending, executed: false};
    return this.#finish('student_walk_preview_refused', true);
  }
  cancel(reason, context) { return this.#finish(reason, this.sameOwner(context)); }
  review() { return {episode: this.episode, revision: this.revision, startControl: this.startControl,
    controls: this.controls, plan: copy(this.plan), records: copy(this.#records), refusedPreview: copy(this.refusedPreview), ended: this.ended}; }
}
