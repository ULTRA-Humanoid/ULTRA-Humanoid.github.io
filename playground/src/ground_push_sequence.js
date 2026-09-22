// A single complete push, followed by the established settling/exit lifecycle.
// Final task precision inherits the existing10 cm measured placement criterion.
// This module does not change contact policy and is not enabled by main.
import { CarryGoalSequenceController, planCarrySegments } from './teacher_carry_sequence.js';
import { TeacherGroundPushController } from './teacher_ground_push_controller.js';

export class GroundPushGoalSequenceController extends CarryGoalSequenceController {
  constructor(skill, goalWorld, { maxCorrection = .05, initialStanceFrames = 90,
    warpStartFrame = skill?.pushInterval?.[0], warpEndFrame = skill?.pushInterval?.[1],
    alternativeSkills = [], maxSegments = 1, ...options } = {}) {
    if (maxSegments !== 1 || alternativeSkills.length) throw new Error('The initial ground-push path executes one declared complete source');
    const pushOptions = { ...options, maxCorrection, initialStanceFrames, warpStartFrame, warpEndFrame };
    new TeacherGroundPushController(skill, goalWorld, pushOptions);
    super(skill, goalWorld, { ...pushOptions, alternativeSkills: [], maxSegments: 1 });
    this.manipulationKind = 'ground_push';
  }

  startFromMeasuredEndpoint(proprio) {
    if (['approach', 'teacher', 'settling', 'settling_quiet', 'awaiting_exit'].includes(this.phase)) {
      throw new Error('A ground-push sequence is already active');
    }
    if (proprio?.objectBodyName !== this.rawSkill.objectBodyName) throw new Error(`This skill requires ${this.rawSkill.objectBodyName}`);
    const finite = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
    if (!finite(proprio.rootPosWorld, 3) || !finite(proprio.rootQuatXyzwWorld, 4)
        || !finite(proprio.rootVelWorld, 3) || !finite(proprio.objPosWorld, 3)) {
      throw new Error('Finite measured root and selected-object state is required');
    }
    const plan = planCarrySegments(proprio.objPosWorld, this.requestedGoalWorld, this.rawSkill, this.planOptions);
    this.reset(); this.plan = plan;
    if (!plan.supported) { this._complete(plan.reason); return; }
    if (!this._liveDistanceSupported(proprio)) { this._complete('unsupported_live_distance'); return; }
    const entry = this.references[this.activeReferenceIndex];
    this.triedReferences = new Set([this.activeReferenceIndex]);
    this.child = new TeacherGroundPushController(entry.skill, plan.goals[this.segmentIndex], this._segmentOptions(entry));
    this.child.startFromMeasuredEndpoint(proprio); this.phase = 'teacher'; this.settlingCount = 0;
  }

  _startSegment(proprio) {
    if (!this._liveDistanceSupported(proprio)) { this._complete('unsupported_live_distance'); return; }
    const target = this.plan.goals[this.segmentIndex], entry = this.references[this.activeReferenceIndex];
    this.triedReferences = new Set([this.activeReferenceIndex]);
    this.child = new TeacherGroundPushController(entry.skill, target, this._segmentOptions(entry));
    this.child.start(proprio); this.phase = 'approach'; this.settlingCount = 0;
  }

  _tryAlternative() { return false; }
}
