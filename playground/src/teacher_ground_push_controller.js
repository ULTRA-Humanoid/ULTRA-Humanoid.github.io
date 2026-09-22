// Complete ground manipulation references. This module is not enabled by main.
// The actor tracks the complete source; a hand push has no carry lift minimum.
import { CarryGoalController } from './teacher_carry_controller.js';

export class TeacherGroundPushController extends CarryGoalController {
  constructor(skill, goalWorld, { maxCorrection = .05, initialStanceFrames = 90,
    warpStartFrame = skill?.pushInterval?.[0], warpEndFrame = skill?.pushInterval?.[1], ...options } = {}) {
    if (skill?.manipulationKind !== 'ground_push' || !['bilateral', 'left_hand', 'right_hand'].includes(skill.pushStyle)) {
      throw new Error('A separately declared complete hand-push source is required');
    }
    if (!Number.isFinite(maxCorrection) || maxCorrection <= 0 || maxCorrection > .05) {
      throw new Error('This initial ground-push experiment permits at most5 cm source correction');
    }
    super(skill, goalWorld, { ...options, maxCorrection, initialStanceFrames, warpStartFrame, warpEndFrame });
    this.outcomeRequirements = { minLiftM: 0, maxFinalObjectHeightM: .25, minRootHeightM: .45, minUpright: .5 };
    this.manipulationKind = 'ground_push';
    this.pushStyle = skill.pushStyle;
  }

  startFromMeasuredEndpoint(proprio) {
    if (this.phase === 'approach' || this.phase === 'teacher') throw new Error('A ground-push skill is already active');
    if (proprio?.objectBodyName !== this.skill.objectBodyName) throw new Error(`This skill requires ${this.skill.objectBodyName}`);
    const finite = (value, length) => value?.length === length && Array.from(value).every(Number.isFinite);
    if (!finite(proprio.rootPosWorld, 3) || !finite(proprio.rootQuatXyzwWorld, 4)
        || !finite(proprio.rootVelWorld, 3) || !finite(proprio.objPosWorld, 3)) {
      throw new Error('Finite measured root and selected-object state is required');
    }
    const plan = this._referencePlan(proprio);
    if (!Array.isArray(plan?.frames) || plan.frames.length < this.skill.sourceFrames + 16) {
      throw new Error('Complete measured-endpoint push reference is required');
    }
    this.reset(); this.referencePlan = plan; this.worldFrames = plan.frames;
    this.phase = 'teacher'; this.referenceIndex = 0; this._entered = true;
  }
}
