// Execute an explicitly planned sequence of different complete carry skills.
// The expanded demo retains the original destination through every segment.
import { CarryGoalSequenceController, planCarrySegments, CARRY_PLACEMENT_TOLERANCE_M } from './teacher_carry_sequence.js';
import { CarryGoalController } from './teacher_carry_controller.js';

const finite = p => p?.length === 3 && Array.from(p).every(Number.isFinite);
const same = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const frozenPosition = p => Object.freeze(Array.from(p));

function retainPlan(plan) {
  if (plan?.supported !== true || !finite(plan.requestedGoalWorld)
      || !finite(plan.initialObjectPositionWorld) || !Array.isArray(plan.segments)
      || plan.segments.length < 1 || plan.segments.length > 3) {
    throw new Error('A supported plan with one to three complete skill segments is required');
  }
  const objectBodyName = plan.segments[0].skill?.objectBodyName;
  let predecessor = plan.initialObjectPositionWorld;
  const segments = plan.segments.map(segment => {
    if (!segment.candidateId || segment.skill?.objectBodyName !== objectBodyName
        || !finite(segment.goalWorld) || !finite(segment.plannedStartObjectPositionWorld)
        || !same(segment.plannedStartObjectPositionWorld, predecessor)) {
      throw new Error('Every segment must retain the same object and exact preceding destination');
    }
    const options = { ...segment.carryOptions };
    // Validate each source and its own correction interval before execution.
    const ownPlan = planCarrySegments(predecessor, segment.goalWorld, segment.skill,
      { maxSegments: 1, maxCorrection: options.maxCorrection ?? .25 });
    // Match the planner's numerical interval tolerance. Absolute world-space
    // subtraction can move an exact declared boundary by a few ulps; this
    // tolerance is 0.1 nanometres and does not alter the physical warp budget.
    const intervalResidual = Math.max(0, ownPlan.minSegmentDistanceM - ownPlan.distanceM,
      ownPlan.distanceM - ownPlan.maxSegmentDistanceM);
    if (!ownPlan.supported && intervalResidual > 1e-10)
      throw new Error('A mixed segment exceeds its own complete reference travel range');
    new CarryGoalController(segment.skill, segment.goalWorld, options);
    const retained = Object.freeze({ ...segment, carryOptions: Object.freeze(options),
      goalWorld: frozenPosition(segment.goalWorld), plannedStartObjectPositionWorld: frozenPosition(predecessor),
      supportedDistanceIntervalM: Object.freeze([ownPlan.minSegmentDistanceM, ownPlan.maxSegmentDistanceM]) });
    predecessor = segment.goalWorld;
    return retained;
  });
  const exactGoal = same(predecessor, plan.requestedGoalWorld);
  if (plan.terminalGoalToleranceM !== undefined && plan.terminalGoalToleranceM !== CARRY_PLACEMENT_TOLERANCE_M)
    throw new Error('The declared original-goal region must be exactly10 cm');
  if (plan.plannedFinalGoalWorld !== undefined && !same(plan.plannedFinalGoalWorld, predecessor))
    throw new Error('The declared planned endpoint must equal the final segment destination');
  if (!exactGoal && (plan.terminalGoalToleranceM !== CARRY_PLACEMENT_TOLERANCE_M
      || !same(plan.plannedFinalGoalWorld, predecessor) || predecessor[2] !== plan.requestedGoalWorld[2]
      || Math.hypot(predecessor[0] - plan.requestedGoalWorld[0], predecessor[1] - plan.requestedGoalWorld[1])
        > CARRY_PLACEMENT_TOLERANCE_M + 1e-10))
    throw new Error('A nonexact endpoint requires the explicit10 cm original-goal region');
  return Object.freeze({ ...plan, requestedGoalWorld: frozenPosition(plan.requestedGoalWorld),
    ...(plan.plannedFinalGoalWorld ? { plannedFinalGoalWorld: frozenPosition(plan.plannedFinalGoalWorld) } : {}),
    initialObjectPositionWorld: frozenPosition(plan.initialObjectPositionWorld), segments: Object.freeze(segments),
    goals: Object.freeze(segments.map(segment => segment.goalWorld)) });
}

/** Reuse the existing measured placement, complete exit, cancellation and
 * student approach lifecycle. Only the reference selected for each untouched
 * segment changes. Neither this controller nor its planner changes physics.
 */
export class MixedCarryGoalSequenceController extends CarryGoalSequenceController {
  #providedPlan;
  #carryDefaults;
  constructor(plan, { settlingSteps = 180, maxLiveGoalResidualM = .05,
    checkEntryReference = null, requireSegmentExit = true, quietSettling = null, ...carryDefaults } = {}) {
    const retained = retainPlan(plan), first = retained.segments[0];
    if ('alternativeSkills' in carryDefaults) throw new Error('Mixed plans explicitly select each segment reference');
    // Sequence-level settling options stay with the parent; per-segment child
    // controllers only receive their own complete carry options.
    super(first.skill, retained.requestedGoalWorld, { ...carryDefaults, ...first.carryOptions,
      settlingSteps, maxLiveGoalResidualM, checkEntryReference, requireSegmentExit, quietSettling, maxSegments: 3 });
    this.#providedPlan = retained; this.#carryDefaults = { ...carryDefaults };
  }

  get segmentCandidateId() { return this.plan?.segments[this.segmentIndex]?.candidateId ?? null; }

  start(proprio) {
    if (['approach', 'teacher', 'settling', 'settling_quiet', 'awaiting_exit'].includes(this.phase)) throw new Error('A carry sequence is already active');
    if (proprio.objectBodyName !== this.#providedPlan.segments[0].skill.objectBodyName)
      throw new Error('The planned scene object is required');
    this.reset(); this.plan = this.#providedPlan;
    this._startSegment(proprio);
  }

  _startSegment(proprio) {
    const segment = this.plan.segments[this.segmentIndex];
    if (!segment) throw new Error('An explicitly planned skill is required for this segment');
    const options = { ...this.#carryDefaults, ...segment.carryOptions };
    const range = planCarrySegments([0, 0, 0], [1, 0, 0], segment.skill,
      { maxSegments: 1, maxCorrection: options.maxCorrection ?? .25 });
    this.rawSkill = segment.skill;
    this.references = [{ skill: segment.skill, options, range }];
    this.activeReferenceIndex = 0;
    // The inherited live-range check and reference alignment use the actual
    // measured box at this point; planned start positions are never applied.
    super._startSegment(proprio);
  }

  step(proprio) {
    const resultCount = this.segmentResults.length;
    const result = super.step(proprio);
    for (const record of this.segmentResults.slice(resultCount)) {
      const segment = this.plan.segments[record.segmentIndex];
      record.candidateId = segment.candidateId;
      record.sourceSkillName = segment.skill.name ?? null;
      record.rawSourceFrames = segment.skill.sourceFrames;
    }
    return { ...result, segmentCandidateId: this.segmentCandidateId,
      mixedCarry: true, plannedCandidateIds: this.plan?.segments.map(segment => segment.candidateId) ?? [] };
  }
}
