// Request-time sampling among explicitly qualified complete programs. No live
// actions, goals, source clocks or correction budgets are changed here.
import { CARRY_PLACEMENT_TOLERANCE_M } from './teacher_carry_sequence.js';

const position = p => p?.length === 3 && Array.from(p).every(Number.isFinite);
const same = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
const metric = n => Number.isFinite(n) && n >= 0;
const compare = (a, b) => a.endpointResidualM - b.endpointResidualM || a.approachCostM - b.approachCostM
  || a.correctionCost - b.correctionCost || (a.order < b.order ? -1 : a.order > b.order ? 1 : 0);

/** Preserve the actual geometry result passed back to the deterministic
 * planner while collecting its reviewed alternatives for later sampling. */
export function captureCarryPlanChecks(checkPlan) {
  if (typeof checkPlan !== 'function') throw new Error('An actual synchronous geometry checker is required');
  const records = [];
  return Object.freeze({ checkPlan: plan => {
    const geometry = checkPlan(plan);
    if (geometry?.then) throw new Error('Carry geometry must complete synchronously');
    records.push(Object.freeze({ plan, geometry: Object.freeze({ supported: geometry?.supported === true,
      reason: geometry?.reason ?? null, predictedEndpointResidualM: geometry?.predictedEndpointResidualM ?? null,
      approachCostM: geometry?.approachCostM ?? geometry?.predictedApproachCostM ?? null }) }));
    return geometry;
  }, getEvaluatedPlans: () => Object.freeze([...records]) });
}

/**
 * eligibility(plan, requestContext) must explicitly return
 * {eligible:true,evidence:[...]} for this complete source/command context.
 * Descriptive tags are copied separately and never confer eligibility.
 *
 * One uniform seeded draw selects an ordered source program; extra allocation
 * variants do not increase that program's probability. Sampling happens only
 * after pickup/residual/approach comparability filtering. Missing qualification
 * fails closed. A one-option result does not claim diversity or consume a draw.
 */
export function sampleCarryPlan({ evaluatedPlans, originalGoalWorld, requestContext = null, seed, eligibility,
  objectBodyName = evaluatedPlans?.[0]?.plan?.segments?.[0]?.skill?.objectBodyName,
  maxExtraEndpointResidualM = .01, maxExtraApproachM = .30 } = {}) {
  if (!Array.isArray(evaluatedPlans) || !position(originalGoalWorld) || typeof eligibility !== 'function'
      || !Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff
      || !metric(maxExtraEndpointResidualM) || maxExtraEndpointResidualM > CARRY_PLACEMENT_TOLERANCE_M
      || !metric(maxExtraApproachM)) throw new Error('Reviewed plans, explicit eligibility, bounded costs and a uint32 seed are required');
  const original = Object.freeze(Array.from(originalGoalWorld)), exclusions = [], admitted = [], checkedSkills = new Set();
  const exclude = (index, reason, evidence = []) => exclusions.push(Object.freeze({ index, reason, evidence: Object.freeze([...evidence]) }));
  function completeSkill(skill) {
    if (checkedSkills.has(skill)) return true;
    if (!skill || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 2
        || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames + 16
        || !skill.frames.every(row => row?.length === 747 && Array.from(row).every(Number.isFinite))) return false;
    checkedSkills.add(skill); return true;
  }
  for (const [index, record] of evaluatedPlans.entries()) {
    const p = record?.plan, geometry = record?.geometry;
    if (p?.supported !== true || !same(p.requestedGoalWorld, original) || !Array.isArray(p.segments)
        || p.segments.length < 1 || p.segments.length > 3 || geometry?.supported !== true
        || !Array.isArray(p.goals) || p.goals.length !== p.segments.length
        || p.segments.some((segment, i) => !same(segment.goalWorld, p.goals[i]))) {
      exclude(index, 'same_original_goal_and_supported_geometry_required'); continue;
    }
    let previous = p.initialObjectPositionWorld, body = p.segments[0].skill?.objectBodyName, correctionCost = 0;
    const valid = position(previous) && p.segments.every(segment => {
      const skill = segment.skill, budget = segment.carryOptions?.maxCorrection;
      const options = segment.carryOptions;
      if (!body || body !== objectBodyName || skill?.objectBodyName !== body || !completeSkill(skill) || !segment.candidateId
          || !same(segment.plannedStartObjectPositionWorld, previous) || !position(segment.goalWorld)
          || !metric(budget) || budget <= 0 || budget > .25
          || !Number.isInteger(options.warpStartFrame) || !Number.isInteger(options.warpEndFrame)
          || options.warpStartFrame < 0 || options.warpStartFrame >= options.warpEndFrame
          || options.warpEndFrame >= skill.sourceFrames) return false;
      const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
      const travel = Math.hypot(last[71] - first[71], last[72] - first[72]);
      const distance = Math.hypot(segment.goalWorld[0] - previous[0], segment.goalWorld[1] - previous[1]);
      const correction = distance - travel;
      if (travel <= budget || Math.abs(correction) > budget + 1e-10) return false;
      correctionCost += (correction / budget) ** 2; previous = segment.goalWorld; return true;
    });
    if (!valid) { exclude(index, 'complete_same_object_sources_and_unchanged_corrections_required'); continue; }
    const terminalResidual = Math.hypot(previous[0] - original[0], previous[1] - original[1]);
    if (previous[2] !== original[2] || terminalResidual > CARRY_PLACEMENT_TOLERANCE_M + 1e-10
        || (!same(previous, original) && (p.terminalGoalToleranceM !== CARRY_PLACEMENT_TOLERANCE_M
          || !same(p.plannedFinalGoalWorld, previous)))) {
      exclude(index, 'explicit_original_goal_region_required'); continue;
    }
    const residual = geometry.predictedEndpointResidualM, approach = geometry.approachCostM ?? geometry.predictedApproachCostM;
    if (!metric(residual) || !metric(approach)) { exclude(index, 'comparable_geometry_metrics_required'); continue; }
    const qualified = eligibility(p, requestContext);
    if (qualified?.then) throw new Error('Physical eligibility must be a synchronous decision');
    if (qualified?.eligible !== true || !Array.isArray(qualified.evidence) || qualified.evidence.length < 1
        || !qualified.evidence.every(value => typeof value === 'string' && value.length)) {
      exclude(index, qualified?.reason ?? 'physical_eligibility_not_established',
        Array.isArray(qualified?.evidence) ? qualified.evidence.filter(v => typeof v === 'string') : []); continue;
    }
    const tags = record.descriptiveTags ?? [];
    if (!Array.isArray(tags) || !tags.every(tag => typeof tag === 'string')) throw new Error('Descriptive source tags must be strings');
    const sourceIds = Object.freeze(p.segments.map(segment => segment.candidateId));
    admitted.push({ index, plan: p, sourceIds, program: JSON.stringify(sourceIds), pickupCount: p.segments.length,
      endpointResidualM: terminalResidual + residual, approachCostM: approach, correctionCost,
      order: JSON.stringify(p.goals), evidence: Object.freeze([...qualified.evidence]), descriptiveTags: Object.freeze([...tags]) });
  }
  const minimumPickups = Math.min(...admitted.map(row => row.pickupCount));
  const sameCount = admitted.filter(row => row.pickupCount === minimumPickups);
  const minimumResidualM = Math.min(...sameCount.map(row => row.endpointResidualM));
  const nearResidual = sameCount.filter(row => row.endpointResidualM <= Math.min(CARRY_PLACEMENT_TOLERANCE_M,
    minimumResidualM + maxExtraEndpointResidualM) + 1e-10);
  const minimumApproachM = Math.min(...nearResidual.map(row => row.approachCostM));
  const comparable = nearResidual.filter(row => row.approachCostM <= minimumApproachM + maxExtraApproachM + 1e-10);
  const comparableIndices = new Set(comparable.map(row => row.index));
  for (const row of admitted) if (!comparableIndices.has(row.index)) exclude(row.index, 'outside_comparable_cost_band', row.evidence);
  const groups = new Map();
  for (const row of comparable) if (!groups.has(row.program) || compare(row, groups.get(row.program)) < 0) groups.set(row.program, row);
  const choices = [...groups.values()].sort((a, b) => a.program < b.program ? -1 : a.program > b.program ? 1 : 0);
  const common = { seed, objectBodyName, eligibleSourcePrograms: choices.length, originalGoalWorld: original,
    exclusions: Object.freeze(exclusions), physicalEligibilitySuppliedExternally: true,
    physicalPlacementSuccessPredicted: false,
    comparableCostLimits: Object.freeze({ maxExtraEndpointResidualM, maxExtraApproachM }) };
  if (!choices.length) return Object.freeze({ ...common, supported: false, reason: 'no_eligible_source_program',
    plan: null, sampled: false, draw: null, nextSeed: seed, sourcePrograms: Object.freeze([]) });
  let nextSeed = seed, draw = null, chosen = 0;
  if (choices.length > 1) {
    // Mulberry32: explicit numeric state, one replayable draw per request.
    nextSeed = (seed + 0x6D2B79F5) >>> 0;
    let value = nextSeed; value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    draw = ((value ^ value >>> 14) >>> 0) / 4294967296;
    chosen = Math.floor(draw * choices.length);
  }
  return Object.freeze({ ...common, supported: true, reason: choices.length > 1 ? 'sampled_eligible_program' : 'only_one_eligible_program',
    plan: choices[chosen].plan, sampled: choices.length > 1, draw, nextSeed,
    selectedEvaluationIndex: choices[chosen].index,
    selectedSourceIds: choices[chosen].sourceIds, selectedProbability: 1 / choices.length,
    sourcePrograms: Object.freeze(choices.map(row => Object.freeze({ sourceIds: row.sourceIds, probability: 1 / choices.length,
      endpointResidualM: row.endpointResidualM, approachCostM: row.approachCostM, evidence: row.evidence,
      descriptiveTags: row.descriptiveTags }))) });
}
