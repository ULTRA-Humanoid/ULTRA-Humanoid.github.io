// Bounded planning in a terminal goal region. This never declares physical
// success: the sequence must execute its full sources and measure placement.
import { planMixedCarry, mixedCarryDistanceCoverage, compareMixedCarryScores, summarizeAttemptFailures } from './mixed_carry_planner.js';
import { CARRY_PLACEMENT_TOLERANCE_M } from './teacher_carry_sequence.js';

const EPS = 1e-10, METRIC_EPS = 1e-6;
const position = p => p?.length === 3 && Array.from(p).every(Number.isFinite);
const freezePosition = p => Object.freeze(Array.from(p));
const xyDistance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function metric(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return Math.abs(a - b) <= METRIC_EPS ? 0 : a - b;
}
function retainOriginal(plan, original, originalDistance) {
  const endpoint = plan.goals.at(-1) ?? original;
  const residual = xyDistance(endpoint, original);
  if (residual > CARRY_PLACEMENT_TOLERANCE_M + EPS || endpoint[2] !== original[2])
    throw new Error('Every planned endpoint must remain inside the original planar goal region');
  return Object.freeze({ ...plan, requestedGoalWorld: original,
    plannedFinalGoalWorld: freezePosition(endpoint), terminalGoalToleranceM: CARRY_PLACEMENT_TOLERANCE_M,
    plannedDistanceM: plan.distanceM, distanceM: originalDistance,
    nominalFinalGoalResidualM: residual });
}
function score(plan) {
  // The geometry callback's endpoint residual describes reference-to-segment
  // error. Add the declared terminal offset as a conservative original-goal
  // bound; neither term predicts physical policy tracking error.
  const ownResidual = plan.score.predictedEndpointResidualM;
  return Object.freeze({ ...plan.score, nominalFinalGoalResidualM: plan.nominalFinalGoalResidualM,
    predictedOriginalGoalResidualM: plan.nominalFinalGoalResidualM + (ownResidual ?? 0),
    predictedOriginalGoalResidualUsesGeometry: ownResidual !== null });
}
// Same budget-guard and ranking precedence as the mixed planner; the original
// goal residual replaces the per-segment endpoint residual.
function makeCompare(rankingOptions) {
  return (a, b) => compareMixedCarryScores(
    { ...a.score, predictedEndpointResidualM: null, approachCostM: null, minimumClearanceM: null, squaredFractionalCorrection: 0 },
    { ...b.score, predictedEndpointResidualM: null, approachCostM: null, minimumClearanceM: null, squaredFractionalCorrection: 0 }, rankingOptions)
    || metric(a.score.predictedOriginalGoalResidualM, b.score.predictedOriginalGoalResidualM)
    || metric(a.score.approachCostM, b.score.approachCostM)
    || -metric(a.score.minimumClearanceM, b.score.minimumClearanceM)
    || metric(a.score.squaredFractionalCorrection, b.score.squaredFractionalCorrection)
    || (a.order < b.order ? -1 : a.order > b.order ? 1 : 0);
}

// Nominal sums and their feasible interval boundaries provide a small declared
// endpoint set. Endpoints remain on the same ray; this is not a disk search.
function endpointsForCount(initial, goal, candidates, count) {
  const distance = xyDistance(initial, goal), low = Math.max(0, distance - CARRY_PLACEMENT_TOLERANCE_M);
  const high = distance + CARRY_PLACEMENT_TOLERANCE_M;
  const lengths = [distance], uses = new Map();
  function add(value) {
    if (value <= 0 || value < low - EPS || value > high + EPS) return;
    value = Math.max(low, Math.min(high, value));
    if (!lengths.some(old => Math.abs(old - value) < EPS)) lengths.push(value);
  }
  function visit(depth, nominal, correction) {
    if (depth === count) {
      const start = Math.max(low, nominal - correction), end = Math.min(high, nominal + correction);
      if (start > end + EPS) return;
      add(start); add(end); add(Math.max(start, Math.min(end, nominal)));
      add(Math.max(start, Math.min(end, distance)));
      return;
    }
    for (const candidate of candidates) {
      const used = uses.get(candidate.id) ?? 0;
      if (used >= (candidate.maxUses ?? count)) continue;
      const first = candidate.skill.frames[0], last = candidate.skill.frames[candidate.skill.sourceFrames - 1];
      const travel = Math.hypot(last[71] - first[71], last[72] - first[72]);
      uses.set(candidate.id, used + 1);
      visit(depth + 1, nominal + travel, correction + (candidate.maxCorrection ?? candidate.carryOptions.maxCorrection));
      uses.set(candidate.id, used);
    }
  }
  visit(0, 0, 0);
  return lengths.sort((a, b) => Math.abs(a - distance) - Math.abs(b - distance) || a - b)
    .map(length => length === distance ? Array.from(goal) : [initial[0] + (goal[0] - initial[0]) * length / distance,
      initial[1] + (goal[1] - initial[1]) * length / distance, goal[2]]);
}

/** Prefer an exact one-pickup plan. Otherwise compare declared endpoints
 * within10 cm, first minimizing pickups, then original-goal geometric residual
 * before approach cost. Physical correction budgets and maxUses do not change.
 * checkPlan always sees both the immutable original click and the planned end.
 * The total plan-check budget is shared across every endpoint and pickup count.
 * `ranking` and `budget` pass through to the mixed planner and its precedence.
 */
export function planCarryGoalRegion(objectPosition, originalGoalWorld, candidates,
  { maxSegments = 3, checkPlan = null, maxPlanChecks = 256, exactFallbackPlan = null, ranking = 'default', budget = null } = {}) {
  if (!position(objectPosition) || !position(originalGoalWorld)
      || (checkPlan !== null && typeof checkPlan !== 'function')
      || !Number.isInteger(maxPlanChecks) || maxPlanChecks < 1 || maxPlanChecks > 16384)
    throw new Error('Finite original goals, synchronous geometry and a bounded check count are required');
  const ranges = mixedCarryDistanceCoverage(candidates, { maxSegments }); // Also validates every complete source.
  const rankingOptions = { ranking, guard: budget?.guard === true }, compare = makeCompare(rankingOptions);
  const initial = freezePosition(objectPosition), original = freezePosition(originalGoalWorld);
  const distance = xyDistance(initial, original), attempts = [], endpoints = [];
  let checked = 0, exhausted = false, reusedChecks = 0;
  const common = () => {
    const failures = summarizeAttemptFailures(attempts);
    return { ...ranges, checkedPlans: checked, attempts: Object.freeze([...attempts]),
      reviewedEndpointCandidates: Object.freeze([...endpoints]), planningBudgetExhausted: exhausted,
      reusedExactPlan: exactFallbackPlan !== null, newCheckedPlans: checked - reusedChecks,
      ranking, dominantReason: failures.dominantReason, dominantObstacle: failures.dominantObstacle, failureSummary: failures,
      physicalPlacementSuccessPredicted: false, placementMeasurementRequired: true,
      endpointSearch: 'Declared collinear endpoint candidates; no exhaustive two-dimensional region claim',
      endpointSearchExhaustive: false, terminalGoalToleranceM: CARRY_PLACEMENT_TOLERANCE_M };
  };
  const refused = reason => Object.freeze({ supported: false, reason, requestedGoalWorld: original,
    initialObjectPositionWorld: initial, distanceM: distance, segments: Object.freeze([]), goals: Object.freeze([]), score: null,
    budget: null, budgetRisk: false, ...common() });
  // Geometry alone cannot establish that a nearby box is grounded and still.
  if (distance <= CARRY_PLACEMENT_TOLERANCE_M + 1e-12) return refused('placement_check_required');
  function evaluate(endpoint, count) {
    if (checked >= maxPlanChecks) { exhausted = true; return null; }
    const plan = planMixedCarry(initial, endpoint, candidates, { maxSegments: count, maxPlanChecks: maxPlanChecks - checked, ranking, budget,
      checkPlan: candidate => checkPlan ? checkPlan(retainOriginal(candidate, original, distance)) : { supported: true } });
    checked += plan.checkedPlans; exhausted ||= plan.planningBudgetExhausted;
    endpoints.push(Object.freeze({ goalWorld: freezePosition(endpoint), maxPickups: count,
      supported: plan.supported, reason: plan.reason, checkedPlans: plan.checkedPlans }));
    for (const attempt of plan.attempts) attempts.push(Object.freeze({ ...attempt, requestedGoalWorld: original,
      plannedFinalGoalWorld: freezePosition(endpoint), nominalFinalGoalResidualM: xyDistance(endpoint, original) }));
    if (!plan.supported) return null;
    const retained = retainOriginal(plan, original, distance), ranked = score(retained);
    return { plan: retained, score: ranked, order: retained.segments.map(s => s.candidateId).join('\u0000')
      + '\u0001' + endpoint.map(v => v.toPrecision(17)).join(',') };
  }
  // Establish the current exact plan before spending checks on alternatives.
  // A crowded region search must never discard an already supported command.
  let exact;
  if (exactFallbackPlan !== null) {
    const p = exactFallbackPlan, same = (a, b) => a?.length === b?.length && Array.from(a ?? []).every((v, i) => v === b[i]);
    if (p.supported !== true || !same(p.requestedGoalWorld, original) || !same(p.initialObjectPositionWorld, initial)
        || !Array.isArray(p.segments) || p.segments.length < 1 || p.segments.length > maxSegments
        || !same(p.goals?.at(-1), original) || p.score?.pickupCount !== p.segments.length
        || !Number.isSafeInteger(p.checkedPlans) || p.checkedPlans < 0
        || p.segments.some(segment => !candidates.some(c => c.id === segment.candidateId && c.skill === segment.skill
          && (c.maxCorrection ?? c.carryOptions?.maxCorrection) === segment.carryOptions?.maxCorrection)))
      throw new Error('A reused exact plan must retain this unchanged source library, live start and original goal');
    checked = reusedChecks = p.checkedPlans; exhausted = Boolean(p.planningBudgetExhausted) || checked >= maxPlanChecks;
    attempts.push(...p.attempts); endpoints.push(Object.freeze({ goalWorld: original, maxPickups: maxSegments,
      supported: true, reason: null, checkedPlans: p.checkedPlans, reused: true }));
    const retained = retainOriginal(p, original, distance);
    exact = { plan: retained, score: score(retained), order: retained.segments.map(s => s.candidateId).join('\u0000') };
  } else exact = evaluate(original, maxSegments);
  if (exact?.score.pickupCount === 1) return Object.freeze({ ...exact.plan, score: exact.score, ...common(), selection: 'exact_one_pickup',
    geometricRankingExhaustiveForSelectedPickupCount: !exhausted });
  let best = exact;
  // An exact geometric endpoint already leaves maximal physical-error margin.
  // Equal-count offsets cannot improve it, so do not repeat those expensive
  // complete-source geometry checks merely to return the same accepted plan.
  const maximumAlternativePickups = exact && exact.score.predictedOriginalGoalResidualM <= METRIC_EPS
    ? exact.score.pickupCount - 1 : exact?.score.pickupCount ?? maxSegments;
  for (let count = 1; count <= maximumAlternativePickups && !exhausted; count++) {
    for (const endpoint of endpointsForCount(initial, original, candidates, count)) {
      if (xyDistance(endpoint, original) < EPS) continue; // Exact full-count search already checked.
      const candidate = evaluate(endpoint, count);
      if (candidate && (!best || compare(candidate, best) < 0)) best = candidate;
      if (exhausted) break;
    }
    if (best && best.score.pickupCount <= count) break; // Fewer pickups take precedence over every longer plan.
  }
  if (!best) return refused(exhausted ? 'planning_budget_exhausted' : attempts.length ? 'no_clear_plan' : 'unsupported_distance');
  return Object.freeze({ ...best.plan, score: best.score, ...common(), selection: best === exact ? 'exact_mixed_fallback' : 'bounded_goal_region',
    geometricRankingExhaustiveForSelectedPickupCount: false });
}

export const ALREADY_PLACED_LIMITS = Object.freeze({ objectLinearSpeedMps: .05, objectAngularSpeedRadps: .1,
  rootLinearSpeedMps: .1, minimumRootHeightM: .65, minimumUpright: .8 });

/** A current measured idle-request check, not a moving carry completion rule.
 * Grounded must come from actual object-floor contact, never height alone.
 * This is a snapshot check; it does not claim a sustained quiet interval. */
export function checkAlreadyPlacedGoal({ objectPositionWorld, requestedGoalWorld, idle, objectGrounded,
  objectLinearSpeedMps, objectAngularSpeedRadps, rootLinearSpeedMps, rootHeightM, upright } = {}) {
  if (!position(objectPositionWorld) || !position(requestedGoalWorld)) throw new Error('Finite object and original goal positions are required');
  const remainingDistanceM = xyDistance(objectPositionWorld, requestedGoalWorld), limits = ALREADY_PLACED_LIMITS;
  const finiteMeasurements = [objectLinearSpeedMps, objectAngularSpeedRadps, rootLinearSpeedMps, rootHeightM, upright].every(Number.isFinite)
    && [objectLinearSpeedMps, objectAngularSpeedRadps, rootLinearSpeedMps].every(v => v >= 0) && upright >= -1 && upright <= 1;
  const reason = idle !== true ? 'task_active' : !finiteMeasurements ? 'measured_motion_required'
    : objectGrounded !== true ? 'object_not_grounded'
    : objectLinearSpeedMps > limits.objectLinearSpeedMps || objectAngularSpeedRadps > limits.objectAngularSpeedRadps
      || rootLinearSpeedMps > limits.rootLinearSpeedMps ? 'motion_not_quiet'
    : rootHeightM < limits.minimumRootHeightM || upright < limits.minimumUpright ? 'root_not_balanced'
    : remainingDistanceM > CARRY_PLACEMENT_TOLERANCE_M + 1e-12 ? 'outside_placement_tolerance' : 'already_placed';
  return Object.freeze({ canSkipCarry: reason === 'already_placed', reason, remainingDistanceM,
    toleranceM: CARRY_PLACEMENT_TOLERANCE_M, requestedGoalWorld: freezePosition(requestedGoalWorld),
    objectPositionWorld: freezePosition(objectPositionWorld), measurement: 'Current idle contact and velocity snapshot' });
}
