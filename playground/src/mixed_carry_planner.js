// Pure geometric planning of complete reference sequences. Nothing here changes
// live state, crops a source, or estimates physical placement success.
import { assessCarryPlanBudget } from './carry_time_budget.js';

const EPS = 1e-10;
const position = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
const freezePosition = value => Object.freeze(Array.from(value));
export const PLANNER_RANKINGS = Object.freeze(['default', 'reliability']);

function readCandidates(candidates, maxSegments) {
  if (!Array.isArray(candidates) || !candidates.length || candidates.length > 16
      || new Set(candidates.map(c => c.id)).size !== candidates.length) {
    throw new Error('One to sixteen distinct complete carry candidates are required');
  }
  const objectName = candidates[0].skill?.objectBodyName;
  return candidates.map(candidate => {
    const { skill } = candidate;
    const correction = candidate.maxCorrection ?? candidate.carryOptions?.maxCorrection;
    const maxUses = candidate.maxUses ?? maxSegments;
    const reliabilityPenalty = candidate.reliabilityPenalty ?? 0;
    const singleSegmentOnly = candidate.singleSegmentOnly === true;
    const finalSegmentOnly = candidate.finalSegmentOnly === true;
    if (candidate.finalSegmentOnly !== undefined && typeof candidate.finalSegmentOnly !== 'boolean') throw new Error('finalSegmentOnly must be boolean');
    if (candidate.lastLegAfter !== undefined && (!Array.isArray(candidate.lastLegAfter) || !candidate.lastLegAfter.length
        || !candidate.lastLegAfter.every(id => typeof id === 'string' && id.length))) throw new Error('lastLegAfter must list candidate ids');
    const lastLegAfter = candidate.lastLegAfter === undefined ? null : Object.freeze([...candidate.lastLegAfter]);
    if (candidate.singleSegmentOnly !== undefined && typeof candidate.singleSegmentOnly !== 'boolean') throw new Error('singleSegmentOnly must be boolean');
    if (typeof candidate.id !== 'string' || !candidate.id.length || !objectName
        || skill?.objectBodyName !== objectName || !Number.isInteger(skill.sourceFrames)
        || skill.sourceFrames < 2 || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames
        || !Number.isFinite(correction) || correction < 0 || correction > .25
        || !Number.isInteger(maxUses) || maxUses < 0 || maxUses > 3
        || !Number.isFinite(reliabilityPenalty) || reliabilityPenalty < 0 || reliabilityPenalty > 1
        || (candidate.carryOptions?.maxCorrection !== undefined && candidate.carryOptions.maxCorrection !== correction)) {
      throw new Error('Each same-object full source needs an explicit correction within0–.25m and bounded use count');
    }
    const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
    if (first?.length !== 747 || last?.length !== 747
        || ![first[71], first[72], last[71], last[72]].every(Number.isFinite)) {
      throw new Error('Full747 source endpoints are required');
    }
    const nominal = Math.hypot(last[71] - first[71], last[72] - first[72]);
    if (!(nominal > correction)) throw new Error('Each correction must be smaller than its positive source travel');
    const carryOptions = { ...candidate.carryOptions, maxCorrection: correction };
    if (carryOptions.warpStartFrame === undefined) carryOptions.warpStartFrame = skill.carryInterval?.[0];
    if (carryOptions.warpEndFrame === undefined) carryOptions.warpEndFrame = skill.carryInterval?.[1];
    if (!Number.isInteger(carryOptions.warpStartFrame) || !Number.isInteger(carryOptions.warpEndFrame)
        || carryOptions.warpStartFrame < 0 || carryOptions.warpEndFrame <= carryOptions.warpStartFrame
        || carryOptions.warpEndFrame >= skill.sourceFrames) throw new Error('An explicit valid full-source carry warp interval is required');
    return Object.freeze({ id: candidate.id, skill, correction, maxUses, nominal, reliabilityPenalty, singleSegmentOnly, lastLegAfter, finalSegmentOnly,
      min: nominal - correction, max: nominal + correction, carryOptions: Object.freeze(carryOptions) });
  }).sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
}

// A candidate with maxUses 0 stays in the library record but never enters a
// sequence (excludeLong ranking); coverage reflects the usable candidates only.
function sequences(candidates, maxSegments) {
  const groups = Array.from({ length: maxSegments }, () => []), uses = new Map();
  function append(prefix) {
    if (prefix.length) groups[prefix.length - 1].push({ entries: prefix,
      min: prefix.reduce((sum, c) => sum + c.min, 0), max: prefix.reduce((sum, c) => sum + c.max, 0) });
    if (prefix.length === maxSegments) return;
    // A singleSegmentOnly candidate may only form a one-segment plan: never after a
    // preceding carry, never followed by another carry (opt-in long clips).
    // A lastLegAfter candidate may be the whole plan, or the LAST leg directly after one listed
    // predecessor (two-segment plan); never after any other clip, never followed by another carry.
    // finalSegmentOnly: the candidate may sit at ANY position but is never followed by another carry (single or final leg).
    if (prefix.some(c => c.singleSegmentOnly || c.lastLegAfter || c.finalSegmentOnly)) return;
    for (const candidate of candidates) {
      const count = uses.get(candidate.id) ?? 0;
      if (count >= candidate.maxUses) continue;
      if (candidate.singleSegmentOnly && prefix.length) continue;
      if (candidate.lastLegAfter && prefix.length && !(prefix.length === 1 && candidate.lastLegAfter.includes(prefix[0].id))) continue;
      uses.set(candidate.id, count + 1); append([...prefix, candidate]); uses.set(candidate.id, count);
    }
  }
  append([]); return groups;
}

function mergedIntervals(intervals) {
  const ordered = intervals.map(i => [...i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]), merged = [];
  for (const interval of ordered) {
    const previous = merged.at(-1);
    if (previous && interval[0] <= previous[1] + EPS) previous[1] = Math.max(previous[1], interval[1]);
    else merged.push(interval);
  }
  return Object.freeze(merged.map(Object.freeze));
}

function coverage(groups) {
  const intervals = mergedIntervals(groups.flatMap(group => group.map(s => [s.min, s.max])));
  return { supportedDistanceIntervalsM: intervals,
    supportedDistanceIntervalsByPickupCount: Object.freeze(groups.map((group, index) => Object.freeze({ pickupCount: index + 1,
      intervalsM: mergedIntervals(group.map(s => [s.min, s.max])) }))),
    unsupportedDistanceGapsM: Object.freeze(intervals.slice(1).map((interval, i) => Object.freeze([intervals[i][1], interval[0]]))) };
}

function validateBound(maxSegments) {
  if (!Number.isInteger(maxSegments) || maxSegments < 1 || maxSegments > 3)
    throw new Error('One to three complete carry segments are supported');
}

function validateOptions({ ranking, budget }) {
  if (!PLANNER_RANKINGS.includes(ranking)) throw new Error('Carry planner ranking must be default or reliability');
  if (budget !== null && (typeof budget !== 'object' || !Number.isInteger(budget.remainingControls) || budget.remainingControls < 0
      || (budget.guard !== undefined && typeof budget.guard !== 'boolean')))
    throw new Error('A planning budget needs whole nonnegative remaining controls and a boolean guard');
}

/** Mathematical interval coverage, not a physical qualification of a source. */
export function mixedCarryDistanceCoverage(candidates, { maxSegments = 3 } = {}) {
  validateBound(maxSegments);
  return Object.freeze(coverage(sequences(readCandidates(candidates, maxSegments), maxSegments)));
}

// The minimum maximum fractional correction allocation, plus each feasible
// polytope vertex and its midpoint with that allocation. These bounded choices
// allow preflight to compare intermediate placement locations for one skill order.
function allocations(entries, distance) {
  const nominal = entries.reduce((sum, c) => sum + c.nominal, 0);
  const budget = entries.reduce((sum, c) => sum + c.correction, 0);
  const central = entries.map(c => c.nominal + (budget ? (distance - nominal) * c.correction / budget : 0));
  const values = [];
  function add(lengths) {
    const corrected = [...lengths];
    corrected[corrected.length - 1] = distance - corrected.slice(0, -1).reduce((sum, v) => sum + v, 0);
    if (corrected.some((v, i) => v < entries[i].min - EPS || v > entries[i].max + EPS)) return;
    if (!values.some(old => old.every((v, i) => Math.abs(v - corrected[i]) < EPS))) values.push(corrected);
  }
  add(central);
  for (let free = 0; free < entries.length; free++) {
    const fixed = entries.map((_, i) => i).filter(i => i !== free);
    for (let bits = 0; bits < 2 ** fixed.length; bits++) {
      const vertex = entries.map(c => c.nominal);
      fixed.forEach((index, bit) => { vertex[index] = bits & (1 << bit) ? entries[index].max : entries[index].min; });
      vertex[free] = distance - fixed.reduce((sum, index) => sum + vertex[index], 0);
      if (vertex[free] < entries[free].min - EPS || vertex[free] > entries[free].max + EPS) continue;
      add(vertex); add(vertex.map((v, i) => (v + central[i]) / 2));
    }
  }
  return values;
}

function structuralPlan(initial, goal, entries, lengths, distance) {
  let previous = freezePosition(initial), cumulative = 0;
  const segments = entries.map((entry, index) => {
    cumulative += lengths[index];
    const destination = freezePosition(index + 1 === entries.length ? goal :
      [initial[0] + (goal[0] - initial[0]) * cumulative / distance,
        initial[1] + (goal[1] - initial[1]) * cumulative / distance, goal[2]]);
    const segment = Object.freeze({ candidateId: entry.id, skill: entry.skill,
      goalWorld: destination, plannedStartObjectPositionWorld: previous,
      nominalTravelM: entry.nominal, plannedTravelM: lengths[index], correctionM: lengths[index] - entry.nominal,
      supportedDistanceIntervalM: Object.freeze([entry.min, entry.max]), carryOptions: entry.carryOptions });
    previous = destination; return segment;
  });
  return Object.freeze({ supported: true, requestedGoalWorld: freezePosition(goal), initialObjectPositionWorld: freezePosition(initial),
    distanceM: distance, segments: Object.freeze(segments), goals: Object.freeze(segments.map(s => s.goalWorld)) });
}

function finiteMetric(value, label, { nonnegative = true } = {}) {
  if (value === null || value === undefined) return null;
  if (!Number.isFinite(value) || (nonnegative && value < 0)) throw new Error(`Invalid geometric metric ${label}`);
  return value;
}
const correctionDifference = (a, b) => Math.abs(a - b) < EPS ? 0 : a - b;
// Geometry transforms use FP32 reference endpoints. Preserve reported values,
// but do not rank a sub-micrometre residual above a materially smaller warp.
const GEOMETRIC_RANK_TOLERANCE_M = 1e-6;
function geometricDifference(a, b, descending = false) {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return Math.abs(a - b) <= GEOMETRIC_RANK_TOLERANCE_M ? 0 : (descending ? b - a : a - b);
}
/** Ranking order. With the budget guard, plans estimated to fit the remaining
 * controls come first. `reliability` then ranks the summed measured failure
 * penalty before pickup count; `default` ranks pickup count first. The
 * remaining geometric tie-breaks are unchanged. */
export function compareMixedCarryScores(a, b, { ranking = 'default', guard = false } = {}) {
  return (guard ? Number(b.fitsBudget === true) - Number(a.fitsBudget === true) : 0)
    || (ranking === 'reliability' ? correctionDifference(a.reliabilityPenalty ?? 0, b.reliabilityPenalty ?? 0) : 0)
    || a.pickupCount - b.pickupCount
    || geometricDifference(a.predictedEndpointResidualM, b.predictedEndpointResidualM)
    || geometricDifference(a.approachCostM, b.approachCostM)
    || geometricDifference(a.minimumClearanceM, b.minimumClearanceM, true)
    || correctionDifference(a.squaredFractionalCorrection, b.squaredFractionalCorrection);
}
function compare(a, b, options) {
  return compareMixedCarryScores(a.score, b.score, options)
    || (a.orderKey < b.orderKey ? -1 : a.orderKey > b.orderKey ? 1 : 0);
}

// The failing geometry check may name the obstacle in a destination footprint
// collision or a reference path violation; accept an explicit name as well.
function describeObstacle(geometry) {
  if (!geometry || geometry.supported === true) return null;
  if (typeof geometry.obstacleName === 'string') return geometry.obstacleName;
  const check = Array.isArray(geometry.checks) ? geometry.checks.at(-1) : null;
  const collision = check?.destination?.collisions?.[0]?.obstacleName;
  if (typeof collision === 'string') return collision;
  const violation = check?.path?.firstViolation?.object;
  return typeof violation === 'string' ? violation : null;
}

/** The most frequent failing reason across attempts, and the most frequent
 * obstacle named with that reason. Ties keep the earliest attempt. */
export function summarizeAttemptFailures(attempts) {
  const reasons = new Map(), obstacles = new Map();
  for (const attempt of attempts) {
    if (attempt.supported || !attempt.reason) continue;
    reasons.set(attempt.reason, (reasons.get(attempt.reason) ?? 0) + 1);
    if (attempt.obstacle) {
      const byReason = obstacles.get(attempt.reason) ?? new Map();
      byReason.set(attempt.obstacle, (byReason.get(attempt.obstacle) ?? 0) + 1);
      obstacles.set(attempt.reason, byReason);
    }
  }
  let dominantReason = null, reasonCount = 0;
  for (const [reason, count] of reasons) if (count > reasonCount) { dominantReason = reason; reasonCount = count; }
  let dominantObstacle = null, obstacleCount = 0;
  for (const [obstacle, count] of obstacles.get(dominantReason) ?? []) if (count > obstacleCount) { dominantObstacle = obstacle; obstacleCount = count; }
  return Object.freeze({ dominantReason, dominantReasonCount: reasonCount, dominantObstacle, dominantObstacleCount: obstacleCount,
    failedAttempts: [...reasons.values()].reduce((sum, v) => sum + v, 0),
    reasonCounts: Object.freeze(Object.fromEntries(reasons)) });
}

/** Choose mixed complete references while preserving the exact original goal.
 * checkPlan(plan) synchronously returns {supported,reason,minimumClearanceM?,
 * approachCostM?,predictedEndpointResidualM?}. These are geometric predictions.
 * Missing metrics remain null. Each complete count is considered before more
 * pickups. A finite check budget is explicit and never reported as exhaustive.
 * `budget` ({remainingControls, guard}) attaches a control-count estimate to
 * every plan; with `guard` the estimated-to-fit plans rank first. `ranking`
 * `reliability` continues to later pickup counts while a lower summed penalty
 * remains possible; `default` stops at the first count with a clear plan.
 */
export function planMixedCarry(objectPosition, goalWorld, candidates,
  { maxSegments = 3, checkPlan = null, maxPlanChecks = 256, ranking = 'default', budget = null } = {}) {
  validateBound(maxSegments);
  validateOptions({ ranking, budget });
  if (!position(objectPosition) || !position(goalWorld) || (checkPlan !== null && typeof checkPlan !== 'function')
      || !Number.isInteger(maxPlanChecks) || maxPlanChecks < 1 || maxPlanChecks > 16384)
    throw new Error('Finite absolute positions, synchronous preflight and a bounded check budget are required');
  const initial = Array.from(objectPosition), goal = Array.from(goalWorld);
  const read = readCandidates(candidates, maxSegments);
  const groups = sequences(read, maxSegments), ranges = coverage(groups);
  const distance = Math.hypot(goal[0] - initial[0], goal[1] - initial[1]);
  const guard = budget?.guard === true, rankingOptions = { ranking, guard };
  const usable = read.filter(c => c.maxUses > 0);
  const minimumPenalty = usable.length ? Math.min(...usable.map(c => c.reliabilityPenalty)) : 0;
  const attempts = []; let best = null, exhausted = false, eligibleOrders = 0;
  for (const [groupIndex, group] of groups.entries()) {
    const candidatesForCount = [];
    for (const sequence of group) {
      if (distance < sequence.min - EPS || distance > sequence.max + EPS) continue;
      eligibleOrders++;
      for (const lengths of allocations(sequence.entries, distance)) {
        const correctionCost = sequence.entries.reduce((sum, c, i) => sum + (c.correction ? ((lengths[i] - c.nominal) / c.correction) ** 2 : 0), 0);
        candidatesForCount.push({ entries: sequence.entries, lengths, correctionCost,
          orderKey: sequence.entries.map(c => c.id).join('\u0000') + '\u0001' + lengths.map(v => v.toPrecision(17)).join(',') });
      }
    }
    candidatesForCount.sort((a, b) => correctionDifference(a.correctionCost, b.correctionCost) || (a.orderKey < b.orderKey ? -1 : a.orderKey > b.orderKey ? 1 : 0));
    for (const candidate of candidatesForCount) {
      if (attempts.length === maxPlanChecks) { exhausted = true; break; }
      const plan = structuralPlan(initial, goal, candidate.entries, candidate.lengths, distance);
      const geometry = checkPlan ? checkPlan(plan) : { supported: true };
      if (geometry?.then) throw new Error('Carry geometry preflight must be synchronous');
      const supported = geometry?.supported === true;
      const approachCostM = finiteMetric(geometry?.approachCostM, 'approach cost');
      const assessment = budget ? assessCarryPlanBudget(plan, budget.remainingControls, { approachCostM }) : null;
      const score = Object.freeze({ pickupCount: plan.segments.length,
        predictedEndpointResidualM: finiteMetric(geometry?.predictedEndpointResidualM, 'predicted endpoint residual'),
        approachCostM,
        minimumClearanceM: finiteMetric(geometry?.minimumClearanceM, 'minimum clearance', { nonnegative: false }),
        nominalFinalGoalResidualM: Math.hypot(plan.goals.at(-1)[0] - goal[0], plan.goals.at(-1)[1] - goal[1]),
        squaredFractionalCorrection: candidate.correctionCost,
        reliabilityPenalty: candidate.entries.reduce((sum, c) => sum + c.reliabilityPenalty, 0),
        estimatedControls: assessment?.estimatedControls ?? null,
        fitsBudget: assessment ? assessment.fits : null });
      const reason = supported ? null : geometry?.reason ?? 'reference_check_unavailable';
      attempts.push(Object.freeze({ candidateIds: Object.freeze(plan.segments.map(s => s.candidateId)),
        goals: plan.goals, segmentLengthsM: Object.freeze([...candidate.lengths]), supported, reason,
        obstacle: describeObstacle(geometry), score }));
      const evaluated = { plan, score, orderKey: candidate.orderKey, assessment };
      if (supported && (!best || compare(evaluated, best, rankingOptions) < 0)) best = evaluated;
    }
    if (exhausted) break;
    if (!best) continue;
    // Fewer pickups always cost fewer controls, so a non-fitting best cannot be
    // rescued by a longer plan. Only the reliability ranking keeps searching,
    // and only while a longer plan could still lower the summed penalty.
    if (ranking !== 'reliability') break;
    const nextCount = groupIndex + 2;
    if (nextCount > maxSegments || best.score.reliabilityPenalty <= nextCount * minimumPenalty + EPS) break;
  }
  const failures = summarizeAttemptFailures(attempts);
  const common = { ...ranges, attempts: Object.freeze(attempts), checkedPlans: attempts.length,
    planningBudgetExhausted: exhausted, geometricRankingExhaustiveForSelectedPickupCount: !exhausted,
    ranking, dominantReason: failures.dominantReason, dominantObstacle: failures.dominantObstacle, failureSummary: failures,
    physicalPlacementSuccessPredicted: false, geometricRankingToleranceM: GEOMETRIC_RANK_TOLERANCE_M, skillDataOwnership: 'Complete existing skill objects retained by reference; no source rows mutated or removed.' };
  if (best) return Object.freeze({ ...best.plan, reason: null, score: best.score,
    budget: best.assessment ? Object.freeze({ ...best.assessment, guard }) : null, budgetRisk: best.assessment ? best.assessment.budgetRisk : false, ...common });
  return Object.freeze({ supported: false, reason: exhausted ? 'planning_budget_exhausted' : eligibleOrders ? 'no_clear_plan' : 'unsupported_distance',
    requestedGoalWorld: freezePosition(goal), initialObjectPositionWorld: freezePosition(initial), distanceM: distance,
    segments: Object.freeze([]), goals: Object.freeze([]), score: null, budget: null, budgetRisk: false, ...common });
}
