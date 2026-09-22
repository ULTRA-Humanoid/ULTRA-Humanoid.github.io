// Tiered carry planning: plans whose pickup poses are reachable rank first.
// Default passes, in order: require 'first' (the first segment's pickup pose,
// from the actual root) with the unchanged library, then with the extended
// library (extra pickup-side sources such as the alternate one-metre carry),
// and finally 'none' with the unchanged library: the pre-search ranking, whose
// plan carries pickupPoseReachable:false so the executing loop knows. The
// optional 'all' pass also requires later segments from their nominal post-exit
// roots; on the frozen panel those nominal predictions re-rank a completed
// request (H082), so it is not part of the default. Within a pass the existing order (pickups,
// residual, approach cost, clearance) is untouched, so a request whose
// unchanged first choice is reachable keeps exactly that choice. This never
// adds a refusal: a request refused here is refused by the same geometry the
// 'none' pass applies. Pure planning; no physics.
import { planMixedCarry } from './mixed_carry_planner.js';
import { planCarryGoalRegion } from './carry_goal_region_planner.js';

export const PICKUP_POSE_SEARCH_TIERS = Object.freeze(['all', 'first', 'none']);
export const PICKUP_POSE_SEARCH_DEFAULT_TIERS = Object.freeze(['first', 'none']);

/** planOnce (optional): ({ library, checkPlan }) => { plan, exactPlan?, region?, ... }
 * runs the caller's own request-time planning path (ranking modes, budget guard,
 * review hooks) for one pass; the default runs the exact mixed plan then the
 * bounded goal region. The chosen pass's full return value is `chosen.planned`. */
export function planCarryWithPickupPoseSearch({ initialObjectPositionWorld, goalWorld, library, extendedLibrary = null, makeCheckPlan,
  planOnce = null, maxSegments = 3, goalRegionEnabled = true, regionResidualLimitM = .01, tiers = PICKUP_POSE_SEARCH_DEFAULT_TIERS } = {}) {
  if (typeof makeCheckPlan !== 'function' || !Array.isArray(tiers) || !tiers.length || (planOnce !== null && typeof planOnce !== 'function')
      || tiers.some(tier => !PICKUP_POSE_SEARCH_TIERS.includes(tier)) || tiers.at(-1) !== 'none')
    throw new Error('Pickup pose search needs a check factory and tiers ending with the unchanged ranking');
  const passes = tiers.flatMap(require => require === 'none' || !extendedLibrary || extendedLibrary === library
    ? [{ require, library, extended: false }] : [{ require, library, extended: false }, { require, library: extendedLibrary, extended: true }]);
  const defaultPlanOnce = ({ library: passLibrary, checkPlan }) => {
    const exactPlan = planMixedCarry(initialObjectPositionWorld, goalWorld, passLibrary, { maxSegments, checkPlan });
    let plan = exactPlan, region = null, priorRegionSelection = null;
    if (goalRegionEnabled) {
      region = planCarryGoalRegion(initialObjectPositionWorld, goalWorld, passLibrary,
        { maxSegments, checkPlan, exactFallbackPlan: exactPlan.supported ? exactPlan : null });
      // Leave most of the 10 cm success tolerance for physical tracking error.
      priorRegionSelection = region.supported && region.nominalFinalGoalResidualM <= .01 + 1e-12 ? region : exactPlan;
      if (region.supported && region.nominalFinalGoalResidualM <= regionResidualLimitM + 1e-12) plan = region;
    }
    return { plan, exactPlan, region, priorRegionSelection };
  };
  const tierResults = [];
  let chosen = null;
  for (const pass of passes) {
    const checkPlan = makeCheckPlan(pass.require);
    const planned = (planOnce ?? defaultPlanOnce)({ library: pass.library, checkPlan, require: pass.require });
    if (!planned?.plan) throw new Error('planOnce must return the pass plan');
    const { plan, exactPlan = null, region = null, priorRegionSelection = null } = planned;
    const unreachableAttempts = [...(exactPlan?.attempts ?? (exactPlan === null ? plan.attempts ?? [] : [])), ...(region?.attempts ?? [])]
      .filter(attempt => attempt.reason === 'pickup_pose_unreachable').length;
    chosen = { require: pass.require, extendedLibrary: pass.extended, checkPlan, planned, exactPlan, region, priorRegionSelection, plan, unreachableAttempts };
    tierResults.push({ require: pass.require, extendedLibrary: pass.extended, supported: plan.supported, reason: plan.reason,
      checkedPlans: plan.checkedPlans ?? null, unreachableAttempts, candidateIds: plan.segments?.map(segment => segment.candidateId) ?? [] });
    if (plan.supported) break;
  }
  const clearance = chosen.plan.supported ? chosen.checkPlan(chosen.plan) : null;
  let plan = chosen.plan;
  if (clearance && clearance.pickupPoseSearchRequirement) plan = Object.freeze({ ...plan,
    pickupPoseReachable: clearance.pickupPoseReachable, firstPickupReachable: clearance.firstPickupReachable,
    allPickupPosesPredictedReachable: clearance.allPickupPosesPredictedReachable,
    pickupPoseReachability: clearance.pickupPoseReachability, pickupPoseSearchTier: chosen.require,
    pickupPoseSearchExtendedLibrary: chosen.extendedLibrary });
  return { plan, clearance, chosen, tierResults };
}
