// Geometric command support for the initial recorded-motion controller.
// It preserves the requested destination and never changes physical state.
import { planNavigationPath } from './navigation_planner.js';

function position(value, label) {
  if (!value || value.length !== 3 || !Array.from(value).every(Number.isFinite)) {
    throw new Error(`${label} must contain three finite coordinates`);
  }
  return Array.from(value);
}

export function planRestrictedFloorGoal(rootPosition, requestedGoal, bounds,
  { clearance = .55, maxGoalDistance = 2, maxPathDistance = 4 } = {}) {
  const root = position(rootPosition, 'Root position');
  const goal = position(requestedGoal, 'Floor destination');
  if (!Array.isArray(bounds) || bounds.some(bound => !['minX', 'maxX', 'minY', 'maxY'].every(key => Number.isFinite(bound[key]))
      || bound.minX > bound.maxX || bound.minY > bound.maxY)
      || ![clearance, maxGoalDistance, maxPathDistance].every(Number.isFinite)
      || clearance < .4 || maxGoalDistance <= 0 || maxPathDistance < maxGoalDistance) {
    throw new Error('Finite physical bounds and positive restricted navigation limits are required');
  }
  const distanceM = Math.hypot(goal[0] - root[0], goal[1] - root[1]);
  const result = { requestedGoalWorld: goal, supported: false, reason: null,
    finalGoalWorld: null, waypoints: [], distanceM, pathDistanceM: null,
    clearanceM: clearance, maxGoalDistanceM: maxGoalDistance, maxPathDistanceM: maxPathDistance };
  if (distanceM > maxGoalDistance) return { ...result, reason: 'goal_distance' };
  const plan = planNavigationPath(root, goal, bounds, clearance);
  if (!plan.reachableGoal || Math.hypot(plan.reachableGoal[0] - goal[0], plan.reachableGoal[1] - goal[1]) > 1e-6) {
    return { ...result, reason: 'goal_clearance' };
  }
  if (!plan.path.length) return { ...result, reason: 'route_unavailable' };
  let previous = root, pathDistanceM = 0;
  for (const point of plan.path) {
    pathDistanceM += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
    previous = point;
  }
  if (pathDistanceM > maxPathDistance) return { ...result, pathDistanceM, reason: 'route_distance' };
  return { ...result, supported: true, pathDistanceM, finalGoalWorld: [...goal],
    waypoints: plan.path.slice(0, -1).map(point => [point[0], point[1], goal[2]]) };
}
