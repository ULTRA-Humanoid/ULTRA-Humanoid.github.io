// Optional geometric navigation experiment. The policy still controls every
// joint; this module proposes floor waypoints around simulated object bounds.

const EPS = 1e-7;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const inside = (p, r) => p[0] > r.minX + EPS && p[0] < r.maxX - EPS
  && p[1] > r.minY + EPS && p[1] < r.maxY - EPS;

export function segmentEntersRectangle(a, b, r) {
  // Intersect the open interior. Tangency to the clearance boundary is legal.
  let low = 0, high = 1;
  for (let axis = 0; axis < 2; axis++) {
    const min = (axis === 0 ? r.minX : r.minY) + EPS;
    const max = (axis === 0 ? r.maxX : r.maxY) - EPS;
    const delta = b[axis] - a[axis];
    if (Math.abs(delta) < EPS) {
      if (a[axis] <= min || a[axis] >= max) return false;
    } else {
      const t1 = (min - a[axis]) / delta, t2 = (max - a[axis]) / delta;
      low = Math.max(low, Math.min(t1, t2));
      high = Math.min(high, Math.max(t1, t2));
      if (low >= high) return false;
    }
  }
  return low < high && high > 0 && low < 1;
}

export function planNavigationPath(start, goal, bounds, clearance = 0.45) {
  const rectangles = bounds.map(r => ({
    minX: r.minX - clearance, maxX: r.maxX + clearance,
    minY: r.minY - clearance, maxY: r.maxY + clearance,
    physical: r,
  }));
  const corners = rectangles.flatMap(r => [
    [r.minX, r.minY], [r.minX, r.maxY], [r.maxX, r.minY], [r.maxX, r.maxY],
  ]).filter(p => !rectangles.some(r => inside(p, r)));
  let reachableGoal = goal.slice(0, 2);
  if (rectangles.some(r => inside(reachableGoal, r))) {
    const candidates = rectangles.flatMap(r => [
      [r.minX, Math.max(r.minY, Math.min(r.maxY, goal[1]))],
      [r.maxX, Math.max(r.minY, Math.min(r.maxY, goal[1]))],
      [Math.max(r.minX, Math.min(r.maxX, goal[0])), r.minY],
      [Math.max(r.minX, Math.min(r.maxX, goal[0])), r.maxY],
    ]).concat(corners).filter(p => !rectangles.some(r => inside(p, r)));
    candidates.sort((a, b) => distance(a, goal) - distance(b, goal));
    if (!candidates.length) return { path: [], reachableGoal: null };
    reachableGoal = candidates[0];
  }
  const nodes = [start.slice(0, 2), reachableGoal, ...corners];
  const clear = (a, b) => !rectangles.some(r => {
    // A robot already within the conservative margin may leave it, provided
    // the route does not cross the physical box or move deeper into its centre.
    if (inside(a, r) && !inside(b, r)) {
      const centre = [(r.minX + r.maxX) / 2, (r.minY + r.maxY) / 2];
      const outward = (b[0] - a[0]) * (a[0] - centre[0])
        + (b[1] - a[1]) * (a[1] - centre[1]);
      return outward < 0 || segmentEntersRectangle(a, b, r.physical);
    }
    return segmentEntersRectangle(a, b, r);
  });
  const costs = nodes.map(() => Infinity), previous = nodes.map(() => -1);
  const visited = nodes.map(() => false);
  costs[0] = 0;
  for (let k = 0; k < nodes.length; k++) {
    let current = -1;
    for (let i = 0; i < nodes.length; i++) {
      if (!visited[i] && (current < 0 || costs[i] < costs[current])) current = i;
    }
    if (current < 0 || !Number.isFinite(costs[current]) || current === 1) break;
    visited[current] = true;
    for (let next = 0; next < nodes.length; next++) {
      if (visited[next] || !clear(nodes[current], nodes[next])) continue;
      const cost = costs[current] + distance(nodes[current], nodes[next]);
      if (cost < costs[next]) { costs[next] = cost; previous[next] = current; }
    }
  }
  if (!Number.isFinite(costs[1])) return { path: [], reachableGoal };
  const path = [];
  for (let index = 1; index !== 0; index = previous[index]) path.unshift(nodes[index]);
  return { path, reachableGoal };
}

export class NavigationPlanner {
  constructor({ clearance = 0.45, waypointRadius = 0.22, maxSegmentLength = Infinity } = {}) {
    if (!(maxSegmentLength > waypointRadius)) throw new Error('Navigation segment must exceed waypoint radius');
    this.clearance = clearance;
    this.waypointRadius = waypointRadius;
    this.maxSegmentLength = maxSegmentLength;
    this.reset();
  }

  reset() { this.goal = null; this.bounds = []; this.path = []; this.reachableGoal = null; }

  step(root, goal, bounds) {
    const moved = bounds.length !== this.bounds.length || bounds.some((r, i) =>
      ['minX', 'maxX', 'minY', 'maxY'].some(key => Math.abs(r[key] - this.bounds[i][key]) > 0.05));
    // Settling boxes must not repeatedly flip the chosen route around them.
    // Replan a moved scene only when the existing route has lost its margin.
    let routeBlocked = this.path.length === 0;
    if (moved) {
      let previous = root;
      const retainedClearance = Math.max(0, this.clearance - 0.10);
      for (const point of this.path) {
        routeBlocked ||= bounds.some(r => segmentEntersRectangle(previous, point, {
          minX: r.minX - retainedClearance, maxX: r.maxX + retainedClearance,
          minY: r.minY - retainedClearance, maxY: r.maxY + retainedClearance,
        }));
        previous = point;
      }
      this.bounds = bounds.map(r => ({ ...r }));
    }
    if (!this.goal || distance(goal, this.goal) > 1e-3 || (moved && routeBlocked)) {
      this.goal = Array.from(goal);
      this.bounds = bounds.map(r => ({ ...r }));
      Object.assign(this, planNavigationPath(root, goal, bounds, this.clearance));
      const subdivided = [];
      let previous = root;
      for (const point of this.path) {
        const count = Math.max(1, Math.ceil(distance(previous, point) / this.maxSegmentLength));
        for (let i = 1; i <= count; i++) subdivided.push([
          previous[0] + (point[0] - previous[0]) * i / count,
          previous[1] + (point[1] - previous[1]) * i / count,
        ]);
        previous = point;
      }
      this.path = subdivided;
    }
    while (this.path.length > 1 && distance(root, this.path[0]) < this.waypointRadius) this.path.shift();
    const target = this.path[0] || root;
    return new Float32Array([target[0], target[1], goal[2] ?? 0]);
  }
}
