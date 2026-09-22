// Experimental box approach waypoints. The final pregrasp target belongs to
// the skill controller and is never projected to an obstacle boundary.
import { planNavigationPath, segmentEntersRectangle } from './navigation_planner.js';

const EPS = 1e-7;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const inside = (p, r) => p[0] > r.minX + EPS && p[0] < r.maxX - EPS
  && p[1] > r.minY + EPS && p[1] < r.maxY - EPS;
const inflate = (r, margin) => ({ minX: r.minX - margin, maxX: r.maxX + margin,
  minY: r.minY - margin, maxY: r.maxY + margin });
const validPoint = p => p && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
const validBounds = bounds => Array.isArray(bounds) && bounds.every(r =>
  ['minX', 'maxX', 'minY', 'maxY'].every(key => Number.isFinite(r[key]))
  && r.minX < r.maxX && r.minY < r.maxY);
const crosses = (a, b, bounds) => bounds.some(r => segmentEntersRectangle(a, b, r));

/** Read the actual compiled collision meshes, including MuJoCo's mesh recentering
 * and each geom's world transform. A point-cloud bbox at the body origin is not
 * equivalent when a source mesh or geom is offset. Only explicitly supplied
 * object bodies are included, never the floor or humanoid.
 */
export class ObjectMeshBounds {
  constructor(model, bodyIds) {
    if (!Array.isArray(bodyIds) || !bodyIds.length || bodyIds.some(id => !Number.isInteger(id) || id <= 0 || id >= model.nbody)) {
      throw new Error('Object bounds require valid object body IDs');
    }
    this.objects = bodyIds.map(bodyId => {
      const meshes = [];
      for (let geomId = 0; geomId < model.ngeom; geomId++) {
        if (model.geom_bodyid[geomId] !== bodyId || (model.geom_contype[geomId] === 0 && model.geom_conaffinity[geomId] === 0)) continue;
        // mjGEOM_MESH = 7. Reject other collision shapes instead of understating
        // an object's footprint by silently omitting them.
        if (model.geom_type[geomId] !== 7) throw new Error('Object bounds currently require mesh collision geoms');
        const meshId = model.geom_dataid[geomId], first = model.mesh_vertadr[meshId] * 3;
        meshes.push({ geomId, vertices: Float64Array.from(model.mesh_vert.slice(first, first + model.mesh_vertnum[meshId] * 3)) });
      }
      if (!meshes.length) throw new Error('Object body has no collision mesh');
      return { bodyId, meshes };
    });
  }

  read(data) {
    // MuJoCo exposes these arrays through WASM getters. Read each transform
    // once per geom; the synchronous vertex loop keeps the same arithmetic.
    const positions = data.geom_xpos, matrices = data.geom_xmat;
    return this.objects.map(({ bodyId, meshes }) => {
      const r = { bodyId, minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity };
      for (const { geomId, vertices } of meshes) {
        const p = geomId * 3, m = geomId * 9;
        const px = positions[p], py = positions[p + 1], pz = positions[p + 2];
        const m0 = matrices[m], m1 = matrices[m + 1], m2 = matrices[m + 2];
        const m3 = matrices[m + 3], m4 = matrices[m + 4], m5 = matrices[m + 5];
        const m6 = matrices[m + 6], m7 = matrices[m + 7], m8 = matrices[m + 8];
        for (let i = 0; i < vertices.length; i += 3) {
          const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
          const wx = px + m0 * x + m1 * y + m2 * z;
          const wy = py + m3 * x + m4 * y + m5 * z;
          const wz = pz + m6 * x + m7 * y + m8 * z;
          r.minX = Math.min(r.minX, wx); r.maxX = Math.max(r.maxX, wx);
          r.minY = Math.min(r.minY, wy); r.maxY = Math.max(r.maxY, wy);
          r.minZ = Math.min(r.minZ, wz); r.maxZ = Math.max(r.maxZ, wz);
        }
      }
      return r;
    });
  }
}

/** Construct a route when the direct centre path crosses physical geometry,
 * or optionally its transit clearance. If the final target lies within that
 * margin, approach it from a staging point on its own side of the physical box.
 */
export function planBoxApproach(root, goal, bounds, { clearance = 0.55, respectTransitClearance = false } = {}) {
  if (!validPoint(root) || !validPoint(goal) || !validBounds(bounds) || !Number.isFinite(clearance) || !(clearance > 0)) throw new Error('Invalid box approach geometry');
  if (typeof respectTransitClearance !== 'boolean') throw new Error('Transit clearance routing must be explicitly enabled or disabled');
  const directBlocked = crosses(root, goal, bounds);
  const expanded = bounds.map(r => inflate(r, clearance));
  const clearanceBlocked = crosses(root, goal, expanded);
  const result = { directBlocked, clearanceBlocked, routed: false, supported: true, path: [], stagingGoal: null,
    finalGoal: Array.from(goal), pathLengthM: distance(root, goal) };
  if (bounds.some(r => inside(goal, r))) return { ...result, supported: false, reason: 'occupied_pregrasp_target' };
  if (!directBlocked && !(respectTransitClearance && clearanceBlocked)) return result;
  const containing = bounds.filter((r, i) => inside(goal, expanded[i]));
  let stages = [];
  if (!containing.length) stages = [goal.slice(0, 2)];
  else for (const r of containing) {
    // Only use faces on the target's side; never stage across the box and
    // subsequently cross it on the final uninflated approach.
    if (goal[0] <= r.minX) stages.push([r.minX - clearance, goal[1]]);
    if (goal[0] >= r.maxX) stages.push([r.maxX + clearance, goal[1]]);
    if (goal[1] <= r.minY) stages.push([goal[0], r.minY - clearance]);
    if (goal[1] >= r.maxY) stages.push([goal[0], r.maxY + clearance]);
  }
  const candidates = [];
  for (const stage of stages) {
    if (expanded.some(r => inside(stage, r)) || crosses(stage, goal, bounds)) continue;
    const navigation = planNavigationPath(root, stage, bounds, clearance);
    // The generic navigator can project an occupied destination. A skill's
    // staging point must be reached exactly instead of accepting that change.
    if (!navigation.path.length || !navigation.reachableGoal || distance(navigation.reachableGoal, stage) > EPS) continue;
    let previous = root, length = 0;
    for (const point of navigation.path) { length += distance(previous, point); previous = point; }
    length += distance(stage, goal);
    candidates.push({ path: navigation.path, stage, length });
  }
  candidates.sort((a, b) => a.length - b.length);
  if (!candidates.length) return { ...result, supported: false, reason: 'no_clear_approach_route' };
  const best = candidates[0];
  return { ...result, routed: true, path: best.path.map(p => [...p]), stagingGoal: [...best.stage], pathLengthM: best.length };
}

export class BoxApproachPlanner {
  constructor({ clearance = 0.55, waypointRadius = 0.28, respectTransitClearance = false } = {}) {
    if (!Number.isFinite(clearance) || !Number.isFinite(waypointRadius) || !(clearance > waypointRadius && waypointRadius > 0.25)) throw new Error('Approach waypoint radius must exceed student arrival radius and leave positive clearance');
    if (typeof respectTransitClearance !== 'boolean') throw new Error('Transit clearance routing must be explicitly enabled or disabled');
    this.clearance = clearance; this.waypointRadius = waypointRadius;
    this.respectTransitClearance = respectTransitClearance; this.reset();
  }

  reset() { this.goal = null; this.plan = null; this.waypoints = []; this.phase = 'direct'; this.initialDirectBlocked = null; }

  step(root, goal, bounds) {
    if (!validPoint(root) || !validPoint(goal) || !validBounds(bounds)) throw new Error('Invalid live approach geometry');
    const directBlocked = crosses(root, goal, bounds);
    const transitBlocked = directBlocked || this.respectTransitClearance
      && crosses(root, goal, bounds.map(r => inflate(r, this.clearance)));
    const changedGoal = !this.goal || distance(this.goal, goal) > 1e-3;
    // Keep an existing detour while its physical segments remain clear, even
    // when small settling motions make another route fractionally shorter.
    let routeBlocked = false, previous = root;
    for (const point of this.waypoints) { routeBlocked ||= crosses(previous, point, bounds); previous = point; }
    if (this.waypoints.length) routeBlocked ||= crosses(previous, goal, bounds);
    if (changedGoal || routeBlocked || this.phase === 'blocked' || (this.phase === 'direct' && transitBlocked)) {
      this.goal = Array.from(goal); this.plan = planBoxApproach(root, goal, bounds, this);
      this.waypoints = this.plan.path.map(p => [...p]);
      this.phase = !this.plan.supported ? 'blocked' : this.plan.routed ? 'staging' : 'direct';
      if (changedGoal) this.initialDirectBlocked = this.plan.directBlocked;
    }
    if (this.phase === 'staging') {
      while (this.waypoints.length && distance(root, this.waypoints[0]) < this.waypointRadius) this.waypoints.shift();
      if (!this.waypoints.length) this.phase = 'final';
    }
    if (this.phase === 'final' && directBlocked) {
      this.phase = 'direct';
      return this.step(root, goal, bounds);
    }
    const commandGoal = this.phase === 'blocked' ? [root[0], root[1], goal[2] ?? 0]
      : this.waypoints.length ? [...this.waypoints[0], goal[2] ?? 0] : goal;
    return { goal: commandGoal, phase: this.phase, directBlocked, initialDirectBlocked: this.initialDirectBlocked,
      supported: this.plan.supported, reason: this.plan.reason || null, routed: this.plan.routed,
      finalGoal: this.goal, stagingGoal: this.plan.stagingGoal, waypoints: this.waypoints.map(p => [...p]),
      pathLengthM: this.plan.pathLengthM };
  }
}
