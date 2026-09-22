// WS-C teacher-descent contact hold (opt-in). Pure owner state + geometry; the caller (main.js) owns preview, physics, history
// and the carry controller's held clock. Evidence (offline hold study 2026-09-16, saved prefixes H046-int17/H064/H067): a
// 17-substep preview of the teacher's own action whose predicted knee/hip origin comes within 6 cm of the box surface, followed
// by a 30-control hold on the live-FK target and a resume at the same source index, removed the recorded knee contact on 3/3.
export const TEACHER_DESCENT_HOLD_LIMITS = Object.freeze({
  onsetMetricM: .06, holdControls: 30, triggerBodies: Object.freeze(['left_knee_link', 'right_knee_link', 'left_hip_yaw_link', 'right_hip_yaw_link']),
});
const count = v => Number.isSafeInteger(v) && v >= 0;
const finite = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);

/** Signed distance from a body origin to the box collision geom surface (OBB in the geom frame; negative = inside).
 *  Pure: takes the geom pose arrays (xpos 3, xmat 9 row-major) and the mesh half extents/center offset. */
export function originToBoxSurfaceM(bodyPosition, geomPosition, geomMatrix, boxCenter, boxHalf) {
  if (!finite(bodyPosition, 3) || !finite(geomPosition, 3) || !finite(geomMatrix, 9) || !finite(boxCenter, 3) || !finite(boxHalf, 3)) throw new Error('Finite box geometry required');
  const r = [bodyPosition[0] - geomPosition[0], bodyPosition[1] - geomPosition[1], bodyPosition[2] - geomPosition[2]], R = geomMatrix;
  const local = [R[0] * r[0] + R[3] * r[1] + R[6] * r[2], R[1] * r[0] + R[4] * r[1] + R[7] * r[2], R[2] * r[0] + R[5] * r[1] + R[8] * r[2]];
  const q = local.map((v, a) => Math.abs(v - boxCenter[a]) - boxHalf[a]);
  const outside = Math.hypot(...q.map(v => Math.max(v, 0)));
  return outside > 0 ? outside : Math.max(...q);
}

/** The selected box's single collision geom (MuJoCo mesh geom) with its mesh half extents and center offset in the geom frame. */
export function boxCollisionGeometry(model, bodyId) {
  const geoms = [];
  for (let i = 0; i < model.ngeom; i++) if (model.geom_bodyid[i] === bodyId && (model.geom_contype[i] !== 0 || model.geom_conaffinity[i] !== 0)) geoms.push(i);
  if (geoms.length !== 1 || model.geom_type[geoms[0]] !== 7) throw new Error('The selected box needs exactly one mesh collision geom');
  const geom = geoms[0], meshId = model.geom_dataid[geom], first = model.mesh_vertadr[meshId] * 3, count = model.mesh_vertnum[meshId];
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) for (let a = 0; a < 3; a++) { const v = model.mesh_vert[first + i * 3 + a]; lo[a] = Math.min(lo[a], v); hi[a] = Math.max(hi[a], v); }
  return Object.freeze({ geom, center: Object.freeze(lo.map((v, a) => (v + hi[a]) / 2)), half: Object.freeze(lo.map((v, a) => (hi[a] - v) / 2)) });
}

/** Minimum predicted trigger-body distance over the previewed substeps; `samples` = [{body, metricM}] from onSubstepData. */
export function holdOnsetDecision(samples, { onsetMetricM = TEACHER_DESCENT_HOLD_LIMITS.onsetMetricM } = {}) {
  if (!Array.isArray(samples) || !samples.length) return { trigger: false, reason: 'no_proximity_samples', metricM: null, body: null };
  let best = null;
  for (const s of samples) { if (!Number.isFinite(s?.metricM)) return { trigger: false, reason: 'nonfinite_proximity_sample', metricM: null, body: null }; if (!best || s.metricM < best.metricM) best = s; }
  return { trigger: best.metricM < onsetMetricM, reason: best.metricM < onsetMetricM ? 'predicted_knee_hip_box_proximity' : null, metricM: best.metricM, body: best.body };
}

/** Owned bounded hold: the carry parent keeps ownership and its source clock is held by the caller for each committed hold control. */
export class TeacherDescentContactHold {
  #ended = null; #records = [];
  constructor({ parent, episode, requestId, physicalControl, segmentIndex, referenceIndex, frame, trigger, holdControls = TEACHER_DESCENT_HOLD_LIMITS.holdControls }) {
    if (!parent || parent.phase !== 'teacher' || !count(episode) || !count(requestId) || requestId === 0 || !count(physicalControl)
        || !count(segmentIndex) || !count(referenceIndex) || !finite(frame, 747) || !trigger || trigger.trigger !== true
        || !Number.isInteger(holdControls) || holdControls < 1) throw new Error('Teacher descent hold requires an owned teacher carry, a live-FK frame and a positive trigger');
    this.parent = parent; this.episode = episode; this.requestId = requestId; this.segmentIndex = segmentIndex; this.referenceIndex = referenceIndex;
    this.startControl = this.lastControl = physicalControl; this.frame = Float32Array.from(frame); this.trigger = structuredClone(trigger); this.holdControls = holdControls;
  }
  get active() { return this.#ended === null; }
  get controls() { return this.lastControl - this.startControl; }
  get ended() { return this.#ended ? structuredClone(this.#ended) : null; }
  get referenceFrames() { return [this.frame, this.frame]; }
  isOwnedBy({ parent, episode, requestId, latestRequestId, queuedRequestId } = {}) {
    return parent === this.parent && episode === this.episode && requestId === this.requestId && latestRequestId === this.requestId
      && (queuedRequestId === null || queuedRequestId === undefined) && parent.phase === 'teacher' && !parent.finishRequested && !parent.cancelRequested
      && parent.segmentIndex === this.segmentIndex && parent.referenceIndex === this.referenceIndex;
  }
  #finish(reason, fields = {}) { if (!this.#ended) this.#ended = Object.freeze({ reason, atControl: this.lastControl, heldControls: this.controls, ...fields }); return this.ended; }
  /** Before inference: the hold either continues (frames supplied), or ends by budget (resume) or ownership loss. */
  observe(context) {
    if (!this.active) return { active: false, ...this.ended };
    if (!this.isOwnedBy(context)) return this.#finish(context?.episode !== this.episode ? 'episode_changed' : 'owner_or_clock_changed');
    if (context.physicalControl !== this.lastControl) return this.#finish('physical_clock_skipped');
    if (this.controls >= this.holdControls) return this.#finish('hold_budget_exhausted', { resume: true });
    return { active: true, remainingControls: this.holdControls - this.controls, referenceFrames: this.referenceFrames };
  }
  /** The hold action's own hard preview refused: end the hold; the caller keeps the ordinary teacher path (no cancel). */
  refuseUnsafeHold(preview, context) {
    if (!this.active) return this.ended;
    if (preview?.supported !== false || typeof preview.reason !== 'string') throw new Error('An unsupported hold preview is required');
    return this.#finish('hold_action_previewed_unsafe', { previewReason: preview.reason, resume: true });
  }
  /** After the real 17 substeps committed against the held frame. */
  commit({ physicalControl, preview, physicsSubsteps, ...context }) {
    if (!this.active || !this.isOwnedBy(context)) throw new Error('Cannot commit an inactive or differently owned hold');
    if (physicalControl !== this.lastControl + 1 || physicsSubsteps !== 17 || preview?.supported !== true) throw new Error('A hold control commits exactly one previewed real control');
    this.#records.push({ physicalControl, preview: structuredClone(preview) }); this.lastControl = physicalControl;
    if (this.controls >= this.holdControls) this.#finish('hold_budget_exhausted', { resume: true });
  }
  review() { return { startControl: this.startControl, controls: this.controls, holdControls: this.holdControls, trigger: structuredClone(this.trigger), ended: this.ended, records: structuredClone(this.#records) }; }
}
