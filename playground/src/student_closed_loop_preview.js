// Opt-in STUDENT-only closed-loop lookahead (experimental, headless evaluation path; not a browser real-time claim).
// From the live state it applies the already-inferred candidate action, then for every further control it rebuilds the
// body observation on PRIVATE physics with a PRIVATE copy of the observation history, samples the finite student window
// at the future clock (read-only window copy), queries the stateless policy, evolves the private action/target history
// and steps private physics through the existing ControlPreview (same PD targets, substeps and hard fall/contact rules).
// Nothing live is written: live MjData, the live body-history builder, the live window/controllers, targetQ/lastAction,
// selected reference and the episode latent are only read. Only the existing hard preview criteria can refuse; any
// internal error is reported as `status:'error'` and never authorizes or refuses an action by itself.
import { BoundedStageGoalWindow } from './stage_goal.js';

const finiteVector = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);

export const STUDENT_CLOSED_LOOP_PREVIEW_HORIZON = 32;
// Only these nested ControlPreview reasons are PHYSICAL refusals of the forecast. Every other unsupported result
// (preview_error, preview_disposed, preview_invalid_*, preview_timestep, preview_nonfinite_*) is an unavailable forecast.
export const PHYSICAL_PREVIEW_REFUSALS = Object.freeze(['preview_balance', 'preview_contact']);
export const CLOSED_LOOP_UNAVAILABLE_REASON = 'closed_loop_preview_unavailable';

/** The commit decision the caller must apply after a triggered lookahead. Pure; no state.
 *  - stale owner (the awaited queries outlived the command)      → cancel (bounded existing abort; nothing committed)
 *  - status 'error' / unavailable / incomplete forecast          → refuse through the existing preview-refusal fallback with an
 *                                                                  explicit fault reason (never commit on an unknown forecast)
 *  - physical refusal inside the horizon                         → refuse with the nested physical reason
 *  - full horizon legal, or legal until the window's own expiry  → commit (coverage labelled; expiry is NOT a full horizon and
 *                                                                  says nothing about the later ownership handoff) */
export function decideClosedLoopCommit({ report, oneStepPreview, ownerCurrent }) {
  if (ownerCurrent !== true) return { action: 'cancel', reason: 'command_changed_during_closed_loop_preview', preview: null };
  const closedLoop = structuredClone(report);
  const refuse = reason => ({ action: 'refuse', reason, preview: { ...oneStepPreview, supported: false, reason, closedLoop } });
  if (!report || report.status !== 'ok') return refuse(CLOSED_LOOP_UNAVAILABLE_REASON);
  if (report.refused) {
    if (!PHYSICAL_PREVIEW_REFUSALS.includes(report.reason)) return refuse(CLOSED_LOOP_UNAVAILABLE_REASON);
    return refuse(report.reason);
  }
  if (report.coverage === 'full_horizon' && report.evaluatedControls === report.horizonControls) return { action: 'commit', coverage: 'full_horizon', preview: null };
  if (report.coverage === 'window_expiry' && report.expiredAtControl !== null) return { action: 'commit', coverage: 'window_expiry', preview: null };
  return refuse(CLOSED_LOOP_UNAVAILABLE_REASON);
}

export class StudentClosedLoopPreview {
  constructor({ mujoco, model, controlPreview, createBodyObsBuilder, policy, actionDim, actionScale, pelvisId,
    horizonControls = STUDENT_CLOSED_LOOP_PREVIEW_HORIZON }) {
    if (!mujoco || !model || !controlPreview || typeof createBodyObsBuilder !== 'function' || typeof policy?.infer !== 'function'
        || !Number.isInteger(actionDim) || actionDim <= 0 || !Number.isFinite(actionScale) || !Number.isInteger(pelvisId) || pelvisId < 1
        || !Number.isInteger(horizonControls) || horizonControls < 1) throw new Error('Closed-loop preview requires the live runtime pieces');
    Object.assign(this, { mujoco, model, controlPreview, createBodyObsBuilder, policy, actionDim, actionScale, pelvisId, horizonControls });
    this.specification = mujoco.mjtState.mjSTATE_INTEGRATION.value;
    this.calls = 0; this.policyQueries = 0; this.totalElapsedMs = 0;
  }

  /** Forecast the continued student window. `window` is the live controller (read: plan/startControl/horizonControls/episode).
   * `candidateTarget`/`candidateAction` are this control's already-inferred (and one-control-previewed) action. */
  async run({ liveData, liveBodyObsBuilder, window, buildObservation, candidateTarget, candidateAction, lastAction, targetQ,
    smoothingAlpha, noise, episode, physicalControl, objectBodyId, trigger }) {
    const started = performance.now();
    const report = { status: 'ok', coverage: null, refused: false, reason: null, horizonControls: this.horizonControls, evaluatedControls: 0,
      refusedAtControl: null, expiredAtControl: null, policyQueries: 0, minRootHeightM: Infinity, minUpright: Infinity,
      peakUnwantedForceN: 0, unwantedContactCount: 0, trigger: structuredClone(trigger ?? null), startControl: physicalControl,
      final: null, elapsedMs: null };
    let privateData = null, privateBuilder = null;
    try {
      if (!finiteVector(candidateTarget, this.actionDim) || !finiteVector(candidateAction, this.actionDim)
          || !finiteVector(lastAction, this.actionDim) || !finiteVector(targetQ, this.actionDim) || !Number.isFinite(smoothingAlpha)
          || !Number.isInteger(episode) || !Number.isInteger(physicalControl) || !Number.isInteger(objectBodyId) || objectBodyId < 1
          || typeof buildObservation !== 'function' || !window?.plan) throw new Error('Closed-loop preview inputs incomplete');
      if (window.episode !== episode) throw new Error('Closed-loop preview window episode mismatch');
      // Private copies: physics state, observation history, window clock, action/target history.
      const { mujoco, model } = this;
      privateData = new mujoco.MjData(model);
      const size = mujoco.mj_stateSize(model, this.specification), buffer = new mujoco.DoubleBuffer(size);
      try {
        mujoco.mj_getState(model, liveData, buffer, this.specification);
        mujoco.mj_setState(model, privateData, Array.from(buffer.GetView()), this.specification);
      } finally { buffer.delete(); }
      privateData.qacc_warmstart.set(liveData.qacc_warmstart);
      mujoco.mj_forward(model, privateData);
      privateBuilder = this.createBodyObsBuilder();
      privateBuilder.copyHistoryFrom(liveBodyObsBuilder);
      const privateWindow = new BoundedStageGoalWindow(window.plan, { episode, physicalControl: window.startControl,
        horizonControls: window.horizonControls });
      const privateLastAction = Float32Array.from(lastAction), privateTargetQ = Float32Array.from(targetQ);
      let target = Float32Array.from(candidateTarget), action = Float32Array.from(candidateAction);
      for (let k = 1; k <= this.horizonControls; k++) {
        if (k > 1) {
          const bodyObs = privateBuilder.build(privateData, privateLastAction);
          const p = this.pelvisId, q = privateData.xquat.slice(p * 4, p * 4 + 4);
          const encoded = privateWindow.sample({ rootPositionWorld: Array.from(privateData.xpos.slice(p * 3, p * 3 + 3)),
            rootQuaternionWorld: [q[1], q[2], q[3], q[0]],
            objectPositionWorld: Array.from(privateData.xpos.slice(objectBodyId * 3, objectBodyId * 3 + 3)) },
            { episode, physicalControl: physicalControl + k - 1 });
          if (encoded.expired) { report.expiredAtControl = k; report.coverage = 'window_expiry'; break; }
          const obs = buildObservation(encoded, bodyObs);
          globalThis.__studentClosedLoopPreviewQuery = true;
          let mu;
          try { mu = await this.policy.infer(obs, noise); } finally { globalThis.__studentClosedLoopPreviewQuery = false; }
          this.policyQueries++; report.policyQueries++;
          if (!finiteVector(mu, this.actionDim)) throw new Error('Closed-loop preview policy returned invalid actions');
          const next = new Float32Array(this.actionDim), nextAction = new Float32Array(this.actionDim);
          for (let i = 0; i < this.actionDim; i++) {
            const m = Math.max(-1, Math.min(1, mu[i]));
            next[i] = smoothingAlpha * (this.actionScale * m) + (1 - smoothingAlpha) * privateTargetQ[i]; nextAction[i] = m;
          }
          target = next; action = nextAction;
        }
        const result = this.controlPreview.evaluate(privateData, target, { captureState: true });
        report.evaluatedControls = k;
        report.minRootHeightM = Math.min(report.minRootHeightM, result.minRootHeightM);
        report.minUpright = Math.min(report.minUpright, result.minUpright);
        report.peakUnwantedForceN = Math.max(report.peakUnwantedForceN, result.peakUnwantedForceN ?? 0);
        report.unwantedContactCount += result.unwantedContactCount ?? 0;
        if (!result.supported) {
          if (!PHYSICAL_PREVIEW_REFUSALS.includes(result.reason))
            throw new Error('Closed-loop preview unavailable: nested preview ' + (result.reason ?? 'without reason') + (result.error ? ': ' + result.error : ''));
          report.refused = true; report.reason = result.reason; report.refusedAtControl = k; report.coverage = 'physical_refusal';
          report.final = { rootZ: result.minRootHeightM, upright: result.minUpright }; break;
        }
        if (!result.endpoint?.integration) throw new Error('Closed-loop preview endpoint state missing');
        mujoco.mj_setState(model, privateData, result.endpoint.integration, this.specification);
        privateData.qacc_warmstart.set(this.controlPreview.lastWarmstart());
        mujoco.mj_forward(model, privateData);
        privateTargetQ.set(target); privateLastAction.set(action);
        const p = this.pelvisId, q = privateData.xquat.slice(p * 4, p * 4 + 4);
        report.final = { rootZ: privateData.xpos[p * 3 + 2], upright: 1 - 2 * (q[1] ** 2 + q[2] ** 2) };
        if (k === this.horizonControls) report.coverage = 'full_horizon';
      }
      if (report.coverage === null) throw new Error('Closed-loop preview ended without covering the horizon, a refusal or the window expiry');
    } catch (error) {
      report.status = 'error'; report.coverage = 'unavailable'; report.error = String(error?.message ?? error); report.refused = false; report.reason = null;
    } finally {
      globalThis.__studentClosedLoopPreviewQuery = false;
      privateBuilder?.dispose?.(); privateData?.delete?.();
      report.elapsedMs = performance.now() - started; this.calls++; this.totalElapsedMs += report.elapsedMs;
    }
    return report;
  }
}
