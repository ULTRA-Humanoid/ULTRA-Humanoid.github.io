// Experimental request-time choice of complete one-pick carry styles.
// Geometry establishes availability, not physical task success. This module
// is separate from the unconnected physically-qualified whole-plan sampler.
import { planMixedCarry } from './mixed_carry_planner.js';
import { planCarryGoalRegion } from './carry_goal_region_planner.js';

const STYLE_IDS = Object.freeze(['staged', 'medium_1224']);
const issuedProposals = new WeakMap();
const validPosition = p => p?.length === 3 && Array.from(p).every(Number.isFinite);
const samePosition = (a, b) => validPosition(a) && validPosition(b) && Array.from(a).every((v, i) => v === b[i]);
const uint32 = n => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff;
const identity = n => Number.isSafeInteger(n) && n >= 0;

export function readCarryStyleSampling(params, { libraryEnabled, studentLiftPreviewEnabled = false }) {
  if (typeof params?.get !== 'function') throw new Error('Page query parameters are required');
  if (libraryEnabled !== true || studentLiftPreviewEnabled !== false || params.get('carryStylePreview') !== 'sample') return null;
  const value = params.get('carryStyleSeed') ?? '1213';
  if (!/^(0|[1-9][0-9]{0,9})$/.test(value) || !uint32(Number(value))) return null;
  return Object.freeze({ mode: 'sample', seed: Number(value), experimental: true });
}

/** Evaluate both original complete candidates at the same synchronous state.
 * The ordinary full-library plan is computed first by the caller and retained
 * verbatim when neither one-pick style is available. Nothing retries after a
 * physical miss, changes the original click, or re-plans a sampled result.
 */
export function planCarryStyleChoices({ requestToken, initialObjectPositionWorld, originalGoalWorld, library,
  baselinePlan, checkPlan, goalRegionEnabled }) {
  if (!validPosition(initialObjectPositionWorld) || !validPosition(originalGoalWorld)
      || !samePosition(baselinePlan?.requestedGoalWorld, originalGoalWorld)
      || !samePosition(baselinePlan?.initialObjectPositionWorld, initialObjectPositionWorld)
      || !samePosition(requestToken?.originalGoalWorld, originalGoalWorld)
      || !Array.isArray(baselinePlan?.segments)
      || typeof checkPlan !== 'function' || !Array.isArray(library)) throw new Error('An owned baseline plan and synchronous geometry checker are required');
  if (library.some(candidate => candidate.skill?.objectBodyName !== requestToken.objectBodyName)
      || baselinePlan.segments.some(segment => segment.skill?.objectBodyName !== requestToken.objectBodyName)) throw new Error('Plans must retain the request object');
  const choices = [], evaluatedStyles = [];
  for (const candidateId of STYLE_IDS) {
    const candidates = library.filter(candidate => candidate.id === candidateId);
    if (candidates.length !== 1) throw new Error('Each sampled style requires its original complete candidate');
    let plan = planMixedCarry(initialObjectPositionWorld, originalGoalWorld, candidates, { maxSegments: 1, checkPlan });
    if (goalRegionEnabled === true) {
      const region = planCarryGoalRegion(initialObjectPositionWorld, originalGoalWorld, candidates,
        { maxSegments: 1, checkPlan, exactFallbackPlan: plan.supported ? plan : null });
      if (region.supported && region.nominalFinalGoalResidualM <= .01 + 1e-12) plan = region;
    }
    evaluatedStyles.push(Object.freeze({ candidateId, supported: plan.supported, reason: plan.reason,
      checkedPlans: plan.checkedPlans ?? null, pickupCount: plan.segments.length,
      plannedOriginalGoalResidualM: plan.nominalFinalGoalResidualM ?? 0,
      approachCostM: plan.score?.approachCostM ?? null }));
    if (plan.supported) {
      if (plan.segments.length !== 1 || plan.segments[0].candidateId !== candidateId
          || plan.segments[0].skill !== candidates[0].skill || !samePosition(plan.requestedGoalWorld, originalGoalWorld)) {
        throw new Error('A style choice must retain its one original source and requested goal');
      }
      choices.push(Object.freeze({ candidateId, plan }));
    }
  }
  const proposal = Object.freeze({ baselinePlan, choices: Object.freeze(choices), evaluatedStyles: Object.freeze(evaluatedStyles),
    physicalPlacementSuccessPredicted: false });
  issuedProposals.set(proposal, requestToken);
  return proposal;
}

import { DEFAULT_OBJECT_BODY } from './object_profiles.js';

/** One draw is reserved synchronously for every explicit-destination carry,
 * including later cancellation/refusal or an already-placed outcome. Queued
 * re-entry reuses its token. Reset revokes ownership but preserves the stream;
 * reloading the same seed and replaying the command/reset history reproduces it.
 */
export class CarryStyleRequestSampler {
  constructor({ seed, episodeVersion, objectBodyName = DEFAULT_OBJECT_BODY }) {
    if (!uint32(seed) || !identity(episodeVersion)) throw new Error('A uint32 seed and episode are required');
    if (typeof objectBodyName !== 'string' || !objectBodyName.length) throw new Error('An explicit request object is required');
    this.initialSeed = seed; this.nextSeed = seed; this.episodeVersion = episodeVersion;
    this.objectBodyName = objectBodyName;
    this.reservationCount = 0; this.pending = new Map(); this.records = [];
  }

  reserve({ episodeVersion, requestId, kind, originalGoalWorld }) {
    if (kind !== 'carry' || originalGoalWorld === null) return null;
    if (episodeVersion !== this.episodeVersion || !identity(requestId) || !validPosition(originalGoalWorld)) throw new Error('A current owned carry request is required');
    const existing = this.pending.get(requestId);
    if (existing) {
      if (existing.status !== 'reserved' || !samePosition(existing.token.originalGoalWorld, originalGoalWorld)) throw new Error('Queued reuse requires the same pending goal and owner');
      return existing.token;
    }
    this.discardPending('newer_carry_request');
    // Mulberry32, independent of diagnostic request numbers and student noise.
    const seedBefore = this.nextSeed;
    this.nextSeed = (seedBefore + 0x6D2B79F5) >>> 0;
    let value = this.nextSeed; value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    const draw = ((value ^ value >>> 14) >>> 0) / 4294967296;
    const token = Object.freeze({ episodeVersion, requestId, objectBodyName: this.objectBodyName, reservation: ++this.reservationCount,
      seedBefore, seedAfter: this.nextSeed, draw, originalGoalWorld: Object.freeze(Array.from(originalGoalWorld)) });
    const record = { token, status: 'reserved', selection: null, discardReason: null };
    this.pending.set(requestId, record); this.records.push(record);
    if (this.records.length > 64) this.records.shift();
    return token;
  }

  select(token, proposal) {
    const record = token && this.pending.get(token.requestId);
    if (!record || record.token !== token || record.status !== 'reserved' || token.episodeVersion !== this.episodeVersion) throw new Error('Only the current reserved request may select once');
    if (issuedProposals.get(proposal) !== token
        || !samePosition(proposal?.baselinePlan?.requestedGoalWorld, token.originalGoalWorld)
        || !Array.isArray(proposal.choices) || proposal.choices.length > 2
        || proposal.physicalPlacementSuccessPredicted !== false) throw new Error('Original-goal geometric choices are required');
    let lastStyle = -1;
    for (const choice of proposal.choices) {
      const style = STYLE_IDS.indexOf(choice.candidateId);
      if (style <= lastStyle || choice.plan?.supported !== true || choice.plan.segments?.length !== 1
          || choice.plan.segments[0].candidateId !== choice.candidateId
          || !samePosition(choice.plan.requestedGoalWorld, token.originalGoalWorld)) throw new Error('Choices must be distinct supported original one-pick styles in fixed order');
      lastStyle = style;
    }
    const count = proposal.choices.length;
    const chosen = count ? proposal.choices[Math.floor(token.draw * count)] : null;
    const plan = chosen?.plan ?? proposal.baselinePlan;
    const diagnostic = Object.freeze({ ...token, reason: count > 1 ? 'sampled_style' : count === 1 ? 'only_one_supported_style' : 'no_style_choice',
      sampled: count > 1, drawUsedForChoice: count > 1, supportedStyleCount: count,
      selectedStyleId: chosen?.candidateId ?? null, selectedProbability: count ? 1 / count : null,
      selectedSourceIds: Object.freeze(plan.segments.map(segment => segment.candidateId)),
      baselinePlanRetained: count === 0, evaluatedStyles: proposal.evaluatedStyles,
      physicalPlacementSuccessPredicted: false });
    record.status = 'selected'; record.selection = diagnostic;
    return Object.freeze({ plan, diagnostic });
  }

  discard(requestId, reason) {
    const record = this.pending.get(requestId);
    if (record?.status === 'reserved') { record.status = 'discarded'; record.discardReason = reason; }
  }

  discardPending(reason) {
    for (const requestId of this.pending.keys()) this.discard(requestId, reason);
  }

  resetEpisode(episodeVersion) {
    if (!identity(episodeVersion) || episodeVersion <= this.episodeVersion) throw new Error('Reset requires a new episode');
    this.discardPending('episode_reset'); this.pending.clear(); this.episodeVersion = episodeVersion;
  }

  snapshot() {
    return { initialSeed: this.initialSeed, nextSeed: this.nextSeed, reservationCount: this.reservationCount,
      episodeVersion: this.episodeVersion, records: this.records.map(record => ({ ...record.token,
        status: record.status, discardReason: record.discardReason, selection: record.selection })) };
  }
}
