// Complete source candidates for the expanded web carry library.
// Correction budgets describe planning, not physical guarantees throughout
// each interval. Original two-reference controls remain available separately.

/** Ranking modes for the mixed planner (URL `carryRanking=`). `default` keeps
 * the pickup-count-first ordering; `excludeLong` removes `long` from plans for
 * requests within the rest of the library's reach; `reliability` ranks a
 * per-candidate failure penalty before pickup count. */
export const CARRY_RANKINGS = Object.freeze(['default', 'excludeLong', 'reliability']);
export const EXCLUDE_LONG_MAX_DISTANCE_M = 4.2; // without `long` the library covers 0.084–4.42 m (2 pickups to 2.95 m, 3 to 4.42 m)

// Teacher-phase segments finished / started per source on the frozen P100
// (`combined`, eval_results/web_historical_coverage_20260914, verify_V3.md).
// Segments that never reached the teacher phase (approach refusals,
// needs_facing) are not counted. These numbers are a planning prior measured
// on ONE panel and must be confirmed on the unseen panel before any claim.
export const CARRY_CANDIDATE_RELIABILITY = Object.freeze({
  long: Object.freeze({ finished: 4, started: 21 }),
  staged: Object.freeze({ finished: 19, started: 22 }),
  medium_1224: Object.freeze({ finished: 18, started: 21 }),
  short_0295: Object.freeze({ finished: 13, started: 18 }),
  short_0184: Object.freeze({ finished: 7, started: 9 }),
});

export function plannerRankingFor(ranking) {
  if (!CARRY_RANKINGS.includes(ranking)) throw new Error(`Unknown carry ranking ${ranking}`);
  return ranking === 'reliability' ? 'reliability' : 'default';
}

function reliabilityOf(id) {
  const record = CARRY_CANDIDATE_RELIABILITY[id];
  if (!record) return { finished: null, started: null, finishRate: null, penalty: 0 };
  const finishRate = record.finished / record.started;
  return { finished: record.finished, started: record.started, finishRate, penalty: 1 - finishRate };
}

/** Attach the measured reliability prior to every candidate and apply a
 * ranking mode's library-level adjustment. `excludeLong` sets `long` to
 * maxUses 0 when the requested distance is within reach without it; other
 * modes leave every use count unchanged. Returns a new frozen list. */
export function applyCarryRanking(candidates, { ranking = 'default', requestedDistanceM = null } = {}) {
  if (!CARRY_RANKINGS.includes(ranking)) throw new Error(`Unknown carry ranking ${ranking}`);
  if (requestedDistanceM !== null && !(Number.isFinite(requestedDistanceM) && requestedDistanceM >= 0))
    throw new Error('A requested distance must be finite and nonnegative');
  const excludeLong = ranking === 'excludeLong' && requestedDistanceM !== null && requestedDistanceM <= EXCLUDE_LONG_MAX_DISTANCE_M;
  return Object.freeze(candidates.map(candidate => {
    const reliability = reliabilityOf(candidate.id);
    const maxUses = excludeLong && candidate.id === 'long' ? 0 : candidate.maxUses;
    return Object.freeze({ ...candidate, maxUses, ranking,
      reliability: Object.freeze({ finished: reliability.finished, started: reliability.started, finishRate: reliability.finishRate,
        source: 'frozen P100 combined census; unconfirmed on the unseen panel' }),
      reliabilityPenalty: reliability.penalty,
      ...(excludeLong && candidate.id === 'long' ? { excludedForDistanceM: requestedDistanceM } : {}) });
  }));
}

export function createCarrySkillLibrary({ shortPlacement, smallPlacement, stagedCarry, longCarry, mediumPlacement = null,
  alternateCarry = null, longClips = [], longClipsSingleSegmentOnly = false, longClipsLastLegAfter = null }, { stagedInitialStanceFrames = 30, ranking = 'default', requestedDistanceM = null } = {}) {
  if (!Number.isInteger(stagedInitialStanceFrames) || stagedInitialStanceFrames < 0)
    throw new Error('A whole nonnegative initial stance duration is required');
  const candidate = (id, skill, maxCorrection, maxUses, initialStanceFrames) => {
    if (!skill || skill.locomotionOnly || !Array.isArray(skill.carryInterval)
        || skill.carryInterval.length !== 2 || !Number.isInteger(skill.sourceFrames)
        || skill.frames?.length < skill.sourceFrames + 16)
      throw new Error('Each library entry requires a complete carry source and terminal lookahead');
    return Object.freeze({ id, skill, maxCorrection, maxUses,
      carryOptions: Object.freeze({ initialStanceFrames, maxCorrection,
        warpStartFrame: skill.carryInterval[0], warpEndFrame: skill.carryInterval[1] }) });
  };
  const candidates = [
    candidate('short_0184', shortPlacement, .10, 3, 90),
    candidate('short_0295', smallPlacement, .10, 3, 90),
    candidate('staged', stagedCarry, .25, 3, stagedInitialStanceFrames),
    candidate('long', longCarry, .25, 1, 90),
  ];
  if (mediumPlacement) candidates.push(candidate('medium_1224', mediumPlacement, .25, 3, 90));
  // The alternate one-metre carry picks up from the destination side of the box
  // (pickup heading ~180 deg from its travel), so it offers a different pickup
  // pose for the same destination. Same budgets as the other one-metre sources.
  if (alternateCarry) candidates.push(candidate('alternate', alternateCarry, .25, 3, 90));
  // Opt-in longer teacher clips (M5, holdout-qualified on the current entry: sub16_010 2.96 m 5/10
  // contact-free, sub8_042 2.76 m 4/10). Same object, one pickup for 2.5–3.2 m destinations; maxUses 1
  // like `long`; never part of the default library (main.js longClipLibrary=1).
  // longClipsSingleSegmentOnly (main.js longClipSingleSegment=1): the long clip may only be the whole plan —
  // never the second leg after a short carry (P100 H078 / bundle knee-contact mechanism) and never followed by one.
  // Optional per-clip correction (`clip.maxCorrection`, default .25): mid-range clips (~0.4–0.8 m) use the short-clip
  // convention .10 so a warp never exceeds ~25 % of their travel.
  for (const clip of longClips) if (clip?.skill) candidates.push(Object.freeze({ ...candidate(clip.id, clip.skill, clip.maxCorrection ?? .25, 1, 90),
    ...(longClipsSingleSegmentOnly ? { singleSegmentOnly: true } : {}),
    // longClipsLastLegAfter (main.js longClipAfterStaged=1 → ['staged']): the long clip may be the whole plan or the
    // last leg directly after a listed clip; P100 bundle evidence: knee contacts only after short_0184 (H078).
    ...(longClipsLastLegAfter ? { lastLegAfter: [...longClipsLastLegAfter] } : {}),
    // per-clip `finalSegmentOnly` (mid clip opt-in midClipFinalOnly=1): single or final leg only, never an intermediate pickup.
    ...(clip.finalSegmentOnly ? { finalSegmentOnly: true } : {}) }));
  if (new Set(candidates.map(c => c.skill.objectBodyName)).size !== 1)
    throw new Error('Library references must manipulate the same scene object');
  return applyCarryRanking(candidates, { ranking, requestedDistanceM });
}
