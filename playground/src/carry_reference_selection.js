// Select a complete carry program before any approach or pickup starts.
// Each candidate retains the original destination and its existing warp limit.
import { planCarrySegments } from './teacher_carry_sequence.js';

export function chooseCarryReference(objectPosition, goalWorld, candidates, { checkReference = null } = {}) {
  if (!Array.isArray(candidates) || !candidates.length
      || candidates.some(c => !c.id || !c.skill)
      || new Set(candidates.map(c => c.id)).size !== candidates.length
      || (checkReference !== null && typeof checkReference !== 'function')) {
    throw new Error('Distinct complete carry candidates and an optional reference check are required');
  }
  const planned = candidates.map((candidate, index) => ({ ...candidate, index,
    plan: planCarrySegments(objectPosition, goalWorld, candidate.skill,
      { maxSegments: candidate.maxSegments ?? 3 }) }));
  const intervals = planned.flatMap(c => c.plan.supportedDistanceIntervalsM.map(range => [...range]));
  const eligible = planned.filter(c => c.plan.supported).sort((a, b) =>
    a.plan.goals.length - b.plan.goals.length || a.index - b.index);
  const attempts = [];
  let firstRefusal = null;
  for (const candidate of eligible) {
    const clearance = checkReference?.(candidate) ?? null;
    const supported = !checkReference || clearance?.supported === true;
    attempts.push({ id: candidate.id, segmentCount: candidate.plan.goals.length,
      distanceM: candidate.plan.distanceM, supported, reason: supported ? null : clearance?.reason ?? 'reference_check_unavailable' });
    if (supported) return { supported: true, reason: null, id: candidate.id, skill: candidate.skill,
      plan: candidate.plan, maxSegments: candidate.maxSegments ?? 3, clearance, attempts,
      supportedDistanceIntervalsM: intervals };
    firstRefusal ??= { candidate, clearance };
  }
  return { supported: false, reason: firstRefusal ? firstRefusal.clearance?.reason ?? 'reference_check_unavailable' : 'unsupported_distance',
    id: null, skill: null, plan: firstRefusal?.candidate.plan ?? planned[0].plan,
    clearance: firstRefusal?.clearance ?? null, attempts, supportedDistanceIntervalsM: intervals };
}
