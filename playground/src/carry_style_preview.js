// Explicit review choices for two complete dataset carries. This is not a
// physical-success predictor or an automatic sampler. Default selection keeps
// the whole library and its existing deterministic planner unchanged.
const STYLES = Object.freeze({
  lower: Object.freeze({ mode: 'lower', candidateId: 'staged', label: 'Lower carry', experimental: true }),
  higher: Object.freeze({ mode: 'higher', candidateId: 'medium_1224', label: 'Higher lift', experimental: true }),
});

/** Read once at page initialization, so a queued request cannot change style
 * during asset loading. Keep this comparison separate from the loaded-student
 * lift experiment: its measured source pairs used the ordinary teacher lift.
 */
export function readCarryStylePreview(params, { libraryEnabled, studentLiftPreviewEnabled = false }) {
  if (typeof params?.get !== 'function') throw new Error('Page query parameters are required');
  if (libraryEnabled !== true || studentLiftPreviewEnabled !== false) return null;
  const mode = params.get('carryStylePreview');
  return Object.hasOwn(STYLES, mode) ? STYLES[mode] : null;
}

/** Only candidate availability changes. The original destination, preparation,
 * complete source arrays, use limits, correction limits and clearance checker
 * remain owned by the ordinary library/planner. An unsupported style request
 * stays unsupported; it does not silently fall back to another carry height.
 */
export function selectCarryStyleCandidates(candidates, preview) {
  if (preview === null) return candidates;
  if (!preview || STYLES[preview.mode] !== preview) throw new Error('An explicit page-scoped carry style is required');
  if (!Array.isArray(candidates)) throw new Error('The complete carry library is required');
  const chosen = candidates.filter(candidate => candidate.id === preview.candidateId);
  if (chosen.length !== 1) throw new Error('The requested carry style must have exactly one complete library candidate');
  return Object.freeze(chosen);
}
