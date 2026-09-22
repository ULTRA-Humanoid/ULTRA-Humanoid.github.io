// Control-count estimates for a planned carry against the panel's fixed
// post-click budget. Pure arithmetic over a plan and measured medians; nothing
// here predicts physical success or changes the 10 cm placement tolerance.
//
// Every constant below was measured on the frozen P100 (`combined`,
// eval_results/web_historical_coverage_20260914/joint100_cells, see the
// scratchpad verify_V3.md census). They must be re-confirmed on the unseen
// panel before being treated as anything but a planning estimate.
export const CARRY_CONTROL_BUDGET_CONTROLS = 5820; // post-click controls per panel cell (97 s at 60 Hz)

export const CARRY_TIME_COSTS = Object.freeze({
  settlingControls: 180,          // CarryGoalSequenceController settlingSteps
  exitControls: 439,              // TeacherBoxExitController: 60 hold + 199 retreat + 180 hold
  finalStandingControls: 120,     // panel portable metric: stable controls after the final exit
  studentHandoffStanceFrames: 90, // prepareStudentHandoff prepends a 90-frame stance to later segments
  // Approach model: 180 + 600 controls per metre of predicted path length.
  // Measured first approaches: 662–3099 (median 1534 for ~2–3 m); second
  // approaches: 11–2785 (median 272 when the ≤0.5 m student window admits,
  // ~1000–2800 when recorded turns/steps run).
  approachBaseControls: 180,
  approachControlsPerMetre: 600,
  defaultFirstApproachControls: 1534,
  defaultLaterApproachControls: 1000,
});

const finiteNonnegative = value => Number.isFinite(value) && value >= 0;

function segmentSourceControls(segment, index) {
  const skill = segment.skill;
  if (!Number.isInteger(skill?.sourceFrames) || skill.sourceFrames < 1) throw new Error('Each planned segment needs a complete source length');
  const stance = segment.carryOptions?.initialStanceFrames ?? 0;
  if (!Number.isInteger(stance) || stance < 0) throw new Error('Each planned segment needs a whole nonnegative initial stance');
  // Later segments receive the student-handoff stance whenever an exit completed
  // before them; the measured staged carries ran 484 (not 424) in that position.
  const applied = index === 0 ? stance : Math.max(stance, CARRY_TIME_COSTS.studentHandoffStanceFrames);
  return skill.sourceFrames + applied;
}

/** Estimate the controls a plan consumes from its first approach to the
 * panel's stable ending. `approachCostM` is the planner's predicted total
 * approach path length (all segments); when it is unavailable the measured
 * medians are used instead. */
export function estimateCarryPlanControls(plan, { approachCostM = null } = {}) {
  const segments = Array.isArray(plan?.segments) ? plan.segments : null;
  if (!segments || !segments.length || segments.length > 3) throw new Error('One to three planned segments are required for a time estimate');
  if (approachCostM !== null && !finiteNonnegative(approachCostM)) throw new Error('Approach path length must be finite and nonnegative');
  const costs = CARRY_TIME_COSTS, count = segments.length;
  const sourceControls = segments.reduce((sum, segment, index) => sum + segmentSourceControls(segment, index), 0);
  const transitionControls = (count - 1) * (costs.settlingControls + costs.exitControls);
  const endingControls = costs.settlingControls + costs.exitControls + costs.finalStandingControls;
  const approachControls = approachCostM !== null
    ? count * costs.approachBaseControls + Math.round(costs.approachControlsPerMetre * approachCostM)
    : costs.defaultFirstApproachControls + (count - 1) * costs.defaultLaterApproachControls;
  return Object.freeze({ pickupCount: count, sourceControls, transitionControls, endingControls, approachControls,
    approachEstimate: approachCostM !== null ? 'predicted_path_length' : 'measured_medians',
    approachCostM, totalControls: sourceControls + transitionControls + endingControls + approachControls });
}

/** Compare an estimate against the controls remaining before the post-click
 * deadline. `fits` is an estimate; `budgetRisk` is its negation and is meant
 * to be recorded, never used to add a refusal at request time. */
export function assessCarryPlanBudget(plan, remainingControls, { approachCostM = null } = {}) {
  if (!Number.isInteger(remainingControls) || remainingControls < 0) throw new Error('Remaining controls must be a whole nonnegative count');
  const estimate = estimateCarryPlanControls(plan, { approachCostM });
  const fits = estimate.totalControls <= remainingControls;
  return Object.freeze({ ...estimate, estimatedControls: estimate.totalControls, remainingControls,
    marginControls: remainingControls - estimate.totalControls, fits, budgetRisk: !fits });
}

/** Controls remaining for a request whose `received` event happened at
 * `requestControlStep`, measured on the same control clock. */
export function remainingCarryControls({ requestControlStep, currentControlStep, budgetControls = CARRY_CONTROL_BUDGET_CONTROLS }) {
  if (![requestControlStep, currentControlStep, budgetControls].every(Number.isInteger) || currentControlStep < requestControlStep || budgetControls < 1)
    throw new Error('A whole request clock, a later current clock and a positive budget are required');
  const elapsedControls = currentControlStep - requestControlStep;
  return Object.freeze({ budgetControls, requestControlStep, currentControlStep, elapsedControls,
    deadlineControlStep: requestControlStep + budgetControls, remainingControls: Math.max(0, budgetControls - elapsedControls) });
}
