/** Small browser-integration primitives for the default-off retarget path. */
const finiteGoal = value => value?.length === 3 && Array.from(value).every(Number.isFinite);

/**
 * The shared step boundary. Existing physics owns the writer first; otherwise
 * a zero-control retarget preview/publication must finish before one step can
 * start. Multiple callers released by the same retarget converge on the one
 * active-step promise returned by startStep.
 */
export function runStepWithRetargetBarrier({ readActiveStep, readActiveRetarget, startStep }) {
  if (typeof readActiveStep !== 'function' || typeof readActiveRetarget !== 'function'
      || typeof startStep !== 'function') throw new Error('Step barrier callbacks are required');
  const activeStep = readActiveStep();
  if (activeStep) return activeStep;
  const activeRetarget = readActiveRetarget();
  if (activeRetarget) return Promise.resolve(activeRetarget).then(() =>
    runStepWithRetargetBarrier({ readActiveStep, readActiveRetarget, startStep }));
  return startStep();
}

/**
 * A picker supplies an object-origin height derived from the object's current
 * quaternion. During an existing carry that height is not a new setdown
 * request: preserve the active requested setdown Z and use only picker XY.
 */
export function planarPickerRetargetGoal(pickerObjectGoalWorld, activeRequestedGoalWorld) {
  if (!finiteGoal(pickerObjectGoalWorld) || !finiteGoal(activeRequestedGoalWorld)) {
    throw new Error('Finite picker and active carry goals are required');
  }
  return Object.freeze([pickerObjectGoalWorld[0], pickerObjectGoalWorld[1], activeRequestedGoalWorld[2]]);
}

/** A synchronous receipt whose completion reports the real async outcome. */
export function pendingRetargetReceipt({ receiptId, goalWorld, selectedObject, operation }) {
  if (!Number.isSafeInteger(receiptId) || receiptId < 1 || !finiteGoal(goalWorld)
      || typeof selectedObject !== 'string' || !selectedObject || !operation?.then) {
    throw new Error('Retarget receipt requires id, goal, selected object, and promise');
  }
  const completion = Promise.resolve(operation).then(review => Object.freeze({
    receiptId, requestId: review?.requestId ?? null, delivered: true, retarget: true,
    disposition: review?.applied === true ? 'started' : 'refused',
    reason: review?.applied === true ? 'mid_carry_retarget_applied' : 'mid_carry_retarget_not_applied',
    goalWorld: Array.from(goalWorld), selectedObject, review,
  }), error => Object.freeze({
    receiptId, requestId: null, delivered: true, retarget: true,
    disposition: 'refused', reason: error?.message || 'mid_carry_retarget_error',
    goalWorld: Array.from(goalWorld), selectedObject, error: error?.message || String(error),
  }));
  return Object.freeze({ receiptId, requestId: null, delivered: true, retarget: true,
    disposition: 'pending', reason: 'mid_carry_retarget_pending',
    goalWorld: Object.freeze(Array.from(goalWorld)), selectedObject, completion });
}
