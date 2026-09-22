/**
 * Bounded no-reset retarget of an already loaded carry reference.
 *
 * The caller supplies the existing commonTranslationWarp implementation.  No
 * simulator state, controller clock, policy history, action, grasp identity or
 * reference source is replaced.  This helper is opt-in and only operates on
 * the active final segment while authoritative live contact says the selected
 * object remains grasped.
 */
const REF_DIM = 747;
const OBS_DIM = 4052;
const ACTION_DIM = 29;
const finite = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);
const same = (a, b) => a?.length === b?.length && Array.from(a).every((v, i) => v === b[i]);
const freeze = value => {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze); Object.freeze(value);
  }
  return value;
};
const frameDelta = (a, b, indices) => Math.max(...indices.map(i => Math.abs(a[i] - b[i])));
const bodyPositionChannels = Object.freeze(Array.from({ length: 39 * 3 }, (_, i) => 84 + i));
const rotationChannels = Object.freeze([
  ...Array.from({ length: 4 }, (_, i) => 3 + i),
  ...Array.from({ length: 4 }, (_, i) => 74 + i),
  ...Array.from({ length: 39 * 4 }, (_, i) => 201 + i),
]);
const velocityChannels = Object.freeze([
  ...Array.from({ length: 3 }, (_, i) => 7 + i),
  ...Array.from({ length: 3 }, (_, i) => 78 + i),
  ...Array.from({ length: 39 * 3 }, (_, i) => 357 + i),
]);

function clonePlanForGoal(sequence, goal) {
  const plan = sequence.plan;
  if (!plan || !Array.isArray(plan.goals) || sequence.segmentIndex !== plan.goals.length - 1) {
    throw new Error('No-reset retarget requires the active final carry segment');
  }
  const goals = plan.goals.map((entry, i) => Object.freeze(i === sequence.segmentIndex
    ? Array.from(goal) : Array.from(entry)));
  const replacement = { ...plan, requestedGoalWorld: Object.freeze(Array.from(goal)), goals: Object.freeze(goals) };
  if (Array.isArray(plan.segments)) {
    replacement.segments = Object.freeze(plan.segments.map((segment, i) => i === sequence.segmentIndex
      ? Object.freeze({ ...segment, goalWorld: Object.freeze(Array.from(goal)) }) : segment));
    replacement.plannedFinalGoalWorld = Object.freeze(Array.from(goal));
  }
  if (finite(plan.initialObjectPositionWorld, 3)) replacement.distanceM = Math.hypot(
    goal[0] - plan.initialObjectPositionWorld[0], goal[1] - plan.initialObjectPositionWorld[1]);
  return Object.freeze(replacement);
}

/**
 * Apply a quintic common translation to only the unexecuted portion of the
 * active world reference.  The current frame has zero position and velocity
 * offset; the ramp endpoint has the full offset and zero added velocity.
 */
export function spliceLoadedCarryGoal({ sequence, requestedGoalWorld, objectBodyName,
  controllerOwner, expectedControllerOwner, teacherBuilder, expectedTeacherBuilder,
  commandGeneration, expectedCommandGeneration, historyGeneration, expectedHistoryGeneration,
  graspRetained, loadedPhase, commonTranslationWarp, publish = true }) {
  if (!sequence?.child || sequence.phase !== 'teacher' || sequence.child.phase !== 'teacher'
      || sequence.child !== controllerOwner || controllerOwner !== expectedControllerOwner
      || teacherBuilder !== expectedTeacherBuilder || commandGeneration !== expectedCommandGeneration
      || historyGeneration !== expectedHistoryGeneration) {
    throw new Error('Current controller, teacher history owner and generations are required');
  }
  if (loadedPhase !== 'teacher_loaded_transport' || graspRetained !== true) {
    throw new Error('Authoritative retained grasp in loaded transport is required');
  }
  const child = sequence.child, skill = child.skill;
  if (objectBodyName !== skill.objectBodyName || objectBodyName !== sequence.rawSkill.objectBodyName) {
    throw new Error('Selected object identity cannot change inside one grasp');
  }
  if (!finite(requestedGoalWorld, 3) || typeof commonTranslationWarp !== 'function') {
    throw new Error('Finite goal and existing common translation warp are required');
  }
  if (!Array.isArray(child.worldFrames) || child.worldFrames.length < skill.sourceFrames + 16
      || child.worldFrames.some(row => !finite(row, REF_DIM))) {
    throw new Error('Complete active 747-channel reference with +16 lookahead is required');
  }
  const sourceFrames = skill.sourceFrames, startFrame = child.referenceIndex;
  const endFrame = Math.min(child.warpEndFrame, sourceFrames - 1);
  if (!Number.isInteger(startFrame) || !Number.isInteger(endFrame)
      || startFrame < child.warpStartFrame || endFrame - startFrame < 16) {
    throw new Error('Retarget needs at least sixteen unexecuted loaded ramp controls before descent');
  }
  const oldFrames = child.worldFrames;
  const oldEndpoint = Array.from(oldFrames[sourceFrames - 1].slice(71, 74));
  const oldRequestedGoalWorld = Array.from(sequence.requestedGoalWorld);
  // A task goal is the desired resting object-origin position, while the
  // reference tail is not required to be the stable setdown sample (and can
  // retain a small vertical residual).  Planarity therefore means preserving
  // the request Z and the existing reference-tail residual, not forcing the
  // request Z to equal the tail Z.  The mid-carry update is the delta between
  // old and new intent; applying newGoal-oldEndpoint would silently consume an
  // existing planner residual as an additional correction.
  if (Math.abs(requestedGoalWorld[2] - oldRequestedGoalWorld[2]) > 1e-5) {
    throw new Error('Loaded retarget is planar and cannot change setdown height');
  }
  const displacement = [requestedGoalWorld[0] - oldRequestedGoalWorld[0],
    requestedGoalWorld[1] - oldRequestedGoalWorld[1], 0];
  const priorCorrection = finite(child.referencePlan?.correctionWorld, 3)
    ? Array.from(child.referencePlan.correctionWorld) : [0, 0, 0];
  const cumulativeCorrection = priorCorrection.map((v, i) => v + displacement[i]);
  if (!Number.isFinite(child.maxCorrection)
      || Math.hypot(cumulativeCorrection[0], cumulativeCorrection[1]) > child.maxCorrection + 1e-10) {
    throw new Error('Retarget exceeds the existing total carry correction bound');
  }
  // Validate parent plan before producing or publishing the replacement bank.
  const newPlan = clonePlanForGoal(sequence, requestedGoalWorld);
  const warped = commonTranslationWarp(oldFrames, displacement, startFrame, endFrame);
  if (!Array.isArray(warped?.frames) || warped.frames.length !== oldFrames.length) {
    throw new Error('Existing common translation warp returned an invalid frame bank');
  }
  const newFrames = warped.frames;
  // The boundary frame must be exactly unchanged, including rotations and all
  // velocity fields.  This also guards against a substituted warp helper.
  if (!same(newFrames[startFrame], oldFrames[startFrame])) {
    throw new Error('Retarget changed the splice boundary frame');
  }
  if (frameDelta(newFrames[endFrame], oldFrames[endFrame], rotationChannels) !== 0
      || frameDelta(newFrames[startFrame], oldFrames[startFrame], velocityChannels) !== 0
      || frameDelta(newFrames[endFrame], oldFrames[endFrame], velocityChannels) !== 0) {
    throw new Error('Retarget changed rotation or endpoint velocity continuity');
  }
  const oldRelative = bodyPositionChannels.map((channel, i) => oldFrames[startFrame][channel]
    - oldFrames[startFrame][71 + i % 3]);
  const newRelative = bodyPositionChannels.map((channel, i) => newFrames[startFrame][channel]
    - newFrames[startFrame][71 + i % 3]);
  if (!same(oldRelative, newRelative)) throw new Error('Retarget changed hand/body-to-object geometry at splice');
  let relativeGeometryRemainderMaxAbs = 0, commonPositionOffsetMaxAbs = 0;
  let commonVelocityOffsetMaxAbs = 0, rotationRemainderMaxAbs = 0;
  for (let frame = startFrame; frame < newFrames.length; frame++) {
    const old = oldFrames[frame], next = newFrames[frame];
    const positionOffset = [0, 1, 2].map(axis => next[71 + axis] - old[71 + axis]);
    const velocityOffset = [0, 1, 2].map(axis => next[78 + axis] - old[78 + axis]);
    rotationRemainderMaxAbs = Math.max(rotationRemainderMaxAbs, frameDelta(next, old, rotationChannels));
    for (const base of [0, ...Array.from({ length: 39 }, (_, body) => 84 + body * 3)]) {
      for (let axis = 0; axis < 3; axis++) {
        const delta = next[base + axis] - old[base + axis];
        commonPositionOffsetMaxAbs = Math.max(commonPositionOffsetMaxAbs,
          Math.abs(delta - positionOffset[axis]));
        const oldRelativePosition = old[base + axis] - old[71 + axis];
        const newRelativePosition = next[base + axis] - next[71 + axis];
        relativeGeometryRemainderMaxAbs = Math.max(relativeGeometryRemainderMaxAbs,
          Math.abs(newRelativePosition - oldRelativePosition));
      }
    }
    for (const base of [7, ...Array.from({ length: 39 }, (_, body) => 357 + body * 3)]) {
      for (let axis = 0; axis < 3; axis++) {
        const delta = next[base + axis] - old[base + axis];
        commonVelocityOffsetMaxAbs = Math.max(commonVelocityOffsetMaxAbs,
          Math.abs(delta - velocityOffset[axis]));
      }
    }
  }
  if (rotationRemainderMaxAbs !== 0 || relativeGeometryRemainderMaxAbs > 2e-7
      || commonPositionOffsetMaxAbs > 2e-7 || commonVelocityOffsetMaxAbs > 2e-7) {
    throw new Error('Retarget did not preserve rigid body/object geometry over the remaining reference');
  }

  const oldReferencePlan = child.referencePlan;
  const newReferencePlan = {
    ...oldReferencePlan, frames: newFrames,
    requestedGoalWorld: Array.from(requestedGoalWorld),
    referenceGoalWorld: Array.from(newFrames[sourceFrames - 1].slice(71, 74)),
    remainingDistance: Math.hypot(requestedGoalWorld[0] - newFrames[sourceFrames - 1][71],
      requestedGoalWorld[1] - newFrames[sourceFrames - 1][72]),
    correctionWorld: cumulativeCorrection,
  };
  if (typeof publish !== 'boolean') throw new Error('Retarget publish mode must be boolean');
  const record = freeze({ index: sequence.loadedRetargets?.length ?? 0, objectBodyName,
    referenceIndex: startFrame, endFrame, oldRequestedGoalWorld,
    requestedGoalWorld: Array.from(requestedGoalWorld), displacementWorld: displacement,
    cumulativeCorrectionWorld: cumulativeCorrection,
    controllerIdentityPreserved: sequence.child === controllerOwner,
    teacherBuilderIdentityPreserved: teacherBuilder === expectedTeacherBuilder,
    commandGeneration, historyGeneration, physicsControlsConsumed: 0,
    positionBoundaryMaxAbs: frameDelta(newFrames[startFrame], oldFrames[startFrame],
      [0, 1, 2, 71, 72, 73, ...bodyPositionChannels]),
    rotationBoundaryMaxAbs: frameDelta(newFrames[startFrame], oldFrames[startFrame], rotationChannels),
    velocityBoundaryMaxAbs: frameDelta(newFrames[startFrame], oldFrames[startFrame], velocityChannels),
    relativeGeometryBoundaryMaxAbs: Math.max(...newRelative.map((v, i) => Math.abs(v - oldRelative[i]))),
    relativeGeometryRemainderMaxAbs, commonPositionOffsetMaxAbs,
    commonVelocityOffsetMaxAbs, rotationRemainderMaxAbs,
    published: publish, physicalGoalUpdateQualified: false });
  if (publish) {
    child.worldFrames = newFrames;
    child.requestedGoalWorld = Object.freeze(Array.from(requestedGoalWorld));
    child.referencePlan = newReferencePlan;
    sequence.plan = newPlan;
    sequence.requestedGoalWorld = Object.freeze(Array.from(requestedGoalWorld));
    sequence.loadedRetargets ??= [];
    sequence.loadedRetargets.push(record);
  }
  return { record, oldFrames, newFrames, newPlan, newReferencePlan,
    referenceFrames: [newFrames[startFrame + 1], newFrames[startFrame + 16]] };
}

/** A zero-control contract probe; buildObservation must not step physics. */
export function probeLoadedRetargetObservation({ data, beforeFrames, afterFrames,
  referenceIndex, action, torque, historyBefore, buildObservation }) {
  if (!Array.isArray(beforeFrames) || !Array.isArray(afterFrames)
      || !finite(action, ACTION_DIM) || !finite(torque, ACTION_DIM)
      || !historyBefore || typeof buildObservation !== 'function') {
    throw new Error('Complete zero-control observation inputs are required');
  }
  const qposBefore = Array.from(data.qpos), qvelBefore = Array.from(data.qvel);
  const historySnapshot = structuredClone(historyBefore);
  const refs = frames => [frames[referenceIndex + 1], frames[referenceIndex + 16]];
  const before = buildObservation(data, refs(beforeFrames), action, torque, historySnapshot);
  const after = buildObservation(data, refs(afterFrames), action, torque, historySnapshot);
  if (!finite(before, OBS_DIM) || !finite(after, OBS_DIM)
      || !same(qposBefore, data.qpos) || !same(qvelBefore, data.qvel)
      || JSON.stringify(historyBefore) !== JSON.stringify(historySnapshot)) {
    throw new Error('Zero-control retarget probe changed physics/history or broke the 4052-D contract');
  }
  let changed = 0, maxAbs = 0;
  for (let i = 0; i < OBS_DIM; i++) {
    const delta = Math.abs(after[i] - before[i]);
    if (delta !== 0) changed++; maxAbs = Math.max(maxAbs, delta);
  }
  return freeze({ observationDimension: OBS_DIM, referenceDimension: REF_DIM,
    actionDimension: ACTION_DIM, changedObservationChannels: changed,
    maxObservationAbsDelta: maxAbs, qposUnchanged: true, qvelUnchanged: true,
    historyUnchanged: true, physicsControlsConsumed: 0 });
}
