/** Optional bounded observations for comparing actual task starting states.
 * This module never forwards, steps, queries a policy, or restores a state.
 */
export class TaskCoverageCapture {
  constructor({readState, maximumRecords = 64}) {
    if (typeof readState !== 'function' || !Number.isSafeInteger(maximumRecords) || maximumRecords < 1)
      throw new Error('Coverage capture requires a reader and positive record limit');
    this.readState = readState;
    this.maximumRecords = maximumRecords;
    this.records = [];
    this.totalCaptured = 0;
    this.droppedRecords = 0;
    this.errors = [];
  }
  capture({stage, requestId, task, originalGoalWorld, clock}) {
    // Diagnostic failure must not reject or alter an otherwise valid command.
    try {
      if (!['request_received', 'planning_input', 'execution_start'].includes(stage))
        throw new Error('Unknown task coverage capture stage');
      const record = structuredClone({stage, requestId, task,
        originalGoalWorld: originalGoalWorld ? Array.from(originalGoalWorld) : null,
        ...clock, state: this.readState()});
      this.records.push(record);
      this.totalCaptured++;
      if (this.records.length > this.maximumRecords) {
        this.records.shift(); this.droppedRecords++;
      }
    } catch (error) {
      this.errors.push({stage, requestId, ...clock, message: String(error)});
      if (this.errors.length > 16) this.errors.shift();
    }
  }
  snapshot() {
    return structuredClone({maximumRecords: this.maximumRecords,
      totalCaptured: this.totalCaptured, droppedRecords: this.droppedRecords,
      records: this.records, errors: this.errors,
      timing: {
        request_received: 'Synchronous request registration; a queued request has not started.',
        planning_input: 'After reference loading and active-step/episode checks, before idle placement and route selection.',
        execution_start: 'After controller initialization, before its first policy query or physical control.',
      },
      scope: 'Measured task inputs, not successful coverage cells or a complete physics replay. Join request outcomes and complete endings separately.'});
  }
}

export function readTaskCoverageState({mujoco, model, data, rootId, objectIds, history, context}) {
  const pose = id => {
    const joint = model.body_jntadr[id];
    const free = joint >= 0 && model.jnt_type[joint] === mujoco.mjtJoint.mjJNT_FREE.value;
    const q = free ? model.jnt_qposadr[joint] : null, v = free ? model.jnt_dofadr[joint] : null;
    return {positionWorld: Array.from(data.xpos.slice(id * 3, id * 3 + 3)),
      quaternionWxyzWorld: Array.from(data.xquat.slice(id * 4, id * 4 + 4)),
      freeJoint: free ? {qposAddress: q, qvelAddress: v,
        positionWorld: Array.from(data.qpos.slice(q, q + 3)),
        quaternionWxyzWorld: Array.from(data.qpos.slice(q + 3, q + 7)),
        velocity: Array.from(data.qvel.slice(v, v + 6))} : null};
  };
  const specification = mujoco.mjtState.mjSTATE_INTEGRATION.value;
  const buffer = new mujoco.DoubleBuffer(mujoco.mj_stateSize(model, specification));
  try {
    mujoco.mj_getState(model, data, buffer, specification);
    return {simulationTime: data.time,
      qpos: Array.from(data.qpos), qvel: Array.from(data.qvel), ctrl: Array.from(data.ctrl),
      integration: {specification, state: Array.from(buffer.GetView())},
      qaccWarmstart: Array.from(data.qacc_warmstart),
      cachedRoot: pose(rootId),
      cachedObjects: Object.fromEntries(objectIds.map(([name, id]) => [name, pose(id)])),
      history: structuredClone(history), context: structuredClone(context),
      poseConvention: 'qpos/qvel are current generalized state; cached world body poses are the actual simulator cache, which may precede final integration. No mj_forward was called.',
      replayLimit: 'Reference assets, selected source rows, policy inputs, full contacts and future commands are separate evidence.'};
  } finally { buffer.delete(); }
}
