// Optional reference-tracking actor. Normalization and action clipping are
// contained in the exported graph, exactly as in the native teacher evaluator.
import { TEACHER_OBS_DIM } from './teacher_obs.js';

export class TeacherPolicy {
  constructor(runtime = globalThis.ort) {
    this.runtime = runtime;
    this.session = null;
  }

  async load(url) {
    if (!this.runtime) throw new Error('ONNX Runtime Web is not loaded');
    const session = await this.runtime.InferenceSession.create(url, {
      executionProviders: ['wasm'], graphOptimizationLevel: 'all',
    });
    if (session.inputNames.length !== 1 || session.inputNames[0] !== 'teacher_obs'
        || !session.outputNames.includes('action')) {
      await session.release();
      throw new Error('Expected the exported reference teacher actor');
    }
    const previous = this.session;
    this.session = session;
    await previous?.release();
  }

  async infer(observation) {
    if (!this.session) throw new Error('Reference teacher is not loaded');
    if (observation.length !== TEACHER_OBS_DIM || !observation.every(Number.isFinite)) {
      throw new Error('Teacher observation must contain 4052 finite values');
    }
    const tensor = new this.runtime.Tensor('float32', observation, [1, TEACHER_OBS_DIM]);
    const outputs = await this.session.run({ teacher_obs: tensor });
    const action = new Float32Array(outputs.action.data);
    if (action.length !== 29 || !action.every(Number.isFinite)) throw new Error('Invalid teacher action');
    return action;
  }

  async dispose() {
    const session = this.session;
    this.session = null;
    await session?.release();
  }
}
