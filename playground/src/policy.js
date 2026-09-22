// policy.js — ONNX inference for the student's forward_deploy graph.
// Inputs:  obs (1, 1422), vae_noise (1, 64)
// Output:  action mu (1, 29)
//
// Uses onnxruntime-web via the WASM execution provider.

import { OBS_DIM, VAE_DIM, ACTION_DIM } from './obs_builder.js';

export class OnnxPolicy {
  constructor() {
    this.session = null;
  }

  async load(onnxUrl) {
    // ort is loaded globally from CDN in index.html.
    // For local dev: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    if (typeof ort === 'undefined') {
      throw new Error("onnxruntime-web not loaded. Add it via CDN in index.html.");
    }
    this.session = await ort.InferenceSession.create(onnxUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const inputs = this.session.inputNames;
    if (!inputs.includes('obs') || !inputs.includes('vae_noise')) {
      throw new Error(
        `ONNX inputs must include 'obs' and 'vae_noise'; got [${inputs.join(', ')}]`
      );
    }
    console.log(`[OnnxPolicy] loaded ${onnxUrl}, inputs=[${inputs.join(', ')}]`);
  }

  async infer(obs, vaeNoise) {
    if (this.session === null) throw new Error("Call load() first.");
    if (obs.length !== OBS_DIM) {
      throw new Error(`obs must be ${OBS_DIM}D, got ${obs.length}`);
    }
    if (vaeNoise.length !== VAE_DIM) {
      throw new Error(`vae_noise must be ${VAE_DIM}D, got ${vaeNoise.length}`);
    }
    const feeds = {
      obs: new ort.Tensor('float32', obs, [1, OBS_DIM]),
      vae_noise: new ort.Tensor('float32', vaeNoise, [1, VAE_DIM]),
    };
    const result = await this.session.run(feeds);
    const mu = result['mu'];   // (1, 29)
    return new Float32Array(mu.data);   // length 29
  }
}
