// smoother.js — residual + EMA action smoothing.
// target = α·(prev + scale·action) + (1−α)·prev

const DEFAULT_ALPHA = 0.7;
const DEFAULT_SCALE = 3.0;   // matches control.action_scale in env yaml

export class ActionSmoother {
  constructor(defaultPose, alpha = DEFAULT_ALPHA, scale = DEFAULT_SCALE) {
    this.defaultPose = new Float32Array(defaultPose);
    this.alpha = alpha;
    this.scale = scale;
    this.prevTarget = new Float32Array(this.defaultPose);
  }

  reset() {
    this.prevTarget.set(this.defaultPose);
  }

  update(action) {
    const n = action.length;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const raw = this.prevTarget[i] + this.scale * action[i];
      out[i] = this.alpha * raw + (1 - this.alpha) * this.prevTarget[i];
    }
    this.prevTarget.set(out);
    return out;
  }
}
