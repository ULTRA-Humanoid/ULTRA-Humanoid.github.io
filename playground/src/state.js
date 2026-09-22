// state.js — UserState + Mode enum. Mirror of Python UserState in
// intermimic/sim2sim_vae_interactive.py.

export const Mode = Object.freeze({
  IDLE: 'idle',
  LOCO: 'loco',
  HOI_FULL: 'hoi_full',
  HOI_OBJ_ONLY: 'hoi_obj_only',
});

export class UserState {
  constructor() {
    // WASD/QE held flags
    this.w = false;
    this.s = false;
    this.a = false;
    this.d = false;
    this.q = false;
    this.e = false;
    // Planar locomotion goal in world frame; z is for floor visualization.
    this.humanGoalWorld = null;     // Float32Array(3) | null
    // Object goal in world frame; null = no goal
    this.objGoalWorld = null;       // Float32Array(3) | null
    this.activeObjName = null;       // string | null

    // Episode-held VAE latent. The control loop must never resample this.
    this.vaeNoise = null;
    this.deterministic = false;       // F1 toggle
  }

  get wasdActive() {
    return this.w || this.s || this.a || this.d || this.q || this.e;
  }

  releaseKeys() {
    this.w = this.s = this.a = this.d = this.q = this.e = false;
  }

  get mode() {
    const hasHuman = this.humanGoalWorld !== null;
    const hasObj = this.objGoalWorld !== null && this.activeObjName !== null;
    if ((this.wasdActive || hasHuman) && hasObj) return Mode.HOI_FULL;
    if (this.wasdActive || hasHuman) return Mode.LOCO;
    if (hasObj) return Mode.HOI_OBJ_ONLY;
    return Mode.IDLE;
  }
}
