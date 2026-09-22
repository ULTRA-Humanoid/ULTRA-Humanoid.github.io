// keyboard.js — WASD / QE / R / F1 / Space / P / 1-3 handlers. Mutates a UserState
// in place.

export function attachKeyboard(userState, callbacks = {}) {
  // callbacks: { onReset, onResampleNoise, onPush, onToggleDeterministic,
  //              onSetSmoothing(alpha) }

  window.addEventListener('keydown', (e) => {
    if (callbacks.onUserCommand?.(e.code) === false) {
      e.preventDefault();
      return;
    }
    // Holding a toggle key must not repeatedly reset/resample the policy.
    if (e.repeat && !['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE'].includes(e.code)) {
      if (['Space', 'F1', 'KeyV', 'KeyG'].includes(e.code)) e.preventDefault();
      return;
    }
    if (['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE'].includes(e.code)) {
      // Direct controls take over from click-to-walk immediately.
      userState.humanGoalWorld = null;
    }
    switch (e.code) {
      case 'KeyW': userState.w = true; break;
      case 'KeyS': userState.s = true; break;
      case 'KeyA': userState.a = true; break;
      case 'KeyD': userState.d = true; break;
      case 'KeyQ': userState.q = true; break;
      case 'KeyE': userState.e = true; break;
      case 'KeyR':
        callbacks.onReset?.();
        break;
      case 'KeyP':
        callbacks.onPush?.();
        break;
      case 'F1':
        e.preventDefault();
        callbacks.onToggleDeterministic?.();
        break;
      case 'Space':
        e.preventDefault();
        // Samples one latent vector; main.js holds it across subsequent frames.
        callbacks.onResampleNoise?.();
        break;
      // 1 / 2 / 3 — toggle PD target smoothing alpha.
      //   1 = 1.0 (pass-through, matches training)
      //   2 = 0.5 (mild low-pass)
      //   3 = 0.3 (heavy low-pass)
      case 'Digit1': callbacks.onSetSmoothing?.(1.0); break;
      case 'Digit2': callbacks.onSetSmoothing?.(0.5); break;
      case 'Digit3': callbacks.onSetSmoothing?.(0.3); break;
      // V toggles RPG-style follow camera (anchors the camera target to
      // the robot's body so the view trails the humanoid).
      case 'KeyV':
        e.preventDefault();
        callbacks.onToggleFollow?.();
        break;
      // G toggles goal-translator visualization (markers at the human +
      // object target positions the policy is being told to reach).
      case 'KeyG':
        e.preventDefault();
        callbacks.onToggleGoalViz?.();
        break;
      // Esc clears click goals and object selection.
      case 'Escape':
        userState.humanGoalWorld = null;
        userState.objGoalWorld = null;
        userState.activeObjName = null;
        break;
    }
    if (['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE', 'Escape'].includes(e.code)) callbacks.onKeysChanged?.();
  });

  window.addEventListener('keyup', (e) => {
    switch (e.code) {
      case 'KeyW': userState.w = false; break;
      case 'KeyS': userState.s = false; break;
      case 'KeyA': userState.a = false; break;
      case 'KeyD': userState.d = false; break;
      case 'KeyQ': userState.q = false; break;
      case 'KeyE': userState.e = false; break;
    }
    if (['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyQ', 'KeyE'].includes(e.code)) callbacks.onKeysChanged?.();
  });

  // Browsers may never deliver keyup after focus leaves the page.
  const release = () => { userState.releaseKeys(); callbacks.onKeysChanged?.(); };
  window.addEventListener('blur', release);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) release();
  });
}
