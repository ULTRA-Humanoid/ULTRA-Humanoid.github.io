// v16 presentation-only release surface.
//
// This module owns only what the player can see and select. It deliberately
// does not remove MuJoCo bodies or alter controller, arbitration, planning, or
// safety-monitor inputs: Plasticbox and Smallbox stay in the validated physics
// scene but are absent from rendering and ordinary interaction.

export const RELEASE_AVAILABLE_OBJECTS = Object.freeze([
  'active_largebox_080_080_080',
  'active_suitcase_080_080_080',
]);

export const RELEASE_HIDDEN_OBJECTS = Object.freeze([
  'active_plasticbox_080_080_080',
  'active_smallbox_080_080_080',
]);

const AVAILABLE = new Set(RELEASE_AVAILABLE_OBJECTS);
const HIDDEN = new Set(RELEASE_HIDDEN_OBJECTS);

export function releasePresentationEnabled(params) {
  return params?.get?.('presentation') !== '0';
}

export function releaseCameraMode(params, enabled = releasePresentationEnabled(params)) {
  const explicit = params?.get?.('cameraMode');
  return explicit || (enabled ? 'player' : null);
}

export function releaseInteractiveNames(allNames, enabled = true) {
  if (!enabled) return new Set(allNames);
  return new Set(Array.from(allNames).filter(name => AVAILABLE.has(name)));
}

export function applyReleaseObjectPresentation(bodyGroups, enabled = true) {
  const hidden = [];
  if (!enabled) return Object.freeze(hidden);
  for (const group of bodyGroups ?? []) {
    const name = group?.userData?.bodyName;
    if (!HIDDEN.has(name)) continue;
    group.visible = false;
    hidden.push(name);
  }
  return Object.freeze(hidden.sort());
}

export function releasePresentationContract({ enabled, cameraMode, interactiveNames, hiddenNames }) {
  const interactive = Array.from(interactiveNames ?? []).sort();
  const hidden = Array.from(hiddenNames ?? []).sort();
  return Object.freeze({
    schema: 'web_release_presentation_v1',
    enabled: Boolean(enabled),
    cameraMode,
    availableObjects: RELEASE_AVAILABLE_OBJECTS,
    hiddenObjects: RELEASE_HIDDEN_OBJECTS,
    interactiveObjects: Object.freeze(interactive),
    renderedHiddenObjects: Object.freeze(hidden),
    controllersChanged: false,
    arbitrationChanged: false,
    physicsChanged: false,
  });
}
