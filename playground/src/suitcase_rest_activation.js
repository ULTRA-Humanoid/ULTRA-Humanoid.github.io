export const SUITCASE_BODY = 'active_suitcase_080_080_080';

export function shouldActivateSuitcaseRestPose({ routingEnabled, bodyName, alreadyActivated }) {
  if (typeof routingEnabled !== 'boolean' || typeof alreadyActivated !== 'boolean') {
    throw new Error('Suitcase rest activation state must be explicit');
  }
  return routingEnabled && bodyName === SUITCASE_BODY && !alreadyActivated;
}

// Preserve the selected body's constructed world yaw while changing only its
// rest axis from v15 flat (+z up) to the certified standing Suitcase (+y up).
export function standingSuitcaseQuaternionWxyz(flatLikeQuaternionWxyz) {
  if (!Array.isArray(flatLikeQuaternionWxyz) && !ArrayBuffer.isView(flatLikeQuaternionWxyz)) {
    throw new Error('Suitcase quaternion must be array-like');
  }
  const [w, x, y, z] = Array.from(flatLikeQuaternionWxyz);
  if (![w, x, y, z].every(Number.isFinite)) throw new Error('Suitcase quaternion must be finite');
  const norm = Math.hypot(w, x, y, z);
  if (Math.abs(norm - 1) > 1e-3) throw new Error('Suitcase quaternion must be normalized');
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const c = Math.cos(yaw / 2), s = Math.sin(yaw / 2), h = Math.SQRT1_2;
  return Object.freeze([c * h, c * h, s * h, s * h]);
}
