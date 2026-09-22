// Explicit opt-in capability for one inspected complete medium reference.
// Source identity comes from this fixed library slot/URI declaration. The
// original JSON has no ancestry field; its length or display name alone is
// never sufficient to enable this capability.
import { OBJECT_PROFILES } from './object_profiles.js';

export const STUDENT_LIFT_PROFILE = Object.freeze({
  id: 'medium_1224_loaded_lift54',
  libraryId: 'medium_1224',
  referenceUrl: 'public/teacher_carry_medium_1224_reference.json',
  sourceFamily: 'omomo:sub16_largebox_026_083_080_078_080_080_080.pt',
  objectBodyName: OBJECT_PROFILES.largebox.bodyName,   // source026 clip is a largebox carry; other classes opt out via this check
  rawSourceControls: 388, rawBankRows: 408, initialStanceControls: 90,
  rawInterval: Object.freeze([133, 187]),
  preparedInterval: Object.freeze([223, 277]),
  preparedSourceControls: 478, horizonControls: 54,
  carryInterval: Object.freeze([120, 296]),
});

const wrappedSkills = new WeakMap();
const same = (a, b) => a?.length === b.length && Array.from(a).every((v, i) => v === b[i]);

/** Called only by the explicit opt-in path after the ordinary skill load.
 * The cache, original skill, and every full reference/point array stay intact.
 */
export function withStudentLiftProfile(skill, { libraryId, referenceUrl } = {}) {
  const profile = STUDENT_LIFT_PROFILE;
  if (libraryId !== profile.libraryId || referenceUrl !== profile.referenceUrl
      || skill?.objectBodyName !== profile.objectBodyName || skill.locomotionOnly
      || skill.sourceFrames !== profile.rawSourceControls || skill.frames?.length !== profile.rawBankRows
      || skill.objectPointsLocal?.length !== 256 || !same(skill.carryInterval, profile.carryInterval)
      || skill.studentTransportInterval != null || skill.studentTransportProfile != null) {
    throw new Error('Student lift preview requires its declared complete medium library reference');
  }
  if (!wrappedSkills.has(skill)) wrappedSkills.set(skill, {
    ...skill, studentTransportInterval: profile.rawInterval, studentTransportProfile: profile.id,
  });
  return wrappedSkills.get(skill);
}

export function hasPreparedStudentLiftProfile(skill) {
  return skill?.studentTransportProfile === STUDENT_LIFT_PROFILE.id
    && skill.sourceFrames === STUDENT_LIFT_PROFILE.preparedSourceControls
    && same(skill.studentTransportInterval, STUDENT_LIFT_PROFILE.preparedInterval);
}
