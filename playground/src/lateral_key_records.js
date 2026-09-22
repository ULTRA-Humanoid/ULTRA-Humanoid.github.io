// lateral_key_records.js — wire the two complete lateral teacher records to the
// A/D keys of the recorded supervisor. They enter RestrictedLocomotionController as
// `additionalKeySkills` exactly like the backward record: directional key steps pick
// the complete record whose travel direction matches the held key within 15°, the
// whole aligned record is admitted against the live box AABBs through its private
// collision sweep, and the record terminal is retained for standing.
import { bindMotionSweep } from './restricted_motion_geometry.js';

export const LATERAL_KEY_REFERENCES = Object.freeze(['lateral_left', 'lateral_right']);
export const LATERAL_SWEEP_KEYS = Object.freeze({ lateral_left: 'lateral_left_561', lateral_right: 'lateral_right_270' });
export const LATERAL_SWEEP_ASSET = 'public/restricted_lateral_sweeps.json';
export const LATERAL_SIDE_BAND_RAD = Object.freeze({ min: 75 * Math.PI / 180, max: 105 * Math.PI / 180 });

const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const yaw = q => Math.atan2(2 * (q[0] * q[1] + q[3] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2));

/** Travel geometry of a complete locomotion record: which side it steps to, relative to its own heading. */
export function describeLateralRecord(skill) {
  if (!skill || skill.locomotionOnly !== true || !Number.isInteger(skill.sourceFrames) || skill.sourceFrames < 2
      || !Array.isArray(skill.frames) || skill.frames.length < skill.sourceFrames + 16) throw new Error('A complete locomotion record is required');
  const first = skill.frames[0], last = skill.frames[skill.sourceFrames - 1];
  const sourceYawRad = yaw(first.slice(3, 7)), travelM = Math.hypot(last[0] - first[0], last[1] - first[1]);
  const travelDirectionRad = Math.atan2(last[1] - first[1], last[0] - first[0]);
  const lateralOffsetRad = wrap(travelDirectionRad - sourceYawRad);
  const magnitude = Math.abs(lateralOffsetRad);
  const side = magnitude >= LATERAL_SIDE_BAND_RAD.min && magnitude <= LATERAL_SIDE_BAND_RAD.max ? (lateralOffsetRad > 0 ? 'left' : 'right') : null;
  return { travelM, sourceYawRad, travelDirectionRad, lateralOffsetRad, side, yawChangeRad: wrap(yaw(last.slice(3, 7)) - sourceYawRad) };
}

/** Validate both records step to their named side and bind their private sweeps. */
export function bindLateralKeyRecords({ left, right }, asset) {
  const records = [];
  for (const [name, skill, key] of [['left', left, LATERAL_SWEEP_KEYS.lateral_left], ['right', right, LATERAL_SWEEP_KEYS.lateral_right]]) {
    const geometry = describeLateralRecord(skill);
    if (geometry.side !== name) throw new Error(`The ${name} lateral record must travel to the ${name} of its heading`);
    if (geometry.travelM < .1) throw new Error('Lateral records must travel at least 10 cm');
    records.push({ side: name, skill, sweep: bindMotionSweep(skill, asset, key), geometry });
  }
  return records;
}
