// Motion data loader for the bounded pickup/setdown experiment.
import { TEACHER_REFERENCE_DIM } from './teacher_obs.js';

export async function loadTeacherSkill(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load pickup reference (${response.status})`);
  const data = await response.json();
  const finiteRow = (row, n) => Array.isArray(row) && row.length === n && row.every(Number.isFinite);
  if (typeof data.object_body !== 'string'
      || !Array.isArray(data.object_points256_local) || data.object_points256_local.length !== 256
      || !data.object_points256_local.every(row => finiteRow(row, 3))
      || !Array.isArray(data.reference_frames747)
      || !data.reference_frames747.every(row => finiteRow(row, TEACHER_REFERENCE_DIM))
      || !Number.isInteger(data.source_frames) || data.source_frames < 1
      || data.source_frames > data.reference_frames747.length - 16) {
    throw new Error('Invalid pickup reference data');
  }
  const carryInterval = data.carry_interval_frames ?? null;
  if (carryInterval !== null && (!Array.isArray(carryInterval) || carryInterval.length !== 2
      || !carryInterval.every(Number.isInteger) || carryInterval[0] < 0
      || carryInterval[0] >= carryInterval[1] || carryInterval[1] >= data.source_frames)) {
    throw new Error('Carry interval must lie within the original reference frames');
  }
  const studentTransportInterval = data.student_transport_interval_frames ?? null;
  if (studentTransportInterval !== null && (!Array.isArray(studentTransportInterval)
      || studentTransportInterval.length !== 2 || !studentTransportInterval.every(Number.isInteger)
      || studentTransportInterval[0] < 0 || studentTransportInterval[1] >= data.source_frames
      || studentTransportInterval[1] - studentTransportInterval[0] !== 90
      || data.locomotion_only === true)) {
    throw new Error('A loaded student interval must contain 90 original carry controls');
  }
  return {
    name: data.name, objectBodyName: data.object_body, sourceFrames: data.source_frames,
    objectPointsLocal: data.object_points256_local,
    frames: data.reference_frames747.map(row => Float32Array.from(row)),
    carryInterval: carryInterval === null ? null : Object.freeze(Array.from(carryInterval)),
    studentTransportInterval: studentTransportInterval === null ? null : Object.freeze(Array.from(studentTransportInterval)),
    locomotionOnly: data.locomotion_only === true,
  };
}

/** Give the teacher time to reach an initial standing pose before lifting.
 * The initial pose/contacts stay fixed, with all reference velocities zero.
 * Only reference data change; the simulated robot and box remain continuous.
 */
export function withInitialStance(skill, stanceFrames) {
  if (!Number.isInteger(stanceFrames) || stanceFrames < 0) throw new Error('Initial stance duration must be whole frames');
  if (stanceFrames === 0) return skill;
  const first = Float32Array.from(skill.frames[0]);
  for (const [start, end] of [[7, 13], [42, 71], [78, 84], [357, 591]]) first.fill(0, start, end);
  return { ...skill, sourceFrames: skill.sourceFrames + stanceFrames,
    studentTransportInterval: skill.studentTransportInterval
      ? Object.freeze(skill.studentTransportInterval.map(index => index + stanceFrames)) : null,
    frames: [...Array.from({ length: stanceFrames }, () => Float32Array.from(first)), ...skill.frames] };
}
