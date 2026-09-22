// math.js — quaternion + rot6d helpers.
// Must produce identical numerical results to the Python reference in
// intermimic/sim2sim_vae_interactive.py (functions of the same name).
// Quaternions are xyzw throughout (matching IsaacGym / training).
// MuJoCo natively uses wxyz; convert at the boundary via wxyz_to_xyzw().

const EPS = 1e-9;

export function quatNormalize(q) {
  const n = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  const inv = 1 / Math.max(n, EPS);
  return [q[0]*inv, q[1]*inv, q[2]*inv, q[3]*inv];
}

export function wxyzToXyzw(qWxyz) {
  return [qWxyz[1], qWxyz[2], qWxyz[3], qWxyz[0]];
}

// Yaw-only inverse — mirror of torch_utils.calc_heading_quat_inv.
// Extract yaw by rotating [1,0,0] through q, then invert about z.
export function calcHeadingQuatInv(rootQuatXyzw) {
  const q = quatNormalize(rootQuatXyzw);
  const [x, y, z, w] = q;
  const fx = 1 - 2 * (y*y + z*z);
  const fy = 2 * (x*y + w*z);
  const yaw = Math.atan2(fy, fx);
  const half = -yaw * 0.5;
  return [0, 0, Math.sin(half), Math.cos(half)];
}

// Rodrigues' formula: v + 2w*(u×v) + 2u×(u×v)
export function quatRotateOne(qXyzw, v) {
  const [qx, qy, qz, qw] = qXyzw;
  const [vx, vy, vz] = v;
  // u × v
  const cx = qy*vz - qz*vy;
  const cy = qz*vx - qx*vz;
  const cz = qx*vy - qy*vx;
  // u × (u × v)
  const ccx = qy*cz - qz*cy;
  const ccy = qz*cx - qx*cz;
  const ccz = qx*cy - qy*cx;
  return [
    vx + 2*qw*cx + 2*ccx,
    vy + 2*qw*cy + 2*ccy,
    vz + 2*qw*cz + 2*ccz,
  ];
}

export function quatMulXyzw(q1, q2) {
  const [x1, y1, z1, w1] = q1;
  const [x2, y2, z2, w2] = q2;
  return [
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
  ];
}

// rot6d = first two rows of the rotation matrix (NOT quat_to_tan_norm —
// matches `_quat_to_rot6d` in interactive task L59-80).
export function quatToRot6d(qXyzw) {
  const [x, y, z, w] = qXyzw;
  const xx = x*x, yy = y*y, zz = z*z;
  const xy = x*y, xz = x*z, yz = y*z;
  const xw = x*w, yw = y*w, zw = z*w;
  return [
    1 - 2*(yy + zz),  // r00
    2*(xy - zw),       // r01
    2*(xz + yw),       // r02
    2*(xy + zw),       // r10
    1 - 2*(xx + zz),  // r11
    2*(yz - xw),       // r12
  ];
}

export function yawQuat(yawRad) {
  const half = yawRad * 0.5;
  return [0, 0, Math.sin(half), Math.cos(half)];
}
