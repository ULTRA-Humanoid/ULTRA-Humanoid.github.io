// scene_builder.js — build a three.js scene from a compiled MuJoCo model.
//
// Walks `model.geom_*` arrays, creates a `THREE.Group` per MuJoCo body, and
// attaches one `THREE.Mesh` per geom. Supports the geom types our G1 scene
// uses: mesh, plane, sphere, box, capsule, cylinder, ellipsoid.
//
// Per-frame: call `syncBodyTransforms(data, bodyGroups)` to update group
// poses from `data.xpos[bid]` and `data.xquat[bid]`.
//
// Coordinate conventions:
//   - MuJoCo uses Z-up. three.js default is Y-up. We mount everything in a
//     parent group rotated -90° about X so that MuJoCo Z = three.js Y.
//     (This is the same approach zalo's example uses.)
//   - MuJoCo quaternion = (w, x, y, z); three.js = (x, y, z, w). Convert.

import * as THREE from 'three';

// MuJoCo geom type constants (from mjtGeom enum)
const mjGEOM_PLANE = 0;
const mjGEOM_HFIELD = 1;
const mjGEOM_SPHERE = 2;
const mjGEOM_CAPSULE = 3;
const mjGEOM_ELLIPSOID = 4;
const mjGEOM_CYLINDER = 5;
const mjGEOM_BOX = 6;
const mjGEOM_MESH = 7;
const mjGEOM_SDF = 9;

// --- Mesh extraction ---------------------------------------------------- //

function getMeshBufferGeometry(model, meshId) {
  // Extract vertex + face data for mesh `meshId` from model arrays.
  // model.mesh_vert is a flat Float32Array of all mesh vertices.
  // model.mesh_face is a flat Int32Array of all mesh face indices.
  const vertAdr = model.mesh_vertadr[meshId];
  const nVert = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const nFace = model.mesh_facenum[meshId];

  const verts = new Float32Array(nVert * 3);
  for (let i = 0; i < nVert * 3; i++) {
    verts[i] = model.mesh_vert[(vertAdr * 3) + i];
  }
  const faces = new Uint32Array(nFace * 3);
  for (let i = 0; i < nFace * 3; i++) {
    faces[i] = model.mesh_face[(faceAdr * 3) + i];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geom.setIndex(new THREE.BufferAttribute(faces, 1));
  geom.computeVertexNormals();
  return geom;
}

// --- Primitive geometries ---------------------------------------------- //

function makePrimitiveGeometry(model, geomId) {
  const t = model.geom_type[geomId];
  const sx = model.geom_size[geomId * 3 + 0];
  const sy = model.geom_size[geomId * 3 + 1];
  const sz = model.geom_size[geomId * 3 + 2];
  switch (t) {
    case mjGEOM_PLANE:
      // Plane in MJCF: infinite in xy. Use a large but finite plane.
      return new THREE.PlaneGeometry(20, 20);
    case mjGEOM_SPHERE:
      return new THREE.SphereGeometry(sx, 16, 12);
    case mjGEOM_CAPSULE: {
      // MJCF capsule: radius=sx, half-length=sy (cylinder part + 2 hemispheres).
      const cap = new THREE.CapsuleGeometry(sx, sy * 2, 4, 8);
      return cap;
    }
    case mjGEOM_ELLIPSOID:
      // Approximate with a stretched sphere.
      const e = new THREE.SphereGeometry(1, 16, 12);
      e.scale(sx, sy, sz);
      return e;
    case mjGEOM_CYLINDER:
      // MJCF cylinder: radius=sx, half-length=sy along z.
      return new THREE.CylinderGeometry(sx, sx, sy * 2, 16);
    case mjGEOM_BOX:
      return new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
    default:
      // Fallback for HFIELD / SDF / unknown.
      return new THREE.BoxGeometry(0.05, 0.05, 0.05);
  }
}

// --- Materials ---------------------------------------------------------- //

function makeMaterial(model, geomId) {
  // Try the material slot first; fall back to geom_rgba.
  const matId = model.geom_matid[geomId];
  let r, g, b, a;
  if (matId >= 0) {
    r = model.mat_rgba[matId * 4 + 0];
    g = model.mat_rgba[matId * 4 + 1];
    b = model.mat_rgba[matId * 4 + 2];
    a = model.mat_rgba[matId * 4 + 3];
  } else {
    r = model.geom_rgba[geomId * 4 + 0];
    g = model.geom_rgba[geomId * 4 + 1];
    b = model.geom_rgba[geomId * 4 + 2];
    a = model.geom_rgba[geomId * 4 + 3];
  }
  const color = new THREE.Color(r, g, b);
  const mat = new THREE.MeshStandardMaterial({
    color: color,
    roughness: 0.7,
    metalness: 0.0,
    transparent: a < 1.0,
    opacity: a,
  });
  return mat;
}

// --- Build scene -------------------------------------------------------- //

/**
 * Build a three.js scene from the compiled mujoco model.
 *
 * @returns {{root: THREE.Group, bodyGroups: THREE.Group[]}}
 *   `root` is the top-level group to add to a THREE.Scene (pre-rotated to
 *   handle MuJoCo Z-up → three.js Y-up). `bodyGroups[bid]` is the per-body
 *   group whose transform you update each frame from data.xpos/xquat.
 */
export function buildSceneFromModel(model) {
  // Top group: rotate -90° about X to convert MuJoCo's Z-up to three.js Y-up.
  const root = new THREE.Group();
  root.name = 'MuJoCoRoot';
  root.rotation.x = -Math.PI / 2;

  // One group per body. Tag each with bodyId + bodyName so the mouse
  // picker can walk up from a mesh hit to identify the body that was
  // clicked.
  const namesBlob = model.names;
  const bodyNameAdr = model.name_bodyadr;
  const readName = (offset) => {
    let s = '';
    for (let k = offset; k < namesBlob.length; k++) {
      const c = namesBlob[k];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  };
  const bodyGroups = new Array(model.nbody);
  for (let bid = 0; bid < model.nbody; bid++) {
    const g = new THREE.Group();
    const name = readName(bodyNameAdr[bid]);
    g.name = `body_${bid}_${name}`;
    g.userData = { bodyId: bid, bodyName: name };
    root.add(g);
    bodyGroups[bid] = g;
  }

  // Walk geoms; attach mesh to its body group.
  let meshCount = 0, primCount = 0;
  for (let gid = 0; gid < model.ngeom; gid++) {
    const bid = model.geom_bodyid[gid];
    const t = model.geom_type[gid];
    let geometry;
    if (t === mjGEOM_MESH) {
      const meshId = model.geom_dataid[gid];
      if (meshId < 0) continue;
      geometry = getMeshBufferGeometry(model, meshId);
      meshCount++;
    } else {
      geometry = makePrimitiveGeometry(model, gid);
      primCount++;
    }
    const material = makeMaterial(model, gid);
    const mesh = new THREE.Mesh(geometry, material);

    // Geom-local pose (relative to the body).
    const lx = model.geom_pos[gid * 3 + 0];
    const ly = model.geom_pos[gid * 3 + 1];
    const lz = model.geom_pos[gid * 3 + 2];
    mesh.position.set(lx, ly, lz);
    // geom_quat is (w, x, y, z) MuJoCo native.
    const qw = model.geom_quat[gid * 4 + 0];
    const qx = model.geom_quat[gid * 4 + 1];
    const qy = model.geom_quat[gid * 4 + 2];
    const qz = model.geom_quat[gid * 4 + 3];
    mesh.quaternion.set(qx, qy, qz, qw);

    // Plane normals: MuJoCo plane lies in xy with normal +z (object frame).
    // three.js PlaneGeometry default lies in xy with normal +z too — good.
    mesh.castShadow = (t !== mjGEOM_PLANE);
    mesh.receiveShadow = true;

    bodyGroups[bid].add(mesh);
  }
  console.log(
    `[scene_builder] built scene: ${model.nbody} bodies, ` +
    `${primCount} primitive geoms, ${meshCount} mesh geoms`
  );
  return { root, bodyGroups };
}

// --- Per-frame sync ----------------------------------------------------- //

/**
 * Update each body group's transform from data.xpos / data.xquat.
 *
 * Quaternion convention: MuJoCo data.xquat is (w, x, y, z) per body.
 * three.js Quaternion is (x, y, z, w).
 */
export function syncBodyTransforms(data, bodyGroups) {
  const xpos = data.xpos;
  const xquat = data.xquat;
  for (let bid = 1; bid < bodyGroups.length; bid++) {
    // bid=0 is the world body; we skip it (root group transform handles it).
    const g = bodyGroups[bid];
    g.position.set(xpos[bid * 3 + 0], xpos[bid * 3 + 1], xpos[bid * 3 + 2]);
    g.quaternion.set(
      xquat[bid * 4 + 1],   // x
      xquat[bid * 4 + 2],   // y
      xquat[bid * 4 + 3],   // z
      xquat[bid * 4 + 0],   // w (MuJoCo's leading component → three.js trailing)
    );
  }
}
