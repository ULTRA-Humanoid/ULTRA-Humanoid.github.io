// mujoco_loader.js — load mujoco-wasm and mount the demo scene + its mesh
// dependencies into the WASM virtual filesystem.
//
// The WASM module is a self-contained Emscripten bundle (mujoco-js@0.0.7,
// 11 MB) — no separate .wasm file is fetched, the binary is inlined.
//
// Flow:
//   1. Load the mujoco_wasm module (returns a `load_mujoco` factory).
//   2. Call factory() → mujoco module with .FS, .MjModel, .MjData, etc.
//   3. Fetch the scene XML text. Walk its <mesh file="..."/> entries.
//   4. Fetch each mesh as ArrayBuffer.
//   5. Mount everything under /working/ in the WASM FS, preserving directory
//      structure (so MJCF relative paths resolve).
//   6. Compile: mujoco.MjModel.loadFromXML('/working/g1_scene.xml').
//
// Returns { mujoco, model, data } ready for use by scene_builder.

import load_mujoco from '../public/lib/mujoco_wasm.js';

export const WORKING_DIR = '/working';

// --- Internal helpers --------------------------------------------------- //

function ensureDir(fs, absPath) {
  // Create a virtual directory recursively (idempotent).
  const parts = absPath.split('/').filter(Boolean);
  let cur = '';
  for (const p of parts) {
    cur += '/' + p;
    try { fs.mkdir(cur); } catch (_) {}
  }
}

async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} failed (${r.status})`);
  return await r.text();
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} failed (${r.status})`);
  return new Uint8Array(await r.arrayBuffer());
}

function extractMeshPaths(xmlText) {
  // Find all `file="..."` attrs on <mesh> elements. The XML is well-formed
  // (built by our Python script), so a regex over <mesh ...> is safe.
  const meshRe = /<mesh\b[^>]*\bfile\s*=\s*"([^"]+)"/g;
  const out = [];
  let m;
  while ((m = meshRe.exec(xmlText)) !== null) {
    out.push(m[1]);
  }
  return out;
}

// --- Public API --------------------------------------------------------- //

/**
 * Load mujoco-wasm + mount scene + meshes + compile model.
 *
 * @param {string} sceneUrl - URL of the scene XML (e.g. 'public/g1_scene.xml').
 * @param {function(string): void} [logFn] - optional progress logger.
 * @returns {Promise<{mujoco, model, data}>}
 */
export async function loadMujocoScene(sceneUrl, logFn = console.log) {
  logFn(`[mujoco_loader] loading WASM module (~11 MB)...`);
  const mujoco = await load_mujoco();
  logFn(`[mujoco_loader] WASM loaded.`);

  // Set up the virtual FS.
  const fs = mujoco.FS;
  try { fs.mkdir(WORKING_DIR); } catch (_) {}
  // Note: zalo's example mounts MEMFS at /working. We rely on the default
  // FS (which is already mounted) and just create directories under /working.

  // Fetch the scene XML and write to /working/g1_scene.xml.
  logFn(`[mujoco_loader] fetching scene XML from ${sceneUrl}`);
  const xmlText = await fetchText(sceneUrl);
  const sceneFilename = sceneUrl.split('/').pop();
  fs.writeFile(`${WORKING_DIR}/${sceneFilename}`, xmlText);

  // Extract mesh paths (relative to the scene URL).
  const meshPaths = extractMeshPaths(xmlText);
  logFn(`[mujoco_loader] found ${meshPaths.length} mesh references`);

  // The scene XML lives at sceneUrl (e.g. 'public/g1_scene.xml'); its assets
  // are referenced relative to that file. We resolve each mesh path against
  // the scene URL's parent.
  const sceneDirUrl = sceneUrl.substring(0, sceneUrl.lastIndexOf('/'));

  let n_fetched = 0;
  for (const mp of meshPaths) {
    const url = `${sceneDirUrl}/${mp}`;
    const bytes = await fetchBytes(url);
    // Mirror the directory structure under /working/.
    const wasmPath = `${WORKING_DIR}/${mp}`;
    const dir = wasmPath.substring(0, wasmPath.lastIndexOf('/'));
    ensureDir(fs, dir);
    fs.writeFile(wasmPath, bytes);
    n_fetched += 1;
    if (n_fetched % 16 === 0 || n_fetched === meshPaths.length) {
      logFn(`[mujoco_loader] fetched ${n_fetched}/${meshPaths.length} meshes`);
    }
  }

  // Compile the model.
  logFn(`[mujoco_loader] compiling MJCF...`);
  const model = mujoco.MjModel.loadFromXML(`${WORKING_DIR}/${sceneFilename}`);
  const data = new mujoco.MjData(model);
  logFn(
    `[mujoco_loader] compiled: nq=${model.nq}, nv=${model.nv}, ` +
    `nbody=${model.nbody}, ngeom=${model.ngeom}, nu=${model.nu}, ` +
    `nmesh=${model.nmesh}, nkey=${model.nkey}`
  );

  // Reset to default_pose keyframe if present.
  if (model.nkey > 0) {
    mujoco.mj_resetDataKeyframe(model, data, 0);
    mujoco.mj_forward(model, data);
    logFn(`[mujoco_loader] reset to keyframe 0`);
  }
  return { mujoco, model, data };
}
