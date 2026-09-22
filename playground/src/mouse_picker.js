// mouse_picker.js — single-button selection (Mac-friendly).
//
// Behavior:
//   * Nothing selected: hovering a SELECTABLE object body highlights it and
//     clicking it selects it (sets user.activeObjName). Clicking the ground
//     sets the human/root locomotion target.
//   * Object selected (destination mode, `selectionOwnsDestination`, the
//     default): EVERY click is a floor destination for the selected object —
//     even when the ray also hits the selected box or another selectable
//     body — using the ground-plane intersection under the pointer. The
//     cursor is a crosshair (not-allowed when the ground ray misses) and the
//     hover tint is suppressed. Selection changes only through explicit
//     actions: Esc (keyboard.js), the Deselect button or api.selectObject().
//   * `selectionOwnsDestination: false` restores the legacy toggle: a click
//     on a selectable body toggles selection (same object → deselect, other
//     object → switch) and only body-free clicks reach the ground branch.
//   * Esc (handled in keyboard.js) → clear selection + target.
//
// "Selectable object" = any MJCF body whose name appears in the
// `selectableNames` Set passed at attach time. This is populated from the
// keys of `object_pointclouds.json` (loaded by main.js), so we never
// mistakenly treat a robot link like 'left_hip_pitch_link' as selectable.
//
// Coord conversion: the three.js scene's MuJoCoRoot is rotated -π/2 about
// X so that MuJoCo z-up == three.js y-up. Raycaster hits are in WORLD
// three.js coordinates AFTER that rotation. To get the corresponding
// MuJoCo world point we apply the inverse rotation:
//     (x_mj, y_mj, z_mj) = (x_three, -z_three, y_three)
// We use the MuJoCoRoot's worldToLocal() to do this rigorously.

import * as THREE from 'three';

const HIGHLIGHT_COLOR = 0x4cc9f0;   // cyan tint
const HIGHLIGHT_EMISSIVE = 0x152a40;
const TARGET_COLOR = 0xffcc33;
const REFUSED_TARGET_COLOR = 0xe55353;

export function attachMousePicker(opts) {
  const {
    canvas,
    camera,
    scene,
    rootGroup,        // the MuJoCoRoot group (so we can convert world → MuJoCo)
    bodyGroups,
    selectableNames,  // Set<string>  — names that count as "objects"
    user,             // UserState (we mutate activeObjName / humanGoalWorld / objGoalWorld)
    onSelect,         // callback(name | null) — fires when selection changes
    objectGoalFromGround, // convert floor support point to object-frame origin
    // Destination mode (default on): with an object selected every click is a
    // floor destination; selection never toggles on click. `false` = legacy.
    selectionOwnsDestination = true,
  } = opts;

  // --- Invisible ground plane for "click on ground" raycasts ---------- //
  // Plane lives in the three.js scene (NOT under rootGroup): we want it
  // perpendicular to gravity in three.js space (which is y-up after the
  // root rotation), so it lies in the XZ plane at y=0.
  const groundGeom = new THREE.PlaneGeometry(40, 40);
  groundGeom.rotateX(-Math.PI / 2);   // XY → XZ (normal = +y)
  const groundMat = new THREE.MeshBasicMaterial({
    visible: false, side: THREE.DoubleSide,
  });
  const groundPicker = new THREE.Mesh(groundGeom, groundMat);
  groundPicker.name = 'GroundPicker';
  groundPicker.userData = { isGround: true };
  scene.add(groundPicker);

  // --- Target marker: visualizes the user's click target ------------- //
  // Lives UNDER rootGroup so we can position it in MuJoCo (z-up) coords
  // directly. A thin ring on the ground + a small vertical pillar. Refused
  // task destinations stay at the requested position and turn red.
  const targetMarker = new THREE.Group();
  targetMarker.name = 'TargetMarker';
  const ringGeom = new THREE.RingGeometry(0.18, 0.22, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: TARGET_COLOR, side: THREE.DoubleSide, transparent: true, opacity: 0.9,
    depthTest: false,
  });
  const ring = new THREE.Mesh(ringGeom, ringMat);
  // RingGeometry is in xy plane (normal +z) by default — perfect for
  // MuJoCo z-up. Lift slightly above ground (z=0.005) to avoid z-fighting.
  ring.position.z = 0.005;
  targetMarker.add(ring);
  // Pillar: thin vertical line from ring center up to ~0.6m so the user
  // can find the target even with an overhead camera angle.
  // Cylinder default axis is local Y. Inside rootGroup (MuJoCo frame),
  // we want the pillar along local Z (vertical in world). So:
  //   1. translate(0, 0.3, 0): center → bottom at y=0, top at y=0.6
  //   2. rotateX(π/2):  Y axis → Z axis (cylinder now extends along local Z)
  // After (2) the (0, 0.3, 0) offset becomes (0, 0, 0.3) — base at z=0,
  // top at z=0.6, centered on the ring's xy.
  const pillarGeom = new THREE.CylinderGeometry(0.008, 0.008, 0.6, 8);
  pillarGeom.translate(0, 0.3, 0);
  pillarGeom.rotateX(Math.PI / 2);
  const pillarMat = new THREE.MeshBasicMaterial({
    color: TARGET_COLOR, transparent: true, opacity: 0.75,
    depthTest: false,
  });
  const pillar = new THREE.Mesh(pillarGeom, pillarMat);
  targetMarker.add(pillar);
  targetMarker.visible = false;
  rootGroup.add(targetMarker);

  /**
   * Show / hide / reposition the target marker based on the active user
   * target. Called every frame via syncFromUserState() and also directly
   * after a successful ground click.
   */
  function updateTargetMarker() {
    const taskGoal = opts.getTaskGoal?.();
    const g = taskGoal || user.objGoalWorld || user.humanGoalWorld;
    // Main owns the request/episode match. A stale task refusal cannot tint
    // an ordinary walking or object target when no task goal is displayed.
    const refused = !!taskGoal && opts.getTaskGoalRefused?.(taskGoal) === true;
    const color = refused ? REFUSED_TARGET_COLOR : TARGET_COLOR;
    ringMat.color.setHex(color);
    pillarMat.color.setHex(color);
    if (g === null || g === undefined) {
      targetMarker.visible = false;
      return;
    }
    targetMarker.position.set(g[0], g[1], 0);
    targetMarker.visible = true;
  }

  // --- Raycaster + mouse vec (reused) -------------------------------- //
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  function setMouseFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    ndc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    ndc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  // --- Walk a hit mesh up to find the body group it belongs to ------- //
  function findBodyGroup(obj) {
    let p = obj;
    while (p) {
      if (p.userData && typeof p.userData.bodyName === 'string') return p;
      p = p.parent;
    }
    return null;
  }

  // --- Highlight state ----------------------------------------------- //
  // We swap material color/emissive on hover. Keep originals to restore.
  const originalMaterials = new WeakMap();   // mesh → original color/emissive
  let hoveredGroup = null;
  let selectedGroup = null;

  function applyHighlight(group, color = HIGHLIGHT_COLOR,
                          emissive = HIGHLIGHT_EMISSIVE) {
    group.traverse((node) => {
      if (!node.isMesh || !node.material) return;
      const mat = node.material;
      if (!originalMaterials.has(node)) {
        originalMaterials.set(node, {
          color: mat.color.clone(),
          emissive: mat.emissive ? mat.emissive.clone() : null,
        });
      }
      // Use a CLONED material to avoid leaking the tint to other meshes
      // sharing the same Material instance.
      if (!node.userData._origMaterial) {
        node.userData._origMaterial = mat;
        node.material = mat.clone();
      }
      node.material.color.setHex(color);
      if (node.material.emissive) node.material.emissive.setHex(emissive);
    });
  }

  function clearHighlight(group) {
    if (!group) return;
    group.traverse((node) => {
      if (!node.isMesh) return;
      if (node.userData._origMaterial) {
        node.material.dispose();
        node.material = node.userData._origMaterial;
        node.userData._origMaterial = null;
      }
    });
  }

  // --- Destination mode --------------------------------------------- //
  // While an object is selected (and `selectionOwnsDestination`), the canvas
  // shows a crosshair (not-allowed when the ground ray misses), suppresses
  // the hover tint and explains how to change selection.
  const selectionTitle = canvas.title ?? '';
  let destinationMode = false;
  let destinationGroundAvailable = null;
  function destinationModeActive() {
    return selectionOwnsDestination && user.activeObjName !== null;
  }
  function syncDestinationIntent(groundAvailable) {
    const active = destinationModeActive();
    if (!active && !destinationMode) return;
    if (groundAvailable !== undefined) destinationGroundAvailable = groundAvailable;
    if (active) {
      if (hoveredGroup && hoveredGroup !== selectedGroup) clearHighlight(hoveredGroup);
      hoveredGroup = null;
      canvas.style.cursor = destinationGroundAvailable === false ? 'not-allowed' : 'crosshair';
      canvas.title = 'Click the floor to choose a destination for the selected object. Esc or Deselect clears the selection.';
    } else {
      canvas.style.cursor = 'default';
      canvas.title = selectionTitle;
      destinationGroundAvailable = null;
    }
    destinationMode = active;
  }

  /**
   * Submit a MuJoCo floor point as the selected object's destination. This
   * is the ONLY path that turns a floor point into a carry request: the
   * same normalization (objectGoalFromGround) and the same onObjectGoal
   * callback for a mouse click and for the picker-free api path.
   */
  function submitDestination(floorPoint) {
    user.objGoalWorld = objectGoalFromGround ? objectGoalFromGround(floorPoint) : floorPoint;
    user.humanGoalWorld = null;
    opts.onObjectGoal?.(Array.from(user.objGoalWorld));
    updateTargetMarker();
  }

  // --- Hover ---------------------------------------------------------- //
  canvas.addEventListener('mousemove', (event) => {
    setMouseFromEvent(event);
    raycaster.setFromCamera(ndc, camera);
    if (destinationModeActive()) {
      syncDestinationIntent(raycaster.intersectObject(groundPicker, true).length > 0);
      return;
    }
    syncDestinationIntent();
    const hits = raycaster.intersectObject(rootGroup, true);
    let newHover = null;
    for (const h of hits) {
      const grp = findBodyGroup(h.object);
      if (grp && selectableNames.has(grp.userData.bodyName)) {
        newHover = grp;
        break;
      }
    }
    if (newHover === hoveredGroup) return;
    // Clear previous hover (unless it's the currently selected one — keep
    // selection highlight on top).
    if (hoveredGroup && hoveredGroup !== selectedGroup) {
      clearHighlight(hoveredGroup);
    }
    if (newHover && newHover !== selectedGroup) {
      applyHighlight(newHover, 0x90e0ef, 0x0a1a26);   // subtle hover tint
    }
    hoveredGroup = newHover;
    canvas.style.cursor = newHover ? 'pointer' : 'default';
  });

  // --- Click ---------------------------------------------------------- //
  let pointerDown = null;
  canvas.addEventListener('pointerdown', (event) => {
    pointerDown = [event.clientX, event.clientY];
  });
  canvas.addEventListener('click', (event) => {
    // OrbitControls also starts with a left-button press. Dragging the
    // camera must not submit a navigation goal at the release location.
    if (pointerDown && Math.hypot(event.clientX - pointerDown[0], event.clientY - pointerDown[1]) > 5) {
      pointerDown = null;
      return;
    }
    pointerDown = null;
    setMouseFromEvent(event);
    raycaster.setFromCamera(ndc, camera);

    // First check: did the click hit a selectable object body?
    const bodyHits = raycaster.intersectObject(rootGroup, true);
    let bodyGroup = null;
    for (const h of bodyHits) {
      const grp = findBodyGroup(h.object);
      if (grp && selectableNames.has(grp.userData.bodyName)) {
        bodyGroup = grp;
        break;
      }
    }

    // Always do the ground raycast too, so we can log it for debugging
    // even when no object is selected.
    const groundHits = raycaster.intersectObject(groundPicker, true);
    syncDestinationIntent(groundHits.length > 0);
    if (!bodyGroup && groundHits.length === 0) return;
    if (opts.canInteract?.() === false) return;

    // Diagnostic — comment this out once picker is stable.
    console.log(`[picker] click ndc=(${ndc.x.toFixed(2)},${ndc.y.toFixed(2)})  ` +
                `body_hits=${bodyHits.length}  ` +
                `selectable_hit=${bodyGroup ? bodyGroup.userData.bodyName : 'none'}  ` +
                `ground_hits=${groundHits.length}  ` +
                `active=${user.activeObjName || 'none'}`);

    if (bodyGroup && !destinationModeActive()) {
      // Selection toggle. In destination mode this branch is reached only
      // while nothing is selected (so it can only select, never deselect).
      const name = bodyGroup.userData.bodyName;
      if (user.activeObjName === name) {
        // Deselect
        clearHighlight(selectedGroup);
        selectedGroup = null;
        user.activeObjName = null;
        user.objGoalWorld = null;
        onSelect?.(null);
        console.log(`[picker] deselected '${name}'`);
      } else {
        // Select. Clear any previous selection highlight first.
        if (selectedGroup) clearHighlight(selectedGroup);
        selectedGroup = bodyGroup;
        applyHighlight(selectedGroup, HIGHLIGHT_COLOR, HIGHLIGHT_EMISSIVE);
        user.activeObjName = name;
        // Don't auto-set a target — wait for ground click.
        user.objGoalWorld = null;
        user.humanGoalWorld = null;
        onSelect?.(name);
        syncDestinationIntent(groundHits.length > 0);
        console.log(`[picker] selected '${name}'`);
      }
      return;
    }

    // Selected object → the click is a destination at the ground-plane
    // intersection (in destination mode even when a body was also hit; in
    // legacy mode only when no selectable body was hit).
    if (user.activeObjName !== null) {
      if (groundHits.length > 0) {
        const p = groundHits[0].point;           // three.js world coords
        // rootGroup applies rotation.x = -π/2 (three.js y-up vs MuJoCo z-up);
        // worldToLocal undoes that to give MuJoCo (x, y, z).
        const local = rootGroup.worldToLocal(p.clone());
        const floorPoint = new Float32Array([local.x, local.y, local.z]);
        submitDestination(floorPoint);
        console.log(`[picker] obj target → MuJoCo (${local.x.toFixed(2)}, ` +
                    `${local.y.toFixed(2)}, ${local.z.toFixed(2)})  ` +
                    `[three.js (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})]`);
      } else {
        console.log(`[picker] no ground intersection — ray missed plane`);
      }
    } else {
      if (groundHits.length > 0) {
        const p = groundHits[0].point;
        const local = rootGroup.worldToLocal(p.clone());
        user.humanGoalWorld = new Float32Array([local.x, local.y, local.z]);
        user.objGoalWorld = null;
        opts.onFloorGoal?.(Array.from(user.humanGoalWorld));
        updateTargetMarker();
        console.log(`[picker] walk target → MuJoCo (${local.x.toFixed(2)}, ` +
                    `${local.y.toFixed(2)}, ${local.z.toFixed(2)})  ` +
                    `[three.js (${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})]`);
      } else {
        console.log('[picker] no ground intersection — ray missed plane');
      }
    }
  });

  // --- Esc (handled by keyboard.js) — but make sure to clear highlight here too.
  // We do this by polling user.activeObjName each frame; main.js can call
  // syncFromUserState() if a UI element changes selection state out-of-band.

  function syncFromUserState() {
    const selectedName = selectedGroup?.userData.bodyName || null;
    if (user.activeObjName !== selectedName) {
      clearHighlight(selectedGroup);
      selectedGroup = user.activeObjName === null ? null
        : Object.values(bodyGroups).find(group => group.userData.bodyName === user.activeObjName) || null;
      if (selectedGroup) applyHighlight(selectedGroup, HIGHLIGHT_COLOR, HIGHLIGHT_EMISSIVE);
      onSelect?.(selectedGroup ? user.activeObjName : null);
    }
    // Keep target marker in sync with user.objGoalWorld every frame so
    // Esc/Reset (which null out objGoalWorld via keyboard.js) hide the
    // marker without an explicit hook here.
    updateTargetMarker();
    if (destinationModeActive() !== destinationMode) syncDestinationIntent();
  }

  /**
   * Picker-free destination request (automation / evaluation arm). Runs the
   * click path from the canInteract gate onward with `floorPoint` (MuJoCo
   * world, z≈0) standing in for the ground-ray intersection: no raycast, no
   * camera, identical normalization and callback. Returns whether the
   * onObjectGoal callback ran and, if not, why.
   */
  function requestDestination(floorPoint) {
    if (user.activeObjName === null) return { delivered: false, reason: 'no_selection' };
    if (opts.canInteract?.() === false) return { delivered: false, reason: 'interaction_blocked' };
    submitDestination(new Float32Array([floorPoint[0], floorPoint[1], floorPoint[2] ?? 0]));
    return { delivered: true, reason: null };
  }

  return { syncFromUserState, requestDestination,
    getDestinationModeState: () => Object.freeze({ enabled: selectionOwnsDestination, active: destinationMode,
      groundAvailable: destinationGroundAvailable, cursor: canvas.style.cursor ?? null }),
    // Copy scalar render properties for inspection without exposing mutable
    // Three objects or command state.
    getTargetMarkerState: () => Object.freeze({ visible: targetMarker.visible,
      position: Object.freeze(targetMarker.position.toArray()),
      ringColor: ringMat.color.getHex(), pillarColor: pillarMat.color.getHex() }),
  };
}
