// Optional review-page observability. It never sends commands or writes policy,
// physics, reference clocks or controller history. No network uploads.
function installReviewSession() {
  const mode = new URLSearchParams(location.search).get('review');
  if (!['baseline', 'development', 'stable', 'student', 'hybrid', 'transport', 'library', 'lift', 'style', 'sample', 'approach', 'descent', 'recovery'].includes(mode)) return;
  const toolbar = document.getElementById('review-toolbar');
  const button = document.getElementById('download-review-session');
  if (!toolbar || !button) return;
  toolbar.hidden = false;
  document.getElementById('review-recording-note').hidden = false;
  document.getElementById('review-version-label').textContent = {
    baseline: 'Earlier restricted controls', development: 'Staged carry preview',
    stable: 'Previous working controls', student: 'Older student controls',
    hybrid: 'Student approach preview',
    transport: new URLSearchParams(location.search).get('restrictedCarryLibrary') === '0'
      ? 'Previous long carry controls' : 'Flexible carry library · development',
    library: 'Flexible carry library · development',
    lift: 'Student lift preview · experimental',
    sample: 'Carry style sampling · checking settings',
    approach: 'Pickup approach preview · experimental',
    descent: 'Placement adjustment preview · experimental',
    recovery: 'Approach recovery preview · experimental',
    style: ({ lower: 'Lower carry', higher: 'Higher lift' }[new URLSearchParams(location.search).get('carryStylePreview')]
      ?? 'Carry style') + ' preview · experimental',
  }[mode];
  const startedAt = performance.now();
  const events = [], frames = [], errors = [];
  let previousControl = null, previousEpisode = null;
  const elapsed = () => Math.round(performance.now() - startedAt);
  const recordError = error => {
    errors.push({wallTimeMs: elapsed(), ...error});
    if (errors.length > 20) errors.shift();
  };
  // Loading failures are useful reports too; exporting does not require a policy.
  button.disabled = false;
  const read = (includeDetails = false) => {
    const api = window.__interactiveDemo;
    if (!api?.getState) return null;
    try {
      const state = api.getState();
      if (mode === 'descent') document.getElementById('review-version-label').textContent =
        state.teacherDescentRematchEnabled ? 'Placement adjustment preview · experimental'
        : 'Carry library · placement adjustment inactive';
      if (mode === 'approach') document.getElementById('review-version-label').textContent =
        state.pickupFacingEntryRegionEnabled ? 'Pickup region preview · experimental'
        : state.pickupFacingApproachEnabled ? 'Pickup approach preview · experimental' : 'Carry library · approach preview inactive';
      if (mode === 'sample') {
        const sampling = state.carryStyleSampling;
        const label = document.getElementById('review-version-label');
        if (!sampling) label.textContent = 'Carry library · sampling inactive';
        else {
          const records = sampling.records.filter(record => record.episodeVersion === state.episodeVersion);
          const active = records.find(record => record.requestId === state.activeBoxTaskRequestId && record.selection);
          const latest = [...records].reverse().find(record => record.selection);
          const selected = active?.selection ?? latest?.selection;
          const pending = records.some(record => record.status === 'reserved');
          let detail = selected ? ({ staged: 'Lower carry', medium_1224: 'Higher lift' }[selected.selectedStyleId]
            ?? 'Library plan') : 'Ready';
          if (selected?.supportedStyleCount === 1) detail += ' · one available style';
          if (pending) detail += active ? ' · next choice queued' : ' · choice pending';
          label.textContent = 'Sampled carry styles · seed ' + sampling.initialSeed + ' · ' + detail + ' · experimental';
        }
      }
      return structuredClone({
        wallTimeMs: elapsed(), control: state.controlStep, episodeControl: state.episodeControlStep,
        episode: state.episodeVersion, simulationTime: state.simTime,
        paused: state.paused, skillLoading: state.skillLoading,
        phase: state.controlPhase, policy: state.policyKind, root: state.rootPosWorld,
        qpos: state.qpos, qvel: state.qvel, ctrl: state.ctrl, upright: state.uprightScore,
        objectPoses: state.objectPoses, selectedObject: state.activeObjName,
        effectiveCommand: state.effectiveCommand, queuedCommand: state.queuedCommand, queueReason: state.queueReason,
        requestedHumanGoal: state.humanGoalWorld, requestedObjectGoal: state.objGoalWorld,
        latestBoxTaskRequestId: state.latestBoxTaskRequestId, activeBoxTaskRequestId: state.activeBoxTaskRequestId,
        boxTaskRequests: includeDetails ? state.boxTaskRequests?.slice(-64) : undefined,
        taskCoverageCaptures: includeDetails ? api.getTaskCoverageCaptures?.() : undefined,
        teacherDescent: includeDetails ? api.getTeacherDescentReview?.() : undefined,
        boxTaskResults: includeDetails ? state.boxTaskResults?.slice(-64) : undefined,
        boxExitResults: includeDetails ? state.boxExitResults?.slice(-64) : undefined,
        carrySegmentResults: includeDetails ? state.carrySegmentResults : undefined,
        carrySegmentExitResults: includeDetails ? state.carrySegmentExitResults : undefined,
        controlPreview: includeDetails ? state.controlPreview : undefined,
        carryRequestClearance: includeDetails ? state.carryRequestClearance : undefined,
        carryEntryClearance: includeDetails ? state.carryEntryClearance : undefined,
        requestedCarryGoal: state.requestedCarryGoal, skillFrame: state.skillFrame,
        pickupFacingApproachEnabled: state.pickupFacingApproachEnabled,
        pickupFacingEntryRegionEnabled: state.pickupFacingEntryRegionEnabled,
        carrySegmentIndex: state.carrySegmentIndex, carrySegmentCount: state.carrySegmentCount,
        carrySegmentExitActive: state.carrySegmentExitActive,
        studentApproachEnabled: state.restrictedStudentApproachEnabled,
        studentApproachActive: state.studentApproachActive,
        studentApproachControls: state.studentApproachControls,
        studentApproachEnded: state.studentApproachEnded,
        studentApproachEntry: includeDetails ? state.studentApproachEntry : undefined,
        carrySegmentPreparations: includeDetails ? state.carrySegmentPreparations : undefined,
        studentApproaches: includeDetails ? api.getStudentApproachReview?.() : undefined,
        studentTransportEnabled: state.restrictedStudentTransportEnabled,
        longCarryEnabled: state.restrictedLongCarryEnabled,
        carryLibraryEnabled: state.restrictedCarryLibraryEnabled,
        studentLiftPreviewEnabled: state.studentLiftPreviewEnabled,
        carryStylePreview: state.carryStylePreview,
        carryStyleSampling: includeDetails ? state.carryStyleSampling : state.carryStyleSampling ? {
          initialSeed: state.carryStyleSampling.initialSeed, nextSeed: state.carryStyleSampling.nextSeed,
          reservationCount: state.carryStyleSampling.reservationCount,
          episodeVersion: state.carryStyleSampling.episodeVersion,
          latestRecord: state.carryStyleSampling.records.at(-1) ?? null,
        } : null,
        carryPlacementToleranceM: state.carryPlacementToleranceM,
        carryPlacementStatus: state.carryPlacementStatus,
        referenceStudentTurnsEnabled: state.referenceStudentTurnsEnabled,
        referenceStudentTurnControls: state.referenceStudentTurnControls,
        referenceStudentTurnEnded: state.referenceStudentTurnEnded,
        heightAwareApproachControls: state.heightAwareApproachControls,
        referenceApproachReviews: includeDetails ? api.getReferenceApproachReviews?.() : undefined,
        pickupFacingApproach: includeDetails ? api.getPickupFacingApproachReview?.() : undefined,
        recoveredFacingTurn: includeDetails ? api.getRecoveredFacingTurnReview?.() : undefined,
        terminalApproachRecovery: includeDetails ? api.getTerminalRefusalRecoveryReview?.() : undefined,
        carryReferenceSelection: includeDetails ? state.carryReferenceSelection : undefined,
        studentTransportActive: state.studentTransportActive,
        studentTransportControls: state.studentTransportControls,
        studentTransportEnded: state.studentTransportEnded,
        studentTransportEntry: includeDetails ? state.studentTransportEntry : undefined,
        studentTransports: includeDetails ? api.getStudentTransportReview?.() : undefined,
        sourceFrames: state.skillSourceFrames, boxExitClock: state.boxExitClock,
        restricted: state.restrictedControl ? {
          phase: state.restrictedControl.phase, referenceIndex: state.restrictedControl.referenceIndex,
          sourceFrames: state.restrictedControl.sourceFrames, skill: state.restrictedControl.skillName,
          reason: state.restrictedControl.reason, suspended: state.restrictedControl.suspended,
          initialStandingPending: state.restrictedControl.initialStandingPending,
          retainedKeyTerminal: state.restrictedControl.retainedKeyTerminal,
          supported: state.restrictedControl.supported, outcome: state.restrictedControl.outcome,
          effectiveGoalWorld: state.restrictedControl.effectiveGoalWorld,
        } : null,
        taskStatus: document.getElementById('task-status')?.textContent,
      });
    } catch (error) {
      recordError({message: String(error), source: 'review_state_sample'});
      return null;
    }
  };
  const record = event => {
    const wallTimeMs = elapsed(), captured = read();
    const entry = {...event, wallTimeMs,
      capturedState: captured ? {
        wallTimeMs: captured.wallTimeMs, control: captured.control, episodeControl: captured.episodeControl,
        episode: captured.episode, simulationTime: captured.simulationTime,
        phase: captured.phase, paused: captured.paused, skillLoading: captured.skillLoading,
        root: captured.root, qpos: captured.qpos, qvel: captured.qvel,
        objectPoses: captured.objectPoses, selectedObject: captured.selectedObject,
        requestedHumanGoal: captured.requestedHumanGoal,
        requestedObjectGoal: captured.requestedObjectGoal,
        requestedCarryGoal: captured.requestedCarryGoal,
        activeBoxTaskRequestId: captured.activeBoxTaskRequestId,
        latestBoxTaskRequestId: captured.latestBoxTaskRequestId,
      } : null,
      afterWallTimeMs: null, after: null};
    events.push(entry);
    if (events.length > 2000) events.shift();
    // A browser may run a microtask between capture and target event listeners.
    // A separate task samples after synchronous input handlers have completed.
    // Asynchronous reference loading can still finish later in sampled frames.
    setTimeout(() => {
      entry.afterWallTimeMs = elapsed();
      entry.after = read();
    }, 0);
  };
  const codes = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyR', 'KeyV',
    'Escape', 'Space', 'Digit1', 'Digit2', 'Digit3']);
  for (const type of ['keydown', 'keyup']) window.addEventListener(type, event => {
    if (codes.has(event.code)) record({type, code: event.code, repeat: event.repeat});
  }, {capture: true, passive: true});
  window.addEventListener('click', event => {
    if (!['mujoco_canvas', 'pickup-button', 'carry-button', 'reset-button', 'long-carry-example-button',
      'short-carry-example-button', 'medium-carry-example-button', 'mixed-carry-example-button', 'style-carry-example-button',
      'angled-carry-example-button', 'angled-medium-carry-example-button', 'entry-region-carry-example-button'].includes(event.target?.id)) return;
    const rect = event.target.getBoundingClientRect();
    record({type: 'click', target: event.target.id, button: event.button,
      normalizedPosition: [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height]});
  }, {capture: true, passive: true});
  window.addEventListener('blur', () => record({type: 'blur'}));
  document.addEventListener('visibilitychange', () => record({type: 'visibility', hidden: document.hidden}));
  window.addEventListener('error', event => {
    recordError({message: event.message, filename: event.filename, line: event.lineno});
  });
  window.addEventListener('unhandledrejection', event => {
    recordError({message: String(event.reason)});
  });
  const pruneFrames = () => {
    while (frames.length > 1200 || frames[0]?.wallTimeMs < elapsed() - 120000) frames.shift();
  };
  const sample = () => {
    pruneFrames();
    const frame = read();
    if (!frame) return;
    if (frame.control === previousControl && frame.episode === previousEpisode) return;
    previousControl = frame.control; previousEpisode = frame.episode;
    frames.push(frame);
    pruneFrames();
  };
  let timer = setInterval(sample, 100);
  window.addEventListener('pagehide', () => {
    clearInterval(timer);
    timer = null;
  });
  window.addEventListener('pageshow', event => {
    if (event.persisted && timer === null) timer = setInterval(sample, 100);
  });
  button.addEventListener('click', () => {
    pruneFrames();
    const report = {
      kind: 'humanoid_control_session', mode, query: Object.fromEntries(new URLSearchParams(location.search)),
      exportedAt: new Date().toISOString(), userAgent: navigator.userAgent,
      pageStatus: document.getElementById('status')?.textContent ?? null,
      description: 'Recent raw input events and sampled actual browser states; this is not an exact physics replay.',
      inputStateTiming: 'Input wallTimeMs and capturedState are sampled in the capture listener before synchronous input handlers; after is sampled in a later task after those handlers. Neither acknowledges asynchronous reference loading.',
      inputPoseTiming: 'capturedState retains the full pose and velocity at input capture, including before Reset or object selection. A queued request may begin later from a different state; this is not its exact execution-start capture.',
      executionPoseTiming: 'When available, current.taskCoverageCaptures separately records request registration, post-load planning input, and initialized execution start before its first control. These bounded records retain request/episode identities and actual policy history; join complete outcomes separately.',
      sampleIntervalMs: 100, maximumFrameLookbackSeconds: 120, maximumInputEvents: 2000,
      maximumBoxRequestRecords: 64,
      current: read(true), events, frames, errors,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `humanoid-${mode}-session.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
installReviewSession();
