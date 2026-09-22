# Interactive G1 Demo — Browser Deployment

Browser-deployable interactive WASD + click demo on G1. The starting mode maps
commands to measured teacher references for walking, box approach, carrying
and stepping clear. It retains the interactive student for task settling and
an explicit comparison mode. MuJoCo physics and ONNX inference run client-side.

The current priority is [flexible, diverse and robust web control](../WEB_DEVELOPMENT_GOAL.md).
The [current development state](../WEB_CURRENT_STATE.md) identifies the delivered
archive and ongoing experiments.

## Architecture

The default supervisor translates commands into complete recorded teacher
references, builds the 4052D privileged observation, and executes teacher
actions through the same PD/physics loop. The diagram below describes the
1422D student path used for the explicit comparison mode and task settling.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (single page, no server)                            │
│                                                              │
│  ┌──────────────┐  state    ┌──────────────────────────┐    │
│  │ mujoco-wasm  │ ────────► │  obs_builder.js (1422D)  │    │
│  │ (physics)    │           │  - NEW_CMD synth (WASD)   │    │
│  └──────────────┘           │  - point cloud (heading)  │    │
│         ▲                   │  - mask block (3 modes)   │    │
│         │ target            └────────┬─────────────────┘    │
│         │                            │ obs                   │
│  ┌──────┴──────────────┐   action    ▼                       │
│  │ smoother.js (EMA)   │ ◄── ┌──────────────────────────┐    │
│  └─────────────────────┘     │ policy.js                │    │
│                              │ - onnxruntime-web        │    │
│                              │ - forward_deploy(obs,    │    │
│                              │   vae_noise) → action    │    │
│                              └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Development review

From the repository root, run `python3 web/serve.py` and open
`http://localhost:8000/review.html`. **Open approach recovery preview** is the
latest optional entry. It preserves ordinary accepted approaches and can finish
settling and turn toward pickup after a clearance refusal. After Reset, try
**Move 1.4 m at −15°**, then choose your own destinations. The measured task
finishes within 7.4 cm after the complete retreat and standing. Its following
backward command moves 30 cm, leaving the box within 7.0 cm; two other measured
requests preserve their ordinary trajectories. This remains finite coverage,
with other directions, poses and later box tasks still to check. See
[the approach review](../WEB_RECOVERED_APPROACH_PREVIEW_REVIEW.md).

**Open pickup region preview** remains a separate development entry.
After Reset, try **Move 1.4 m at +15°**, then choose
your own box destinations. It allows nearby pickup approach positions inside
the existing arrival region while preserving the original box destination and
clearance checks. Two measured requests improve to 5.20/5.67 cm after complete
retreat; the original lower carry remains physically identical. The approach
is slower and other requests still miss or refuse. **Explore measured coverage**
compares human pose, object pose and destination across versions, with unknown
cases and contact qualifications explicit. See the
[delivery notes](../WEB_PICKUP_ENTRY_REGION_DELIVERY.md).

**Planner honesty (WS-B, September 2026 inference-only pass).** Four
behaviours, each behind its own URL parameter (no model or training change):

- `replanRemaining=1` (default on): when a carry sequence ends with
  `unsupported_live_distance` (the measured box left the next fixed source's
  travel range after a short-segment miss), the page consumes the sequence's
  `remainingGoalRequest()` and re-runs the request-time planner from the live
  box toward the **original** destination with the remaining pickup budget
  (at most three pickups per task). The same request id, goal object and
  control accounting are kept; the request record stays open with
  `replan_remaining_goal`. The replan starts only if a supported plan is
  estimated to fit the controls left before the 5820-control post-click
  deadline (`carryControlBudget=` overrides the budget); otherwise the
  refusal stands and names `not_enough_time_left`. `replanRemaining=0`
  restores the previous ending.
- `carryRanking=default|excludeLong|reliability` (default `default`, so
  behaviour is unchanged until evaluated). `excludeLong` gives `long`
  maxUses 0 for requests within 4.2 m (the rest of the library reaches
  4.42 m). `reliability` ranks a summed per-candidate failure penalty before
  pickup count; the penalties are the frozen-P100 teacher-phase finish rates
  in `CARRY_CANDIDATE_RELIABILITY` (`carry_skill_library.js`) and must be
  confirmed on the unseen panel.
- `carryBudgetGuard=1` (default on): every plan carries a control estimate
  (sources + 180 settling + 439 exit per transition + 739 ending + an
  approach model); estimated-to-fit plans rank first and `budgetRisk` is
  recorded in the plan and `carryReferenceSelection`. The guard never adds a
  refusal and never changes the 10 cm tolerance or the 5820 budget.
- Truthful refusal reasons: when the mixed planner exhausts its check budget
  or finds no clear plan, the plan reports `dominantReason` and
  `dominantObstacle` (for example `occupied_carry_destination` +
  `active_suitcase_080_080_080`), and the request record uses that reason
  instead of `planning_budget_exhausted`. `getState()` exposes
  `carryRefusal`, `carryControlBudget` (remaining controls), `carryReplans`
  and `carryTaskLineage` for automation and status text.

**Flexible carry skill library** retains the preserved baseline. Existing
long-carry links also enable this library;
`restrictedCarryLibrary=0` retains the earlier two-reference selection.
The five complete sources include two short placements, the original carry,
a higher-lift middle-distance carry and the long student-assisted carry.
Mixed sequences use the actual box position while retaining the original click.
Final success requires horizontal error at most 10 cm after complete placement,
settling and retreat. The review page includes matched videos and preserved
comparison controls. Full-range physical robustness is still being evaluated.
The planner can avoid extra pickups using a reference endpoint within 1 cm of
the original click, keeping the full 10 cm tolerance for measured completion.
The measured 0.79 m request uses one pickup and finishes about 1.5 cm away.
The separate **Lower carry or higher lift** entries add an explicit complete
motion choice through `carryStylePreview=lower` or `higher`. Try their
**Move 1.15 m** example after Reset, or select the large box and click a floor
destination. Their full measured central runs finish 1.91/5.39 cm away; the
higher style takes about 16 seconds longer and misses the nearby 1.05 m test
by 11.19 cm. Those links keep a fixed experimental style. The default selection
remains deterministic. See the [style integration review](../WEB_CARRY_STYLE_PREVIEW_REVIEW.md).
The new **Try sampled styles** entry uses `carryStylePreview=sample` and a
`carryStyleSeed` (1213 by default). Try **Move 1.15 m**, Reset and repeat;
choices can repeat, while reloading restarts the sequence. It samples among
geometry-supported complete one-pick lower/higher plans and preserves the
ordinary library plan when neither fits. Both tested first-draw branches
reproduce all 12,000 controls of the explicit comparisons exactly. The toolbar
shows the actual choice, and session downloads retain draws and requests.
The known higher-style miss and longer approach remain. This separate path
does not connect the older physically-qualified whole-plan sampler or sample
student latent noise. See the [sampled-style review](../WEB_SAMPLED_CARRY_STYLE_REVIEW.md).
The separate **Student-assisted higher lift** link enables `studentLiftPreview=1`
with the library and existing student-transport settings. It gives the student
54 controls during the upward portion of the selected higher carry, with the
complete teacher placement and exit afterward. It remains off in the baseline.
The central 1.40 m task finishes 8.94 cm away, and a 1.45 m task still misses
at 11.74 cm. A queued subsequent task also regresses: 13.44 cm with later knee
contact, versus 7.02 cm and no such contact with the option off. The review page
includes full comparisons of these outcomes and two carry styles reaching the
same destination. See the [complete experimental review](../WEB_MEDIUM_STUDENT_LIFT_PREVIEW_PRODUCT_REVIEW.md).
The server also supports seeking within the MP4 comparisons. It serves only
the web folder and binds to localhost by default.

Longer box requests now complete a physical exit between intermediate carries,
then replan the next approach from the actual placed box while retaining the
original destination. A single eastward 2 m click completes both carries and
both exits, ending 0.8 cm from the click. Its first 1,986 physical controls are
exactly unchanged, and all 122,400 solver substeps have zero falls or leg-box
contacts. The new 69.5-second comparison shows every changed motion at 1x.
The recorded second approach takes 42.77 seconds to cover a 48 cm initial gap.
Cancellation during an intermediate exit finishes that exit without starting
another carry. Reset clears the pending sequence. See
[the staged carry review](../WEB_STAGED_CARRY_REVIEW.md).

The new **student approach demo** adds `restrictedStudentApproach=1`. After a
complete intermediate exit, an eligible direct approach within 0.5 m uses a
fixed student root-pose goal for at most 180 physical controls. Goal coordinates,
rotation and remaining time retain the training convention; policy weights and
body histories are unchanged. Each action receives the existing all-box physics
preview. The teacher then executes its full carry with 90 initial stance controls.
Other entries retain the recorded route, and a student timeout replans that route
from the actual state. This feature remains off in the default page and the
separate staged teacher demo.

For the eastward 2 m request, this approach takes 2.05 seconds and the full task
finishes about 40 seconds earlier, ending 1.1 cm from the click. A 90-degree
starting yaw ends 2.1 cm away. Two new videos show the approach speed and
why an extra second of teacher preparation improves placement after student
walking. These results concern short unloaded approaches between carries;
the separately selected long carry preview adds loaded student steering. A third video shows
the 3 m case: 73.58 seconds versus 152.13, but final placement error grows from
2.14 to 8.12 cm. See
[the student approach module](../WEB_STAGED_STUDENT_APPROACH_MODULE.md).

The earlier **long carry demo** includes a **Carry 2.3 m example** button and adds `restrictedLongCarry=1` and
`restrictedStudentTransport=1`. It retains the original short reference and
loads a separate complete 366-control carry. Before approaching, it prefers
one long carry over two or three short carries when the goal lies in its
bounded 2.084–2.584 m range and the whole reference passes existing path and
destination checks. It checks the short alternative if long-reference geometry
refuses. Other distances retain the old staged plan; multiple long carries
are not enabled. This selection does not yet fix a later approach/facing refusal.

After 90 preparation and 150 pickup controls, a measured loaded entry permits
90 student controls toward the paired full root pose and box COM at source240.
The metric goal uses the training heading frame and countdown; the original
clicked destination remains separate. Every action receives all 17 physical
preview substeps, permitting selected-box support on both rubber hands and
the wrist pitch/yaw links. Forearms, wrist roll links, legs and other boxes
remain outside that support set. A refused action consumes no physics or source time; the
teacher continues from the unchanged source. After 90 accepted controls, the
teacher executes all 126 remaining source controls and the complete exit.
These are moving handoffs, not claims that the student has stopped at its goal.

Eastward 2.15 and 2.334 m trials complete with 90 loaded student controls and
about 7 cm final error. The ordinary 2.3 m example also completes all 90 controls,
ends 5.4 cm away and has no falls or leg contacts in its complete dense trial.
At 2.50 m, the same full 90-control window ends 7.0 cm away; its later teacher
placement has 68 knee-contact substeps, peaking at 50.2 N. The earlier narrower
wrist set ended the student window after 41 controls and placed 11.6 cm away.
Both 15-degree angled 2.334 m requests currently refuse during approach
or facing, before pickup. The review page includes the matched 19.9-second
full task video and the two-distance pose comparison. These are actual-main
MuJoCo WASM with CPU ONNX inference evaluations; browser rendering and ORT-Web parity remain
unverified in the sandbox. No checkpoint was changed.

If the project is on a remote machine, run the server there, then use
`ssh -L 8000:127.0.0.1:8000 USER@HOST` from your computer and open the same URL.
The Codex sandbox cannot bind a listening socket, so the command must run in
your normal terminal. No public deployment has been made.

The improved approach link adds routing with the existing 55 cm transit
clearance to the previous backward-step, keyboard-terminal and initial-standing
settings. It also enables `restrictedApproachRecovery=1`: a bounded teacher
continuation after a refused final approach, when that approach actually ended
inside the original arrival region before settling drifted out. This recovered
the tested southward and rotated-start requests and loading-time variants.
Longer routes still add waiting. Reference durations and checkpoints remain
the same. See [the recovery review](../WEB_APPROACH_RECOVERY_PRODUCT_REVIEW.md)
and [additional routed cases](../TEACHER_ROUTED_RECOVERY_PRODUCT_REVIEW.md).

Restricted destination clicks now check the placed box footprint and the
complete humanoid/carried-box reference against the other scene boxes before
walking. Each actual entry replan is checked again before lifting. Rejected
requests retain their clicked point and show a red marker; occupied endpoints
and obstructed carry paths have distinct messages. North/east/northeast remain
exact across10,800 physical controls with both checks. These geometric checks
do not establish dynamic success for arbitrary states or moving obstacles.
The previous approach settings have their own link and use these current
destination checks too. The original archive remains under
`eval_results/web_stable_baseline_20260912/`; the earlier route-only download is
preserved under `eval_results/web_approach_review_delivery_20260912/`.

Approach-terminal retention remains disabled in these review links. Its isolated
northeast test regressed to leg contacts and an incomplete exit, despite quieter
approach settling. Both post-exit comparisons also show separate private
experiments. Browser rendering and ORT-Web performance need human checking.

Review links expose a **Download session** button. It saves up to two minutes
of states sampled at 10 Hz plus recent raw commands and browser errors, locally
as JSON. It does not upload data or alter control. This trace helps inspect
unanticipated command sequences; it is not an exact physics replay.

### Full Browser Demo

```bash
cd web/
python3 -m http.server 8000
# Open http://localhost:8000/
```

The folder is self-contained for local locomotion testing: `public/policy.onnx`
is the current browser-promoted policy, `public/clip_db.{bin,json}` is the
motion-matching command database, and `public/g1_scene.xml` is the MuJoCo scene.
The page still loads `three` and `onnxruntime-web` from CDN.

### Measured walking controls under evaluation

The page starts with the recorded-motion supervisor; `?restrictedControl=1`
also selects it explicitly. `?restrictedControl=0` selects the older student
comparison. The supervisor uses
the reference teacher for complete steps, signed turns, settling and standing,
with finite recorded approaches to boxes and the short post-setdown exit.
WASD selects a direction; Q/E requests complete turns. Releasing or changing
keys queues the new intent until the current record and its settling finish.
The current tiny step takes 2.88 seconds and each turn takes 4.55 seconds,
followed by at least one second of settling. These controls are deliberately
slower than continuous student control while their transition coverage is
being evaluated.

### Public entry presentation (September 2026)

Three presentation flags default ON on `index.html`; each can be switched off
by URL so evaluation arms stay comparable. None changes the controller,
planner or policy.

- `cameraMode=wide` (default) frames the whole working floor (60° fov from
  three.js (1.5, 10, 2) toward (1.5, 0, 0)), so every historical panel
  destination projects inside the canvas; `cameraMode=orbit` restores the
  original close camera. `V` still toggles the follow camera and OrbitControls
  still works (orbit distance 10.2 m < the 12 m limit). Literals live in
  `src/camera_modes.js`.
- `destinationPicker=1` (default): once an object is selected, every canvas
  click is a floor destination for it — even when the click lands on the
  selected box or another box. Selection changes only through explicit
  actions (`Esc`, the **Deselect** button, `api.selectObject`). The cursor is a
  crosshair while an object is selected (not-allowed when the ground ray
  misses). `destinationPicker=0` restores the legacy behaviour where a click
  on a box toggled selection and could swallow the destination.
- `publicUi=1` (default) hides the development buttons (`Pick up & set down`,
  `Carry about 1 m`, the fixed-distance examples), the review toolbar and the
  `Space` / `1 2 3` legend entries. What remains: click box · click floor ·
  Deselect · W A S D · Q E · V · R · Esc · Reset · status line. `publicUi=0`
  restores everything. The status line always reports what happened,
  including a sampled carry style when `carryStylePreview=sample` is on.

`?debug=1&restrictedBackward=1` enables an experimental keyboard backward
step using the existing complete 199-frame box-exit recording. It retains that
recording's fixed terminal stance after release, with fresh collision checks.
Only candidates within 15 degrees of the requested direction use it; floor
navigation and diagonal turns retain their existing choices. In four tested
headings, a short S pulse moved 30.2–30.8 cm backward with 1.1–1.9 cm lateral
error. Release still waited about 3.27 seconds for the complete record and
settling. It remains a debug option while mixed commands and obstacle cases
are evaluated; see [the backward control review](../WEB_BACKWARD_CONTROL_REVIEW.md).

Floor goals retain the requested destination, with a 2 m direct-distance
limit, a 4 m route-length limit and clearance around all scene boxes. A request
outside those limits is reported explicitly. Supported floor routes combine
four unscaled recorded step lengths and aim for 10 cm arrival, followed by a
fixed neutral reference. Keyboard steps retain the direction resolved when the
key combination changed; a held backward key does not reverse itself again
after the robot turns. New box requests wait for the current movement, and
movement requested during carrying waits for setdown. Reset discards both
queued input and any pending action from the old episode.

Whole-record collision checks, box handoffs, response latency and broad command
streams are still being evaluated; bounded
input alone does not establish that every resulting physical state is supported.
With `debug=1`, `window.__interactiveDemo.getState()` exposes `effectiveCommand`,
`queuedCommand`, `queueReason` and `restrictedControl`; `getContactsForObject(name)`
reads forces for each scene object without changing simulation state.

The current implementation checks complete recorded collision sweeps against
the live boxes before executing each record. If the movement is refused but a
neutral stance is supported, it keeps standing. If no supported stance is
available, it suspends physics and requires Reset; repeated debug stepping and
new input cannot reactivate the refused reference. This is a remaining
capability limitation near a recently placed box, not successful task completion.

Restricted mode uses teacher-controlled box approaches with all four step
lengths and teacher settling. It can try smaller complete records after a clearance refusal and
retain an executed stance only after checking it against the current boxes.
The carry resumes inside its original 25 cm pregrasp region after 60 additional
teacher controls and measured quiet standing. Actual-main cases now reach the
box and execute the carry; other approach directions still refuse clearance.
The original approach paths remain available for explicit debug comparisons
with `debug=1&teacherRecordedApproach=0`; `teacherApproach=1` then selects the
older waypoint controller. Those comparisons do not retain the full restricted
box-control behavior.

`debug=1&restrictedApproachRouting=1` also routes when the direct root path
enters the existing transit clearance of any scene box, retaining the exact
pregrasp target. It keeps the final close approach separate from transit, with
unchanged whole-body checks before each complete recording.
`restrictedApproachTerminal=1` is a separate experiment that retains a completed
walking stance inside the parent's existing handoff region. It is disabled in
the improved approach link because its effect on the following carry depends
on the incoming state. See the cardinal and bearing reviews in the repository
for both completed tasks and failures.

The short exit after a completed setdown holds the executed terminal reference
for one second, executes the complete 199-control backward recording, then
settles for three seconds.
New commands queue through that motion. Each exit action is first simulated
for its 17 physics substeps in separate MuJoCo data; a predicted humanoid–box
contact or balance failure suspends before committing the action or history.
This predicts one control interval, not future recoverability. The first
integrated queued-carry stream completes the exit and then refuses the second
approach's turn while maintaining standing. See
[the exit review](../TEACHER_BOX_EXIT_INTEGRATION_REVIEW.md) for measured scope.
`debug=1&teacherBoxExit=0` disables it for comparison. Existing explicit
`teacherRecordedApproach=1&teacherBoxExit=1` queries retain the same behavior.

The restricted display marks the accepted walking destination in green, the
current reference target and heading in blue, and the box destination in orange.
These distinguish queued input from the motion currently executing.

The original twenty saved command streams now cover 50,408 physical controls
with zero falls, box–leg contacts, suspensions or interface faults using the
short exit. This includes a reset fix: resetting during a finite approach must
detach that controller before the next control. Many requests are canceled by
later input; only two finish and one routed approach reaches its turn limit.
Five additional release follow-ups all reach quiet standing, with 3.02–5.43 s
from release to neutral. These results establish neither arbitrary command
coverage nor low response latency. Details are in
`../eval_results/restricted_exit_reset_detach_20260912/original20_review.json`
and `../eval_results/restricted_exit_release_holds_20260912/README.md`.

`debug=1&teacherBoxExit=long` substitutes a complete 249-control backward recording.
It clears a previously refused following turn and permits a repeated carry in
one queued stream. That repeated carry has a reproducible knee contact during
setdown, so the longer exit remains a separate experiment. Exact replay shows
the planned box stays clear while the actual box drifts toward the knee.
An extra second at the original initial carry pose removes that contact in the
captured state but moves the feet substantially; it is not enabled by default.
See [the entry study](../TEACHER_CARRY_ENTRY_REVIEW.md) and
[the held-box study](../TEACHER_HELD_BOX_PAUSE_REVIEW.md) for measured limits.

`debug=1&teacherCarryEntry=1` now exposes that preparation as an optional
experiment within restricted mode. It extends the fixed prefix by60 controls
and preserves the complete carry. A full queued stream with the longer exit
finishes both carries without box–leg contacts; cancellation and reset checks
also pass, including an explicitly refused post-exit turn. This option remains
off by default because placement and foot-motion effects vary by entry. See
[the integration review](../TEACHER_CARRY_ENTRY_INTEGRATION_REVIEW.md).

The first actual-main one-metre northward route held its destination for three
seconds with 3.19 cm final error and zero leg contact against any box. Rejected
goals preserved the accepted hold, and reset during pending teacher inference
discarded the old action. These checks use real MuJoCo WASM and CPU ONNX
inference with rendering stubbed. They do not verify browser rendering or
ONNX Runtime Web. See [the command-stream review](../WEB_COMMAND_SEQUENCE_REVIEW.md)
for broader results and coverage gaps.

### Plan-time pickup-pose reachability (September 2026, lane WS-D2)

Before a carry plan is chosen, `web/src/pickup_pose_reachability.js` predicts whether the
recorded approach can reach each candidate's pickup pose (the carry reference's first-frame
root) from the current root: a pure-geometry dry-run of the approach (route, waypoint radii,
`planRecordedSteps`, heading rule, hull admission with the runtime reserves, `trySmallerStep`,
standing handoff inside 0.25 m) plus the `pickup_facing_approach` programme generalized to every
carry source and entry-region offset. `web/src/pickup_pose_plan_search.js` then ranks plans
whose first pickup pose is reachable first (unchanged library, then the extended library with
the `alternate` one-metre carry), keeping the existing order inside a pass, and otherwise
returns the unchanged plan marked `pickupPoseReachable:false`. Nothing is refused by this step.

| URL param | Default | Effect |
|---|---|---|
| `pickupPoseSearch` | on | `=0` restores the pre-search ranking exactly. |
| `pickupPoseSearchTiers` | `first,none` | `all,first,none` also requires nominal later segments (opt-in; re-ranks a frozen success on the P100 panel). |
| `alternateCarry` | on | `=0` leaves `teacher_carry_alternate_reference.json` out of the extended library. |

The facing-turn refusal after `needs_facing` keeps `needs_facing` as the task reason and
reports `facingErrorDeg`, `turnAdmissible:false`, `turnRefusalReason` in the request outcome.
Tests: `node web/test_pickup_pose_reachability_node.mjs [--quick]` (reconstructed joint100
scenes), benchmark: `node scripts/bench_pickup_pose_reachability.mjs H034 20`.

### Controls

- Click floor: set a planar locomotion goal; the floor height does not command
  the pelvis downward.
- WASD: request a walking direction. Restricted mode completes each recorded
  motion and settling before applying the latest queued command.
- Q/E: request a complete left/right turn in restricted mode.
- R: reset robot, observation history, translated goals and held keys, and
  return to deterministic zero latent.
- Space: sample one VAE latent in the older student comparison mode.
- F1: toggle deterministic latent in the older student comparison mode.
- G: toggle translated-goal markers.
- V: toggle follow camera.

Movement keys take over from a clicked walk goal. Leaving the tab releases
held keys. Dragging the camera does not submit a floor target.
With an object selected, a floor click specifies where its support surface
should land; the target origin height is computed from its rotated surface
points. This avoids requesting that the center of a box lie on the floor.

For testing locomotion only, do not select an object. Object interaction is
still experimental.

### Deploy to GitHub Pages

```bash
# Pushes /web/ subtree to gh-pages branch (after the full demo works)
git subtree push --prefix=web origin gh-pages
```

## File map

| File | Role | Status |
|---|---|---|
| `index.html` | Entry page; canvas + UI; loads all scripts | scaffold |
| `src/main.js` | App entry; async policy/physics loop and benchmark controls | implemented |
| `src/state.js` | `UserState` + `Mode` enum | implemented |
| `src/math.js` | Quaternion, rot6d helpers (mirror Python) | implemented |
| `src/obs_builder.js` | Build 1422D obs from MuJoCo state + UserState | implemented |
| `src/policy.js` | ONNX session; `forward_deploy` inference | implemented |
| `src/smoother.js` | Residual + EMA action smoothing | implemented |
| `src/keyboard.js` | WASD / QE / R / F1 handlers | implemented |
| `src/mouse_picker.js` | Floor targets, object selection and hover | implemented |
| `src/mujoco_loader.js` | Load MuJoCo WASM, meshes and scene | implemented |
| `src/goal_viz.js` | User and translated command markers | implemented |
| `public/policy.onnx` | Exported student model | current deploy policy |
| `public/g1_scene.xml` | G1 + OMOMO objects MJCF | current deploy scene |
| `public/object_pointclouds.json` | 64-point clouds per object | symlinked from `intermimic/data/` |

## Obs layout (MUST match training)

```
[ NEW_CMD(13) | body(1012) | task(192=points only) | NEW_MASK(205) ] = 1422D

NEW_CMD: human_target_pos(3) + human_target_rot(6 rot6d) + obj_target_pos(3) + time_to_target(1)
body:    proprio(92) + 10-frame history(920)
task:    64 surface points × 3 (robot heading frame)
NEW_MASK: obj_points(192) + command(13 = 9 keep_h + 3 keep_o + 1 always)
```

See `src/obs_builder.js` for the exact concatenation order. The reference
Python implementation is in `intermimic/sim2sim_vae_interactive.py`.

## Three deploy modes

| Mode | Trigger | `keep_human_target` | `keep_obj_features` |
|---|---|---|---|
| IDLE | no input, stand anchor | 1 | 0 |
| LOCO | WASD held or floor navigation goal | 1 | 0 |
| HOI-full | WASD + click | 1 | 1 |
| HOI-obj-only | click only | 0 | 1 |

The perception bit (`obj_points` mask) is set when an object goal activates an
HOI mode. Selecting a box only highlights it; standing and navigation keep
object perception off, matching training's locomotion observations. The
selection loader accepts the shipped `{points, bbox, ...}` records and checks
the 64 surface samples before exposing an object's pose and point cloud.

## Golden-state parity test

Before deploying, run a parity test:
1. Log `(mj_state, obs)` pair from the Python sim2sim (`intermimic/sim2sim_vae_interactive.py`)
2. Replay the same `mj_state` in JS via `obs_builder.js`
3. Verify the 1422D obs matches byte-for-byte (tolerance 1e-5)

This catches subtle bugs in heading-frame transforms, point cloud sampling
order, and mask layout differences across the Python/JS port.

## Live browser click benchmark

Open the demo with a benchmark query string to run the actual browser/WASM
control loop and print metrics to the devtools console:

```text
http://localhost:8000/?benchmark=click-grid&benchDurationS=5
```

The grid uses distances `0.5,1.0,1.5m` and directions
`0,45,-45,90,-90,135,-135,180deg` by default. For one click case:

```text
http://localhost:8000/?benchmark=click&benchDist=1.5&benchDirDeg=180&benchDurationS=5
```

Results are also exposed as `window.__interactiveBenchmarkResults` and
`window.__interactiveBenchmarkSummary`.

For browser automation, `?debug=1&paused=1` exposes
`window.__interactiveDemo`. Its `getState()` reports root pose/velocity,
simulation step counters, current command/mask, actions, object poses and
translator diagnostics. It also reads actual solver hand–box normal forces
and loaded foot contact-point tangential speeds; these diagnostics do not
change policy inputs or task decisions. `reset({rootYawRad})` varies only the
episode's initial robot heading for evaluation. `await step(n)` runs exactly `n` policy/physics steps
while paused; `pause()` waits for pending inference and `resume()` resumes the
animation loop. Controls are `reset()`, `setWalkGoal([x,y,0])`,
`setKeys({w:true})`, `selectObject(name)`, `setObjectFloorGoal([x,y,0])`,
`setObjectGoal([x,y,z])`, and `requestCarryDestination([x,y,0])`. The last one
is the picker-free carry request: it runs the mouse click's normalization
(`objectGoalFromGround` → `onObjectGoal`) for a MuJoCo floor point with no
raycast or camera and returns `{requestId, delivered, disposition, reason,
goalWorld, selectedObject}` — the request id when a carry request was
registered, otherwise the refusal disposition (`no_selection`,
`interaction_blocked`, `unsupported_object`).
Object goal coordinates in this low-level API specify the object origin.
The hook is absent from normal sessions without `debug=1` or a benchmark.

When browser launch is unavailable, the repository runner executes the actual
web command, observation and PD modules against bundled MuJoCo WASM:

```bash
node scripts/eval_web_wasm.mjs --out eval_results/web_wasm_local
node web/test_goal_translator_node.mjs
node web/test_keyboard_node.mjs
node web/test_goal_geometry_node.mjs
node web/test_object_selection_node.mjs
node web/test_observation_masks_node.mjs
node web/test_body_obs_wasm.mjs
node web/test_contact_diagnostics_wasm.mjs
```

Run these commands from the repository root. The rollout runner uses Python
ONNX Runtime CPU over stdio, so its results do not validate browser rendering
or ONNX Runtime Web. It records sustained final arrival time in addition to
minimum distance, stability and action clipping.

## Status

- Locomotion: active browser deploy path. The live command translator now
  matches the accepted MuJoCo clip-match path more closely: click goals and
  held keyboard locomotion both recompile from the live root pose instead of
  aging an old fixed clip row. No-input standing uses a latched stand anchor
  with the same short final-receding hold style used by the MuJoCo gate, rather
  than a pure no-goal IDLE mask.
- Current `public/policy.onnx`: `policy_interactive_bt27000_20260531_distill_browser_cmd.onnx`.
  The corrected strict MuJoCo browser-command gates pass 30/30 stable
  scenarios and 24/24 click arrival-hold goals at both `5s` and `10s` scripted
  durations, with no action clipping or NaN/Inf and min pelvis z `0.695m`.
- Object interaction: experimental; validate with MuJoCo gates before treating
  object results as deploy-ready.
- September 2026 runtime fixes: real floor clicks now use planar arrival
  geometry, episode reset clears compiler state and discards outstanding
  pre-reset inference, and body angular velocity uses native MuJoCo buffers.
  The real WASM observation fixture agrees with Python within `2.98e-8`.
  Historical native command benchmarks above did not cover these browser
  implementation defects. Click arrival now latches against the final XY
  target; the old held-row achievement check was unreachable during live
  per-frame re-query. This improves the eight-second clear-scene grid from
  22/24 to 23/24 stable arrivals with final hold, and a 12-case retarget family
  from 5/12 to 10/12. These measurements use MuJoCo WASM and Python ONNX CPU.
  Early reversals before arrival and object interaction still fail in measured
  cases; policy robustness remains in development.
- Object selection now loads the shipped point-cloud records correctly and
  preserves locomotion perception until an interaction goal is active. Five
  measured selection/navigation cases remain upright after this fix, compared
  with one previously; this does not establish successful box manipulation.

## Experimental reference teacher

The **Pick up & set down** button uses the student to approach the large box,
then switches to the reference teacher for a complete lift and setdown, and
returns to student standing. It never resets or teleports physics at handoff.
**Carry about 1 m** uses the same approach to lift, walk with, and set down
the large box. Its short carry direction follows the robot's heading at
handoff. To choose a destination, select the large box and click the floor.
The carry reference rotates toward that point and adjusts each carry's travel
by at most 25 cm. The planner divides the destination into up to three carries,
each about 0.8–1.3 m, with three seconds of student standing after every setdown.
It retains the original destination and each intermediate target. Unsupported
distances, including gaps between the supported intervals, are declined before
walking. If a placement leaves the next target outside the supported range,
the task stops with the box down and asks for a new destination.
If an approach reaches the box with an incompatible heading, the controller
can execute a recorded teacher turn and retry the same carry. The current
turns rotate about 90° in either direction, using separately recorded motions.
The controller chooses the direction that reduces the measured heading error.
There are at most two preparations per task. The original
destination, intermediate targets, physics and observation history remain
continuous. Unsupported approaches stop with a status message. Other object classes do not yet have a
validated carry control.
Movement or Escape cancels an approach or standing interval. During a lift,
the current carry finishes its setdown and cancels the remaining carries.
During a turn, it finishes the recorded turn and cancels the box task;
**Reset scene** remains available immediately.
The current task supports the large box and requires a settled approach. Its
reference follows the robot's current heading at handoff. The approach uses
bounded, half-metre locomotion commands with an upright heading target and
the same 25 cm arrival radius as locomotion. The teacher gets half a second
to reach the initial stance before the lift. The previous click-command setting
is restored after the task. Camera drags leave the approach active. Manual
movement during asset loading cancels the pending task.

`src/teacher_obs.js` is the 4052-channel teacher adapter. It consumes FP32 reference frames
at `t+1` and `t+16`, all 39 body poses and velocities, simulated contacts and
256 object surface points. `TeacherObsBuilder.build()` runs immediately before
physics and retains the previous control step's joint positions and velocities.
Its default feature layout follows teacher training. Explicit diagnostic
options reproduce the older native helper's different contact, object-offset,
torque and previous-joint-state behavior for numerical comparison.

`src/teacher_reference.js` aligns reference object XY to the live box and
reference root heading to the live robot, transforming world positions,
rotations and velocities together. It preserves floor height and heading-relative interaction
features; it does not reposition simulation bodies.

The native helper comparisons use fixtures generated by
`scripts/export_teacher_reference_parity.py`. The corrected-object native
fixture matches the real WASM adapter within `7.15e-7` across five poses. The
actual training comparison also passes: `1.91e-6` maximum error over eight
states, including three with real hand/foot contacts. Reference-initialized
teacher rollouts lift the box in the actual WASM physics. The complete front-box
approach/lift/setdown/standing cycle also passes through the actual `main.js`
control loop in seven starting conditions: zero/one/three/five/ten seconds of
idle time, and after walking 0.6 m left or right. Minimum pelvis heights are
`0.520–0.545 m`, peak box heights `1.034–1.062 m`; all execute 30 initial
stance frames plus 407 original teacher frames and return to student standing.
Task completion checks the measured lift, final box height, and robot balance.
The short carry passes five actual-main starts: immediate, three/ten seconds
of idle, and after either lateral walk. Peak box heights are `0.887–0.914 m`,
minimum pelvis heights `0.579–0.584 m`; all finish the 30 stance + 394 source
frames and return to standing. The exposed carry button also passes actor
reuse, cancellation, and reset-during-inference checks.
Destination-directed actual-main runs toward 30°, 90° and 120° points end
2.3–6.6 cm from their goals after three seconds of initial idle. A real floor
raycast through the mouse handler also completes a carry, with 3.4 cm final
error. Before turn preparation, a 150° floor click declined the handoff and stayed upright. These are
bounded measured cases, not arbitrary-scene success guarantees.
The sequence integration also passes actual floor raycasts at 1/2/3 m after
three seconds idle: final errors 3.4/5.4/2.8 cm, minimum pelvis heights
0.575/0.570/0.572 m, and all 424/848/1272 teacher frames plus the corresponding
180/360/540 student standing steps. Cancellation during the first lift or
standing interval leaves the box down and cancels later carries.
Teacher turn preparation now completes the previously declined northward
two-metre task after ten seconds idle, with 2.92 cm final placement error.
The 120° two-metre and 150° one-metre requests also complete, with
2.19/2.53 cm errors. Each uses 273 turn steps, followed by the original carry
references and student standing intervals; minimum pelvis heights are
0.570–0.580 m. Escape during the turn finishes all 273 steps, cancels the
pending carry and leaves the box on the floor. The three-metre path that
already worked still completes without a turn. Initial approach state and
turn direction remain relevant limitations.
Clockwise preparation also completes the one-metre 30° request from initial
root headings of 60° and 90°, after three seconds idle: 2.39/1.34 cm final
errors, 273 turn + 424 carry steps and 180 student standing steps. The earlier
northward counterclockwise trajectory is unchanged after adding this choice.
Some other directions still time out during approach; walking around the box
and initial settling remain under evaluation.
These tests stub DOM rendering and
run ONNX through Python CPU; they do **not** establish browser rendering or
ONNX Runtime Web parity. Broader approach states remain under evaluation.
Source evaluations live under `eval_results/web_main_teacher_stance30_*`.
Carry records use `web_main_carry_*`, `web_main_mouse_carry*` and
`web_main_turn_carry_*`.
The optional
`public/teacher_policy.onnx` and the requested pickup/carry reference JSON are
loaded on the first corresponding button press. Both tasks reuse one teacher
actor; `public/policy.onnx` remains the student.
Floor-click carries also load `public/teacher_turn_reference.json` and
`public/teacher_turn_clockwise_reference.json`. Their
locomotion-only teacher adapter reproduces the training task's masked object
channels. The alternate carry asset is retained for private comparisons;
the current main loop uses turn preparation with the primary carry.
The optional `carry_interval_frames` in carry data specifies where the common
translation adjustment begins and ends in the original reference. Custom
carry exports require their measured interval via `--carry-interval START END`.

```bash
node web/test_teacher_reference_node.mjs
node web/test_teacher_obs_wasm.mjs
node web/test_teacher_obs_wasm.mjs eval_results/teacher_web_parity_20260912/observation_cases_corrected.json --correct-object-offsets
node --loader ./web/test_support/main_loader.mjs web/test_main_control_wasm.mjs
HUMANOID_TEST_BOX_TASK=carry node --loader ./web/test_support/main_loader.mjs web/test_main_control_wasm.mjs
HUMANOID_TEST_IDLE_STEPS=180 HUMANOID_TEST_CARRY_GOAL=3.2320508076,1,0 HUMANOID_TEST_MOUSE_CARRY=1 node --loader ./web/test_support/main_loader.mjs web/test_main_control_wasm.mjs
```

The main-loop test uses an existing local Three.js build; set
`HUMANOID_TEST_THREE_MODULE` if its default path is unavailable. Rebuild the
optional public teacher assets from the validated evaluation exports with
`.conda/isaacgym/bin/python scripts/export_web_teacher_skill.py`.
Add `--skill carry` to export the short carry reference.

An exact successful teacher initial state also lets us separate initial-pose
problems from student behavior. In the current four-second student test, the
robot remains upright and crouches toward the box but does not lift it. Both a
fixed matched future goal and an advancing reference goal show this limitation.

The current carry planner and repeated-carry experiments are described in
[TEACHER_CARRY_EXPERIMENT.md](TEACHER_CARRY_EXPERIMENT.md).

### Private approach routing

`?debug=1&boxApproachRouting=1` enables a measured-geometry routing experiment.
It reads actual collision-mesh bounds, proposes detour waypoints only when a
direct approach crosses an object, and retains the original pregrasp target.
Transit uses the ordinary matched walking orientation; the final approach
retains the upright orientation used by the box task. This experiment allows
up to twelve seconds per approach. The normal page retains its existing
approach behavior.

Actual-main southward and southwestward one-metre floor clicks complete with
5.66/3.13 cm placement errors, each using a turn, one carry and three seconds
of student standing. A clear 30° path also completes. These nominal results
are insufficient to enable routing by default: a matched-transit diagnostic
changes from success to a fall with a roughly `3e-8` waypoint rounding
difference. Upright transit commands and teacher-supported walking are being
evaluated for more robust behavior. Use `HUMANOID_TEST_BOX_ROUTING=1` with the
main-loop runner to reproduce the experiment; longer routes may require an
explicit `HUMANOID_TEST_EVALUATION_STEPS` to include final standing.

`?debug=1&teacherApproach=1` replaces obstructed transit with recorded teacher
steps and turns, using the same physical geometry and unchanged pregrasp goal.
It loads the medium and short step assets only for destination-directed carries.
Each motion runs on its original reference clock, followed by at least one
second of student standing. The controller replans from the measured endpoint
and finishes a recorded step before honoring cancellation. Clear approaches
retain their existing controls. This option remains experimental and disabled
on the normal page.

Actual-main southward requests after three and ten seconds of idle, and a
southwestward request after three seconds, all complete with no approach
leg–box contact. Placement errors at carry setdown are 4.42, 7.28 and 1.84 cm.
Longer post-task observations expose a separate limitation: student standing
later moves the box in both southward cases. The three-second-idle run records
a brief leg contact at simulation time 61.483 s, increasing final error to
16.18 cm; the ten-second-idle case ends at 20.30 cm. Completion is therefore
not evidence of a stable long-term release. Standing and retreat behavior
remain under evaluation. The clear-path comparison preserves all 241 saved
physics states and controls exactly, and cancellation during a step or its
last action finishes that step without beginning a carry.

Reproduce with `HUMANOID_TEST_TEACHER_APPROACH=1` and the main-loop runner.
Records are under `eval_results/web_main_teacher_approach_*_20260912`.
Private step assets can be compared without replacing public files using
`HUMANOID_TEST_STEP_MEDIUM_REFERENCE` and `HUMANOID_TEST_STEP_SHORT_REFERENCE`.
As with the other main-loop tests, these checks exercise actual WASM physics
and Python ONNX CPU, with rendering stubbed.

`?debug=1&teacherStanding=original` tests fixed teacher standing after a
destination-directed carry completes its existing three-second student
standing interval. The body target is the executed carry's terminal pose,
with all reference velocities zero. The object target is fixed once to the
box actually placed, and its relative surface features are recomputed.
Physics and observation history continue normally. Movement, Escape, a floor
click or a new task returns control immediately; camera dragging preserves
the hold. The alternative `teacherStanding=live-root` first aligns that same
body pose to the live root's XY and heading. Both options are disabled by
default; ordinary pickup and undirected carry are not changed by this option.

In the actual-main cropped southward case after ten seconds idle, both variants
preserve all 411 saved pre-hold states and controls exactly. During 1,739 later
teacher controls, both have zero humanoid–box contact and less than 1.9 mm box
drift, compared with a late leg contact during student idle. The completed
sequence still places the box about 7.8 cm from the original destination; the
hold preserves that achieved placement. Manual movement resumes student
locomotion on the next control step. More starting states and sustained
walk-away behavior remain under evaluation. The stance keeps the terminal
carry's slight lean and arm posture; a neutral recorded stance is also being
tested.
