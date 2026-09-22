# Humanoid Playground v16

This immutable release presents the validated Largebox and Suitcase skills through one interaction: click an object, then click its target. The arbiter chooses safe Push, pickup/carry/place, or locomotion behavior and reports its choice in the lower HUD.

Start locally with `python3 serve.py --bind 127.0.0.1 --port 8000`, then open `/index.html?profile=release`.

Included: Largebox, Suitcase, certified ordinary-interface Push, pickup/carry/place, and continuous locomotion. The shipped runtime is the validated Phase C.1 v4 source plus a presentation-only UI, camera, render-visibility, and ordinary-selection layer. Plasticbox, Smallbox/Kick, and experimental non-one-shot long-carry behavior are excluded from the playable surface. Chromium is not included.

Packaged source: frozen source directory /lustre/fs12/portfolios/nvr/projects/nvr_lpr_digitalhuman/users/siruix/tools/ai-cli/astra-sonic-sprint-20260915-1327/web-claude-hybrid/release-source-20260922/phasec1-v15-suitcase-push-release-v16/web.
