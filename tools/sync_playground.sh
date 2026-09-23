#!/usr/bin/env bash
# Sync a new release of the interactive demo ("Humanoid Playground") into playground/.
#
#   tools/sync_playground.sh ~/Downloads/web-largebox-…/web-largebox-multifunction-candidate
#
# What it does
#   1. rsync only the files the demo needs at runtime (drops dev pages, tests, source maps,
#      the unused ONNX Runtime / three.js builds and G1 meshes the scene does not reference).
#      417 MB -> ~240 MB.
#   2. Re-applies the embed hook (playground/embed.css + one <link> and one <script> in
#      playground/index.html) so the demo can be framed by index.html#playground.
#   3. Re-applies the site's runtime patches (tools/playground-patches/*.patch): planner
#      speed-ups, reach ring + destination snapping, HUD wording, reference prefetch,
#      maxPixelRatio. If a patch no longer applies to the new release, it is reported
#      and must be ported by hand (see README "Interactive demo").
#
# Everything under playground/ that differs from Sirui's release is one of those patches.
set -euo pipefail

SRC="${1:?usage: tools/sync_playground.sh <path to unzipped demo dir containing index.html>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/playground"

[ -f "$SRC/index.html" ] || { echo "no index.html in $SRC" >&2; exit 1; }
[ -f "$SRC/public/g1_scene.xml" ] || { echo "no public/g1_scene.xml in $SRC" >&2; exit 1; }

# keep our own files across syncs
KEEP=(embed.css)
mkdir -p "$DST"
for f in "${KEEP[@]}"; do [ -f "$DST/$f" ] && cp "$DST/$f" "/tmp/.pg_keep_$f"; done

rsync -a --delete \
  --include='/index.html' --include='/style.css' --include='/review.css' --include='/combined_settings.js' \
  --include='/README.md' --include='/RELEASE_README.md' \
  --include='/src/***' \
  --include='/public/' --include='/public/**' \
  --exclude='/public/loaded_anchor_bank.json' \
  --exclude='/public/restricted_phase_successor_sweep.json' \
  --exclude='/public/teacher_walk_phase_successor_reference.json' \
  --include='/vendor/' \
  --include='/vendor/ort/' --include='/vendor/ort/ort.min.js' \
  --include='/vendor/ort/ort-wasm-simd.wasm' --include='/vendor/ort/ort-wasm.wasm' --include='/vendor/ort/README.md' \
  --include='/vendor/three/' --include='/vendor/three/LICENSE' \
  --include='/vendor/three/build/' --include='/vendor/three/build/three.module.js' --include='/vendor/three/build/three.core.js' \
  --include='/vendor/three/examples/' --include='/vendor/three/examples/jsm/' \
  --include='/vendor/three/examples/jsm/controls/' --include='/vendor/three/examples/jsm/controls/OrbitControls.js' \
  --exclude='*' \
  "$SRC/" "$DST/"

for f in "${KEEP[@]}"; do [ -f "/tmp/.pg_keep_$f" ] && mv "/tmp/.pg_keep_$f" "$DST/$f"; done

# drop G1 meshes that public/g1_scene.xml does not reference (mujoco_loader only fetches referenced files)
python3 - "$DST" <<'EOF'
import os, re, sys
dst = sys.argv[1]
xml = open(os.path.join(dst, 'public/g1_scene.xml')).read()
refs = set(re.findall(r'file="([^"]+)"', xml))
removed = 0
for dp, dn, fn in os.walk(os.path.join(dst, 'public/meshes')):
    for f in fn:
        rel = os.path.relpath(os.path.join(dp, f), os.path.join(dst, 'public'))
        if rel not in refs:
            removed += os.path.getsize(os.path.join(dp, f)); os.remove(os.path.join(dp, f))
print(f"dropped {removed/1048576:.0f} MB of unreferenced meshes")
EOF

# sanity: every ort/three file the demo loads must still be there
for f in vendor/ort/ort.min.js vendor/ort/ort-wasm-simd.wasm vendor/three/build/three.module.js \
         vendor/three/examples/jsm/controls/OrbitControls.js public/policy.onnx public/lib/mujoco_wasm.js; do
  [ -f "$DST/$f" ] || { echo "missing $f after sync" >&2; exit 1; }
done
grep -q 'vendor/ort/ort.min.js' "$DST/index.html" || echo "warning: index.html no longer loads vendor/ort/ort.min.js; check the vendor include list" >&2

# embed hook (idempotent): class on <html> when ?embed=1, plus embed.css
if ! grep -q 'embed.css' "$DST/index.html"; then
  python3 - "$DST/index.html" <<'EOF'
import sys, re
p = sys.argv[1]; s = open(p).read()
hook = ('  <script>if(new URLSearchParams(location.search).get("embed")==="1")document.documentElement.classList.add("embed");</script>\n'
        '  <link rel="stylesheet" href="embed.css">\n')
s, n = re.subn(r'(\n\s*<link rel="stylesheet" href="review.css">\n)', r'\1' + hook, s, count=1)
if n != 1:
    s, n = re.subn(r'(</head>)', hook + r'\1', s, count=1)
open(p, 'w').write(s)
print('embed hook inserted' if n == 1 else 'WARNING: could not insert embed hook')
EOF
else
  echo "embed hook already present"
fi

# site runtime patches (generated with: git diff HEAD -- playground/src > tools/playground-patches/0001-site-runtime.patch)
PATCHES=("$ROOT"/tools/playground-patches/*.patch)
PATCH_FAILED=0
if [ -e "${PATCHES[0]}" ]; then
  for patch in "${PATCHES[@]}"; do
    if git -C "$ROOT" apply --check "$patch"; then
      git -C "$ROOT" apply "$patch" && echo "applied $(basename "$patch")"
    else
      echo "ERROR: $(basename "$patch") does not apply to this release; port it by hand (git apply --3way \"$patch\" shows the conflicts)" >&2
      PATCH_FAILED=1
    fi
  done
fi

du -sh "$DST"
find "$DST" -type f -size +100M -print -exec echo "  ^ over GitHub's 100 MB file limit" \;
if [ "$PATCH_FAILED" = 1 ]; then
  echo "NOT DONE: playground/ holds the pristine release without the site patches (index.html promises the reach ring); port the patch before committing." >&2
  exit 1
fi
echo "done. Preview: serve the repo root with a Range-capable server and open /playground/index.html?profile=release"
