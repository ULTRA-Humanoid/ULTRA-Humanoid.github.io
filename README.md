# ULTRA project page

Source of <https://ultra-humanoid.github.io/>, the project page for

**ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation**
Xialin He\*, Sirui Xu\*, Xinyao Li, Runpei Dong, Liuyu Bian, Yu-Xiong Wang†, Liang-Yan Gui† (UIUC). IROS 2026. [arXiv 2603.03279](https://arxiv.org/abs/2603.03279)

## Layout

- `index.html` – single page. Hero stage (full-bleed video that scrolls into the title), real-world clips, abstract, result chapters, BibTeX.
- `static/css/index.css` – all styling (no framework).
- `static/js/index.js` – hero scroll animation (GSAP ScrollTrigger), lazy video loading, filmstrips, rail nav, BibTeX copy.
- `static/js/vendor/` – GSAP 3 + ScrollTrigger (vendored, no CDN dependency).
- `static/videos/` – H.264 MP4 clips (30 fps, no audio, faststart). `hero/` holds the opening video in 1080p and 720p (720p is served below 900 px viewports or on slow connections); `scene/` holds the crowd render that is scrubbed by scroll behind the abstract.
- `static/posters/` – one JPEG poster per clip; clips load their MP4 only when scrolled near.
- `static/images/` – favicon and `social_preview.jpg` (1200×630 Open Graph image).
- `static/pdfs/ULTRA.pdf` – paper.

## Editing

- Add a clip: drop the MP4 in `static/videos/<section>/`, create a poster (`ffmpeg -ss 1 -i clip.mp4 -frames:v 1 -vf scale=960:-2 -q:v 4 static/posters/clip.jpg`) and copy one of the `<figure class="clip">` blocks in `index.html`.
- Encode for the web (H.264, no audio, faststart):
  `ffmpeg -i in.mov -an -vf "fps=30,scale=1280:-2,format=yuv420p" -c:v libx264 -preset slow -crf 24 -movflags +faststart out.mp4`
- Preview locally with a server that supports HTTP Range requests (needed for seekable video; `python3 -m http.server` does not), e.g. `npx http-server -p 8765`, then open <http://localhost:8765/>.

The hero animation runs only on viewports at least 901 px wide with motion enabled; smaller screens and `prefers-reduced-motion` get a static hero.

Layout adapted from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
