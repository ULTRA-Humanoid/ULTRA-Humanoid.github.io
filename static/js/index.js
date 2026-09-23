/* ULTRA project page - behaviour
   - hero stage: scroll-driven video -> title transition (GSAP ScrollTrigger, desktop only)
   - clips: lazy source loading + autoplay/pause in view + click to pause
   - filmstrips, rail nav, reading progress, BibTeX copy
*/
(function () {
  'use strict';

  var d = document, w = window;
  var html = d.documentElement;
  var $ = function (s, r) { return (r || d).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || d).querySelectorAll(s)); };
  var supportsIO = 'IntersectionObserver' in w;

  function safePlay(v) { var p = v.play(); if (p && typeof p.catch === 'function') p.catch(function () {}); }
  function ensureSource(v) {
    if (v.dataset.poster && !v.getAttribute('poster')) v.poster = v.dataset.poster;
    if (v.dataset.src && !v.getAttribute('src')) { v.src = v.dataset.src; v.load(); }
  }

  /* ---------------------------------------------------------------- clips */
  function setupClips() {
    var clips = $$('.clip');
    var vids = clips.map(function (c) { return $('video', c); }).filter(Boolean);
    var hero = $('#hero-video');
    var sceneVideo = $('#scene-video');
    if (sceneVideo) {
      /* decorative 5 MB render: phones and data-saver users get the poster only; larger screens loop it in view (anim mode scrubs it instead) */
      var conn = navigator.connection;
      var wantScene = w.matchMedia('(min-width: 700px)').matches && !(conn && conn.saveData);
      if (wantScene) { sceneVideo.loop = true; vids.push(sceneVideo); }
      else if (sceneVideo.dataset.poster) { sceneVideo.poster = sceneVideo.dataset.poster; }
    }

    var loadIO = supportsIO ? new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        var v = e.target;
        ensureSource(v);
        loadIO.unobserve(v);
      });
    }, { rootMargin: '120% 60% 120% 60%' }) : null;

    var playIO = supportsIO ? new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.intersectionRatio >= 0.35) {
          if (v.paused && !v.dataset.userPaused) { ensureSource(v); safePlay(v); }
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { threshold: [0, 0.35, 0.7] }) : null;

    vids.forEach(function (v) {
      if (loadIO) loadIO.observe(v); else ensureSource(v);
      if (playIO) playIO.observe(v);
    });
    if (hero && playIO) playIO.observe(hero);

    clips.forEach(function (fig) {
      var v = $('video', fig), btn = $('.clip-toggle', fig);
      if (!v) return;
      var toggle = function () {
        if (v.paused) {
          delete v.dataset.userPaused;
          fig.classList.remove('is-paused');
          ensureSource(v);
          safePlay(v);
        } else {
          v.dataset.userPaused = '1';
          fig.classList.add('is-paused');
          v.pause();
        }
        if (btn) btn.setAttribute('aria-pressed', v.paused ? 'true' : 'false');
      };
      fig.addEventListener('click', toggle);
    });
  }

  /* --------------------------------------------------------------- strips */
  function setupStrips() {
    $$('[data-strip]').forEach(function (strip) {
      var track = $('.strip-track', strip);
      var items = $$('.strip-item', strip);
      var prev = $('.strip-btn.prev', strip);
      var next = $('.strip-btn.next', strip);
      var count = $('.strip-count', strip);
      if (!track || !items.length) return;

      var step = function () {
        var gap = parseFloat(getComputedStyle(track).columnGap) || 16;
        return items[0].getBoundingClientRect().width + gap;
      };
      var update = function () {
        var max = track.scrollWidth - track.clientWidth;
        var atStart = track.scrollLeft <= 4, atEnd = track.scrollLeft >= max - 4;
        strip.classList.toggle('at-start', atStart);
        strip.classList.toggle('at-end', atEnd);
        if (prev) prev.disabled = atStart;
        if (next) next.disabled = atEnd;
        if (count) {
          var idx = atEnd ? items.length : Math.min(items.length, Math.round(track.scrollLeft / step()) + 1);
          count.textContent = idx + ' / ' + items.length;
        }
        /* cards hidden inside the scroller never intersect the viewport, so prefetch the next screenful by hand */
        var r = strip.getBoundingClientRect();
        if (r.top < w.innerHeight * 1.5 && r.bottom > -w.innerHeight * 0.5) {
          var limit = track.scrollLeft + track.clientWidth * 2;
          items.forEach(function (it) { if (it.offsetLeft < limit) { var v = $('video', it); if (v) ensureSource(v); } });
        }
      };
      if (prev) prev.addEventListener('click', function () { track.scrollBy({ left: -step(), behavior: 'smooth' }); });
      if (next) next.addEventListener('click', function () { track.scrollBy({ left: step(), behavior: 'smooth' }); });
      track.addEventListener('scroll', update, { passive: true });
      w.addEventListener('resize', update);
      w.addEventListener('scroll', update, { passive: true });
      update();
    });
  }

  /* ------------------------------------------- progress, rail, back-to-top */
  function setupChrome() {
    var bar = $('#read-progress'), toTop = $('#to-top'), rail = $('#rail');
    var links = rail ? $$('a', rail) : [];
    var sections = links.map(function (a) { return d.getElementById(a.getAttribute('href').slice(1)); });
    var ticking = false;

    function update() {
      ticking = false;
      var y = w.pageYOffset || html.scrollTop;
      var max = html.scrollHeight - w.innerHeight;
      if (bar) bar.style.width = (max > 0 ? Math.min(100, (y / max) * 100) : 0) + '%';
      if (toTop) toTop.classList.toggle('visible', y > 900);

      var first = sections[0];
      var pastStage = first ? first.getBoundingClientRect().top < w.innerHeight * 0.6 : y > 600;
      if (rail) rail.classList.toggle('visible', pastStage);

      var line = w.innerHeight * 0.45, current = -1;
      sections.forEach(function (s, i) { if (s && s.getBoundingClientRect().top <= line) current = i; });
      if (y + w.innerHeight >= html.scrollHeight - 2) current = sections.length - 1;
      links.forEach(function (a, i) {
        var on = i === current;
        a.classList.toggle('is-active', on);
        if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
      });
    }
    w.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; w.requestAnimationFrame(update); }
    }, { passive: true });
    w.addEventListener('resize', update);
    if (toTop) toTop.addEventListener('click', function () { w.scrollTo({ top: 0, behavior: 'smooth' }); });
    update();
  }

  /* --------------------------------------------------------------- bibtex */
  function setupBibtex() {
    var btn = $('#copy-bib'), code = $('#bibtex-code');
    if (!btn || !code) return;
    var label = $('.copy-label', btn), timer;
    var done = function (ok) {
      btn.classList.toggle('copied', ok);
      label.textContent = ok ? 'Copied' : 'Copy failed';
      clearTimeout(timer);
      timer = setTimeout(function () { btn.classList.remove('copied'); label.textContent = 'Copy'; }, 2000);
    };
    var fallback = function (text) {
      try {
        var ta = d.createElement('textarea');
        ta.value = text; ta.setAttribute('readonly', '');
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        d.body.appendChild(ta); ta.select();
        var ok = d.execCommand('copy');
        d.body.removeChild(ta);
        done(ok);
      } catch (e) { done(false); }
    };
    btn.addEventListener('click', function () {
      var text = code.textContent;
      if (navigator.clipboard && w.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, function () { fallback(text); });
      } else {
        fallback(text);
      }
    });
  }

  /* ------------------------------------------------------------ playground */
  /* The MuJoCo + ONNX demo lives in /playground/ (same origin). Nothing from it is
     fetched until the visitor clicks Launch: the card is a poster + muted teaser loop.
     Ready/loading text is mirrored from the demo's own #status line. */
  function setupPlayground() {
    var play = $('#play');
    if (!play) return;
    var frame = $('#play-frame', play), btn = $('#play-btn', play), teaser = $('.play-teaser', play);
    var note = $('#play-note', play), loadText = $('#play-loading-text', play), errBox = $('#play-error', play);
    var status = $('#play-status', play), full = $('#play-full', play), chip = $('#play-chip', play);
    var conn = navigator.connection;
    var saveData = !!(conn && conn.saveData);
    var canRun = typeof WebAssembly === 'object' && (function () {
      try { var c = d.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch (e) { return false; }
    })();
    var linkOnly = w.matchMedia('(max-width: 900px), (pointer: coarse)').matches || saveData || !canRun;

    /* teaser loop follows the same rules as the other clips: load near the viewport, pause out of it */
    if (teaser && !saveData) {
      if (supportsIO) {
        new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (play.classList.contains('is-live')) return;
            if (e.isIntersecting) { ensureSource(teaser); safePlay(teaser); } else { teaser.pause(); }
          });
        }, { rootMargin: '60% 0px' }).observe(teaser);
      } else { ensureSource(teaser); safePlay(teaser); }
    }

    if (linkOnly || !btn) { /* phones, tablets, data saver, no WebGL/WASM: never build the iframe; the card links to the full page */
      play.classList.add('is-link');
      if (note) note.textContent = !canRun ? 'Needs WebGL and WebAssembly: ultra-humanoid.github.io/playground'
        : 'The interactive demo needs a mouse and keyboard: ultra-humanoid.github.io/playground';
      return;
    }

    var iframe = null, timer = 0, started = 0, live = false, failed = false, pausedByUs = false;
    function demoDoc() { try { return iframe && iframe.contentDocument; } catch (e) { return null; } }
    function demoApi() { try { return iframe && iframe.contentWindow && iframe.contentWindow.__interactiveDemo; } catch (e) { return null; } }
    function readStatus() {
      var doc = demoDoc(), el = doc && doc.getElementById('status');
      return el ? el.textContent.trim().replace(/^\[[^\]]+\]\s*/, '') : '';
    }
    function visibleShare() {
      var r = frame.getBoundingClientRect();
      var vis = Math.min(r.bottom, w.innerHeight) - Math.max(r.top, 0);
      return r.height > 0 ? Math.max(0, vis) / r.height : 0;
    }
    function fail(title, detail) {
      failed = true;
      clearTimeout(timer); timer = 0;
      play.classList.remove('is-loading');
      play.classList.add('is-error');
      if (errBox) { errBox.textContent = ''; var b = d.createElement('b'); b.textContent = title; errBox.appendChild(b); if (detail) errBox.appendChild(d.createTextNode(' ' + detail)); }
    }
    function goLive() {
      live = true;
      play.classList.remove('is-loading');
      play.classList.add('is-live');
      if (teaser) teaser.pause();
      if (full && (d.fullscreenEnabled || d.webkitFullscreenEnabled)) full.hidden = false;
      /* hand the keyboard to the demo only if the visitor is still looking at it */
      var ae = d.activeElement;
      if (visibleShare() >= 0.5 && (!ae || ae === d.body || play.contains(ae))) { try { iframe.contentWindow.focus(); } catch (e) {} }
    }
    function poll() {
      var t = readStatus(), waited = performance.now() - started;
      if (!live) {
        if (t && loadText) loadText.textContent = t;
        if (status) status.textContent = t;
        if (/^ERROR/.test(t)) { fail('The simulation could not start.', t.replace(/^ERROR[^:]*:\s*/, '').split('\n')[0]); return; }
        if (!t && waited > 25000 && loadText) loadText.textContent = 'Still downloading — the first visit fetches about 125 MB.';
        /* "Ready" arrives before the walking motions load and before the first frame; "Running at" is the steady state */
        if (/^Running/.test(t) || waited > 180000) goLive();
      } else if (status) {
        status.textContent = /^ERROR/.test(t) ? t : ''; /* after launch the frame speaks for itself; only surface errors */
      }
      timer = setTimeout(poll, live ? 1000 : 250);
    }
    btn.addEventListener('click', function () {
      if (iframe) return;
      iframe = d.createElement('iframe');
      iframe.src = play.dataset.src;
      iframe.title = 'ULTRA interactive demo: MuJoCo simulation of the humanoid';
      iframe.setAttribute('allow', 'fullscreen');
      iframe.setAttribute('allowfullscreen', '');
      frame.insertBefore(iframe, frame.firstChild); /* under the teaser; the teaser fades once the demo reports Running */
      iframe.addEventListener('load', function () {
        var doc = demoDoc();
        if (doc && !doc.getElementById('mujoco_canvas')) { fail('Nothing is deployed at playground/ yet.'); return; }
        /* Same origin: a plain wheel over the canvas would be eaten by the demo's OrbitControls zoom and trap the page
           scroll. Stop it before it reaches the canvas so the wheel scrolls the page; Ctrl/Cmd + wheel still zooms. */
        if (doc) doc.addEventListener('wheel', function (e) { if (!e.ctrlKey && !e.metaKey) e.stopPropagation(); }, { capture: true, passive: true });
      });
      started = performance.now();
      play.classList.add('is-loading');
      if (chip) chip.textContent = 'Loading';
      if (loadText) loadText.textContent = 'Starting…';
      poll();
    });

    if (full) full.addEventListener('click', function () {
      if (d.fullscreenElement || d.webkitFullscreenElement) { (d.exitFullscreen || d.webkitExitFullscreen).call(d); return; }
      var req = frame.requestFullscreen || frame.webkitRequestFullscreen;
      if (req) req.call(frame);
    });
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (ev) {
      d.addEventListener(ev, function () { if (full) full.textContent = (d.fullscreenElement || d.webkitFullscreenElement) ? 'Exit fullscreen' : 'Fullscreen'; });
    });

    /* Out of view: physics pauses (rendering keeps running, so no blank frame) and the keyboard goes back to the page
       so Space / arrows are not swallowed by the demo while the visitor reads on. Back in view: resume. Errors are never resumed over. */
    function demoOwnsPause() { /* fallen robot ("Reset needed") or a step error: the demo paused itself, leave it alone */
      var doc = demoDoc(), badge = doc && doc.getElementById('mode-badge');
      return /^ERROR/.test(readStatus()) || !!(badge && /reset needed/i.test(badge.textContent));
    }
    function setAway(away) {
      if (!iframe || !live) return;
      var api = demoApi();
      if (away) {
        if (api && !pausedByUs && !demoOwnsPause()) { pausedByUs = true; try { api.pause(); } catch (e) {} }
        if (d.activeElement === iframe) { try { iframe.blur(); w.focus(); } catch (e) {} }
      } else if (api && pausedByUs) {
        pausedByUs = false;
        if (!demoOwnsPause()) { try { api.resume(); } catch (e) {} }
      }
    }
    if (supportsIO) new IntersectionObserver(function (entries) { entries.forEach(function (e) { setAway(!e.isIntersecting); }); }, { threshold: 0 }).observe(frame);
    d.addEventListener('visibilitychange', function () { /* rAF already idles the sim in hidden tabs; stop polling too */
      if (d.hidden) { clearTimeout(timer); timer = 0; } else if (iframe && !timer && !failed) poll();
    });
  }

  /* ---------------------------------------------------------------- stage */
  function setupStage() {
    var stage = $('#stage');
    if (!stage) return;
    if (!w.gsap || !w.ScrollTrigger) { html.classList.remove('stage-anim'); return; }
    gsap.registerPlugin(ScrollTrigger);

    var mm = gsap.matchMedia();
    mm.add('(min-width: 901px) and (min-height: 540px) and (prefers-reduced-motion: no-preference)', function () {
      var videoWrap = $('#stage-video'), scrim = $('#stage-scrim'), brand = $('#stage-brand');
      var kicker = $('#stage-kicker'), cue = $('#scroll-cue'), titleBlock = $('#stage-title');
      var letters = $$('#stage-acronym .ac-letter');
      var targets = $$('#paper-title .tl.ac');
      var colon = $('#paper-title .tl.colon');
      var words = $$('#paper-title .tw');
      var meta = $$('#stage-title .reveal');
      if (!videoWrap || !titleBlock || letters.length !== targets.length || html.classList.contains('stage-locked')) { html.classList.remove('stage-anim'); return; }

      html.classList.add('stage-anim');

      /* measurements (layout metrics ignore transforms, so they are safe at any scroll progress) */
      var M = { video: { scale: 0.7, y: 0, radius: 30 }, letters: letters.map(function () { return { x: 0, y: 0, scale: 0.2 }; }) };
      function layoutRect(el) {
        var x = 0, y = 0, n = el;
        while (n && n !== stage) { x += n.offsetLeft; y += n.offsetTop; n = n.offsetParent; }
        return { x: x, y: y, w: el.offsetWidth, h: el.offsetHeight };
      }
      var measuredAt = -1;
      function fresh() { /* getters run after ScrollTrigger has re-applied the pin with the new viewport size */
        var now = performance.now();
        if (now - measuredAt > 16) { measure(); measuredAt = now; }
        return M;
      }
      function measure() {
        var vw = stage.clientWidth, vh = stage.clientHeight;
        if (!vw || !vh) return;
        var tb = layoutRect(titleBlock);
        var gap = Math.max(22, vh * 0.045), bottom = Math.max(22, vh * 0.05), side = Math.max(24, vw * 0.07);
        var availH = vh - (tb.y + tb.h + gap) - bottom;
        var s = Math.min(availH / vh, (vw - 2 * side) / vw, 0.8);
        s = Math.max(s, 0.3);
        var cardH = vh * s, cy = tb.y + tb.h + gap + cardH / 2;
        M.video = { scale: s, y: cy - vh / 2, radius: 22 / s };
        M.letters = letters.map(function (el, i) {
          var a = layoutRect(el), b = layoutRect(targets[i]);
          var fa = parseFloat(getComputedStyle(el).fontSize) || 1;
          var fb = parseFloat(getComputedStyle(targets[i]).fontSize) || 1;
          return {
            x: (b.x + b.w / 2) - (a.x + a.w / 2),
            y: (b.y + b.h / 2) - (a.y + a.h / 2),
            scale: fb / fa
          };
        });
      }

      gsap.set(targets, { opacity: 0 });
      gsap.set(colon, { opacity: 0 });
      gsap.set(words, { opacity: 0, y: 16 });
      gsap.set(meta, { opacity: 0, y: 18 });
      html.classList.add('stage-ready');

      var tl = gsap.timeline({ defaults: { ease: 'none' } });
      tl.to(cue, { autoAlpha: 0, duration: 0.08 }, 0)
        .to([kicker, brand], { autoAlpha: 0, y: -16, duration: 0.18 }, 0.02)
        .to(videoWrap, {
          scale: function () { return fresh().video.scale; },
          y: function () { return fresh().video.y; },
          borderRadius: function () { return fresh().video.radius + 'px'; },
          ease: 'power2.inOut', duration: 0.58
        }, 0.14)
        .to(scrim, { opacity: 0.25, duration: 0.5 }, 0.14)
        .to(letters, {
          x: function (i) { return fresh().letters[i].x; },
          y: function (i) { return fresh().letters[i].y; },
          scale: function (i) { return fresh().letters[i].scale; },
          ease: 'power2.inOut', duration: 0.58
        }, 0.14)
        .to(letters, { opacity: 0, duration: 0.04 }, 0.72)
        .to(targets, { opacity: 1, duration: 0.04 }, 0.72)
        .to(colon, { opacity: 1, duration: 0.06 }, 0.75)
        .to(words, { opacity: 1, y: 0, duration: 0.14, stagger: 0.018, ease: 'power1.out' }, 0.74)
        .to(meta, { opacity: 1, y: 0, duration: 0.16, stagger: 0.05, ease: 'power1.out' }, 0.84);

      ScrollTrigger.create({
        trigger: stage,
        start: 'top top',
        end: '+=170%',
        pin: true,
        scrub: 0.6,
        animation: tl,
        anticipatePin: 1,
        invalidateOnRefresh: true,
        onRefreshInit: measure
      });
      measure();

      /* scene: the crowd render behind the abstract is scrubbed by scroll instead of playing */
      var scene = $('#abstract'), sceneVideo = $('#scene-video');
      if (scene && sceneVideo) {
        sceneVideo.dataset.userPaused = '1';
        sceneVideo.loop = false;
        sceneVideo.pause();
        var proxy = { t: 0 }, lastT = -1, fellBack = false;
        var fallbackPlay = function () { /* server without Range support: media is not seekable -> just loop it */
          if (fellBack) return;
          fellBack = true;
          delete sceneVideo.dataset.userPaused;
          sceneVideo.loop = true;
          safePlay(sceneVideo);
        };
        var seek = function () {
          if (fellBack || sceneVideo.readyState < 1) return;
          if (sceneVideo.readyState >= 3 && !(sceneVideo.seekable.length && sceneVideo.seekable.end(0) > 1)) { fallbackPlay(); return; }
          var t = proxy.t * (sceneVideo.duration || 0);
          if (Math.abs(t - lastT) > 0.02) { lastT = t; sceneVideo.currentTime = t; }
        };
        sceneVideo.addEventListener('canplay', seek);
        sceneVideo.addEventListener('loadedmetadata', seek);
        ScrollTrigger.create({
          trigger: scene, start: 'top bottom', end: 'bottom bottom', scrub: 0.5,
          animation: gsap.to(proxy, { t: 1, ease: 'none', onUpdate: seek })
        });
      }

      return function () {
        html.classList.remove('stage-anim'); html.classList.remove('stage-ready');
        if (sceneVideo) { delete sceneVideo.dataset.userPaused; sceneVideo.loop = true; }
      };
    });

    if (d.fonts && d.fonts.ready) d.fonts.ready.then(function () { ScrollTrigger.refresh(); });
  }

  function init() {
    setupClips();
    setupStrips();
    setupChrome();
    setupBibtex();
    setupPlayground();
    setupStage();
  }
  if (d.readyState === 'loading') d.addEventListener('DOMContentLoaded', init); else init();
})();
