/* ==========================================================================
   Decart Taste Index — brand.js
   Motion + interaction for brand.css. No dependencies. Call DecartBrand.init()
   (runs automatically on DOMContentLoaded).

   What each behaviour is based on:
     dropdowns      [M] site: instant show/hide, chevron rotates 180° in .2s,
                        closes on outside click / Escape [I]
     tabs           [M] site: segmented pill tabs + .35s opacity crossfade of media
     accordion      [M] site (/lucy): grid-template-rows .35s easeOutSine
     scroll reveal  [M] visual: headline words sweep from faint → grey → ink as the
                        section scrolls through; [I] the exact curve below is mine
     nav on dark    [M] site: header becomes transparent with white .85 links over
                        dark/video heroes; colour change .3s
     marquee        [M] site: duplicated track translates -50%, linear, infinite
   ========================================================================== */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var GREY = [138, 138, 142]; // #8a8a8e  [M]
  var INK = [0, 0, 0];        // #000     [M]

  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  /* ---- Dropdowns: [data-dc-dropdown] containing .dc-nav__link + .dc-dropdown__list ---- */
  function initDropdowns(root) {
    var drops = $all("[data-dc-dropdown]", root);
    function closeAll(except) {
      drops.forEach(function (d) {
        if (d !== except) { d.classList.remove("is-open"); var t = d.querySelector("button"); if (t) t.setAttribute("aria-expanded", "false"); }
      });
    }
    drops.forEach(function (d) {
      var btn = d.querySelector("button");
      if (!btn) return;
      btn.setAttribute("aria-expanded", "false");
      btn.addEventListener("click", function (e) {
        e.stopPropagation();
        var open = !d.classList.contains("is-open");
        closeAll(d);
        d.classList.toggle("is-open", open);
        btn.setAttribute("aria-expanded", String(open));
      });
    });
    document.addEventListener("click", function () { closeAll(null); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeAll(null); });
  }

  /* ---- Tabs: [data-dc-tabs] with .dc-tab buttons (data-target="id"),
          optional panels inside [data-dc-stage] with matching ids ---- */
  function initTabs(root) {
    $all("[data-dc-tabs]", root).forEach(function (group) {
      var tabs = $all(".dc-tab", group);
      group.setAttribute("role", "tablist");
      function select(tab, focus) {
        tabs.forEach(function (t) {
          var on = t === tab;
          t.setAttribute("aria-selected", String(on));
          t.tabIndex = on ? 0 : -1;
          var id = t.getAttribute("data-target");
          var panel = id && document.getElementById(id);
          if (panel) panel.classList.toggle("is-active", on);
        });
        if (focus) tab.focus();
      }
      tabs.forEach(function (t, i) {
        t.setAttribute("role", "tab");
        t.addEventListener("click", function () { select(t, false); });
        t.addEventListener("keydown", function (e) {
          var k = e.key, n = tabs.length;
          if (k === "ArrowRight") { e.preventDefault(); select(tabs[(i + 1) % n], true); }
          if (k === "ArrowLeft")  { e.preventDefault(); select(tabs[(i - 1 + n) % n], true); }
        });
      });
      var initial = tabs.filter(function (t) { return t.getAttribute("aria-selected") === "true"; })[0] || tabs[0];
      if (initial) select(initial, false);
    });
  }

  /* ---- Accordion: .dc-accordion__item with .dc-accordion__trigger ---- */
  function initAccordion(root) {
    $all(".dc-accordion__item", root).forEach(function (item) {
      var trig = item.querySelector(".dc-accordion__trigger");
      if (!trig) return;
      trig.setAttribute("aria-expanded", String(item.classList.contains("is-open")));
      trig.addEventListener("click", function () {
        var open = !item.classList.contains("is-open");
        item.classList.toggle("is-open", open);
        trig.setAttribute("aria-expanded", String(open));
      });
    });
  }

  /* ---- Scroll reveal: [data-dc-reveal] text is split into words.
          Progress 0→1 as the block travels from 85% to 35% of viewport height.
          Each word: opacity .15→1 and colour grey→ink, staggered across the line. ---- */
  function lerp(a, b, t) { return a + (b - a) * t; }
  function clamp(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  function splitWords(el) {
    if (el.getAttribute("data-dc-split") === "1") return $all(".dc-word", el);
    var text = el.textContent.replace(/\s+/g, " ").trim();
    el.setAttribute("aria-label", text);
    el.textContent = "";
    text.split(" ").forEach(function (w, i, arr) {
      var s = document.createElement("span");
      s.className = "dc-word";
      s.setAttribute("aria-hidden", "true");
      s.textContent = w;
      el.appendChild(s);
      if (i < arr.length - 1) el.appendChild(document.createTextNode(" "));
    });
    el.setAttribute("data-dc-split", "1");
    el.classList.add("dc-reveal");
    return $all(".dc-word", el);
  }

  function initReveal(root) {
    var blocks = $all("[data-dc-reveal]", root).map(function (el) {
      var ink = el.getAttribute("data-dc-reveal-ink");
      return { el: el, words: splitWords(el), ink: ink ? hexToRgb(ink) : INK };
    });
    if (!blocks.length) return;
    if (reduceMotion) { blocks.forEach(function (b) { b.words.forEach(function (w) { w.style.opacity = "1"; w.style.color = rgb(b.ink); }); }); return; }

    var ticking = false;
    function paint() {
      ticking = false;
      var vh = window.innerHeight;
      blocks.forEach(function (b) {
        var r = b.el.getBoundingClientRect();
        var start = vh * 0.85, end = vh * 0.35;
        var p = clamp((start - (r.top + r.height / 2)) / (start - end));
        var n = b.words.length;
        b.words.forEach(function (w, i) {
          // each word gets a window; the sweep is ~3 words wide
          var local = clamp(p * (n + 3) - i);
          var t1 = clamp(local * 2);        // fade-in phase: faint → grey
          var t2 = clamp(local * 2 - 1);    // settle phase: grey → ink
          w.style.opacity = String(lerp(0.15, 1, t1));
          w.style.color = rgb([lerp(GREY[0], b.ink[0], t2), lerp(GREY[1], b.ink[1], t2), lerp(GREY[2], b.ink[2], t2)]);
        });
      });
    }
    function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(paint); } }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    paint();
  }

  /* ---- Nav over dark: .dc-nav gets .dc-nav--on-dark while overlapping any [data-dc-dark] ---- */
  function initNav(root) {
    var nav = (root || document).querySelector(".dc-nav");
    var darks = $all("[data-dc-dark]", root);
    if (!nav || !darks.length) return;
    function check() {
      var h = nav.getBoundingClientRect().bottom;
      var over = darks.some(function (d) { var r = d.getBoundingClientRect(); return r.top < h && r.bottom > h / 2; });
      nav.classList.toggle("dc-nav--on-dark", over);
    }
    window.addEventListener("scroll", check, { passive: true });
    check();
  }

  /* ---- Marquee: duplicate children of .dc-marquee__track so -50% loops seamlessly ---- */
  function initMarquee(root) {
    $all(".dc-marquee__track", root).forEach(function (track) {
      if (track.getAttribute("data-dc-cloned") === "1") return;
      $all(":scope > *", track).forEach(function (c) { var k = c.cloneNode(true); k.setAttribute("aria-hidden", "true"); track.appendChild(k); });
      track.setAttribute("data-dc-cloned", "1");
    });
  }

  /* ---- Copy-to-clipboard for swatches: [data-dc-copy="#2548f6"] ---- */
  function initCopy(root) {
    $all("[data-dc-copy]", root).forEach(function (el) {
      el.addEventListener("click", function () {
        var v = el.getAttribute("data-dc-copy");
        var done = function () { el.setAttribute("data-copied", "1"); setTimeout(function () { el.removeAttribute("data-copied"); }, 900); };
        if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(v).then(done, done); else done();
      });
    });
  }

  function hexToRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.replace(/./g, "$&$&"); var n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function rgb(a) { return "rgb(" + Math.round(a[0]) + "," + Math.round(a[1]) + "," + Math.round(a[2]) + ")"; }

  function init(root) {
    initDropdowns(root); initTabs(root); initAccordion(root);
    initReveal(root); initNav(root); initMarquee(root); initCopy(root);
  }

  window.DecartBrand = { init: init, reduceMotion: reduceMotion };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { init(document); });
  else init(document);
})();
