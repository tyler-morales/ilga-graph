/**
 * Magnifying glass lens on advocacy hero headline: follows cursor, magnifies
 * content underneath. Respects prefers-reduced-motion (static over "F") and
 * hides on touch viewports.
 */
(function () {
  var SCALE = 1.8;
  var REDUCED_MOTION_OFFSET_X = 12;
  var REDUCED_MOTION_OFFSET_Y = 10;
  /** Extra distance (px) the intro path extends left and right of the text bounds. */
  var PATH_EXTEND_PX = 100;

  function getScale() {
    var root = document.documentElement;
    var val = root && getComputedStyle(root).getPropertyValue("--magnify-scale");
    if (val) return parseFloat(val.trim()) || SCALE;
    return SCALE;
  }

  /** Single source of truth for lens size so intro and cursor positioning stay in sync with CSS (--magnify-size).
   *  Reads from root when glass is not yet styled (e.g. first frame before layout) so intro uses correct size. */
  function getLensSizePx(glassEl) {
    var doc = glassEl && glassEl.ownerDocument;
    if (!doc) return 120;
    function parseSize(computed) {
      if (!computed) return NaN;
      var val = computed.getPropertyValue("--magnify-size");
      if (!val) return NaN;
      var num = parseFloat(val.trim());
      return isNaN(num) ? NaN : num;
    }
    var fromGlass = parseSize(getComputedStyle(glassEl));
    if (!isNaN(fromGlass)) return fromGlass;
    var fromRoot = parseSize(getComputedStyle(doc.documentElement));
    if (!isNaN(fromRoot)) return fromRoot;
    return glassEl.offsetWidth || 120;
  }

  function buildLens(headline) {
    var wrapper = document.createElement("div");
    wrapper.className = "hero-headline-lens-wrap";
    headline.parentNode.insertBefore(wrapper, headline);
    wrapper.appendChild(headline);

    var lens = document.createElement("div");
    lens.className = "magnify-lens";
    lens.setAttribute("aria-hidden", "true");

    var glass = document.createElement("div");
    glass.className = "magnify-lens-glass";

    var content = document.createElement("div");
    content.className = "magnify-lens-content";

    glass.appendChild(content);
    lens.appendChild(glass);
    wrapper.appendChild(lens);

    return { wrapper: wrapper, lens: lens, glass: glass, content: content };
  }

  function cloneContent(headline, contentEl) {
    var computed = getComputedStyle(headline);
    contentEl.style.fontSize = computed.fontSize;
    contentEl.style.fontFamily = computed.fontFamily;
    contentEl.style.fontWeight = computed.fontWeight;
    contentEl.style.letterSpacing = computed.letterSpacing;
    contentEl.style.lineHeight = computed.lineHeight;
    contentEl.style.color = computed.color;
    contentEl.style.width = headline.offsetWidth + "px";
    contentEl.innerHTML = headline.innerHTML;
  }

  function updatePosition(lensEl, contentEl, x, y, scale) {
    var glass = lensEl.firstElementChild;
    var half = getLensSizePx(glass) / 2;
    var left = x - half;
    var top = y - half;
    var tx = half - x * scale;
    var ty = half - y * scale;
    lensEl.style.left = left + "px";
    lensEl.style.top = top + "px";
    contentEl.style.transform =
      "translate(" + tx + "px, " + ty + "px) scale(" + scale + ")";
  }

  function getLineBoundsInWrapper(wrapper, lineEl) {
    var lineRect = lineEl.getBoundingClientRect();
    var wrapperRect = wrapper.getBoundingClientRect();
    return {
      startX: lineRect.left - wrapperRect.left,
      endX: lineRect.right - wrapperRect.left,
      y: lineRect.top - wrapperRect.top + lineRect.height / 2,
    };
  }

  /** Bounds of the line's text only (Range). Start at first non-whitespace char ("F"), end at last char ("."). */
  function getTextBoundsInWrapper(wrapper, lineEl) {
    var textNodes = [];
    var walk = document.createTreeWalker(lineEl, NodeFilter.SHOW_TEXT, null, false);
    var n;
    while ((n = walk.nextNode())) textNodes.push(n);
    if (textNodes.length === 0) return null;
    var firstNode = null;
    var firstOffset = 0;
    for (var i = 0; i < textNodes.length; i++) {
      var node = textNodes[i];
      var text = node.textContent;
      for (var j = 0; j < text.length; j++) {
        var c = text[j];
        if (c !== " " && c !== "\n" && c !== "\t" && c !== "\r") {
          firstNode = node;
          firstOffset = j;
          break;
        }
      }
      if (firstNode) break;
    }
    if (!firstNode) return null;
    var last = textNodes[textNodes.length - 1];
    var range = document.createRange();
    range.setStart(firstNode, firstOffset);
    range.setEnd(last, last.length);
    var rect = range.getBoundingClientRect();
    var wrapperRect = wrapper.getBoundingClientRect();
    return {
      startX: rect.left - wrapperRect.left,
      endX: rect.right - wrapperRect.left,
      y: rect.top - wrapperRect.top + rect.height / 2,
    };
  }

  function runIntroAnimation(wrapper, headline, lens, content, reducedMotion) {
    if (reducedMotion || window.innerWidth <= 768) return;
    var line1 = headline.querySelector(".hero-headline-line");
    if (!line1 || !line1.textContent || !line1.textContent.trim()) return;

    var elementBounds = getLineBoundsInWrapper(wrapper, line1);
    var textBounds = getTextBoundsInWrapper(wrapper, line1);
    var bounds = textBounds || elementBounds;
    var glass = lens.firstElementChild;
    var half = getLensSizePx(glass) / 2;

    var startX = bounds.startX + half - PATH_EXTEND_PX;
    var textWidth = bounds.endX - bounds.startX;
    var endX = bounds.endX - half + PATH_EXTEND_PX;
    if (startX > endX) {
      startX = endX = bounds.startX + textWidth / 2;
    }

    var y = bounds.y;

    var FADE_IN_MS = 300;
    var INTRO_DURATION_MS = 2200;
    var OUTRO_DURATION_MS = 350;
    var introDurationSec = INTRO_DURATION_MS / 1000;

    var marks = headline.querySelectorAll(".hero-headline-mark");
    var heroEl = wrapper.closest(".advocacy-hero");
    var runSecondThenOutro = null;

    if (marks.length >= 2) {
      function getElementBoundsInWrapper(wrap, el) {
        var r = el.getBoundingClientRect();
        var wr = wrap.getBoundingClientRect();
        return {
          startX: r.left - wr.left,
          endX: r.right - wr.left,
          y: r.top - wr.top + r.height / 2,
        };
      }
      function runSingleSweep(wrap, h, ln, cnt, b, onDone) {
        h.classList.add("hero-headline--lens-active");
        ln.classList.add("magnify-lens--visible");
        ln.style.setProperty("--magnify-intro-duration", introDurationSec + "s");
        updatePosition(ln, cnt, b.startX + half - PATH_EXTEND_PX, b.y, getScale());
        requestAnimationFrame(function () {
          ln.classList.add("magnify-lens--intro");
          ln.offsetHeight;
          requestAnimationFrame(function () {
            updatePosition(ln, cnt, b.endX - half + PATH_EXTEND_PX, b.y, getScale());
            setTimeout(function () {
              ln.classList.remove("magnify-lens--intro");
              if (onDone) onDone();
            }, INTRO_DURATION_MS);
          });
        });
      }
      var bounds0 = getElementBoundsInWrapper(wrapper, marks[0]);
      var bounds1 = getElementBoundsInWrapper(wrapper, marks[1]);
      var totalIntroSec = introDurationSec * 2;
      if (heroEl) {
        heroEl.style.setProperty("--hero-highlight-delay", (FADE_IN_MS / 1000 + totalIntroSec) + "s");
      }
      headline.classList.add("hero-headline--lens-active");
      lens.classList.add("magnify-lens--visible");

      runSecondThenOutro = function () {
        runSingleSweep(wrapper, headline, lens, content, bounds1, function () {
          lens.classList.add("magnify-lens--outro");
          setTimeout(function () {
            headline.classList.remove("hero-headline--lens-active");
            lens.classList.remove("magnify-lens--visible", "magnify-lens--outro");
          }, OUTRO_DURATION_MS);
        });
      };

      setTimeout(function () {
        runSingleSweep(wrapper, headline, lens, content, bounds0, runSecondThenOutro);
      }, FADE_IN_MS);
      return;
    }

    if (heroEl) heroEl.style.setProperty("--hero-highlight-delay", (FADE_IN_MS / 1000 + introDurationSec) + "s");

    headline.classList.add("hero-headline--lens-active");
    lens.style.setProperty("--magnify-intro-duration", introDurationSec + "s");
    lens.classList.add("magnify-lens--visible");
    updatePosition(lens, content, startX, y, getScale());

    setTimeout(function () {
      lens.classList.add("magnify-lens--intro");
      lens.offsetHeight;
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          updatePosition(lens, content, endX, y, getScale());
          setTimeout(function () {
            lens.classList.remove("magnify-lens--intro");
            lens.classList.add("magnify-lens--outro");
            setTimeout(function () {
              headline.classList.remove("hero-headline--lens-active");
              lens.classList.remove("magnify-lens--visible", "magnify-lens--outro");
            }, OUTRO_DURATION_MS);
          }, INTRO_DURATION_MS);
        });
      });
    }, FADE_IN_MS);
  }

  function init() {
    var hero = document.querySelector(".advocacy-hero");
    var headline = hero && hero.querySelector(".hero-headline");
    if (!headline) return;

    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var parts = buildLens(headline);
    var wrapper = parts.wrapper;
    var lens = parts.lens;
    var glass = parts.glass;
    var content = parts.content;

    cloneContent(headline, content);

    function showLens(ev) {
      var rect = wrapper.getBoundingClientRect();
      var x = ev.clientX - rect.left;
      var y = ev.clientY - rect.top;
      x = Math.max(0, Math.min(rect.width, x));
      y = Math.max(0, Math.min(rect.height, y));
      headline.classList.add("hero-headline--lens-active");
      lens.classList.add("magnify-lens--visible");
      if (reducedMotion) {
        updatePosition(lens, content, REDUCED_MOTION_OFFSET_X, REDUCED_MOTION_OFFSET_Y, getScale());
      } else {
        updatePosition(lens, content, x, y, getScale());
      }
    }

    function moveLens(ev) {
      if (reducedMotion) return;
      if (lens.classList.contains("magnify-lens--intro")) return;
      var rect = wrapper.getBoundingClientRect();
      var x = ev.clientX - rect.left;
      var y = ev.clientY - rect.top;
      /* Clamp to wrapper so lens cannot be dragged outside the headline area */
      x = Math.max(0, Math.min(rect.width, x));
      y = Math.max(0, Math.min(rect.height, y));
      updatePosition(lens, content, x, y, getScale());
    }

    function hideLens() {
      headline.classList.remove("hero-headline--lens-active");
      lens.classList.remove("magnify-lens--visible");
    }

    wrapper.addEventListener("mouseenter", showLens);
    wrapper.addEventListener("mousemove", moveLens);
    wrapper.addEventListener("mouseleave", hideLens);

    window.addEventListener("resize", function () {
      cloneContent(headline, content);
    });

    runIntroAnimation(wrapper, headline, lens, content, reducedMotion);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
