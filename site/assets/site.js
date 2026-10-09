/* Shared chrome: theme toggle, nav highlighting, tooltip singleton, and small
   SVG helpers. No dependencies and no build step -- everything here runs from
   a static server or from GitHub Pages unchanged. */

/* One entry per topic tab, in reading order. The index is first. Add a new
   topic here when its page is written and it appears in the top bar, the
   index cards and the prev/next pager on every page at once. Pages are
   organised by topic, not by lecture: a lecture that covers two topics gets
   two pages, and a topic revisited in a later lecture grows its existing page. */
export const CHAPTERS = [
  { href: 'index.html', short: 'Start', title: 'EDPY 506' },
  { href: 'regression.html', short: 'Regression', title: 'Regression' },
  { href: 'trees.html', short: 'Trees & forests', title: 'Decision trees and random forests' },
  { href: 'classification.html', short: 'Classification', title: 'Classification' },
  { href: 'imbalance.html', short: 'Imbalance', title: 'Imbalanced classes' },
  { href: 'features.html', short: 'Features', title: 'Feature engineering' },
  { href: 'tuning.html', short: 'Tuning', title: 'Model tuning' },
];

/* Standalone pages that are not part of the topic sequence, shown in a
   dropdown after the topic tabs. Empty for now, and the dropdown does not
   render while it is. Dr. Bulut's own demos are deliberately NOT listed here:
   they are his work, they live outside this repository, and the site explains
   the same ideas in its own figures rather than republishing them. */
export const EXTRAS = {
  label: 'More',
  items: [],
};

/* ---------- theme ---------- */

const THEME_KEY = 'edpy506-theme';

export function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch { /* private mode */ }
  if (saved === 'light' || saved === 'dark') {
    document.documentElement.setAttribute('data-theme', saved);
  }
  const btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  const isDark = () => document.documentElement.getAttribute('data-theme') === 'dark'
    || (!document.documentElement.hasAttribute('data-theme')
        && matchMedia('(prefers-color-scheme: dark)').matches);
  const paint = () => {
    const dark = isDark();
    btn.textContent = dark ? '☀' : '☾';
    btn.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
  };
  paint();
  btn.addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
    paint();
    window.dispatchEvent(new CustomEvent('themechange'));
  });
}

/* ---------- chrome ---------- */

function currentPage() {
  const f = location.pathname.split('/').pop();
  return f === '' ? 'index.html' : f;
}

export function initChrome() {
  const here = currentPage();
  const inExtras = EXTRAS.items.some((c) => decodeURIComponent(c.href) === decodeURIComponent(here));

  const bar = document.querySelector('.topbar');
  if (bar) {
    const nav = bar.querySelector('.topbar__nav');
    if (nav) {
      nav.innerHTML = CHAPTERS.map((c) => {
        const cur = c.href === here ? ' aria-current="page"' : '';
        return `<a href="${c.href}"${cur}>${c.short}</a>`;
      }).join('');

      /* The dropdown is a details/summary rather than a custom widget: it gets
         keyboard support, Escape-to-close and focus handling from the browser,
         and it still works with no JS at all.

         It is inserted as a SIBLING of .topbar__nav, never inside it. That nav
         sets overflow-x:auto so the tab row can scroll on a phone, and per spec
         an auto overflow-x forces overflow-y to compute to auto as well --
         making the nav a scroll container that clips in BOTH axes. An
         absolutely-positioned panel inside it is silently cropped to nothing. */
      if (EXTRAS.items.length) {
        const items = EXTRAS.items.map((c) => {
          const cur = decodeURIComponent(c.href) === decodeURIComponent(here) ? ' aria-current="page"' : '';
          return `<a href="${c.href}"${cur}>
              <span class="navmenu__ttl">${c.short}</span>
              <span class="navmenu__sub">${c.blurb}</span>
            </a>`;
        }).join('');

        const menu = document.createElement('details');
        menu.className = 'navmenu';
        if (inExtras) menu.setAttribute('data-current', '1');
        menu.innerHTML = `<summary>${EXTRAS.label}<span class="navmenu__caret" aria-hidden="true">▾</span></summary>`;
        nav.insertAdjacentElement('afterend', menu);

        /* The panel lives on <body>, not inside the <details>, and is positioned
           with JS. Two separate things in this topbar would otherwise break it:
             1. .topbar__nav clips in both axes (see above).
             2. .topbar sets backdrop-filter, which makes it the containing block
                for position:fixed descendants, so even a fixed panel would be
                positioned against the topbar rather than the viewport.
           Hanging the panel off <body> sidesteps both. */
        const panel = document.createElement('div');
        panel.className = 'navmenu__panel';
        panel.innerHTML = items;
        document.body.appendChild(panel);

        const place = () => {
          const r = menu.getBoundingClientRect();
          panel.style.top = `${r.bottom + 6}px`;
          const w = panel.offsetWidth;
          const left = Math.min(Math.max(8, r.right - w), innerWidth - w - 8);
          panel.style.left = `${left}px`;
        };
        const setOpen = (on) => {
          menu.open = on;
          if (on) { panel.setAttribute('data-open', ''); place(); }
          else panel.removeAttribute('data-open');
        };

        menu.addEventListener('toggle', () => setOpen(menu.open));
        addEventListener('resize', () => { if (menu.open) place(); });
        addEventListener('scroll', () => { if (menu.open) place(); }, true);
        document.addEventListener('click', (e) => {
          if (!menu.contains(e.target) && !panel.contains(e.target)) setOpen(false);
        });
        document.addEventListener('keydown', (e) => {
          if (e.key === 'Escape' && menu.open) { setOpen(false); menu.querySelector('summary').focus(); }
        });
      }
    }
  }

  /* Prev/next walks the lecture sequence. */
  const pager = document.querySelector('.pager');
  if (pager) {
    const i = CHAPTERS.findIndex((c) => c.href === here);
    const prev = CHAPTERS[i - 1];
    const next = CHAPTERS[i + 1];
    pager.innerHTML = [
      prev ? `<a href="${prev.href}"><div class="dir">← Previous</div><div class="ttl">${prev.title}</div></a>` : '<div></div>',
      next ? `<a class="next" href="${next.href}"><div class="dir">Next →</div><div class="ttl">${next.title}</div></a>` : '<div></div>',
    ].join('');
  }
  initTheme();
}

/* ---------- animating a control to a computed value ---------- */

/**
 * Move one or more range inputs to target values over time, calling `update`
 * on every frame.
 *
 * A "solve it for me" button that assigns the answer teaches nothing: the
 * reader sees a before and an after and has to infer the path. Sweeping the
 * control there shows the error falling as the line rotates into place, which
 * is the thing the button is meant to demonstrate.
 *
 * `items` is [{ el, to }]. Returns a cancel function. Any tween already running
 * on one of these inputs is cancelled first, so repeated clicks do not fight,
 * and a tween cancels itself if the reader grabs the control mid-flight.
 */
const running = new WeakMap();

export function tweenInputs(items, update, opts = {}) {
  for (const { el: input } of items) running.get(input)?.();

  const from = items.map(({ el: input }) => +input.value);
  const spans = items.map(({ el: input, to }, i) => {
    const lo = +input.min || 0;
    const hi = +input.max || 1;
    return Math.abs(to - from[i]) / (hi - lo || 1);
  });
  const reach = Math.max(0, ...spans);

  const settle = () => {
    items.forEach(({ el: input, to }) => { input.value = to; });
    update();
  };

  /* Honour the reader's own setting rather than overriding it. */
  if (reach === 0 || matchMedia('(prefers-reduced-motion: reduce)').matches) {
    settle();
    return () => {};
  }

  /* A short hop should not take as long as a sweep across the whole range. */
  const ms = opts.ms ?? Math.min(750, Math.max(260, 260 + 520 * reach));
  const step = items.map(({ el: input }) => +input.step || 1);
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

  let raf = null;
  const start = performance.now();
  const cancel = () => {
    cancelAnimationFrame(raf);
    items.forEach(({ el: input }) => {
      input.removeEventListener('pointerdown', cancel);
      input.removeEventListener('input', onInput);
      running.delete(input);
    });
  };
  /* Only a real pointer or keyboard interaction stops it; the tween's own
     writes dispatch no input event, so this cannot cancel itself. */
  const onInput = (ev) => { if (ev.isTrusted) cancel(); };
  items.forEach(({ el: input }) => {
    running.set(input, cancel);
    input.addEventListener('pointerdown', cancel);
    input.addEventListener('input', onInput);
  });

  const frame = (now) => {
    const t = Math.min(1, (now - start) / ms);
    const k = ease(t);
    items.forEach(({ el: input, to }, i) => {
      const v = from[i] + (to - from[i]) * k;
      input.value = Math.round(v / step[i]) * step[i];
    });
    update();
    if (t < 1) raf = requestAnimationFrame(frame);
    else { cancel(); settle(); opts.done?.(); }
  };
  raf = requestAnimationFrame(frame);
  return cancel;
}

/** One input, which is the common case. */
export const tweenInput = (input, to, update, opts) =>
  tweenInputs([{ el: input, to }], update, opts);

/* ---------- tooltip ---------- */

let tipEl = null;

export function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement('div');
    tipEl.className = 'tooltip';
    document.body.appendChild(tipEl);
  }
  return {
    show(html, x, y) {
      tipEl.innerHTML = html;
      tipEl.setAttribute('data-show', '');
      const r = tipEl.getBoundingClientRect();
      let left = x + 14;
      let top = y - r.height - 10;
      if (left + r.width > innerWidth - 8) left = x - r.width - 14;
      if (top < 8) top = y + 16;
      tipEl.style.left = `${Math.max(8, left)}px`;
      tipEl.style.top = `${top}px`;
    },
    hide() { tipEl.removeAttribute('data-show'); },
  };
}

/* ---------- svg helpers ---------- */

export const NS = 'http://www.w3.org/2000/svg';

export function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function svgRoot(host, w, h, cls = '') {
  host.innerHTML = '';
  return el('svg', {
    class: `chart${cls ? ' ' + cls : ''}`,
    viewBox: `0 0 ${w} ${h}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
  }, host);
}

export function scale(d0, d1, r0, r1) {
  const m = (r1 - r0) / (d1 - d0 || 1);
  const f = (v) => r0 + (v - d0) * m;
  f.invert = (p) => d0 + (p - r0) / m;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  return f;
}

export function ticks(d0, d1, count = 5) {
  const span = d1 - d0;
  if (span === 0) return [d0];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(d0 / step) * step; v <= d1 + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toFixed(10));
  }
  return out;
}

export function linePath(points) {
  return points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join('');
}

/** Draws grid + axes and returns the group. */
export function frame(svg, w, h, pad, xs, ys, opts = {}) {
  const { xLabel, yLabel, xTicks, yTicks, xFmt = String, yFmt = String } = opts;
  const g = el('g', {}, svg);
  const xt = xTicks || ticks(xs.domain[0], xs.domain[1], 6);
  const yt = yTicks || ticks(ys.domain[0], ys.domain[1], 5);

  for (const v of yt) {
    const y = ys(v);
    el('line', { class: 'grid', x1: pad.l, x2: w - pad.r, y1: y, y2: y }, g);
    el('text', { class: 'tick', x: pad.l - 7, y: y + 3.5, 'text-anchor': 'end' }, g)
      .textContent = yFmt(v);
  }
  for (const v of xt) {
    const x = xs(v);
    el('text', { class: 'tick', x, y: h - pad.b + 14, 'text-anchor': 'middle' }, g)
      .textContent = xFmt(v);
  }
  el('line', { class: 'axis', x1: pad.l, x2: w - pad.r, y1: h - pad.b, y2: h - pad.b }, g);
  el('line', { class: 'axis', x1: pad.l, x2: pad.l, y1: pad.t, y2: h - pad.b }, g);

  if (xLabel) {
    el('text', { class: 'axis-label', x: (pad.l + w - pad.r) / 2, y: h - 2, 'text-anchor': 'middle' }, g)
      .textContent = xLabel;
  }
  if (yLabel) {
    const cy = (pad.t + h - pad.b) / 2;
    el('text', {
      class: 'axis-label', x: 11, y: cy, 'text-anchor': 'middle',
      transform: `rotate(-90 11 ${cy})`,
    }, g).textContent = yLabel;
  }
  return g;
}

/**
 * Draw a column of labels beside a chart without letting them overlap.
 *
 * Every series wants its label at its own last y, but two series that end close
 * together produce two labels drawn on top of each other and neither is
 * readable. This places them at their preferred y where it can, then pushes
 * them apart to a minimum gap and clamps the column to the plot area -- the
 * standard fix, and the reason a legend is not needed on these charts.
 *
 * `items` is [{ y, text, color }], in any order. Returns nothing; it draws.
 */
export function directLabels(svg, items, x, top, bottom, gap = 13) {
  const sorted = items.map((d, i) => ({ ...d, i })).sort((a, b) => a.y - b.y);
  let prev = -Infinity;
  for (const d of sorted) {
    d.y = Math.max(Math.max(top, prev + gap), Math.min(d.y, bottom));
    prev = d.y;
  }
  /* If the push-down ran past the bottom, pull the whole column back up. */
  const over = prev - bottom;
  if (over > 0) {
    let q = Infinity;
    for (let i = sorted.length - 1; i >= 0; i--) {
      sorted[i].y = Math.min(sorted[i].y, q - gap);
      q = sorted[i].y;
    }
    for (const d of sorted) d.y = Math.max(d.y, top);
  }
  for (const d of sorted) {
    el('text', {
      class: 'direct-label', x, y: d.y + 4, fill: d.color,
      'text-anchor': d.anchor || 'start', 'font-size': d.size || null,
    }, svg).textContent = d.text;
  }
}

/**
 * Tick values and a shared multiplier for an axis whose numbers are too large
 * to print in full. Returns { ticks, div, suffix } -- divide each tick by div
 * and put the suffix in the axis label, so "2.5e+10" reads as "25" with a
 * "× 10⁹" on the axis instead.
 */
export function scaledTicks(hi, count = 5, lo = 0) {
  const t = ticks(lo, hi, count);
  const mag = Math.floor(Math.log10(Math.max(...t.map(Math.abs)) || 1));
  const e = Math.floor(mag / 3) * 3;
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  return {
    ticks: t,
    div: 10 ** e,
    suffix: e === 0 ? '' : ` (× 10${String(e).split('').map((c) => sup[+c]).join('')})`,
  };
}

/** A clipPath covering the plot area, so wild curves stay inside the axes. */
export function clipRect(svg, id, pad, w, h) {
  const defs = el('defs', {}, svg);
  const cp = el('clipPath', { id }, defs);
  el('rect', { x: pad.l, y: pad.t, width: w - pad.l - pad.r, height: h - pad.t - pad.b }, cp);
  return `url(#${id})`;
}

/* ---------- small stats ---------- */

export const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export function sd(a) {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1));
}

/* ---------- deterministic RNG (so demos are reproducible) ---------- */

export function rng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

export function gauss(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function shuffle(a, rand = Math.random) {
  const out = a.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/* ---------- formatting ---------- */

export const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');

/** Dollars, with the minus sign outside the symbol rather than after it. */
export function money(v) {
  if (!Number.isFinite(v)) return '—';
  const r = Math.round(v);
  return `${r < 0 ? '−' : ''}$${Math.abs(r).toLocaleString('en-CA')}`;
}

/** Dollars to the nearest thousand, for axis ticks. */
export const money1k = (v) => `${v < 0 ? '−' : ''}$${Math.abs(Math.round(v / 1000)).toLocaleString('en-CA')}k`;

/** Reads a CSS custom property off :root (so charts follow the theme). */
export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Mouse coordinates -> viewBox units.
 *
 * `svg.chart` caps height, so any viewBox tall enough to hit that cap gets
 * scaled down and centred by preserveAspectRatio="xMidYMid meet". Arithmetic
 * on the bounding box then maps the letterbox bars onto real data.
 * getScreenCTM() is the transform the browser actually used, so inverting it
 * is exact wherever the SVG ends up.
 */
export function clientToViewBox(svg, ev) {
  const ctm = svg.getScreenCTM();
  if (!ctm) return { x: NaN, y: NaN };
  const pt = svg.createSVGPoint();
  pt.x = ev.clientX;
  pt.y = ev.clientY;
  const p = pt.matrixTransform(ctm.inverse());
  return { x: p.x, y: p.y };
}

/** Re-run a draw function whenever the theme or size changes. */
export function responsive(host, draw) {
  let raf = null;
  const go = () => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(draw);
  };
  window.addEventListener('themechange', go);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', go);
  new ResizeObserver(go).observe(host);
  draw();
}
