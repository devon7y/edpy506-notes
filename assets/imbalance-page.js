/* Every figure on imbalance.html. Quick sales are made rare with
   marketLabels(), which slows the market rather than deleting homes: the same
   homes and the same relationship, with a lower base rate. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  mean, rng, clipRect, directLabels, tweenInput,
} from './site.js';
import { logisticFit, confusion, classMetrics, atThreshold, split } from './classify.js';
import { growTree, predictTree, countLeaves } from './trees.js';
import { undersample, oversample, smote, smoteUnder, classes } from './imbalance.js';
import { runJobs } from './tuning.js';
import { currentDataset, marketLabels } from './datasets.js';

initChrome();

const ds = currentDataset();
const tip = tooltip();
const pct = (v, d = 1) => (Number.isFinite(v) ? `${(100 * v).toFixed(d)}%` : 'undefined');
const over = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
const MIN = '--series-1';     // the minority class: sold within thirty days
const MAJ = '--series-8';     // the majority class: still listed

/* One set of homes for the whole page, with asking price and distance to the
   LRT as the two features the thirty-day rule depends on. */
const HOMES = ds.sample(5, 1600);
const X = HOMES.map((r) => [r.over, r.lrt]);
const XR = { lo: -24, hi: 28 }, YR = { lo: 0, hi: 8.4 };

/* ===================================================================
   3. How severe: one model, refitted as the market slows
   =================================================================== */
const SHARES = Array.from({ length: 50 }, (_, i) => (i + 1) / 100);
const SPLITS = 6;
/* At 1% the held-out set holds a handful of quick sales, so one split's
   recall is mostly luck. The counts are pooled over six splits before any
   metric is computed, which shows the trend rather than one draw of it. The
   work is spread across frames so the page stays usable while it runs. */
const SWEEP = new Array(SHARES.length).fill(null);
const sweepJob = (k) => () => {
  const share = SHARES[k];
  const { y } = marketLabels(HOMES, share);
  const cm = { tp: 0, fp: 0, tn: 0, fn: 0 };
  const lz = { tp: 0, fp: 0, tn: 0, fn: 0 };
  let nMin = 0, nTest = 0;
  for (let s = 0; s < SPLITS; s++) {
    const sp = split(HOMES, y, 0.7, rng(3 + s * 17));
    const f = logisticFit(sp.train.map((i) => X[i]), sp.train.map((i) => y[i]));
    const yt = sp.test.map((i) => y[i]);
    const c = confusion(yt, atThreshold(sp.test.map((i) => f.prob(X[i])), 0.5));
    const l = confusion(yt, yt.map(() => 0));
    for (const key of ['tp', 'fp', 'tn', 'fn']) { cm[key] += c[key]; lz[key] += l[key]; }
    nMin += yt.filter((v) => v).length; nTest += yt.length;
  }
  SWEEP[k] = { share, cm, m: classMetrics(cm), lazy: classMetrics(lz), nMin, nTest };
};

{
  const host = document.getElementById('sev-chart');
  const sevEl = document.getElementById('sev');
  document.getElementById('sev-sub').textContent =
    `${HOMES.length} homes, 70% for training and 30% held out, split ${SPLITS} different ways with `
    + `the counts pooled. A logistic regression on asking price and distance to the LRT, scored on `
    + `the held-out homes at the default 0.5 threshold.`;
  const SERIES = [
    { key: 'accuracy', label: 'Accuracy', color: '--text-primary', w: 2.6 },
    { key: 'recall', label: 'Recall', color: '--series-8', w: 2.4 },
    { key: 'precision', label: 'Precision', color: '--series-1', w: 2 },
    { key: 'f1', label: 'F1', color: '--series-5', w: 2 },
  ];
  document.getElementById('sev-legend').innerHTML = SERIES.map((s) =>
    `<span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token(s.color)}"></span>${s.key === 'accuracy' ? 'Accuracy, over all homes' : `${s.label}, on the quick sales`}</span>`
  ).join('') + `<span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--text-muted')}"></span>Accuracy of always answering “still listed”</span>`;

  let draw;
  const update = () => {
    const share = +sevEl.value / 100;
    sevEl.nextElementSibling.textContent = pct(share, 0);
    const R = SWEEP[+sevEl.value - 1];
    if (!R || !SWEEP[0] || !SWEEP[49]) {
      document.getElementById('sev-stats').innerHTML =
        `<div class="stat stat--muted"><div class="stat__value" style="font-size:1.2rem">Fitting…</div>`
        + `<div class="stat__label">${SWEEP.filter(Boolean).length} of ${SWEEP.length} markets done</div></div>`;
      draw();
      return;
    }
    const { m, cm, lazy } = R;
    document.getElementById('sev-stats').innerHTML = `
      <div class="stat"><div class="stat__value">${pct(m.accuracy)}</div>
        <div class="stat__label">Accuracy<br><span class="muted">always “still listed” gets ${pct(lazy.accuracy)}</span></div></div>
      <div class="stat ${m.recall < 0.2 ? 'stat--test' : ''}"><div class="stat__value">${pct(m.recall)}</div>
        <div class="stat__label">Recall<br><span class="muted">${cm.tp} of the ${R.nMin} quick sales found, over ${SPLITS} splits</span></div></div>
      <div class="stat"><div class="stat__value">${pct(m.precision)}</div>
        <div class="stat__label">Precision<br><span class="muted">${cm.tp + cm.fp ? `${cm.tp} right of ${cm.tp + cm.fp} flagged` : 'nothing flagged at all'}</span></div></div>
      <div class="stat"><div class="stat__value">${Number.isFinite(m.f1) ? m.f1.toFixed(3) : '0'}</div>
        <div class="stat__label">F1</div></div>`;
    const top = SWEEP[49], low = SWEEP[0];
    document.getElementById('sev-note').innerHTML = share >= 0.35
      ? `At ${pct(share, 0)} the classes are close to balanced and the model finds `
        + `${pct(m.recall, 0)} of the quick sales. Drag toward 1% and watch accuracy climb while `
        + `recall falls.`
      : `From a balanced market to one where 1% of homes sell quickly, accuracy rises from `
        + `<b>${pct(top.m.accuracy)}</b> to <b>${pct(low.m.accuracy)}</b> and recall falls from `
        + `<b>${pct(top.m.recall)}</b> to <b>${pct(low.m.recall)}</b>. The homes, the features and `
        + `the rule connecting them are identical at every setting. Only the base rate changed, `
        + `and a model trained where the minority is rare learns that the safest answer is the `
        + `majority one.`;
    draw();
  };

  draw = () => {
    const share = +sevEl.value / 100;
    const w = 900, h = 300;
    const pad = { l: 56, r: 150, t: 18, b: 44 };
    const svg = svgRoot(host, w, h);
    /* share on a log axis: the interesting range is 1% to 10%, and on a linear
       axis it is a sliver at the left edge */
    const sx = scale(Math.log(0.01), Math.log(0.5), pad.l, w - pad.r);
    const X_ = (s) => sx(Math.log(s));
    const sy = scale(0, 1, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Share of homes that sell within thirty days (log scale)',
      yLabel: 'Score on held-out homes', xTicks: [], yTicks: [0, 0.25, 0.5, 0.75, 1],
      yFmt: (v) => `${Math.round(v * 100)}%`,
    });
    for (const s of [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) {
      el('text', { class: 'tick', x: X_(s), y: h - pad.b + 14, 'text-anchor': 'middle' }, svg)
        .textContent = `${Math.round(s * 100)}%`;
    }
    /* slight and severe, as the lecture names them */
    el('rect', { x: X_(0.01), y: pad.t, width: X_(0.03) - X_(0.01), height: h - pad.t - pad.b,
      fill: token('--critical'), opacity: 0.07 }, svg);
    el('text', { class: 'annot', x: X_(0.01) + 5, y: pad.t + 12, fill: token('--critical'),
      'font-weight': 620 }, svg).textContent = 'severe';
    el('rect', { x: X_(0.35), y: pad.t, width: X_(0.5) - X_(0.35), height: h - pad.t - pad.b,
      fill: token('--good'), opacity: 0.07 }, svg);
    el('text', { class: 'annot', x: X_(0.5) - 5, y: pad.t + 12, 'text-anchor': 'end',
      fill: token('--good'), 'font-weight': 620 }, svg).textContent = 'slight';

    const g = el('g', { 'clip-path': clipRect(svg, 'sev-clip', pad, w, h) }, svg);
    const done = SWEEP.filter(Boolean);
    if (done.length > 1) {
      el('path', { class: 'series-line', stroke: token('--text-muted'), 'stroke-width': 1.6,
        'stroke-dasharray': '5 4', d: linePath(done.map((r) => [X_(r.share), sy(r.lazy.accuracy)])) }, g);
      const labels = [];
      for (const s of SERIES) {
        /* precision is undefined where nothing is flagged, so the line stops
           there rather than pretending the value is zero */
        const pts = done.filter((r) => Number.isFinite(r.m[s.key]) || s.key !== 'precision')
          .map((r) => [X_(r.share), sy(Number.isFinite(r.m[s.key]) ? r.m[s.key] : 0)]);
        if (pts.length > 1) {
          el('path', { class: 'series-line', d: linePath(pts), stroke: token(s.color), 'stroke-width': s.w }, g);
          labels.push({ y: pts[pts.length - 1][1], text: s.label, color: token(s.color) });
        }
      }
      directLabels(svg, labels, w - pad.r + 8, pad.t + 6, h - pad.b);
    }
    el('line', { x1: X_(share), x2: X_(share), y1: pad.t, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 2 }, svg);
  };

  sevEl.addEventListener('input', update);
  document.getElementById('sev-slight').addEventListener('click', () => tweenInput(sevEl, 40, update));
  document.getElementById('sev-severe').addEventListener('click', () => tweenInput(sevEl, 1, update, { ms: 900 }));
  responsive(host, update);
  /* the balanced and the most severe markets first, so the note can be written
     early, then everything between */
  const order = [49, 0, ...SHARES.map((_, k) => k).filter((k) => k !== 0 && k !== 49)];
  runJobs(order.map(sweepJob), () => update());

  /* section 4's table: the always-majority model on a 1% market, which needs
     no fitting */
  const { y: y1 } = marketLabels(HOMES, 0.01);
  const lazy = classMetrics(confusion(y1, y1.map(() => 0)));
  document.getElementById('why-table').innerHTML = `
    <tr><td><b>Accuracy</b></td><td class="num">${pct(lazy.accuracy)}</td>
      <td>Looks excellent, and is the share of homes in the majority class.</td></tr>
    <tr><td><b>Recall</b>, minority class</td><td class="num">${pct(lazy.recall, 0)}</td>
      <td>It catches none of the cases it exists to find.</td></tr>
    <tr><td><b>Precision</b>, minority class</td><td class="num">undefined</td>
      <td>It never predicts the minority class, so there is nothing to be precise about.</td></tr>
    <tr><td><b>F1</b>, minority class</td><td class="num">0</td>
      <td>The measure meant to summarise performance exposes the failure instead.</td></tr>`;
}

/* ===================================================================
   The severely imbalanced training set used by sections 6, 7 and 9
   =================================================================== */
/* the rare-disease example's prevalence, 50 of 1,000 */
const SEVERE_SHARE = 0.05;
const SEVERE = (() => {
  const { y } = marketLabels(HOMES, SEVERE_SHARE);
  const sp = split(HOMES, y, 0.7, rng(3));
  return {
    y,
    Xtr: sp.train.map((i) => X[i]), ytr: sp.train.map((i) => y[i]),
    Xte: sp.test.map((i) => X[i]), yte: sp.test.map((i) => y[i]),
  };
})();

/** Two bars per class count, before and after, as in the lecture's diagrams. */
function countBars(hostId, before, after, opts = {}) {
  const host = document.getElementById(hostId);
  const draw = () => {
    const w = 360, h = 150;
    const svg = svgRoot(host, w, h);
    const maxN = Math.max(before.min, before.maj, after.min, after.maj);
    const sy = scale(0, maxN * 1.12, h - 26, 18);
    const bw = 40;
    const groups = [['Before', before, 30], ['After', after, 200]];
    for (const [label, c, x0] of groups) {
      for (const [k, n, dx, col] of [['minority', c.min, 0, MIN], ['majority', c.maj, bw + 8, MAJ]]) {
        el('rect', { x: x0 + dx, y: sy(n), width: bw, height: sy(0) - sy(n), rx: 3,
          fill: token(col), opacity: 0.8 }, svg);
        el('text', { class: 'tick', x: x0 + dx + bw / 2, y: sy(n) - 4, 'text-anchor': 'middle',
          'font-weight': 620, fill: token('--text-secondary') }, svg).textContent = n.toLocaleString('en-CA');
      }
      el('text', { class: 'axis-label', x: x0 + bw + 4, y: h - 6, 'text-anchor': 'middle' }, svg).textContent = label;
    }
    if (opts.ghost) {
      /* the removed majority rows, drawn as the empty part of the bar */
      el('rect', { x: 200 + bw + 8, y: sy(before.maj), width: bw, height: sy(after.maj) - sy(before.maj),
        rx: 3, fill: 'none', stroke: token(MAJ), 'stroke-dasharray': '4 3', 'stroke-width': 1.4 }, svg);
    }
    el('path', { d: `M${135} ${h / 2 - 6} L${178} ${h / 2 - 6}`, stroke: token('--text-muted'),
      'stroke-width': 1.6, fill: 'none', 'marker-end': 'none' }, svg);
    el('text', { class: 'tick', x: 181, y: h / 2 - 2, fill: token('--text-muted') }, svg).textContent = '›';
  };
  responsive(host, draw);
}

/* ---- section 6 ---- */
{
  const c = classes(SEVERE.ytr);
  const before = { min: c.min.length, maj: c.maj.length };
  const U = undersample(SEVERE.Xtr, SEVERE.ytr, rng(4));
  const O = oversample(SEVERE.Xtr, SEVERE.ytr, rng(4));
  const cu = classes(U.y), co = classes(O.y);
  countBars('us-bars', before, { min: cu.min.length, maj: cu.maj.length }, { ghost: true });
  countBars('os-bars', before, { min: co.min.length, maj: co.maj.length });
  document.getElementById('uo-note').innerHTML =
    `On a training set with <b>${before.min}</b> quick sales and <b>${before.maj.toLocaleString('en-CA')}</b> `
    + `slow ones. Undersampling keeps every quick sale and <b>${cu.maj.length}</b> of the slow ones, `
    + `discarding ${U.dropped.toLocaleString('en-CA')} homes. Oversampling keeps every home and adds `
    + `<b>${O.added.toLocaleString('en-CA')}</b> copies of the ${before.min} quick sales, so each `
    + `real quick sale now appears about ${(co.min.length / before.min).toFixed(0)} times.`;
}

/* ===================================================================
   7. SMOTE, one step at a time
   =================================================================== */
{
  const host = document.getElementById('sm-chart');
  const kEl = document.getElementById('sm-k');
  const c = classes(SEVERE.ytr);
  const minIdx = c.min;
  /* SMOTE is re-run from scratch with the chosen k; its trace is the list of
     synthetic homes in the order they were built. */
  let trace = [];
  let built = 0;        // synthetic homes fully placed
  let phase = 0;        // within the current one: 0 none, 1 A, 2 neighbours, 3 B, 4 placed
  /* The first home is the one walked through step by step, so the seed is
     chosen, deterministically, to be one whose A and B sit visibly apart: a
     pair almost on top of each other draws a segment nobody can see. */
  const spread = (tr) => {
    const [a, b] = [SEVERE.Xtr[tr.a], SEVERE.Xtr[tr.b]];
    return Math.hypot((a[0] - b[0]) / 8, a[1] - b[1]);
  };
  /* Seeds are scrambled because the generator's first draw from a small seed
     is close to zero, which would make every small seed pick the same first
     home -- the first draw here is exactly the one that matters. */
  const rebuild = () => {
    for (let seed = 1; seed < 400; seed++) {
      trace = smote(SEVERE.Xtr, SEVERE.ytr, rng((seed * 2654435761) >>> 0), { k: +kEl.value, ratio: 0.2 }).trace;
      if (spread(trace[0]) > 0.9 && SEVERE.Xtr[trace[0].a][0] > -19) break;
    }
    built = 0; phase = 0;
  };
  rebuild();

  document.getElementById('sm-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(MIN)}"></span>A real quick sale</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(MAJ)};opacity:.35"></span>A slow sale (the majority)</span>
    <span class="legend__item"><span class="legend__swatch" style="background:transparent;border:2px solid ${token('--series-6')};transform:rotate(45deg)"></span>A synthetic quick sale</span>`;

  const STEP_TEXT = [
    'Press “Next step” to build a synthetic home.',
    '<b>Step 1.</b> Pick a real quick sale at random. Call it A.',
    '<b>Step 2.</b> Find A’s k nearest neighbours among the other quick sales.',
    '<b>Step 3.</b> Pick one of those neighbours at random. Call it B, and join A to B.',
    '<b>Step 4.</b> Place the new home at a random point on that segment.',
  ];

  let draw;
  const update = () => {
    kEl.nextElementSibling.textContent = kEl.value;
    const cur = trace[built];
    let html = `<div class="small" style="line-height:1.55">${STEP_TEXT[phase]}</div>`;
    if (cur && phase >= 1) {
      const A = SEVERE.Xtr[cur.a];
      html += `<div class="small muted" style="margin-top:0.6rem">A asks ${over(A[0])} against the local rate and is ${A[1].toFixed(1)} km from the LRT.</div>`;
    }
    if (cur && phase >= 3) {
      const B = SEVERE.Xtr[cur.b];
      html += `<div class="small muted" style="margin-top:0.3rem">B asks ${over(B[0])} and is ${B[1].toFixed(1)} km out.</div>`;
    }
    if (cur && phase >= 4) {
      html += `<div class="small muted" style="margin-top:0.3rem">The new home sits ${Math.round(cur.t * 100)}% of the way from A to B: `
        + `${over(cur.point[0])}, ${cur.point[1].toFixed(1)} km.</div>`;
    }
    html += `<div class="stats stats--1" style="margin-top:0.9rem"><div class="stat">
      <div class="stat__value" style="font-size:1.3rem">${built + (phase === 4 ? 1 : 0)}</div>
      <div class="stat__label">synthetic quick sales built<br><span class="muted">from ${minIdx.length} real ones</span></div></div></div>`;
    document.getElementById('sm-steps').innerHTML = html;
    document.getElementById('sm-note').innerHTML =
      `Every synthetic home lies on a segment between two real quick sales that are neighbours, so `
      + `it lands inside the region real quick sales already occupy rather than on top of one of `
      + `them. That is the difference from random oversampling: a copy repeats a point, a synthetic `
      + `home fills in the space between points. The neighbour search runs on standardised columns, `
      + `because a percentage and a number of kilometres cannot be compared directly.`;
    draw();
  };

  draw = () => {
    const w = 560, h = 400;
    const pad = { l: 56, r: 16, t: 16, b: 46 };
    const svg = svgRoot(host, w, h);
    /* zoomed to where the quick sales are, which is the only place SMOTE works */
    const sx = scale(-24, 12, pad.l, w - pad.r);
    const sy = scale(0, 7, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)', yLabel: 'Distance to the LRT (km)',
      xTicks: [-20, -10, 0, 10], yTicks: [0, 2, 4, 6], xFmt: (v) => `${v > 0 ? '+' : ''}${v}`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'sm-clip', pad, w, h) }, svg);
    c.maj.forEach((i) => el('circle', { cx: sx(SEVERE.Xtr[i][0]), cy: sy(SEVERE.Xtr[i][1]), r: 2,
      fill: token(MAJ), opacity: 0.18 }, g));
    /* every finished synthetic home so far */
    const diamond = (x, y, strong) => el('rect', {
      x: sx(x) - 4.2, y: sy(y) - 4.2, width: 8.4, height: 8.4,
      transform: `rotate(45 ${sx(x)} ${sy(y)})`, fill: strong ? token('--series-6') : 'none',
      stroke: token('--series-6'), 'stroke-width': 1.6, opacity: strong ? 1 : 0.85,
    }, g);
    for (let s = 0; s < built; s++) diamond(trace[s].point[0], trace[s].point[1], false);
    minIdx.forEach((i) => el('circle', { cx: sx(SEVERE.Xtr[i][0]), cy: sy(SEVERE.Xtr[i][1]), r: 4.6,
      fill: token(MIN), stroke: token('--surface-1'), 'stroke-width': 1.2 }, g));

    const cur = trace[built];
    if (!cur || phase === 0) return;
    const A = SEVERE.Xtr[cur.a];
    if (phase >= 2) {
      for (const j of cur.neighbours) {
        const N = SEVERE.Xtr[j];
        el('line', { x1: sx(A[0]), y1: sy(A[1]), x2: sx(N[0]), y2: sy(N[1]),
          stroke: token('--text-muted'), 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, g);
        el('circle', { cx: sx(N[0]), cy: sy(N[1]), r: 8, fill: 'none',
          stroke: token('--text-secondary'), 'stroke-width': 1.3 }, g);
      }
    }
    if (phase >= 3) {
      const B = SEVERE.Xtr[cur.b];
      el('line', { x1: sx(A[0]), y1: sy(A[1]), x2: sx(B[0]), y2: sy(B[1]),
        stroke: token('--series-6'), 'stroke-width': 2.4 }, g);
      el('circle', { cx: sx(B[0]), cy: sy(B[1]), r: 9, fill: 'none', stroke: token('--series-6'), 'stroke-width': 2 }, g);
      el('text', { x: sx(B[0]) + 11, y: sy(B[1]) + 15, 'font-size': 12, 'font-weight': 700,
        fill: token('--series-6') }, g).textContent = 'B';
    }
    el('circle', { cx: sx(A[0]), cy: sy(A[1]), r: 9, fill: 'none', stroke: token('--series-6'), 'stroke-width': 2 }, g);
    el('text', { x: sx(A[0]) - 12, y: sy(A[1]) - 9, 'font-size': 12, 'font-weight': 700,
      'text-anchor': 'end', fill: token('--series-6') }, g).textContent = 'A';
    if (phase >= 4) diamond(cur.point[0], cur.point[1], true);
  };

  document.getElementById('sm-step').addEventListener('click', () => {
    if (built >= trace.length) return;
    if (phase < 4) phase++;
    else { built++; phase = 1; }
    update();
  });
  document.getElementById('sm-ten').addEventListener('click', () => {
    if (phase === 4) built++;
    built = Math.min(trace.length, built + 10);
    phase = 0;
    update();
  });
  document.getElementById('sm-reset').addEventListener('click', () => { built = 0; phase = 0; update(); });
  kEl.addEventListener('input', () => { rebuild(); update(); });
  responsive(host, update);
}

/* ===================================================================
   9. All five training sets, each grown into a tree
   =================================================================== */
const STRATS = {
  original: { name: 'As collected', make: () => ({ X: SEVERE.Xtr, y: SEVERE.ytr }) },
  under: { name: 'Undersample', make: () => undersample(SEVERE.Xtr, SEVERE.ytr, rng(4)) },
  over: { name: 'Oversample', make: () => oversample(SEVERE.Xtr, SEVERE.ytr, rng(4)) },
  smote: { name: 'SMOTE', make: () => smote(SEVERE.Xtr, SEVERE.ytr, rng(4), { k: 5 }) },
  hybrid: { name: 'SMOTE + undersample', make: () => smoteUnder(SEVERE.Xtr, SEVERE.ytr, rng(4), { lift: 0.5, k: 5 }) },
};
const RESULTS = Object.fromEntries(Object.entries(STRATS).map(([key, s]) => {
  const R = s.make();
  /* a fully grown tree: flexible enough that what the training set contains
     changes what it learns, which a straight-line model would largely hide */
  const tree = growTree(R.X, R.y, { maxDepth: 12, minLeaf: 1 });
  const pred = SEVERE.Xte.map((r) => (predictTree(tree, r) >= 0.5 ? 1 : 0));
  const cm = confusion(SEVERE.yte, pred);
  const c = classes(R.y);
  return [key, { ...s, R, tree, cm, m: classMetrics(cm), leaves: countLeaves(tree),
    nMin: c.min.length, nMaj: c.maj.length }];
}));

{
  const host = document.getElementById('cmp-chart');
  const barsHost = document.getElementById('cmp-bars');
  let key = 'original';

  document.getElementById('cmp-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(MIN)}"></span>Quick sale in the training set</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(MAJ)};opacity:.4"></span>Slow sale in the training set</span>
    <span class="legend__item"><span class="legend__swatch" style="background:transparent;border:2px solid ${token('--series-6')};transform:rotate(45deg)"></span>Synthetic quick sale</span>
    <span class="legend__item"><span class="legend__swatch" style="background:${token(MIN)};opacity:.18"></span>Where the tree predicts a quick sale</span>`;

  const table = () => {
    const rows = Object.entries(RESULTS).map(([k, r]) => `
      <tr${k === key ? ' style="background:var(--accent-wash)"' : ''}>
        <td>${r.name}</td>
        <td class="num">${r.R.X.length.toLocaleString('en-CA')}</td>
        <td class="num">${r.leaves}</td>
        <td class="num">${r.cm.tp} of ${r.cm.tp + r.cm.fn}</td>
        <td class="num">${r.cm.fp}</td>
        <td class="num">${pct(r.m.recall)}</td>
        <td class="num">${pct(r.m.precision)}</td>
        <td class="num">${Number.isFinite(r.m.f1) ? r.m.f1.toFixed(3) : '—'}</td>
      </tr>`).join('');
    document.getElementById('cmp-table').innerHTML = `
      <thead><tr><th>Training set</th><th>Homes</th><th>Leaves</th><th>Quick sales caught</th>
        <th>False alarms</th><th>Recall</th><th>Precision</th><th>F1</th></tr></thead>
      <tbody>${rows}</tbody>`;
  };

  let draw;
  const update = () => {
    document.querySelectorAll('#cmp-mode button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.s === key)));
    const r = RESULTS[key];
    document.getElementById('cmp-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${pct(r.m.recall)}</div>
        <div class="stat__label">Recall on held-out homes<br><span class="muted">${r.cm.tp} of ${r.cm.tp + r.cm.fn} quick sales found</span></div></div>
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${pct(r.m.precision)}</div>
        <div class="stat__label">Precision<br><span class="muted">${r.cm.fp} false alarms</span></div></div>`;
    const o = RESULTS.original, ov = RESULTS.over, sm = RESULTS.smote, hy = RESULTS.hybrid, un = RESULTS.under;
    document.getElementById('cmp-note').innerHTML =
      `Grown on the data as collected, the tree finds <b>${o.cm.tp}</b> of the ${o.cm.tp + o.cm.fn} `
      + `quick sales in the held-out homes. <b>Oversampling barely changes that</b> — ${ov.cm.tp} — `
      + `because a fully grown tree already gives each real quick sale its own small region, and `
      + `copies of the same point give it nothing new to split on. That is the overfitting risk the `
      + `lecture attaches to replicated samples, made visible. <b>SMOTE</b> finds ${sm.cm.tp} and the `
      + `<b>hybrid</b> ${hy.cm.tp}, because the synthetic homes fill in the region between real quick `
      + `sales and the tree learns that region as a whole. <b>Undersampling</b> finds ${un.cm.tp} with `
      + `a tree of only ${un.leaves} leaves, grown from ${un.R.X.length} homes out of `
      + `${o.R.X.length.toLocaleString('en-CA')}.<br><br>`
      + `Every method that raises recall also raises the false alarms, from ${o.cm.fp} to between `
      + `${Math.min(un.cm.fp, sm.cm.fp, hy.cm.fp)} and ${Math.max(un.cm.fp, sm.cm.fp, hy.cm.fp)}. `
      + `Whether that trade is worth making depends on what a miss costs against what a false alarm `
      + `costs, which is a property of the problem and not of the method.`;
    table();
    draw();
  };

  draw = () => {
    const r = RESULTS[key];
    const w = 560, h = 420;
    const pad = { l: 56, r: 16, t: 16, b: 46 };
    const svg = svgRoot(host, w, h);
    const sx = scale(XR.lo, XR.hi, pad.l, w - pad.r);
    const sy = scale(YR.lo, YR.hi, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)', yLabel: 'Distance to the LRT (km)',
      xTicks: [-20, -10, 0, 10, 20], yTicks: [0, 2, 4, 6, 8], xFmt: (v) => `${v > 0 ? '+' : ''}${v}`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'cmp-clip', pad, w, h) }, svg);
    /* the tree's predictions, as shaded cells */
    const cw = 7, ch = 7;
    for (let px = pad.l; px < w - pad.r; px += cw) {
      for (let py = pad.t; py < h - pad.b; py += ch) {
        if (predictTree(r.tree, [sx.invert(px + cw / 2), sy.invert(py + ch / 2)]) >= 0.5) {
          el('rect', { x: px, y: py, width: cw, height: ch, fill: token(MIN), opacity: 0.16 }, g);
        }
      }
    }
    const synthStart = SEVERE.Xtr.length;
    const isSynth = (i) => (key === 'smote' && i >= synthStart);
    r.R.X.forEach((p, i) => {
      if (r.R.y[i] === 1) return;
      el('circle', { cx: sx(p[0]), cy: sy(p[1]), r: 2, fill: token(MAJ), opacity: 0.28 }, g);
    });
    /* synthetic homes as diamonds; for the hybrid they are the minority rows not
       among the originals */
    const realMin = new Set(SEVERE.Xtr.filter((_, i) => SEVERE.ytr[i] === 1).map((p) => p.join(',')));
    r.R.X.forEach((p, i) => {
      if (r.R.y[i] !== 1) return;
      const synth = key === 'smote' ? isSynth(i) : key === 'hybrid' ? !realMin.has(p.join(',')) : false;
      if (synth) {
        el('rect', { x: sx(p[0]) - 3.2, y: sy(p[1]) - 3.2, width: 6.4, height: 6.4,
          transform: `rotate(45 ${sx(p[0])} ${sy(p[1])})`, fill: 'none',
          stroke: token('--series-6'), 'stroke-width': 1.2, opacity: 0.75 }, g);
      }
    });
    /* real quick sales on top; an oversampled one shows how many copies it has */
    const copies = new Map();
    r.R.X.forEach((p, i) => { if (r.R.y[i] === 1 && realMin.has(p.join(','))) copies.set(p.join(','), (copies.get(p.join(',')) || 0) + 1); });
    for (const [k2, n] of copies) {
      const [a, b] = k2.split(',').map(Number);
      if (n > 1) {
        el('circle', { cx: sx(a), cy: sy(b), r: 4.5 + Math.sqrt(n) * 1.6, fill: 'none',
          stroke: token(MIN), 'stroke-width': 1, opacity: 0.55 }, g);
      }
      const c = el('circle', { cx: sx(a), cy: sy(b), r: 4.4, fill: token(MIN),
        stroke: token('--surface-1'), 'stroke-width': 1.1 }, g);
      c.addEventListener('pointerenter', (ev) => tip.show(
        `a real quick sale<br>asking ${over(a)}, ${b.toFixed(1)} km from the LRT`
        + `${n > 1 ? `<br><b>${n} copies</b> in this training set` : ''}`, ev.clientX, ev.clientY));
      c.addEventListener('pointerleave', () => tip.hide());
    }
  };

  const drawBars = () => {
    const r = RESULTS[key], o = RESULTS.original;
    const w = 360, h = 150;
    const svg = svgRoot(barsHost, w, h);
    const maxN = Math.max(...Object.values(RESULTS).flatMap((x) => [x.nMin, x.nMaj]));
    const sy = scale(0, maxN * 1.12, h - 26, 18);
    const bw = 40;
    for (const [label, a, b, x0] of [['As collected', o.nMin, o.nMaj, 30], [r.name, r.nMin, r.nMaj, 200]]) {
      for (const [n, dx, col] of [[a, 0, MIN], [b, bw + 8, MAJ]]) {
        el('rect', { x: x0 + dx, y: sy(n), width: bw, height: sy(0) - sy(n), rx: 3, fill: token(col), opacity: 0.8 }, svg);
        el('text', { class: 'tick', x: x0 + dx + bw / 2, y: sy(n) - 4, 'text-anchor': 'middle',
          'font-weight': 620, fill: token('--text-secondary') }, svg).textContent = n.toLocaleString('en-CA');
      }
      el('text', { class: 'axis-label', x: x0 + bw + 4, y: h - 6, 'text-anchor': 'middle' }, svg).textContent = label;
    }
  };

  document.querySelectorAll('#cmp-mode button').forEach((b) =>
    b.addEventListener('click', () => { key = b.dataset.s; update(); drawBars(); }));
  responsive(host, update);
  responsive(barsHost, drawBars);
}

/* ===================================================================
   10. The rare disease example, in numbers
   =================================================================== */
{
  const y = [...Array(950).fill(0), ...Array(50).fill(1)];
  const lazy = classMetrics(confusion(y, y.map(() => 0)));
  document.getElementById('disease-lazy').innerHTML =
    `A model that always predicts <em>healthy</em> is right about <b>${pct(lazy.accuracy, 0)}</b> of `
    + `them and misses <b>every one</b> of the 50 true cases: recall ${pct(lazy.recall, 0)}.`;
  const o = RESULTS.original, h = RESULTS.hybrid;
  document.getElementById('disease-keypoint').innerHTML =
    `<strong>The same resampling ideas apply.</strong> Undersampling the healthy patients, `
    + `oversampling the ill ones, or SMOTE can all rebalance the training data, exactly as for the `
    + `homes. On the homes at the same ${pct(SEVERE_SHARE, 0)} prevalence, the hybrid took recall to `
    + `<b>${pct(h.m.recall, 0)}</b>, against <b>${pct(o.m.recall, 0)}</b> without resampling, and cost <b>${h.cm.fp - o.cm.fp}</b> `
    + `more false alarms. When a miss can cost a life, that is a trade worth making, and when a `
    + `false alarm is the expensive mistake it may not be.`;
}
