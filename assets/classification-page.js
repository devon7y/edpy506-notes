/* Every figure on classification.html. As on the other topic pages, all the
   prose beside a figure is assembled from the same computation that drew it. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  clientToViewBox, mean, money, fmt, rng, clipRect, directLabels, tweenInput,
} from './site.js';
import { ols } from './linreg.js';
import {
  sigmoid, logisticFit, knn, svmFit, confusion, classMetrics, atThreshold,
  roc, auc, accuracyOf, split,
} from './classify.js';
import { currentDataset, labelled, classDesign } from './datasets.js';

initChrome();

const ds = currentDataset();
const SPEC = ds.classification;
const tip = tooltip();
const pct = (v) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—');
const over = (v) => `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
const CLS = ['--series-8', '--series-1'];      // 0 = still listed, 1 = sold fast
const NAMES = SPEC.classes;

/* ===================================================================
   1. A line, and a curve, on a 0/1 target
   =================================================================== */
const NOLINE = labelled(ds, 5, 60);
{
  const host = document.getElementById('noline-chart');
  const xs = NOLINE.rows.map((r) => r.over);
  const ys = NOLINE.rows.map((r) => r.fast);
  const line = ols(xs, ys);
  const curve = logisticFit(xs.map((x) => [x]), ys);
  const probeEl = document.getElementById('probe');
  let mode = 'line';

  document.getElementById('noline-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[1])}"></span>${NAMES[1]} (y = 1)</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[0])}"></span>${NAMES[0]} (y = 0)</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-fit')}"></span>Least-squares line</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--series-5')}"></span>Logistic curve</span>`;

  let draw;
  const update = () => {
    const x = +probeEl.value;
    probeEl.nextElementSibling.textContent = over(x);
    document.querySelectorAll('#noline-mode button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.m === mode)));
    const lin = line.b0 + line.b1 * x;
    const log = curve.prob([x]);
    const bad = lin < 0 || lin > 1;
    document.getElementById('noline-stats').innerHTML = `
      <div class="stat ${bad ? 'stat--test' : ''}">
        <div class="stat__value">${lin.toFixed(2)}</div>
        <div class="stat__label">What the straight line calls the probability<br>
          <span class="muted">${bad ? 'not a probability' : 'inside 0 and 1, for now'}</span></div></div>
      ${mode === 'both' ? `<div class="stat" style="margin-top:0.7rem">
        <div class="stat__value">${log.toFixed(2)}</div>
        <div class="stat__label">What the logistic curve calls it</div></div>` : ''}`;
    document.getElementById('noline-verdict').innerHTML = bad
      ? `<strong>That is not a probability.</strong> A home asking ${over(x)} against the local
         rate gets <b>${lin.toFixed(2)}</b>, and nothing in the model prevents it. The line
         keeps going in both directions, so the further out a home sits the more absurd the
         answer becomes.`
      : `Between about ${over(xAt(1))} and ${over(xAt(0))} the line returns something that could
         pass for a probability. Outside that range it does not, and the range is a property
         of this particular fit rather than of the method.`;
    draw();
  };
  /* Where the line crosses 0 and 1, which is where it stops making sense. */
  const xAt = (p) => (p - line.b0) / line.b1;

  draw = () => {
    const x = +probeEl.value;
    const w = 560, h = 380;
    const pad = { l: 58, r: 16, t: 16, b: 44 };
    const svg = svgRoot(host, w, h);
    const sx = scale(-40, 40, pad.l, w - pad.r);
    const sy = scale(-0.45, 1.45, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)',
      yLabel: 'Sold within thirty days',
      xTicks: [-40, -20, 0, 20, 40], yTicks: [0, 0.5, 1],
      xFmt: (v) => `${v > 0 ? '+' : ''}${v}`, yFmt: (v) => v.toFixed(1),
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'noline-clip', pad, w, h) }, svg);
    /* the band a probability is allowed to live in */
    el('rect', { x: pad.l, y: sy(1), width: w - pad.l - pad.r, height: sy(0) - sy(1),
      fill: token('--good'), opacity: 0.06 }, g);
    for (const p of [0, 1]) {
      el('line', { x1: pad.l, x2: w - pad.r, y1: sy(p), y2: sy(p),
        stroke: token('--baseline'), 'stroke-dasharray': '4 3' }, g);
    }
    el('text', { class: 'annot', x: pad.l + 5, y: sy(1) - 6, fill: token('--text-muted') }, g)
      .textContent = 'a probability cannot go above 1';
    el('text', { class: 'annot', x: pad.l + 5, y: sy(0) + 13, fill: token('--text-muted') }, g)
      .textContent = 'or below 0';

    el('path', { class: 'series-line', stroke: token('--c-fit'), 'stroke-width': 2.5,
      d: linePath([[sx(-40), sy(line.b0 + line.b1 * -40)], [sx(40), sy(line.b0 + line.b1 * 40)]]) }, g);
    if (mode === 'both') {
      const pts = [];
      for (let i = 0; i <= 240; i++) { const v = -40 + (80 * i) / 240; pts.push([sx(v), sy(curve.prob([v]))]); }
      el('path', { class: 'series-line', d: linePath(pts), stroke: token('--series-5'), 'stroke-width': 2.8 }, g);
    }
    xs.forEach((v, i) => el('circle', {
      cx: sx(v), cy: sy(ys[i]) + (i % 2 ? 4 : -4), r: 4,
      fill: token(CLS[ys[i]]), opacity: 0.7,
    }, g));
    el('line', { x1: sx(x), x2: sx(x), y1: pad.t, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 2, 'stroke-dasharray': '5 4' }, g);
    const lin = line.b0 + line.b1 * x;
    el('circle', { cx: sx(x), cy: sy(lin), r: 6, fill: token('--c-fit'),
      stroke: token('--surface-1'), 'stroke-width': 2 }, g);
    if (mode === 'both') {
      el('circle', { cx: sx(x), cy: sy(curve.prob([x])), r: 6, fill: token('--series-5'),
        stroke: token('--surface-1'), 'stroke-width': 2 }, g);
    }
  };

  document.querySelectorAll('#noline-mode button').forEach((b) =>
    b.addEventListener('click', () => { mode = b.dataset.m; update(); }));
  probeEl.addEventListener('input', update);
  document.getElementById('noline-note').innerHTML =
    `${xs.length} homes, each drawn at 0 or 1 and nudged apart so they can be counted. `
    + `The least-squares line crosses 1 at about <b>${over(xAt(1))}</b> and 0 at about `
    + `<b>${over(xAt(0))}</b>. Homes outside that range exist in the sample, so this is not `
    + `a problem that only appears under extrapolation.`;
  responsive(host, update);
}

/* ===================================================================
   2. The two coefficients of a logistic regression
   =================================================================== */
{
  const host = document.getElementById('logit-chart');
  const oddsHost = document.getElementById('logit-odds');
  const b0El = document.getElementById('beta0');
  const b1El = document.getElementById('beta1');
  const xs = NOLINE.rows.map((r) => r.over);
  const ys = NOLINE.rows.map((r) => r.fast);
  const FIT = logisticFit(xs.map((x) => [x]), ys);
  const P = (b0, b1, x) => sigmoid(b0 + b1 * x);

  /* Log-likelihood, which is the quantity the fit maximises. */
  const loglik = (b0, b1) => {
    let s = 0;
    for (let i = 0; i < xs.length; i++) {
      const p = Math.min(1 - 1e-12, Math.max(1e-12, P(b0, b1, xs[i])));
      s += ys[i] ? Math.log(p) : Math.log(1 - p);
    }
    return s;
  };
  const BEST = loglik(FIT.b0, FIT.b[0]);

  let draw, drawOdds;
  const update = () => {
    const b0 = +b0El.value, b1 = +b1El.value;
    b0El.nextElementSibling.textContent = b0.toFixed(2);
    b1El.nextElementSibling.textContent = b1.toFixed(3);
    const ll = loglik(b0, b1);
    const acc = accuracyOf(ys, xs.map((x) => (P(b0, b1, x) >= 0.5 ? 1 : 0)));
    const half = b1 === 0 ? NaN : -b0 / b1;
    document.getElementById('logit-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${Number.isFinite(half) ? over(half) : '—'}</div>
        <div class="stat__label">Where the curve passes 0.5<br><span class="muted">the price at which it is a coin flip</span></div></div>
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${pct(acc)}</div>
        <div class="stat__label">Accuracy on these ${xs.length} homes</div></div>
      <div class="stat ${ll > BEST - 0.5 ? '' : 'stat--muted'}">
        <div class="stat__value" style="font-size:1.25rem">${ll.toFixed(1)}</div>
        <div class="stat__label">Log-likelihood<br><span class="muted">best possible ${BEST.toFixed(1)}</span></div></div>`;
    document.getElementById('logit-note').innerHTML = ll > BEST - 0.5
      ? `<b>That is the fitted model.</b> No other pair of coefficients makes these ${xs.length} `
        + `outcomes more likely. It puts the coin flip at <b>${over(-FIT.b0 / FIT.b[0])}</b> over `
        + `the local rate, and a slope of <b>${FIT.b[0].toFixed(3)}</b> means each extra percent `
        + `asked multiplies the odds of a quick sale by `
        + `<b>${Math.exp(FIT.b[0]).toFixed(3)}</b>.`
      : `The right-hand chart is the same model on the log-odds scale, where the relationship `
        + `is a straight line with intercept <em>β</em><sub>0</sub> and slope `
        + `<em>β</em><sub>1</sub>. The S-shape on the left and the line on the right are the `
        + `same two numbers, which is the reason for writing the model that way.`;
    draw();
    drawOdds();
  };

  draw = () => {
    const b0 = +b0El.value, b1 = +b1El.value;
    const w = 440, h = 330;
    const pad = { l: 52, r: 14, t: 16, b: 44 };
    const svg = svgRoot(host, w, h);
    const sx = scale(-40, 40, pad.l, w - pad.r);
    const sy = scale(-0.12, 1.12, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)', yLabel: 'P(sold in 30 days)',
      xTicks: [-40, -20, 0, 20, 40], yTicks: [0, 0.5, 1],
      xFmt: (v) => `${v > 0 ? '+' : ''}${v}`, yFmt: (v) => v.toFixed(1),
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'logit-clip', pad, w, h) }, svg);
    el('line', { x1: pad.l, x2: w - pad.r, y1: sy(0.5), y2: sy(0.5),
      stroke: token('--baseline'), 'stroke-dasharray': '4 3' }, g);
    const pts = [];
    for (let i = 0; i <= 260; i++) { const v = -40 + (80 * i) / 260; pts.push([sx(v), sy(P(b0, b1, v))]); }
    el('path', { class: 'series-line', d: linePath(pts), stroke: token('--series-5'), 'stroke-width': 2.8 }, g);
    xs.forEach((v, i) => el('circle', {
      cx: sx(v), cy: sy(ys[i]) + (i % 2 ? 3.5 : -3.5), r: 3.6,
      fill: token(CLS[ys[i]]), opacity: 0.65,
    }, g));
  };

  drawOdds = () => {
    const b0 = +b0El.value, b1 = +b1El.value;
    const w = 440, h = 330;
    const pad = { l: 52, r: 14, t: 16, b: 44 };
    const svg = svgRoot(oddsHost, w, h);
    const sx = scale(-40, 40, pad.l, w - pad.r);
    const sy = scale(-8, 8, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)',
      yLabel: 'log odds  =  β₀ + β₁X',
      xTicks: [-40, -20, 0, 20, 40], yTicks: [-8, -4, 0, 4, 8],
      xFmt: (v) => `${v > 0 ? '+' : ''}${v}`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'odds-clip', pad, w, h) }, svg);
    el('line', { x1: pad.l, x2: w - pad.r, y1: sy(0), y2: sy(0),
      stroke: token('--baseline'), 'stroke-dasharray': '4 3' }, g);
    el('path', { class: 'series-line', stroke: token('--series-5'), 'stroke-width': 2.8,
      d: linePath([[sx(-40), sy(b0 + b1 * -40)], [sx(40), sy(b0 + b1 * 40)]]) }, g);
    el('text', { class: 'annot', x: w - pad.r - 6, y: sy(0) - 7, 'text-anchor': 'end',
      fill: token('--text-muted') }, g).textContent = 'log odds 0  =  probability 0.5';
  };

  document.getElementById('logit-fit').addEventListener('click', () => {
    tweenInput(b1El, +FIT.b[0].toFixed(3), update, { ms: 900 });
    tweenInput(b0El, +FIT.b0.toFixed(2), () => {}, { ms: 900 });
  });
  b0El.addEventListener('input', update);
  b1El.addEventListener('input', update);
  responsive(host, update);
  responsive(oddsHost, drawOdds);
}

/* ===================================================================
   3 & 4. k-nearest neighbours
   =================================================================== */
const KNN_DATA = (() => {
  const { rows } = labelled(ds, 5, 260);
  const d = classDesign(ds, rows, ['over', 'lrt']);
  const sp = split(rows, d.y, 0.6, rng(2));
  return {
    rows, ...d, ...sp,
    Xtr: sp.train.map((i) => d.X[i]), ytr: sp.train.map((i) => d.y[i]),
    Xte: sp.test.map((i) => d.X[i]), yte: sp.test.map((i) => d.y[i]),
  };
})();

{
  const host = document.getElementById('knn-chart');
  const kEl = document.getElementById('knn-k');
  let p = 2, standardise = true;
  let query = [-4, 2.5];
  const models = new Map();
  const model = () => {
    const key = `${p}-${standardise}`;
    if (!models.has(key)) models.set(key, knn(KNN_DATA.Xtr, KNN_DATA.ytr, { p, standardise }));
    return models.get(key);
  };

  document.getElementById('knn-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[1])}"></span>${NAMES[1]}</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[0])}"></span>${NAMES[0]}</span>
    <span class="legend__item"><span class="legend__swatch" style="background:${token('--series-6')};border-radius:2px"></span>The home being classified — drag it</span>`;

  let draw;
  const update = () => {
    const k = +kEl.value;
    kEl.nextElementSibling.textContent = k === 1 ? '1 — memorise' : k;
    document.querySelectorAll('#knn-metric button').forEach((b) =>
      b.setAttribute('aria-pressed', String(+b.dataset.p === p)));
    document.querySelectorAll('#knn-scale button').forEach((b) =>
      b.setAttribute('aria-pressed', String((b.dataset.s === '1') === standardise)));
    const m = model();
    const near = m.neighbours(query, k);
    const votes = near.filter((nb) => KNN_DATA.ytr[nb.i] === 1).length;
    const pred = votes * 2 >= k ? 1 : 0;
    document.getElementById('knn-vote').innerHTML = `
      <div class="stat" style="border-left:3px solid ${token(CLS[pred])}">
        <div class="stat__value" style="font-size:1.2rem;color:${token(CLS[pred])}">${NAMES[pred]}</div>
        <div class="stat__label">${k === 1
          ? `its single nearest neighbour ${votes ? 'did' : 'did not'} sell quickly`
          : `${votes} of the ${k} nearest sold quickly`}<br>
          <span class="muted">asking ${over(query[0])}, ${query[1].toFixed(1)} km from the LRT</span></div></div>`;
    document.getElementById('knn-list').innerHTML =
      `<div class="small muted" style="margin-bottom:0.4rem">The ${k === 1 ? 'closest training home' : `${k} closest training homes`}</div>`
      + `<div class="bars">${near.slice(0, 9).map((nb) => {
        const r = KNN_DATA.rows[KNN_DATA.train[nb.i]];
        const c = token(CLS[KNN_DATA.ytr[nb.i]]);
        return `<div class="bars__row" style="grid-template-columns:8.5rem 1fr 4rem">
          <div class="bars__name" style="color:${c}">${over(r.over)}, ${r.lrt.toFixed(1)} km</div>
          <div class="bars__track"><div class="bars__fill" style="width:${Math.min(100, (nb.d / (near[near.length - 1].d || 1)) * 100)}%;background:${c}"></div></div>
          <div class="bars__val">${nb.d.toFixed(2)}</div></div>`;
      }).join('')}</div>${k > 9 ? `<div class="small muted" style="margin-top:0.4rem">and ${k - 9} more</div>` : ''}`;
    document.getElementById('knn-note').innerHTML = standardise
      ? `Distances are computed on standardised columns, so a percentage and a number of `
        + `kilometres count equally. ${p === 2 ? 'Euclidean' : 'Manhattan'} distance. `
        + `Switch to raw units to see what happens without that step.`
      : `<b>Raw units.</b> Asking price runs over roughly 40 percentage points and distance `
        + `to the LRT over about 8 kilometres, so the price column is five times larger and `
        + `dominates every distance. The neighbours chosen are almost the ones nearest in `
        + `price alone, and the second feature has stopped mattering.`;
    draw();
  };

  draw = () => {
    const k = +kEl.value;
    const m = model();
    const w = 560, h = 420;
    const pad = { l: 56, r: 16, t: 16, b: 46 };
    const svg = svgRoot(host, w, h);
    const sx = scale(-26, 30, pad.l, w - pad.r);
    const sy = scale(0, 8.5, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)',
      yLabel: 'Distance to the LRT (km)',
      xTicks: [-20, -10, 0, 10, 20, 30], yTicks: [0, 2, 4, 6, 8],
      xFmt: (v) => `${v > 0 ? '+' : ''}${v}`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'knn-clip', pad, w, h) }, svg);

    if (document.getElementById('knn-boundary').checked) {
      /* The boundary is drawn as a coarse grid of shaded cells rather than a
         contour: KNN's boundary is not a curve with an equation, it is whatever
         the votes happen to carve out, and the blocks say so. */
      const step = 14;
      for (let px = pad.l; px < w - pad.r; px += step) {
        for (let py = pad.t; py < h - pad.b; py += step) {
          const cx = sx.invert(px + step / 2), cy = sy.invert(py + step / 2);
          const c = m.predict([cx, cy], k);
          el('rect', { x: px, y: py, width: step, height: step,
            fill: token(CLS[c]), opacity: 0.1 }, g);
        }
      }
    }

    const near = new Set(m.neighbours(query, k).map((nb) => nb.i));
    KNN_DATA.Xtr.forEach((r, i) => {
      const hot = near.has(i);
      if (hot) {
        el('line', { x1: sx(query[0]), y1: sy(query[1]), x2: sx(r[0]), y2: sy(r[1]),
          stroke: token('--series-6'), 'stroke-width': 1, opacity: 0.55 }, g);
      }
      el('circle', { cx: sx(r[0]), cy: sy(r[1]), r: hot ? 5.5 : 3.4,
        fill: token(CLS[KNN_DATA.ytr[i]]), opacity: hot ? 1 : 0.5,
        stroke: hot ? token('--series-6') : 'none', 'stroke-width': hot ? 1.8 : 0 }, g);
    });
    /* the query home */
    el('rect', { x: sx(query[0]) - 6, y: sy(query[1]) - 6, width: 12, height: 12,
      fill: token('--series-6'), stroke: token('--surface-1'), 'stroke-width': 2,
      transform: `rotate(45 ${sx(query[0])} ${sy(query[1])})` }, g);

    /* dragging */
    const hit = el('rect', { x: pad.l, y: pad.t, width: w - pad.l - pad.r,
      height: h - pad.t - pad.b, fill: 'transparent', style: 'cursor:crosshair' }, svg);
    const move = (ev) => {
      const q = clientToViewBox(svg, ev);
      query = [
        Math.max(-26, Math.min(30, sx.invert(q.x))),
        Math.max(0, Math.min(8.5, sy.invert(q.y))),
      ];
      update();
    };
    hit.addEventListener('pointerdown', (ev) => {
      hit.setPointerCapture(ev.pointerId);
      move(ev);
      hit.addEventListener('pointermove', move);
    });
    hit.addEventListener('pointerup', (ev) => {
      hit.releasePointerCapture(ev.pointerId);
      hit.removeEventListener('pointermove', move);
    });
  };

  kEl.addEventListener('input', update);
  document.getElementById('knn-boundary').addEventListener('change', update);
  document.querySelectorAll('#knn-metric button').forEach((b) =>
    b.addEventListener('click', () => { p = +b.dataset.p; update(); }));
  document.querySelectorAll('#knn-scale button').forEach((b) =>
    b.addEventListener('click', () => { standardise = b.dataset.s === '1'; update(); }));
  responsive(host, update);
}

/* ---- accuracy against k ---- */
{
  const host = document.getElementById('choosek-chart');
  const m = knn(KNN_DATA.Xtr, KNN_DATA.ytr);
  const KS = [];
  for (let k = 1; k <= 61; k += 2) KS.push(k);
  const curve = KS.map((k) => ({
    k,
    train: accuracyOf(KNN_DATA.ytr, KNN_DATA.Xtr.map((r) => m.predict(r, k))),
    test: accuracyOf(KNN_DATA.yte, KNN_DATA.Xte.map((r) => m.predict(r, k))),
  }));
  const best = curve.reduce((a, b) => (b.test > a.test ? b : a));
  const majority = mean(KNN_DATA.ytr) >= 0.5 ? 1 : 0;
  const baseline = accuracyOf(KNN_DATA.yte, KNN_DATA.yte.map(() => majority));

  document.getElementById('choosek-sub').textContent =
    `${KNN_DATA.Xtr.length} training homes, ${KNN_DATA.Xte.length} held out, `
    + `two features, Euclidean distance on standardised columns.`;
  document.getElementById('choosek-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-train')}"></span>Training accuracy</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-test')}"></span>Test accuracy</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--text-muted')}"></span>Always predicting the majority class</span>`;
  document.getElementById('choosek-stats').innerHTML = `
    <div class="stat"><div class="stat__value" style="font-size:1.25rem">${pct(curve[0].train)}</div>
      <div class="stat__label">Training accuracy at k = 1<br><span class="muted">test accuracy ${pct(curve[0].test)}</span></div></div>
    <div class="stat stat--test"><div class="stat__value" style="font-size:1.25rem">k = ${best.k}</div>
      <div class="stat__label">Best on held-out homes<br><span class="muted">${pct(best.test)}</span></div></div>
    <div class="stat stat--muted"><div class="stat__value" style="font-size:1.25rem">${pct(curve[curve.length - 1].test)}</div>
      <div class="stat__label">Test accuracy at k = ${curve[curve.length - 1].k}<br><span class="muted">majority class alone gives ${pct(baseline)}</span></div></div>`;
  document.getElementById('choosek-note').innerHTML =
    `At k = 1 every training home is its own nearest neighbour, so training accuracy is `
    + `<b>${pct(curve[0].train)}</b> while held-out accuracy is only <b>${pct(curve[0].test)}</b>. `
    + `That gap is the overfitting. Held-out accuracy peaks at <b>k = ${best.k}</b> `
    + `(${pct(best.test)}) and falls away again as k grows, because distant homes that have `
    + `nothing to do with the one being priced start voting.`;

  responsive(host, () => {
    const w = 900, h = 330;
    const pad = { l: 62, r: 92, t: 16, b: 44 };
    const svg = svgRoot(host, w, h);
    const sx = scale(1, 61, pad.l, w - pad.r);
    const sy = scale(0.45, 1.02, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Number of neighbours (k)', yLabel: 'Accuracy',
      xTicks: [1, 11, 21, 31, 41, 51, 61], yTicks: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
      yFmt: (v) => `${Math.round(v * 100)}%`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'ck-clip', pad, w, h) }, svg);
    el('line', { x1: pad.l, x2: w - pad.r, y1: sy(baseline), y2: sy(baseline),
      stroke: token('--text-muted'), 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }, g);
    const labels = [{ y: sy(baseline), text: 'Majority class', color: token('--text-muted'), size: 10 }];
    for (const [key, color, name] of [['train', '--c-train', 'Training'], ['test', '--c-test', 'Test']]) {
      const pts = curve.map((c) => [sx(c.k), sy(c[key])]);
      el('path', { class: 'series-line', d: linePath(pts), stroke: token(color), 'stroke-width': 2.2 }, g);
      curve.forEach((c, i) => el('circle', { cx: pts[i][0], cy: pts[i][1], r: 2.6, fill: token(color), opacity: 0.7 }, g));
      labels.push({ y: pts[pts.length - 1][1], text: name, color: token(color) });
    }
    directLabels(svg, labels, w - pad.r + 7, pad.t + 6, h - pad.b);
    el('line', { x1: sx(best.k), x2: sx(best.k), y1: pad.t, y2: h - pad.b,
      stroke: token('--good'), 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
    el('text', { class: 'annot', x: sx(best.k) + 5, y: pad.t + 12, fill: token('--good'),
      'font-weight': 620 }, svg).textContent = `best held-out accuracy: k = ${best.k}`;
    el('text', { class: 'annot', x: pad.l + 4, y: h - pad.b - 8, fill: token('--text-muted') }, svg)
      .textContent = 'overfitting';
    el('text', { class: 'annot', x: w - pad.r - 4, y: h - pad.b - 8, 'text-anchor': 'end',
      fill: token('--text-muted') }, svg).textContent = 'underfitting';
  });
}

/* ===================================================================
   5. Support vector machines
   =================================================================== */
{
  const host = document.getElementById('svm-chart');
  const cEl = document.getElementById('svm-c');
  let which = 'separable';

  /* Two datasets: homes chosen so a line does separate them, and all the homes,
     where no line does. */
  const SETS = {
    separable: (() => {
      /* Homes the generating rule is confident about, labelled by which side of
         that rule they fall rather than by the coin it flipped. The rule is
         linear in these two features, so the result is separable by
         construction and the claim the figure makes about it is true. */
      const { rows } = labelled(ds, 31, 700);
      const keep = rows.filter((r) => r.pFast > 0.9 || r.pFast < 0.1).slice(0, 60);
      const d = classDesign(ds, keep, ['over', 'lrt']);
      return { ...d, y: keep.map((r) => (r.pFast > 0.5 ? 1 : 0)), rows: keep };
    })(),
    real: (() => {
      const { rows } = labelled(ds, 9, 120);
      const d = classDesign(ds, rows, ['over', 'lrt']);
      return { ...d, rows };
    })(),
  };
  const CS = Array.from({ length: 41 }, (_, i) => 0.05 * Math.exp((i / 40) * Math.log(400)));
  const cache = new Map();
  const fitFor = (set, ci) => {
    const key = `${set}-${ci}`;
    if (!cache.has(key)) cache.set(key, svmFit(SETS[set].X, SETS[set].y, { C: CS[ci] }));
    return cache.get(key);
  };

  document.getElementById('svm-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[1])}"></span>${NAMES[1]}</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token(CLS[0])}"></span>${NAMES[0]}</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--text-primary')}"></span>The hyperplane</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--text-muted')}"></span>The margin</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:transparent;border:2px solid ${token('--series-6')}"></span>Support vectors</span>`;

  let draw;
  const update = () => {
    const ci = +cEl.value;
    const C = CS[ci];
    cEl.nextElementSibling.textContent = C < 1 ? C.toFixed(2) : C.toFixed(1);
    document.querySelectorAll('#svm-data button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.d === which)));
    const S = SETS[which];
    const m = fitFor(which, ci);
    const acc = accuracyOf(S.y, S.X.map((r) => m.predict(r)));
    document.getElementById('svm-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${m.support.length}</div>
        <div class="stat__label">Support vectors, of ${S.X.length} homes<br>
          <span class="muted">the only ones holding the boundary up</span></div></div>
      <div class="stat" style="margin-top:0.7rem"><div class="stat__value" style="font-size:1.25rem">${pct(acc)}</div>
        <div class="stat__label">Correctly classified${acc >= 0.9999 ? '<br><span class="muted">every home, on the right side</span>' : ''}</div></div>`;
    document.getElementById('svm-explain').innerHTML = which === 'separable'
      ? `<strong>A line does separate these homes.</strong> Infinitely many do. The one drawn
         is the one whose margin is widest, which is the maximal margin classifier, and only
         <b>${m.support.length}</b> homes determine where it sits. Drag any of the others
         anywhere on its own side of the margin and the line does not move.`
      : `<strong>No line separates all the homes.</strong> The two groups overlap, so the fit
         allows points inside the margin and charges <em>C</em> for each one. At this setting
         <b>${m.support.length}</b> homes are on or inside the margin, and
         <b>${pct(1 - acc)}</b> end up on the wrong side of the line entirely.`;
    document.getElementById('svm-note').innerHTML =
      `<b>C is the price of letting a home sit inside the margin.</b> Turn it down and the `
      + `margin widens, admitting more homes and taking more of them into account. Turn it up `
      + `and the margin narrows onto the few homes closest to the boundary, which fits the `
      + `training data harder and rests the whole model on fewer observations. `
      + `Here the margin spans <b>${m.width.toFixed(2)}</b> standardised units and rests on `
      + `<b>${m.support.length}</b> homes.`;
    draw();
  };

  draw = () => {
    const S = SETS[which];
    const m = fitFor(which, +cEl.value);
    const w = 560, h = 420;
    const pad = { l: 56, r: 16, t: 16, b: 46 };
    const svg = svgRoot(host, w, h);
    const xsAll = S.X.map((r) => r[0]), ysAll = S.X.map((r) => r[1]);
    const sx = scale(Math.min(...xsAll) - 3, Math.max(...xsAll) + 3, pad.l, w - pad.r);
    const sy = scale(-0.4, 8.6, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Asking price against the local rate (%)',
      yLabel: 'Distance to the LRT (km)',
      yTicks: [0, 2, 4, 6, 8], xFmt: (v) => `${v > 0 ? '+' : ''}${Math.round(v)}`,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'svm-clip', pad, w, h) }, svg);

    /* The boundary and the two margin edges, solved in data units: the model
       works on standardised columns, so each line is traced by walking x and
       solving for the y that puts the margin at the wanted value. */
    const yFor = (x, target) => {
      const zx = (x - m.mu[0]) / m.sd[0];
      if (Math.abs(m.w[1]) < 1e-9) return NaN;
      const zy = (target - m.b - m.w[0] * zx) / m.w[1];
      return zy * m.sd[1] + m.mu[1];
    };
    const xlo = sx.domain[0], xhi = sx.domain[1];
    for (const [target, stroke, width, dash] of [
      [0, token('--text-primary'), 2.6, null],
      [1, token('--text-muted'), 1.4, '5 4'],
      [-1, token('--text-muted'), 1.4, '5 4'],
    ]) {
      const a = yFor(xlo, target), b = yFor(xhi, target);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      el('path', { class: 'series-line', d: linePath([[sx(xlo), sy(a)], [sx(xhi), sy(b)]]),
        stroke, 'stroke-width': width, 'stroke-dasharray': dash }, g);
    }
    /* shade the margin band */
    {
      const a1 = yFor(xlo, 1), b1 = yFor(xhi, 1), a2 = yFor(xlo, -1), b2 = yFor(xhi, -1);
      if ([a1, b1, a2, b2].every(Number.isFinite)) {
        el('path', { d: `M${sx(xlo)} ${sy(a1)} L${sx(xhi)} ${sy(b1)} L${sx(xhi)} ${sy(b2)} L${sx(xlo)} ${sy(a2)} Z`,
          fill: token('--text-muted'), opacity: 0.08 }, g);
      }
    }

    const sup = new Set(m.support);
    S.X.forEach((r, i) => {
      const isSup = sup.has(i);
      const c = el('circle', { cx: sx(r[0]), cy: sy(r[1]), r: isSup ? 6 : 4,
        fill: token(CLS[S.y[i]]), opacity: isSup ? 0.95 : 0.45,
        stroke: isSup ? token('--series-6') : 'none', 'stroke-width': isSup ? 2 : 0 }, g);
      c.addEventListener('pointerenter', (ev) => tip.show(
        `asking <b>${over(r[0])}</b>, ${r[1].toFixed(1)} km from the LRT<br>`
        + `${NAMES[S.y[i]]}<br><span style="opacity:.7">${isSup ? 'a support vector' : 'outside the margin, holds nothing up'}</span>`,
        ev.clientX, ev.clientY));
      c.addEventListener('pointerleave', () => tip.hide());
    });
  };

  cEl.addEventListener('input', update);
  document.querySelectorAll('#svm-data button').forEach((b) =>
    b.addEventListener('click', () => { which = b.dataset.d; update(); }));
  responsive(host, update);
}

/* ===================================================================
   6-9. Evaluation: confusion matrix, metrics, threshold, ROC
   =================================================================== */
const EVAL = (() => {
  const { rows } = labelled(ds, 5, 400);
  const d = classDesign(ds, rows, ['over', 'lrt', 'size', 'age']);
  const sp = split(rows, d.y, 0.65, rng(7));
  const f = logisticFit(sp.train.map((i) => d.X[i]), sp.train.map((i) => d.y[i]));
  const y = sp.test.map((i) => d.y[i]);
  const s = sp.test.map((i) => f.prob(d.X[i]));
  /* a deliberately weaker model, fitted on one feature, for the ROC comparison */
  const fWeak = logisticFit(sp.train.map((i) => [d.X[i][1]]), sp.train.map((i) => d.y[i]));
  const sWeak = sp.test.map((i) => fWeak.prob([d.X[i][1]]));
  return { rows, ...d, ...sp, f, y, s, sWeak, A: auc(y, s), Aweak: auc(y, sWeak) };
})();

/** The four cells, drawn as the 2 x 2 grid from the lecture. */
function confusionGrid(hostId, cm, opts = {}) {
  const { highlight = null } = opts;
  const cell = (label, n, kind, sub) => {
    const right = kind === 'tp' || kind === 'tn';
    return `<div class="cm__cell${right ? ' cm__cell--right' : ''}${highlight === kind ? ' cm__cell--on' : ''}">
      <div class="cm__n">${n}</div>
      <div class="cm__label">${label}</div>
      <div class="cm__sub">${sub}</div>
    </div>`;
  };
  document.getElementById(hostId).innerHTML = `
    <div class="cm">
      <div></div>
      <div class="cm__axis">Predicted: sold fast</div>
      <div class="cm__axis">Predicted: still listed</div>
      <div class="cm__axis cm__axis--side">Actually sold fast</div>
      ${cell('True positive', cm.tp, 'tp', 'said it would sell, and it did')}
      ${cell('False negative', cm.fn, 'fn', 'said it would not, but it did · Type II')}
      <div class="cm__axis cm__axis--side">Actually still listed</div>
      ${cell('False positive', cm.fp, 'fp', 'said it would sell, and it did not · Type I')}
      ${cell('True negative', cm.tn, 'tn', 'said it would not, and it did not')}
    </div>`;
}

{
  const cm = confusion(EVAL.y, atThreshold(EVAL.s, 0.5));
  document.getElementById('cm-n').textContent = EVAL.y.length;
  document.getElementById('cm-sub').innerHTML =
    `A logistic regression on four features, at the default threshold of 0.5.`;
  confusionGrid('cm-grid', cm);
  const m = classMetrics(cm);
  document.getElementById('cm-words').innerHTML = `
    <p class="small">Of the ${EVAL.y.length} held-out homes, <b>${cm.tp + cm.tn}</b> were
      classified correctly and <b>${cm.fp + cm.fn}</b> were not.</p>
    <p class="small" style="margin-top:0.7rem">The two kinds of mistake are not
      interchangeable. <b>${cm.fp}</b> homes were predicted to sell quickly and did not,
      which for a seller means a price set too low on the strength of a bad forecast.
      <b>${cm.fn}</b> were predicted to linger and sold at once, which means a price set
      too high and money left behind.</p>
    <p class="small" style="margin-top:0.7rem">Accuracy counts those two together and
      reports <b>${pct(m.accuracy)}</b>. Whether that is the right summary depends on
      which mistake costs more, and accuracy does not know.</p>`;
  document.getElementById('cm-note').innerHTML =
    `Positive means sold within thirty days, which is a choice: name the other class positive `
    + `and the true positives and true negatives swap, as do the two kinds of error. Precision `
    + `and recall are computed with respect to whichever class was named positive.`;

  /* the lecture's worked example, recomputed */
  const actual = ['dog', 'cat', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'cat', 'dog',
    'dog', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat'];
  const pred = ['dog', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat', 'cat', 'cat', 'cat',
    'dog', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat', 'dog', 'dog', 'cat'];
  const ycat = actual.map((v) => (v === 'cat' ? 1 : 0));
  const pcat = pred.map((v) => (v === 'cat' ? 1 : 0));
  const c2 = confusion(ycat, pcat);
  const m2 = classMetrics(c2);
  const c3 = { tp: 20, tn: 70, fp: 5, fn: 5, n: 100 };
  const m3 = classMetrics(c3);
  document.getElementById('cm-worked').innerHTML = `
    <p class="small">Twenty animals, classified as cat or not by weight, with cat as the
      positive class:</p>
    <div class="table-scroll"><table class="data">
      <tbody>
        <tr><td>True positives</td><td class="num">${c2.tp}</td></tr>
        <tr><td>False negatives</td><td class="num">${c2.fn}</td></tr>
        <tr><td>False positives</td><td class="num">${c2.fp}</td></tr>
        <tr><td>True negatives</td><td class="num">${c2.tn}</td></tr>
        <tr><td><b>Accuracy</b></td><td class="num"><b>${m2.accuracy.toFixed(3)}</b></td></tr>
        <tr><td><b>Precision</b></td><td class="num"><b>${m2.precision.toFixed(3)}</b></td></tr>
        <tr><td><b>Recall</b></td><td class="num"><b>${m2.recall.toFixed(3)}</b></td></tr>
        <tr><td><b>F1</b></td><td class="num"><b>${m2.f1.toFixed(3)}</b></td></tr>
      </tbody></table></div>
    <p class="small" style="margin-top:0.8rem">And the credit card example: 20 fraudulent
      transactions caught, 70 legitimate ones correctly cleared, 5 legitimate flagged as
      fraud, 5 fraudulent missed, out of 100.</p>
    <div class="table-scroll"><table class="data">
      <tbody>
        <tr><td><b>Accuracy</b></td><td class="num"><b>${m3.accuracy.toFixed(2)}</b></td></tr>
        <tr><td><b>Precision</b></td><td class="num"><b>${m3.precision.toFixed(2)}</b></td></tr>
        <tr><td><b>Recall</b></td><td class="num"><b>${m3.recall.toFixed(2)}</b></td></tr>
        <tr><td><b>F1</b></td><td class="num"><b>${m3.f1.toFixed(2)}</b></td></tr>
      </tbody></table></div>
    <p class="small muted" style="margin-top:0.6rem">Both tables are computed in the browser
      from the counts, not transcribed.</p>`;
}

/* ---- the accuracy paradox ---- */
{
  const el0 = document.getElementById('balance');
  const update = () => {
    const share = +el0.value / 100;
    el0.nextElementSibling.textContent = pct(share);
    const n = 200;
    const pos = Math.round(n * share);
    const y = [...Array(n - pos).fill(0), ...Array(pos).fill(1)];
    const lazy = classMetrics(confusion(y, y.map(() => 0)));
    document.getElementById('balance-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${pct(lazy.accuracy)}</div>
        <div class="stat__label">Accuracy of a model that always says “still listed”</div></div>
      <div class="stat stat--muted"><div class="stat__value" style="font-size:1.3rem">${Number.isFinite(lazy.precision) ? pct(lazy.precision) : 'undefined'}</div>
        <div class="stat__label">Precision<br><span class="muted">it never predicts positive</span></div></div>
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.3rem">${pct(lazy.recall)}</div>
        <div class="stat__label">Recall<br><span class="muted">it catches none of the ${pos} quick sales</span></div></div>`;
    document.getElementById('balance-note').innerHTML = share <= 0.12
      ? `<b>${pct(lazy.accuracy)} accurate, and worthless.</b> The model has no parameters, `
        + `looks at nothing, and answers the same way every time. On a problem where only `
        + `${pct(share)} of homes sell quickly it still beats most honest attempts on accuracy `
        + `alone, which is why accuracy alone is not enough. Recall says what is actually `
        + `happening: it finds none of them.`
      : `With the classes near balanced, the always-negative model scores about `
        + `${pct(lazy.accuracy)} and the problem is not obvious. Drag the share down toward `
        + `5% and watch its accuracy climb while it keeps finding nothing.`;
  };
  el0.addEventListener('input', update);
  update();
}

/* ---- the threshold ---- */
{
  const thrEl = document.getElementById('thr');
  const stripHost = document.getElementById('thr-strip');
  const GRID = Array.from({ length: 97 }, (_, i) => (i + 2) / 100);
  const AT = GRID.map((t) => {
    const cm = confusion(EVAL.y, atThreshold(EVAL.s, t));
    return { t, cm, m: classMetrics(cm) };
  });
  const bestF1 = AT.reduce((a, b) => ((b.m.f1 || 0) > (a.m.f1 || 0) ? b : a));
  /* the highest threshold that still catches every positive */
  const fullRecall = AT.filter((a) => a.m.recall >= 0.999).reduce((a, b) => (b.t > a.t ? b : a), AT[0]);

  let drawStrip;
  const update = () => {
    const t = +thrEl.value / 100;
    thrEl.nextElementSibling.textContent = t.toFixed(2);
    const { cm, m } = AT[Math.round(t * 100) - 2];
    confusionGrid('thr-cm', cm);
    document.getElementById('thr-metrics').innerHTML = [
      ['Accuracy', m.accuracy, ''],
      ['Precision', m.precision, ''],
      ['Recall', m.recall, ''],
      ['F1', m.f1, Math.abs(t - bestF1.t) < 0.005 ? 'stat--train' : ''],
    ].map(([n, v, c]) => `<div class="stat ${c}">
      <div class="stat__value" style="font-size:1.3rem">${Number.isFinite(v) ? v.toFixed(3) : '—'}</div>
      <div class="stat__label">${n}</div></div>`).join('');
    document.getElementById('thr-verdict').innerHTML = t <= 0.2
      ? `<strong>A low threshold calls almost everything a quick sale.</strong> Recall is
         ${pct(m.recall)}, because almost nothing is missed. Precision is ${pct(m.precision)},
         because most of what it flags is wrong.`
      : t >= 0.8
        ? `<strong>A high threshold only commits when it is sure.</strong> Precision is
           ${pct(m.precision)}, so a flagged home usually does sell fast. Recall is
           ${pct(m.recall)}: it misses ${cm.fn} of the ${cm.tp + cm.fn} homes that sold quickly.`
        : `At ${t.toFixed(2)} the model flags ${cm.tp + cm.fp} homes, of which ${cm.tp} really
           did sell quickly. Precision ${pct(m.precision)}, recall ${pct(m.recall)}.`;
    document.getElementById('thr-note').innerHTML =
      `The model has not changed and neither have its probabilities. Only the line between `
      + `"call it a quick sale" and "do not" has moved. F1 is highest at a threshold of `
      + `<b>${bestF1.t.toFixed(2)}</b> (${bestF1.m.f1.toFixed(3)}), not at 0.5 — the default `
      + `is a convention rather than an answer.`;
    drawStrip();
  };

  drawStrip = () => {
    const t = +thrEl.value / 100;
    const w = 900, h = 120;
    const pad = { l: 46, r: 16, t: 20, b: 34 };
    const svg = svgRoot(stripHost, w, h);
    const sx = scale(0, 1, pad.l, w - pad.r);
    el('line', { x1: pad.l, x2: w - pad.r, y1: h - pad.b, y2: h - pad.b,
      stroke: token('--baseline') }, svg);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      el('text', { class: 'tick', x: sx(v), y: h - pad.b + 15, 'text-anchor': 'middle' }, svg)
        .textContent = v.toFixed(2);
    }
    el('text', { class: 'axis-label', x: (pad.l + w - pad.r) / 2, y: h - 3,
      'text-anchor': 'middle' }, svg).textContent = 'Predicted probability of selling within thirty days';
    /* every held-out home, placed by its predicted probability */
    EVAL.s.forEach((p, i) => {
      el('circle', { cx: sx(p), cy: h - pad.b - 12 - (i % 5) * 7, r: 3,
        fill: token(CLS[EVAL.y[i]]), opacity: 0.6 }, svg);
    });
    el('line', { x1: sx(t), x2: sx(t), y1: pad.t - 6, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 2.5 }, svg);
    el('text', { class: 'annot', x: sx(t), y: pad.t - 10, 'text-anchor': 'middle',
      fill: token('--series-6'), 'font-weight': 640 }, svg).textContent = `threshold ${t.toFixed(2)}`;
    el('text', { class: 'annot', x: sx(t) - 8, y: h - pad.b - 3, 'text-anchor': 'end',
      fill: token('--text-muted') }, svg).textContent = '← called still listed';
    el('text', { class: 'annot', x: sx(t) + 8, y: h - pad.b - 3,
      fill: token('--text-muted') }, svg).textContent = 'called sold fast →';
  };

  thrEl.addEventListener('input', update);
  document.getElementById('thr-half').addEventListener('click', () => tweenInput(thrEl, 50, update));
  document.getElementById('thr-f1').addEventListener('click', () =>
    tweenInput(thrEl, Math.round(bestF1.t * 100), update));
  document.getElementById('thr-recall').addEventListener('click', () =>
    tweenInput(thrEl, Math.round(fullRecall.t * 100), update));
  responsive(stripHost, update);
}

/* ---- ROC and AUC ---- */
{
  const host = document.getElementById('roc-chart');
  const thrEl = document.getElementById('roc-thr');
  const cmpEl = document.getElementById('roc-compare');
  const PTS = roc(EVAL.y, EVAL.s);
  const PTS_WEAK = roc(EVAL.y, EVAL.sWeak);

  let draw;
  const update = () => {
    const t = +thrEl.value / 100;
    thrEl.nextElementSibling.textContent = t.toFixed(2);
    const cm = confusion(EVAL.y, atThreshold(EVAL.s, t));
    const fpr = cm.fp / (cm.fp + cm.tn || 1);
    const tpr = cm.tp / (cm.tp + cm.fn || 1);
    document.getElementById('roc-stats').innerHTML = `
      <div class="stat"><div class="stat__value">${EVAL.A.toFixed(3)}</div>
        <div class="stat__label">AUC<br><span class="muted">1.0 is perfect, 0.5 is a coin flip</span></div></div>
      ${cmpEl.checked ? `<div class="stat stat--muted" style="margin-top:0.7rem">
        <div class="stat__value">${EVAL.Aweak.toFixed(3)}</div>
        <div class="stat__label">AUC of the one-feature model</div></div>` : ''}`;
    document.getElementById('roc-read').innerHTML =
      `At a threshold of <b>${t.toFixed(2)}</b> the model catches <b>${pct(tpr)}</b> of the `
      + `homes that did sell quickly, and wrongly flags <b>${pct(fpr)}</b> of the ones that did `
      + `not. That pair of numbers is one point on the curve. Every other threshold is another `
      + `point, and the curve is all of them.`;
    document.getElementById('roc-note').innerHTML =
      `AUC has a reading that needs no curve at all: <b>take one home that sold quickly and one `
      + `that did not, at random, and AUC is the chance the model scores the quick one higher.</b> `
      + `At ${EVAL.A.toFixed(3)} it gets that ordering right about ${pct(EVAL.A)} of the time. `
      + `Because the question is about ordering rather than counting, AUC is unaffected by how `
      + `common each class is, which is the failing of accuracy from section 8.`;
    draw();
  };

  draw = () => {
    const t = +thrEl.value / 100;
    const w = 440, h = 400;
    const pad = { l: 56, r: 16, t: 16, b: 48 };
    const svg = svgRoot(host, w, h);
    const sx = scale(0, 1, pad.l, w - pad.r);
    const sy = scale(0, 1, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'False positive rate', yLabel: 'True positive rate',
      xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [0, 0.25, 0.5, 0.75, 1],
      xFmt: (v) => v.toFixed(2), yFmt: (v) => v.toFixed(2),
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'roc-clip', pad, w, h) }, svg);
    /* the area under the curve, shaded */
    el('path', {
      d: `M${sx(0)} ${sy(0)} ` + PTS.map((p) => `L${sx(p.fpr)} ${sy(p.tpr)}`).join(' ')
        + ` L${sx(1)} ${sy(0)} Z`,
      fill: token('--accent'), opacity: 0.12,
    }, g);
    el('path', { class: 'series-line', stroke: token('--baseline'), 'stroke-width': 1.5,
      'stroke-dasharray': '5 4', d: linePath([[sx(0), sy(0)], [sx(1), sy(1)]]) }, g);
    el('text', { class: 'annot', x: sx(0.62), y: sy(0.52), fill: token('--text-muted'),
      transform: `rotate(-45 ${sx(0.62)} ${sy(0.52)})` }, g).textContent = 'a model guessing at random';
    if (cmpEl.checked) {
      el('path', { class: 'series-line', stroke: token('--text-muted'), 'stroke-width': 2,
        d: linePath(PTS_WEAK.map((p) => [sx(p.fpr), sy(p.tpr)])) }, g);
    }
    el('path', { class: 'series-line', stroke: token('--accent'), 'stroke-width': 2.8,
      d: linePath(PTS.map((p) => [sx(p.fpr), sy(p.tpr)])) }, g);
    const cm = confusion(EVAL.y, atThreshold(EVAL.s, t));
    const fpr = cm.fp / (cm.fp + cm.tn || 1);
    const tpr = cm.tp / (cm.tp + cm.fn || 1);
    el('line', { x1: sx(fpr), x2: sx(fpr), y1: sy(0), y2: sy(tpr),
      stroke: token('--series-6'), 'stroke-width': 1, 'stroke-dasharray': '3 3' }, g);
    el('circle', { cx: sx(fpr), cy: sy(tpr), r: 7, fill: token('--series-6'),
      stroke: token('--surface-1'), 'stroke-width': 2 }, g);
    el('text', { class: 'annot', x: sx(fpr) + 11, y: sy(tpr) + 4, fill: token('--series-6'),
      'font-weight': 620 }, g).textContent = `threshold ${t.toFixed(2)}`;
    el('text', { class: 'annot', x: sx(0.04), y: sy(0.96), fill: token('--accent'),
      'font-weight': 640 }, g).textContent = `AUC ${EVAL.A.toFixed(3)}`;
  };

  thrEl.addEventListener('input', update);
  cmpEl.addEventListener('change', update);
  responsive(host, update);
}
