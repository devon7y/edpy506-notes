/* Everything interactive on regression.html. Each figure is a self-contained
   block: read its controls, compute with the modules in this folder, draw.
   Nothing is hard-coded -- every number quoted in the page's prose comes from
   these computations, so the text cannot drift away from the figures. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  clientToViewBox, mean, sd, money, money1k, fmt, rng, gauss, clipRect,
  directLabels, scaledTicks, tweenInput, tweenInputs,
} from './site.js';
import {
  ols, metrics, lstsq, polyfit, standardize, regPath, crossValidate, logGrid,
  lassoLambdaMax,
} from './linreg.js';
import { currentDataset, pairs, linearDesign, labelOf } from './datasets.js';

initChrome();

const ds = currentDataset();
const tip = tooltip();
const k$ = money1k;

/* ===================================================================
   1. Where regression sits
   =================================================================== */
{
  const host = document.getElementById('taxonomy');
  const boxes = [
    { x: 8, y: 92, w: 96, h: 40, t: 'Machine learning', fill: '--series-6' },
    { x: 140, y: 40, w: 96, h: 36, t: 'Supervised', sub: 'labelled data', fill: '--series-1' },
    { x: 140, y: 148, w: 96, h: 36, t: 'Unsupervised', sub: 'unlabelled data', fill: '--baseline' },
    { x: 276, y: 8, w: 104, h: 32, t: 'Regression', hi: true },
    { x: 276, y: 56, w: 104, h: 32, t: 'Classification' },
    { x: 276, y: 116, w: 104, h: 28, t: 'Clustering' },
    { x: 276, y: 152, w: 104, h: 28, t: 'Dimensionality reduction' },
    { x: 276, y: 188, w: 104, h: 28, t: 'Association' },
  ];
  const links = [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6], [2, 7]];
  const notes = [[3, 'a number'], [4, 'a category']];

  responsive(host, () => {
    const w = 480, h = 232;
    const svg = svgRoot(host, w, h);
    for (const [a, b] of links) {
      const A = boxes[a], B = boxes[b];
      const x1 = A.x + A.w, y1 = A.y + A.h / 2, x2 = B.x, y2 = B.y + B.h / 2;
      el('path', {
        d: `M${x1} ${y1} C${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`,
        fill: 'none', stroke: token('--baseline'), 'stroke-width': 1.3,
      }, svg);
    }
    boxes.forEach((b) => {
      el('rect', {
        x: b.x, y: b.y, width: b.w, height: b.h, rx: 6,
        fill: b.hi ? token('--accent-wash') : token('--surface-1'),
        stroke: token(b.fill || (b.hi ? '--accent' : '--border-strong')),
        'stroke-width': b.hi ? 2 : 1.2,
      }, svg);
      const cy = b.y + b.h / 2 + (b.sub ? -3 : 3.5);
      const t = el('text', {
        x: b.x + b.w / 2, y: cy, 'text-anchor': 'middle', 'font-size': 10.5,
        'font-weight': b.hi ? 700 : 560, fill: b.hi ? token('--accent') : token('--text-primary'),
      }, svg);
      /* Long labels wrap by hand: SVG text has no line breaking. */
      const words = b.t.split(' ');
      if (b.t.length > 16 && words.length > 1) {
        t.textContent = '';
        el('tspan', { x: b.x + b.w / 2, dy: -5 }, t).textContent = words[0];
        el('tspan', { x: b.x + b.w / 2, dy: 12 }, t).textContent = words.slice(1).join(' ');
      } else {
        t.textContent = b.t;
      }
      if (b.sub) {
        el('text', {
          x: b.x + b.w / 2, y: b.y + b.h / 2 + 11, 'text-anchor': 'middle',
          'font-size': 8.5, fill: token('--text-muted'),
        }, svg).textContent = `(${b.sub})`;
      }
    });
    for (const [i, label] of notes) {
      const b = boxes[i];
      el('text', {
        x: b.x + b.w + 8, y: b.y + b.h / 2 + 3.5, 'font-size': 9.5,
        fill: token('--text-muted'),
      }, svg).textContent = `output is ${label}`;
    }
  });
}

/* ===================================================================
   2. Fitting a line by hand
   =================================================================== */
const LINE = pairs(ds, 'linear', 7, 30);
const LINE_FIT = ols(LINE.xs, LINE.ys);
const LINE_BEST = metrics(LINE.ys, LINE.xs.map((x) => LINE_FIT.b0 + LINE_FIT.b1 * x));
{
  const host = document.getElementById('line-chart');
  const b0El = document.getElementById('b0');
  const b1El = document.getElementById('b1');
  const b0Out = document.getElementById('b0-out');
  const b1Out = document.getElementById('b1-out');
  const statsEl = document.getElementById('line-stats');
  const barEl = document.getElementById('sse-bar');
  const noteEl = document.getElementById('line-note');
  const { xs, ys, feature } = LINE;
  const xlo = Math.min(...xs), xhi = Math.max(...xs);

  let draw;
  const update = () => {
    const b0 = +b0El.value, b1 = +b1El.value;
    b0Out.textContent = money(b0);
    b1Out.textContent = `$${b1}/sq ft`;
    const yhat = xs.map((x) => b0 + b1 * x);
    const m = metrics(ys, yhat);
    const ratio = LINE_BEST.sse / m.sse;
    statsEl.innerHTML = `
      <div class="stat"><div class="stat__value">${money(m.rmse)}</div>
        <div class="stat__label">Your line is typically this far out<br><span class="muted">on ${xs.length} homes</span></div></div>
      <div class="stat ${ratio > 0.999 ? '' : 'stat--muted'}"><div class="stat__value">${ratio > 0.999 ? '1.00×' : `${(1 / ratio).toFixed(2)}×`}</div>
        <div class="stat__label">Your total squared error, against the smallest possible<br><span class="muted">1.00× is the least-squares line</span></div></div>`;
    barEl.innerHTML = `
      <div class="bars"><div class="bars__row">
        <div class="bars__name">How close</div>
        <div class="bars__track"><div class="bars__fill" style="width:${(ratio * 100).toFixed(1)}%"></div></div>
        <div class="bars__val">${(ratio * 100).toFixed(0)}%</div>
      </div></div>`;
    noteEl.innerHTML = ratio > 0.999
      ? `<b>That is the least-squares line.</b> No other pair of numbers gives a smaller total. `
        + `It prices a home at ${money(LINE_FIT.b0)} plus $${LINE_FIT.b1.toFixed(0)} for every square foot, `
        + `and is typically ${money(LINE_BEST.rmse)} out.`
      : `Your line leaves <b>${(1 / ratio).toFixed(2)}×</b> as much squared error as the best one does. `
        + `Squaring is why the far-off homes pull so hard: the residual drawn longest on the chart `
        + `contributes more than the several shortest ones put together.`;
    draw();
  };

  draw = () => {
    const b0 = +b0El.value, b1 = +b1El.value;
    const w = 620, h = 400;
    const pad = { l: 66, r: 16, t: 14, b: 42 };
    const svg = svgRoot(host, w, h);
    const sx = scale(xlo - 60, xhi + 60, pad.l, w - pad.r);
    const sy = scale(Math.min(...ys) - 60000, Math.max(...ys) + 60000, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: labelOf(feature), yLabel: `${ds.target.label} ($)`,
      xFmt: (v) => Math.round(v).toLocaleString('en-CA'), yFmt: ds.target.short,
    });
    const clip = clipRect(svg, 'line-clip', pad, w, h);
    const g = el('g', { 'clip-path': clip }, svg);

    /* residuals first, so the dots and the line sit on top */
    xs.forEach((x, i) => {
      const yh = b0 + b1 * x;
      el('line', {
        x1: sx(x), x2: sx(x), y1: sy(ys[i]), y2: sy(yh),
        stroke: token('--text-muted'), 'stroke-width': 1.4, opacity: 0.55,
      }, g);
    });
    el('path', {
      class: 'series-line',
      d: linePath([[sx(xlo - 60), sy(b0 + b1 * (xlo - 60))], [sx(xhi + 60), sy(b0 + b1 * (xhi + 60))]]),
      stroke: token('--series-6'), 'stroke-width': 2.5,
    }, g);
    xs.forEach((x, i) => {
      const c = el('circle', {
        cx: sx(x), cy: sy(ys[i]), r: 5, fill: token('--series-1'), opacity: 0.85,
        stroke: token('--surface-1'), 'stroke-width': 1.2,
      }, g);
      c.addEventListener('pointerenter', (ev) => tip.show(
        `<b>${feature.fmt(x)} sq ft</b><br>sold for ${money(ys[i])}`
        + `<br>your line says ${money(b0 + b1 * x)}`
        + `<br><span style="opacity:.7">off by ${money(Math.abs(ys[i] - b0 - b1 * x))}</span>`,
        ev.clientX, ev.clientY));
      c.addEventListener('pointerleave', () => tip.hide());
    });
  };

  b0El.addEventListener('input', update);
  b1El.addEventListener('input', update);
  /* Both sliders travel together, so the line rotates and slides into place and
     the total error can be watched falling as it goes. */
  document.getElementById('solve').addEventListener('click', () => {
    tweenInputs([
      { el: b0El, to: Math.round(LINE_FIT.b0 / 5000) * 5000 },
      { el: b1El, to: Math.round(LINE_FIT.b1 / 5) * 5 },
    ], update);
  });
  document.getElementById('flat').addEventListener('click', () => {
    tweenInputs([
      { el: b0El, to: Math.round(mean(LINE.ys) / 5000) * 5000 },
      { el: b1El, to: 0 },
    ], update);
  });
  responsive(host, update);
}

/* ===================================================================
   3. The worked least-squares numbers
   =================================================================== */
{
  const { xs, ys } = LINE;
  const xb = mean(xs), yb = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) { num += (ys[i] - yb) * (xs[i] - xb); den += (xs[i] - xb) ** 2; }
  document.getElementById('ols-work').innerHTML = `
    <p class="small">For the ${xs.length} homes in the figure above:</p>
    <div class="table-scroll"><table class="data">
      <tbody>
        <tr><td>Average size <em>x̄</em></td><td class="num">${xb.toFixed(0)} sq ft</td></tr>
        <tr><td>Average price <em>ȳ</em></td><td class="num">${money(yb)}</td></tr>
        <tr><td>Σ (<em>y</em> − <em>ȳ</em>)(<em>x</em> − <em>x̄</em>)</td><td class="num">${num.toExponential(3)}</td></tr>
        <tr><td>Σ (<em>x</em> − <em>x̄</em>)²</td><td class="num">${den.toExponential(3)}</td></tr>
        <tr><td><strong>Slope <em>b</em><sub>1</sub></strong></td><td class="num"><strong>$${(num / den).toFixed(1)} per sq ft</strong></td></tr>
        <tr><td><strong>Intercept <em>b</em><sub>0</sub></strong></td><td class="num"><strong>${money(yb - (num / den) * xb)}</strong></td></tr>
      </tbody></table></div>
    <p class="small muted">The intercept is what the line says a home of zero square feet
      would sell for, which is not a meaningful home. That is normal: an intercept is
      often outside the range the data actually cover, and is there to position the line
      rather than to be read on its own.</p>`;
}

/* ===================================================================
   4. Multiple regression: raw slopes against standardised betas
   =================================================================== */
const MR_N = 150;
{
  const rows = ds.sample(5, MR_N);
  const { X, y, names, noise } = linearDesign(ds, rows);
  const A = X.map((r) => [1, ...r]);
  const raw = lstsq(A, y);
  const { Z } = standardize(X);
  const yb = mean(y);
  const beta = lstsq(Z, y.map((v) => v - yb));
  const order = names.map((n, j) => j).sort((a, b) => Math.abs(beta[b]) - Math.abs(beta[a]));
  const sds = standardize(X).sd;

  document.getElementById('mr-n').textContent = MR_N;
  document.getElementById('mr-table').innerHTML = `
    <thead><tr><th>Predictor</th><th>Raw slope <em>b</em></th><th>Standardised <em>β</em></th></tr></thead>
    <tbody>${order.map((j) => {
      const f = ds.features.find((ft) => labelOf(ft) === names[j]);
      const unit = names[j].includes(':') ? 'vs. the first level'
        : `per ${f && f.unit ? (f.unit === 'sq ft' ? 'sq ft' : f.unit.replace(/s$/, '')) : 'unit'}`;
      return `<tr>
        <td>${names[j]}</td>
        <td class="num">${money(raw[j + 1])} <span class="muted small">${unit}</span></td>
        <td class="num ${Math.abs(beta[j]) < 8000 ? 'zero' : ''}">${money(beta[j])}</td>
      </tr>`;
    }).join('')}
    <tr><td><em>Intercept</em></td><td class="num">${money(raw[0])}</td><td class="zero">—</td></tr></tbody>`;

  const top = names[order[0]];
  const tiny = order.filter((j) => noise[j]);
  const worstNoise = Math.max(...tiny.map((j) => Math.abs(beta[j])));
  document.getElementById('mr-note').innerHTML =
    `Sorted by the standardised <em>β</em>, which is the column you can compare down. `
    + `<b>${top}</b> comes out the strongest predictor. `
    + `Read the raw column instead and the ranking is different and meaningless, because `
    + `each row is in its own units.<br><br>`
    + `At the bottom of the table sit door colour and the house number. These homes were `
    + `generated so that neither has <b>any effect on price whatsoever</b>, and the model still `
    + `hands them coefficients — the largest is ${money(worstNoise)}. Least squares gives every `
    + `feature it is offered a non-zero slope, because a coefficient of exactly zero is `
    + `almost never the arrangement that minimises squared error on a finite sample. `
    + `That is the overfitting problem in miniature, and <a href="#ridge">regularization</a> `
    + `is the fix.`
    + `<br><br><b>Age</b> sits down among the noise too, with a `
    + `<em>β</em> of ${money(beta[names.indexOf('Age (years)')])}, and age genuinely does move `
    + `the price of these homes — a lot. A linear model cannot see it, because the effect is `
    + `not a straight line: prices fall with age, bottom out, then recover, and a single slope `
    + `through that shape averages out to almost nothing. `
    + `<a href="#fitting">The polynomial figure</a> draws that curve, and `
    + `<a href="trees.html">trees and forests</a> fit it.`;
}

/* ===================================================================
   5. Four scores, one line
   =================================================================== */
{
  const host = document.getElementById('score-chart');
  const tiltEl = document.getElementById('tilt');
  const outEl = document.getElementById('tilt-out');
  const statsEl = document.getElementById('score-stats');
  const noteEl = document.getElementById('score-note');
  const { xs, ys, feature } = LINE;
  const xb = mean(xs);
  const yb = mean(ys);

  /* Tilt pivots the line about the centre of the data, so the intercept follows
     the slope and the line keeps passing through (x̄, ȳ). One control, and the
     fit gets monotonically worse. */
  const params = (t) => {
    const b1 = LINE_FIT.b1 * (1 - 0.9 * (t / 100));
    return { b0: yb - b1 * xb, b1 };
  };

  let draw;
  const update = () => {
    const t = +tiltEl.value;
    const { b0, b1 } = params(t);
    outEl.textContent = t === 0 ? 'best fit' : `${t}% flatter`;
    const m = metrics(ys, xs.map((x) => b0 + b1 * x));
    statsEl.innerHTML = [
      ['R²', fmt(m.r2, 3), 'explained share of variance', m.r2 < 0.3 ? 'stat--muted' : ''],
      ['MAE', money(m.mae), 'mean absolute error', ''],
      ['MSE', m.mse.toExponential(2), 'mean squared error', ''],
      ['RMSE', money(m.rmse), 'root mean squared error', ''],
    ].map(([n, v, l, c]) => `<div class="stat ${c}"><div class="stat__value" style="font-size:1.25rem">${v}</div>
      <div class="stat__label"><b>${n}</b> · ${l}</div></div>`).join('');
    const base = metrics(ys, xs.map((x) => LINE_FIT.b0 + LINE_FIT.b1 * x));
    noteEl.innerHTML = t === 0
      ? `This is the least-squares fit: R² of <b>${fmt(base.r2, 2)}</b>, typically `
        + `${money(base.rmse)} out. Drag the slider and watch MSE climb faster than MAE — `
        + `squaring is what makes the difference, and it is the same squaring that chose this line.`
      : `MAE has grown <b>${(m.mae / base.mae).toFixed(2)}×</b>, MSE <b>${(m.mse / base.mse).toFixed(2)}×</b>. `
        + `MSE reacts harder because it squares each miss before averaging, so the homes now `
        + `furthest from the line dominate it. RMSE is MSE's square root, which is why it tracks `
        + `MAE closely while staying in dollars.`;
    draw();
  };

  draw = () => {
    const { b0, b1 } = params(+tiltEl.value);
    const w = 560, h = 380;
    const pad = { l: 66, r: 16, t: 14, b: 42 };
    const svg = svgRoot(host, w, h);
    const xlo = Math.min(...xs), xhi = Math.max(...xs);
    const sx = scale(xlo - 60, xhi + 60, pad.l, w - pad.r);
    const sy = scale(Math.min(...ys) - 60000, Math.max(...ys) + 60000, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: labelOf(feature), yLabel: `${ds.target.label} ($)`,
      xFmt: (v) => Math.round(v).toLocaleString('en-CA'), yFmt: ds.target.short,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'score-clip', pad, w, h) }, svg);
    xs.forEach((x, i) => {
      el('line', {
        x1: sx(x), x2: sx(x), y1: sy(ys[i]), y2: sy(b0 + b1 * x),
        stroke: token('--series-8'), 'stroke-width': 1.3, opacity: 0.4,
      }, g);
    });
    el('path', {
      class: 'series-line',
      d: linePath([[sx(xlo - 60), sy(b0 + b1 * (xlo - 60))], [sx(xhi + 60), sy(b0 + b1 * (xhi + 60))]]),
      stroke: token('--series-6'), 'stroke-width': 2.5,
    }, g);
    xs.forEach((x, i) => el('circle', {
      cx: sx(x), cy: sy(ys[i]), r: 4.5, fill: token('--series-1'), opacity: 0.8,
      stroke: token('--surface-1'), 'stroke-width': 1.1,
    }, g));
  };

  tiltEl.addEventListener('input', update);
  document.getElementById('tilt-reset').addEventListener('click', () => tweenInput(tiltEl, 0, update));
  responsive(host, update);
}

/* ===================================================================
   6 & 8. The polynomial demo and the bias-variance decomposition
   ===================================================================
   Both figures come from the same experiment, so the U-curve in section 7 is
   the measured behaviour of the model in section 8 rather than a schematic.

   For each degree, many training sets of the same size are drawn, a polynomial
   is fitted to each, and the predictions are compared at a grid of ages:
     bias²    = (average prediction − the truth)²
     variance = spread of the predictions around their own average
     noise    = the variance the homes were generated with
   Their sum is the expected squared error on a fresh home, which is what the
   total curve plots. */
const DEG_MAX = 11;
const TRAIN_SEED = 21, TEST_SEED = 22, N_POINTS = 12;
const POLY_TRAIN = pairs(ds, 'curved', TRAIN_SEED, N_POINTS);
const POLY_TEST = pairs(ds, 'curved', TEST_SEED, N_POINTS);

const BV = (() => {
  const REPS = 120;
  const grid = Array.from({ length: 41 }, (_, i) => (80 * i) / 40);
  /* The truth: the same homes with the noise term removed, which the dataset
     gives us by drawing a very large sample and averaging at each age. */
  const truthRows = ds.sample(999, 4000, ds.views.curved.fixed);
  const truthAt = (age) => {
    let s = 0, n = 0;
    for (const r of truthRows) if (Math.abs(r.age - age) < 2.5) { s += r.price; n++; }
    return n ? s / n : NaN;
  };
  const truth = grid.map(truthAt);
  const preds = Array.from({ length: DEG_MAX }, () => grid.map(() => []));
  for (let r = 0; r < REPS; r++) {
    const s = pairs(ds, 'curved', 1000 + r * 7, N_POINTS);
    for (let d = 1; d <= DEG_MAX; d++) {
      const { predict } = polyfit(s.xs, s.ys, d);
      grid.forEach((x, gi) => preds[d - 1][gi].push(predict(x)));
    }
  }
  const noiseVar = 20000 ** 2;
  const rows = [];
  for (let d = 1; d <= DEG_MAX; d++) {
    let b2 = 0, va = 0, ok = 0;
    grid.forEach((x, gi) => {
      if (!Number.isFinite(truth[gi])) return;
      const p = preds[d - 1][gi];
      const mu = mean(p);
      /* Predictions from a high-degree fit can be astronomically large near the
         edges. Those are real, but one of them swamps the average, so the
         decomposition is summarised over the middle 90% of the age range. */
      if (x < 4 || x > 76) return;
      b2 += (mu - truth[gi]) ** 2;
      va += mean(p.map((v) => (v - mu) ** 2));
      ok++;
    });
    rows.push({ d, bias2: b2 / ok, variance: va / ok, noise: noiseVar });
  }
  return rows.map((r) => ({ ...r, total: r.bias2 + r.variance + r.noise }));
})();

{
  const host = document.getElementById('bv-chart');
  const series = [
    { key: 'bias2', label: 'Bias²', color: '--series-1' },
    { key: 'variance', label: 'Variance', color: '--series-8' },
    { key: 'noise', label: 'Noise', color: '--text-muted' },
    { key: 'total', label: 'Total error', color: '--series-6', wide: true },
  ];
  document.getElementById('bv-legend').innerHTML = series.map((s) =>
    `<span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token(s.color)}"></span>${s.label}</span>`
  ).join('') + `<span class="legend__item muted">measured over 120 refits per degree</span>`;

  responsive(host, () => {
    const w = 900, h = 340;
    const pad = { l: 66, r: 96, t: 16, b: 40 };
    const svg = svgRoot(host, w, h);
    /* Every quantity here is an error in dollars squared, and they span six
       orders of magnitude between them -- bias at its smallest is a few hundred
       dollars, variance at degree 11 is billions. A linear axis shows one of
       them and flattens the rest, so the axis is the root of the error (which
       puts it in dollars) on a log scale (which fits the range). */
    const root = (v) => Math.sqrt(Math.max(v, 1));
    const lo = 300, hi = 3e6;
    const sx = scale(1, DEG_MAX, pad.l, w - pad.r);
    const sy = scale(Math.log10(lo), Math.log10(hi), h - pad.b, pad.t);
    const Y = (v) => sy(Math.log10(Math.min(Math.max(root(v), lo), hi)));
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Model complexity  (degree of the polynomial)',
      yLabel: 'Typical error, log scale',
      xTicks: BV.map((r) => r.d),
      yTicks: [3, 4, 5, 6].map((e) => e), yFmt: (v) => k$(10 ** v),
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'bv-clip', pad, w, h) }, svg);
    const labels = [];
    for (const s of series) {
      const pts = BV.map((r) => [sx(r.d), Y(r[s.key])]);
      el('path', {
        class: 'series-line', d: linePath(pts), stroke: token(s.color),
        'stroke-width': s.wide ? 3 : 2, 'stroke-dasharray': s.key === 'noise' ? '4 4' : null,
      }, g);
      labels.push({ y: pts[pts.length - 1][1], text: s.label, color: token(s.color) });
    }
    directLabels(svg, labels, w - pad.r + 7, pad.t + 8, h - pad.b);
    /* mark the minimum of the total curve */
    const best = BV.reduce((a, b) => (b.total < a.total ? b : a));
    el('line', {
      x1: sx(best.d), x2: sx(best.d), y1: pad.t, y2: h - pad.b,
      stroke: token('--good'), 'stroke-width': 1.5, 'stroke-dasharray': '3 3',
    }, svg);
    el('text', {
      class: 'annot', x: sx(best.d) + 6, y: pad.t + 12, fill: token('--good'), 'font-weight': 600,
    }, svg).textContent = `smallest total error: degree ${best.d}`;
  });
}

/* ---- section 8: the degree slider ---- */
{
  const host = document.getElementById('poly-chart');
  const curveHost = document.getElementById('poly-curve');
  const degEl = document.getElementById('degree');
  const outEl = document.getElementById('degree-out');
  const statsEl = document.getElementById('poly-stats');
  const verdictEl = document.getElementById('poly-verdict');
  const noteEl = document.getElementById('poly-note');

  document.getElementById('curve-desc').textContent = ds.views.curved.describe;
  document.getElementById('poly-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token('--c-train')}"></span>Training homes (${N_POINTS})</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token('--c-test')}"></span>Test homes (${N_POINTS}, never fitted)</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-fit')}"></span>The fitted polynomial</span>`;

  /* Precompute every degree once: the fits never change. */
  const FITS = [];
  for (let d = 1; d <= DEG_MAX; d++) {
    const { predict } = polyfit(POLY_TRAIN.xs, POLY_TRAIN.ys, d);
    FITS.push({
      d,
      predict,
      train: metrics(POLY_TRAIN.ys, POLY_TRAIN.xs.map(predict)),
      test: metrics(POLY_TEST.ys, POLY_TEST.xs.map(predict)),
    });
  }
  const BEST = FITS.reduce((a, b) => (b.test.rmse < a.test.rmse ? b : a));

  let drawScatter, drawCurves;
  const update = () => {
    const d = +degEl.value;
    const f = FITS[d - 1];
    outEl.textContent = d;
    document.querySelectorAll('#regime-jump button').forEach((b) =>
      b.setAttribute('aria-pressed', String(+b.dataset.d === d)));
    const gap = f.test.rmse / f.train.rmse;
    statsEl.innerHTML = `
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.35rem">${f.train.rmse < 1 ? '$0' : money(f.train.rmse)}</div>
        <div class="stat__label">Training RMSE<br><span class="muted">homes it was fitted on</span></div></div>
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.35rem">${f.test.rmse > 5e6 ? f.test.rmse.toExponential(1) : money(f.test.rmse)}</div>
        <div class="stat__label">Test RMSE<br><span class="muted">homes it has never seen</span></div></div>`;
    let verdict;
    if (d === 1) {
      verdict = `<strong>Underfit — high bias.</strong> A straight line cannot bend, so it `
        + `misses the dip in the middle and the recovery at the end. Training and test error `
        + `are both high and close together, which is the signature: the model is not `
        + `confused by the training data, it simply never had the capacity to fit it.`;
    } else if (d >= 7) {
      verdict = `<strong>Overfit — high variance.</strong> The curve now passes through or `
        + `near every training home, and training error has collapsed. Between them it does `
        + `whatever it likes, and test error is ${gap > 50 ? 'orders of magnitude' : `${gap.toFixed(0)} times`} `
        + `worse. Those wild excursions are fitting noise, which the next set of homes does not share.`;
    } else if (d === BEST.d || Math.abs(f.test.rmse - BEST.test.rmse) / BEST.test.rmse < 0.08) {
      verdict = `<strong>About right.</strong> Enough flexibility to follow the real shape, `
        + `not enough to chase individual homes. Training and test error are both low and `
        + `within ${(gap).toFixed(1)}× of each other. Some gap is normal and expected.`;
    } else {
      verdict = `<strong>Starting to strain.</strong> Training error keeps falling, but test `
        + `error has stopped following it down. That divergence is the thing to watch for — `
        + `it is the only visible sign that added complexity has stopped buying anything.`;
    }
    verdictEl.innerHTML = verdict;
    noteEl.innerHTML = `The best degree on these homes is <b>${BEST.d}</b>, at a test RMSE of `
      + `${money(BEST.test.rmse)}. Degree 1 manages ${money(FITS[0].test.rmse)} and degree ${DEG_MAX} `
      + `manages ${FITS[DEG_MAX - 1].test.rmse.toExponential(1)} — while fitting its training homes `
      + `perfectly. The homes carry about ${money(20000)} of pure noise, so nothing can do better than that.`;
    drawScatter();
    drawCurves();
  };

  drawScatter = () => {
    const d = +degEl.value;
    const f = FITS[d - 1];
    const w = 560, h = 380;
    const pad = { l: 66, r: 16, t: 14, b: 42 };
    const svg = svgRoot(host, w, h);
    const all = [...POLY_TRAIN.ys, ...POLY_TEST.ys];
    const sx = scale(-2, 82, pad.l, w - pad.r);
    const sy = scale(Math.min(...all) - 50000, Math.max(...all) + 50000, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Age of the home (years)', yLabel: `${ds.target.label} ($)`,
      xTicks: [0, 20, 40, 60, 80], yFmt: ds.target.short,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'poly-clip', pad, w, h) }, svg);
    const pts = [];
    for (let i = 0; i <= 240; i++) {
      const x = -2 + (84 * i) / 240;
      pts.push([sx(x), sy(f.predict(x))]);
    }
    el('path', { class: 'series-line', d: linePath(pts), stroke: token('--c-fit'), 'stroke-width': 2.5 }, g);
    POLY_TEST.xs.forEach((x, i) => el('circle', {
      cx: sx(x), cy: sy(POLY_TEST.ys[i]), r: 5, fill: 'none',
      stroke: token('--c-test'), 'stroke-width': 2,
    }, g));
    POLY_TRAIN.xs.forEach((x, i) => el('circle', {
      cx: sx(x), cy: sy(POLY_TRAIN.ys[i]), r: 5, fill: token('--c-train'),
      stroke: token('--surface-1'), 'stroke-width': 1.2,
    }, g));
  };

  drawCurves = () => {
    const d = +degEl.value;
    const w = 900, h = 250;
    const pad = { l: 66, r: 88, t: 16, b: 40 };
    const svg = svgRoot(curveHost, w, h);
    /* Test RMSE runs from about $20k to several hundred million, so this axis is
       logarithmic. On a linear one the entire interesting region -- every degree
       from 1 to 8 -- is a flat smear along the bottom. */
    const lo = 5000, hi = 1e9;
    const sx = scale(1, DEG_MAX, pad.l, w - pad.r);
    const sy = scale(Math.log10(lo), Math.log10(hi), h - pad.b, pad.t);
    const Y = (v) => sy(Math.log10(Math.min(Math.max(v, lo), hi)));
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Degree', yLabel: 'RMSE, log scale', xTicks: FITS.map((f) => f.d),
      yTicks: [4, 5, 6, 7, 8, 9], yFmt: (v) => k$(10 ** v),
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'pc-clip', pad, w, h) }, svg);
    const labels = [];
    for (const [key, color] of [['train', '--c-train'], ['test', '--c-test']]) {
      const pts = FITS.map((f) => [sx(f.d), Y(f[key].rmse)]);
      el('path', { class: 'series-line', d: linePath(pts), stroke: token(color), 'stroke-width': 2.4 }, g);
      FITS.forEach((f, i) => el('circle', {
        cx: pts[i][0], cy: pts[i][1], r: f.d === d ? 5 : 3,
        fill: token(color), opacity: f.d === d ? 1 : 0.5,
      }, g));
      labels.push({ y: pts[pts.length - 1][1], text: key === 'train' ? 'Training' : 'Test', color: token(color) });
    }
    directLabels(svg, labels, w - pad.r + 7, pad.t + 6, h - pad.b);
    el('line', {
      x1: sx(d), x2: sx(d), y1: pad.t, y2: h - pad.b,
      stroke: token('--text-muted'), 'stroke-width': 1, 'stroke-dasharray': '3 3',
    }, svg);
    el('text', { class: 'annot', x: sx(d) + 5, y: pad.t + 11, fill: token('--text-muted') }, svg)
      .textContent = `degree ${d}`;
  };

  degEl.addEventListener('input', update);
  /* Stepping through the degrees rather than jumping is the whole point of these
     three buttons: the curve visibly stiffens on the way down to 1 and visibly
     tears itself apart on the way up to 11. */
  document.querySelectorAll('#regime-jump button').forEach((b) =>
    b.addEventListener('click', () => tweenInput(degEl, +b.dataset.d, update, { ms: 900 })));
  responsive(host, update);
  responsive(curveHost, drawCurves);
}

/* ===================================================================
   9-11. Ridge, lasso and the choice of lambda
   =================================================================== */
const REG_N = 60;
const REG = (() => {
  const rows = ds.sample(5, REG_N);
  const d = linearDesign(ds, rows);
  const { Z, sd: sds } = standardize(d.X);
  const yb = mean(d.y);
  const yc = d.y.map((v) => v - yb);
  const lmax = lassoLambdaMax(Z, yc);
  /* One grid per method: their natural scales differ by orders of magnitude,
     because ridge's penalty is compared against ZᵀZ and lasso's against Zᵀy. */
  const grids = {
    ridge: logGrid(Math.exp(-3), Math.exp(12), 60),
    lasso: logGrid(Math.exp(4), Math.exp(Math.log(lmax) + 0.6), 60),
  };
  const paths = {
    ridge: regPath(Z, yc, grids.ridge, 'ridge'),
    lasso: regPath(Z, yc, grids.lasso, 'lasso'),
  };
  const cv = {
    ridge: crossValidate(d.X, d.y, grids.ridge, 'ridge', 5, rng(5)),
    lasso: crossValidate(d.X, d.y, grids.lasso, 'lasso', 5, rng(5)),
  };
  return { ...d, Z, sds, yb, yc, lmax, grids, paths, cv };
})();

document.getElementById('ridge-p').textContent = REG.names.length;

function pathFigure(method, ids) {
  const host = document.getElementById(ids.chart);
  const lamEl = document.getElementById(ids.slider);
  const outEl = document.getElementById(ids.out);
  const noteEl = document.getElementById(ids.note);
  const chipsEl = document.getElementById(ids.chips);
  const grid = REG.grids[method];
  const path = REG.paths[method];
  const live = (b) => b.filter((v) => Math.abs(v) > 1e-9).length;

  let draw;
  const update = () => {
    const i = +lamEl.value;
    const b = path[i];
    outEl.textContent = Math.log(grid[i]).toFixed(1);
    const kept = live(b);
    const keptNoise = b.filter((v, j) => REG.noise[j] && Math.abs(v) > 1e-9).length;
    const nNoise = REG.noise.filter(Boolean).length;
    const shrink = Math.sqrt(b.reduce((s, v) => s + v * v, 0))
      / Math.sqrt(path[0].reduce((s, v) => s + v * v, 0));
    noteEl.innerHTML = method === 'ridge'
      ? `<b>${kept} of ${REG.names.length} predictors still in the model</b>, with coefficients `
        + `${(shrink * 100).toFixed(0)}% the size they were at the smallest λ. The lines converge `
        + `toward zero and never arrive.`
      : `<b>${kept} of ${REG.names.length} predictors still in the model</b>. `
        + `${keptNoise} of the ${nNoise} predictors that have no effect on price at all are `
        + `${keptNoise === 0 ? 'now all gone' : 'still in it'}. `
        + `${kept === 0 ? 'At a large enough λ every coefficient is zero and the model predicts the average for every home.'
          : 'Each line that reaches the axis is a feature dropped from the model entirely.'}`;
    chipsEl.innerHTML = REG.names.map((n, j) =>
      `<span class="chip" ${Math.abs(b[j]) > 1e-9 ? 'data-on' : 'data-off'}>${n}</span>`).join('');
    draw();
  };

  draw = () => {
    const i = +lamEl.value;
    const w = 900, h = 340;
    const pad = { l: 66, r: 150, t: 16, b: 42 };
    const svg = svgRoot(host, w, h);
    const flat = path.flat();
    const lim = Math.max(...flat.map(Math.abs)) * 1.08;
    const sx = scale(Math.log(grid[0]), Math.log(grid[grid.length - 1]), pad.l, w - pad.r);
    const sy = scale(-lim, lim, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'log λ', yLabel: 'Standardised coefficient', yFmt: (v) => k$(v),
    });
    el('line', {
      x1: pad.l, x2: w - pad.r, y1: sy(0), y2: sy(0),
      stroke: token('--baseline'), 'stroke-width': 1.2,
    }, svg);
    const g = el('g', { 'clip-path': clipRect(svg, `${method}-clip`, pad, w, h) }, svg);
    /* colour by whether the feature really drives the price */
    REG.names.forEach((n, j) => {
      const pts = grid.map((lam, gi) => [sx(Math.log(lam)), sy(path[gi][j])]);
      el('path', {
        class: 'series-line', d: linePath(pts),
        stroke: token(REG.noise[j] ? '--text-muted' : '--series-1'),
        'stroke-width': REG.noise[j] ? 1.2 : 2, opacity: REG.noise[j] ? 0.55 : 0.85,
      }, g);
    });
    /* label the largest few, at the left edge where they are still separated */
    directLabels(svg, REG.names.map((n, j) => [n, j, Math.abs(path[0][j])])
      .sort((a, b) => b[2] - a[2]).slice(0, 4)
      .map(([n, j]) => ({
        y: sy(path[0][j]) - 7, text: n, size: 10,
        color: token(REG.noise[j] ? '--text-muted' : '--series-1'),
      })), pad.l + 6, pad.t + 10, h - pad.b - 6, 14);
    el('line', {
      x1: sx(Math.log(grid[i])), x2: sx(Math.log(grid[i])), y1: pad.t, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 2,
    }, svg);
    /* the coefficients at the chosen λ, as a row of ticks on the right */
    const b = path[i];
    REG.names.forEach((n, j) => {
      el('circle', {
        cx: sx(Math.log(grid[i])), cy: sy(b[j]), r: 3,
        fill: token(REG.noise[j] ? '--text-muted' : '--series-1'),
      }, g);
    });
    const top = REG.names.map((n, j) => [n, j]).sort((a, c) => Math.abs(b[c[1]]) - Math.abs(b[a[1]])).slice(0, 6);
    top.forEach(([n, j], r) => {
      el('text', {
        x: w - pad.r + 10, y: pad.t + 16 + r * 15, 'font-size': 10,
        fill: Math.abs(b[j]) > 1e-9 ? token('--text-secondary') : token('--text-muted'),
      }, svg).textContent = `${n.length > 22 ? `${n.slice(0, 21)}…` : n}  ${Math.abs(b[j]) > 1e-9 ? k$(b[j]) : '0'}`;
    });
    el('text', {
      x: w - pad.r + 10, y: pad.t + 4, 'font-size': 9.5, 'font-weight': 640,
      fill: token('--text-muted'),
    }, svg).textContent = 'AT THIS λ';
  };

  lamEl.addEventListener('input', update);
  responsive(host, update);
}

pathFigure('ridge', { chart: 'ridge-chart', slider: 'ridge-lam', out: 'ridge-lam-out', note: 'ridge-note', chips: 'ridge-chips' });
pathFigure('lasso', { chart: 'lasso-chart', slider: 'lasso-lam', out: 'lasso-lam-out', note: 'lasso-note', chips: 'lasso-chips' });

{
  const cvL = REG.cv.lasso;
  const bMin = REG.paths.lasso[cvL.iMin], b1se = REG.paths.lasso[cvL.i1se];
  const live = (b) => b.filter((v) => Math.abs(v) > 1e-9).length;
  document.getElementById('lasso-keypoint').innerHTML =
    `<strong>The difference that matters.</strong> Ridge keeps all `
    + `${REG.names.length} predictors at every λ. Lasso, at the λ that `
    + `<a href="#lambda">cross-validation</a> picks, keeps <b>${live(b1se)}</b> — and it drops `
    + `${REG.noise.filter(Boolean).length - b1se.filter((v, j) => REG.noise[j] && Math.abs(v) > 1e-9).length} `
    + `of the ${REG.noise.filter(Boolean).length} predictors that genuinely have no relationship `
    + `with price. A model with fewer predictors in it is also a model someone can read.`;
}

/* ---- cross-validation curve ---- */
{
  const host = document.getElementById('cv-chart');
  const statsEl = document.getElementById('cv-stats');
  const noteEl = document.getElementById('cv-note');
  const subEl = document.getElementById('cv-sub');
  let method = 'lasso';

  document.getElementById('cv-sub').textContent = '';

  let draw;
  const update = () => {
    document.querySelectorAll('#cv-method button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.m === method)));
    const cv = REG.cv[method];
    const grid = REG.grids[method];
    const path = REG.paths[method];
    const live = (i) => path[i].filter((v) => Math.abs(v) > 1e-9).length;
    subEl.innerHTML = `5-fold cross-validation over ${grid.length} candidate values of λ, `
      + `on the same ${REG_N} homes and ${REG.names.length} predictors as the two figures above. `
      + `Vertical bars are ±1 standard error across the folds.`;
    statsEl.innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${Math.log(grid[cv.iMin]).toFixed(1)}</div>
        <div class="stat__label">log λ at the smallest MSE<br><span class="muted">RMSE ${money(Math.sqrt(cv.meanMse[cv.iMin]))} · ${live(cv.iMin)} predictors</span></div></div>
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${Math.log(grid[cv.i1se]).toFixed(1)}</div>
        <div class="stat__label">log λ within one standard error<br><span class="muted">RMSE ${money(Math.sqrt(cv.meanMse[cv.i1se]))} · ${live(cv.i1se)} predictors</span></div></div>
      <div class="stat stat--muted"><div class="stat__value" style="font-size:1.3rem">${money(Math.sqrt(Math.max(...cv.meanMse)))}</div>
        <div class="stat__label">RMSE at the largest λ<br><span class="muted">the model has given up</span></div></div>`;
    const dropped = REG.names.length - live(cv.i1se);
    noteEl.innerHTML = method === 'lasso'
      ? `Moving from the first dashed line to the second costs `
        + `<b>${money(Math.sqrt(cv.meanMse[cv.i1se]) - Math.sqrt(cv.meanMse[cv.iMin]))}</b> of `
        + `typical accuracy and removes <b>${live(cv.iMin) - live(cv.i1se)}</b> more predictors, `
        + `leaving ${live(cv.i1se)} of ${REG.names.length}. That is the trade the one-standard-error `
        + `rule is making: a difference in MSE too small to distinguish from fold-to-fold noise, `
        + `in exchange for a model ${dropped} predictors simpler.`
      : `Ridge's curve has the same shape but the count of predictors never moves: `
        + `${live(cv.iMin)} at the minimum, ${live(cv.i1se)} at the one-standard-error λ, `
        + `${REG.names.length} everywhere. Choosing λ here is choosing how hard to shrink, not what to keep. `
        + `Notice too that ridge's useful λ range sits orders of magnitude below lasso's — `
        + `the two penalties are on different scales, so a λ that is small for one is enormous for the other.`;
    draw();
  };

  draw = () => {
    const cv = REG.cv[method];
    const grid = REG.grids[method];
    const w = 900, h = 330;
    const pad = { l: 70, r: 20, t: 18, b: 42 };
    const svg = svgRoot(host, w, h);
    const lo = Math.min(...cv.meanMse.map((v, i) => v - cv.seMse[i]));
    const hi = Math.max(...cv.meanMse.map((v, i) => v + cv.seMse[i]));
    const sx = scale(Math.log(grid[0]), Math.log(grid[grid.length - 1]), pad.l, w - pad.r);
    const top = hi + (hi - lo) * 0.06;
    const sy = scale(Math.max(0, lo - (hi - lo) * 0.08), top, h - pad.b, pad.t);
    /* MSE here is in dollars squared, so the raw numbers run to 3 × 10¹⁰ and
       every tick would print as an exponential. One shared multiplier on the
       axis label leaves the ticks as small readable integers. */
    const T = scaledTicks(top, 5);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'log λ', yLabel: `Mean squared error${T.suffix}`,
      yTicks: T.ticks, yFmt: (v) => (v / T.div).toFixed(0),
    });
    for (const [i, label, color] of [[cv.iMin, 'smallest MSE', '--good'], [cv.i1se, 'within one SE', '--series-6']]) {
      el('line', {
        x1: sx(Math.log(grid[i])), x2: sx(Math.log(grid[i])), y1: pad.t, y2: h - pad.b,
        stroke: token(color), 'stroke-width': 1.6, 'stroke-dasharray': '5 4',
      }, svg);
      el('text', {
        class: 'annot', x: sx(Math.log(grid[i])) + 5,
        y: pad.t + (label === 'smallest MSE' ? 12 : 26), fill: token(color), 'font-weight': 620,
      }, svg).textContent = label;
    }
    const g = el('g', { 'clip-path': clipRect(svg, 'cv-clip', pad, w, h) }, svg);
    cv.meanMse.forEach((v, i) => {
      const x = sx(Math.log(grid[i]));
      el('line', {
        x1: x, x2: x, y1: sy(v - cv.seMse[i]), y2: sy(v + cv.seMse[i]),
        stroke: token('--series-8'), 'stroke-width': 1, opacity: 0.45,
      }, g);
    });
    el('path', {
      class: 'series-line', d: linePath(cv.meanMse.map((v, i) => [sx(Math.log(grid[i])), sy(v)])),
      stroke: token('--series-8'), 'stroke-width': 1.5, opacity: 0.6,
    }, g);
    cv.meanMse.forEach((v, i) => {
      const c = el('circle', {
        cx: sx(Math.log(grid[i])), cy: sy(v), r: 3, fill: token('--series-8'),
      }, g);
      c.addEventListener('pointerenter', (ev) => tip.show(
        `<b>log λ = ${Math.log(grid[i]).toFixed(2)}</b><br>MSE ${v.toExponential(2)}`
        + `<br>RMSE ${money(Math.sqrt(v))}`
        + `<br><span style="opacity:.7">${REG.paths[method][i].filter((z) => Math.abs(z) > 1e-9).length} predictors kept</span>`,
        ev.clientX, ev.clientY));
      c.addEventListener('pointerleave', () => tip.hide());
    });
  };

  document.querySelectorAll('#cv-method button').forEach((b) =>
    b.addEventListener('click', () => { method = b.dataset.m; update(); }));
  responsive(host, update);
}
