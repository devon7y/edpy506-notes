/* Every figure on tuning.html. The search surfaces are real: each cell is a
   random forest's cross-validated error. They are computed in the background
   in reading order, so the top of the page is usable at once and the lower
   figures fill in as their work finishes. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  mean, sd, money, money1k, rng, clipRect, directLabels,
} from './site.js';
import { growTree, predictTree, randomForest } from './trees.js';
import { standardize, lassoLambdaMax, logGrid, regPath, crossValidate } from './linreg.js';
import { kfold, cvScore, holdout, gridPoints, randomPoints, runJobs } from './tuning.js';
import { currentDataset, treeDesign, linearDesign } from './datasets.js';

initChrome();

const ds = currentDataset();
const tip = tooltip();
const k$ = money1k;
/* xorshift's first draw from a small seed is near zero; scramble seeds */
const seeded = (s) => rng((s * 2654435761) >>> 0);
const rmse = (y, idx, pred) => Math.sqrt(mean(idx.map((i) => (y[i] - pred(i)) ** 2)));

const HOMES = treeDesign(ds, ds.sample(8, 200));
const N = HOMES.y.length;
const TREE = { maxDepth: 5, minLeaf: 3 };

/* ===================================================================
   3. One split against five folds
   =================================================================== */
{
  const schemeHost = document.getElementById('cv-scheme');
  const dotsHost = document.getElementById('cv-dots');
  const single = [], folded = [];
  let draws = 0;

  const drawOnce = () => {
    const s = 100 + draws++;
    const h = holdout(N, 0.8, seeded(s));
    const t = growTree(h.train.map((i) => HOMES.X[i]), h.train.map((i) => HOMES.y[i]), TREE);
    single.push(rmse(HOMES.y, h.test, (i) => predictTree(t, HOMES.X[i])));
    const cv = cvScore(kfold(N, 5, seeded(s)), (tr, te) => {
      const tt = growTree(tr.map((i) => HOMES.X[i]), tr.map((i) => HOMES.y[i]), TREE);
      return rmse(HOMES.y, te, (i) => predictTree(tt, HOMES.X[i]));
    });
    folded.push(cv.mean);
  };
  for (let i = 0; i < 12; i++) drawOnce();

  document.getElementById('cv-sub').textContent =
    `A decision tree of depth ${TREE.maxDepth} on ${N} homes. Each draw shuffles the homes and `
    + `estimates the tree's error twice: once from a single 80/20 split, once from five-fold `
    + `cross-validation.`;

  const drawScheme = () => {
    const w = 440, h = 260;
    const svg = svgRoot(schemeHost, w, h);
    const x0 = 70, bw = 330, cellH = 22;
    el('text', { class: 'axis-label', x: 8, y: 30, 'font-weight': 640 }, svg).textContent = 'One split';
    el('rect', { x: x0, y: 16, width: bw * 0.8 - 2, height: cellH, rx: 3, fill: token('--baseline'), opacity: 0.55 }, svg);
    el('rect', { x: x0 + bw * 0.8, y: 16, width: bw * 0.2, height: cellH, rx: 3, fill: token('--series-6') }, svg);
    el('text', { class: 'tick', x: x0 + bw * 0.4, y: 31, 'text-anchor': 'middle' }, svg).textContent = 'training, 80%';
    el('text', { x: x0 + bw * 0.9, y: 31, 'text-anchor': 'middle', 'font-size': 11, fill: token('--surface-1'), 'font-weight': 700 }, svg).textContent = 'test';
    el('text', { class: 'axis-label', x: 8, y: 72, 'font-weight': 640 }, svg).textContent = 'Five folds';
    const fw = bw / 5;
    for (let r = 0; r < 5; r++) {
      const y = 82 + r * (cellH + 6);
      el('text', { class: 'tick', x: x0 - 8, y: y + 15, 'text-anchor': 'end' }, svg).textContent = `round ${r + 1}`;
      for (let f = 0; f < 5; f++) {
        const test = f === r;
        el('rect', { x: x0 + f * fw + 1, y, width: fw - 2, height: cellH, rx: 3,
          fill: token(test ? '--series-6' : '--baseline'), opacity: test ? 1 : 0.55 }, svg);
        if (test) el('text', { x: x0 + f * fw + fw / 2, y: y + 15, 'text-anchor': 'middle', 'font-size': 11,
          fill: token('--surface-1'), 'font-weight': 700 }, svg).textContent = 'test';
      }
    }
    el('text', { class: 'tick', x: x0 + bw / 2, y: h - 4, 'text-anchor': 'middle' }, svg)
      .textContent = 'every home is in the test fold exactly once; the five scores are averaged';
  };

  let drawDots;
  const update = () => {
    const sS = single.length > 1 ? sd(single) : 0, sF = folded.length > 1 ? sd(folded) : 0;
    document.getElementById('cv-stats').innerHTML = `
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.3rem">± ${money(sS)}</div>
        <div class="stat__label">Spread of the one-split estimates<br><span class="muted">${money(Math.min(...single))} to ${money(Math.max(...single))}</span></div></div>
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.3rem">± ${money(sF)}</div>
        <div class="stat__label">Spread of the five-fold estimates<br><span class="muted">${money(Math.min(...folded))} to ${money(Math.max(...folded))}</span></div></div>`;
    document.getElementById('cv-note').innerHTML =
      `Over <b>${single.length}</b> draws, the one-split estimate of the same tree's error has a `
      + `standard deviation of <b>${money(sS)}</b>, and the five-fold estimate one of <b>${money(sF)}</b>`
      + `${sF > 0 ? `, about ${(sS / sF).toFixed(1)} times smaller` : ''}. Nothing about the tree or the `
      + `homes differs between draws; only the luck of which homes landed in the test set. A `
      + `comparison between two hyperparameter settings made on a single split can be decided by that `
      + `luck, which is why every candidate in a search is scored by cross-validation.`;
    drawDots();
  };

  drawDots = () => {
    const w = 440, h = 260;
    const pad = { l: 96, r: 16, t: 24, b: 44 };
    const svg = svgRoot(dotsHost, w, h);
    const all = [...single, ...folded];
    const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
    const sx = scale(lo - span * 0.08, hi + span * 0.08, pad.l, w - pad.r);
    const sy = scale(0, 1, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, { xLabel: 'Estimated RMSE', yTicks: [], xFmt: k$ });
    const rowY = { single: pad.t + 55, folded: pad.t + 140 };
    el('text', { class: 'axis-label', x: pad.l - 10, y: rowY.single + 4, 'text-anchor': 'end', 'font-weight': 640 }, svg).textContent = 'One split';
    el('text', { class: 'axis-label', x: pad.l - 10, y: rowY.folded + 4, 'text-anchor': 'end', 'font-weight': 640 }, svg).textContent = 'Five folds';
    const beeswarm = (vals, y, color) => {
      const placed = [];
      vals.forEach((v, i) => {
        const x = sx(v);
        let dy = 0, k = 0;
        while (placed.some((p) => Math.abs(p.x - x) < 7 && Math.abs(p.y - (y + dy)) < 7)) {
          k++; dy = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 7;
        }
        placed.push({ x, y: y + dy });
        el('circle', { cx: x, cy: y + dy, r: 3.4, fill: token(color), opacity: i === vals.length - 1 ? 1 : 0.7,
          stroke: i === vals.length - 1 ? token('--text-primary') : 'none', 'stroke-width': 1.4 }, svg);
      });
      const m = mean(vals);
      el('line', { x1: sx(m), x2: sx(m), y1: y - 30, y2: y + 30, stroke: token(color), 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, svg);
    };
    beeswarm(single, rowY.single, '--c-test');
    beeswarm(folded, rowY.folded, '--c-train');
  };

  document.getElementById('cv-one').addEventListener('click', () => { drawOnce(); update(); });
  document.getElementById('cv-many').addEventListener('click', async () => {
    await runJobs(Array.from({ length: 30 }, () => drawOnce), () => update());
  });
  document.getElementById('cv-reset').addEventListener('click', () => {
    single.length = 0; folded.length = 0; draws = 0;
    for (let i = 0; i < 12; i++) drawOnce();
    update();
  });
  responsive(schemeHost, drawScheme);
  responsive(dotsHost, update);
}

/* ===================================================================
   4. Grid search against random search
   =================================================================== */
const FOLDS = kfold(N, 5, seeded(9));
const DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const TREES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60];
const MAXT = 60;
/* SURF[d][t]: cross-validated RMSE at depth DEPTHS[d] with TREES[t] trees. One
   forest of 60 trees per depth and fold gives every smaller count for free,
   by averaging only its first trees. */
const SURF = DEPTHS.map(() => new Array(TREES.length).fill(NaN));
let surfDone = false;
const surfaceJobs = () => DEPTHS.flatMap((d, di) => {
  const sse = new Float64Array(MAXT);
  let n = 0, folds = 0;
  return FOLDS.map((te) => () => {
    const held = new Set(te);
    const tr = FOLDS.flat().filter((i) => !held.has(i));
    const rf = randomForest(tr.map((i) => HOMES.X[i]), tr.map((i) => HOMES.y[i]),
      { nTrees: MAXT, maxDepth: d, minLeaf: 1, rand: seeded(3) });
    const acc = new Float64Array(te.length);
    rf.trees.forEach((t, k) => te.forEach((i, j) => {
      acc[j] += predictTree(t, HOMES.X[i]);
      sse[k] += (HOMES.y[i] - acc[j] / (k + 1)) ** 2;
    }));
    n += te.length;
    if (++folds === FOLDS.length) TREES.forEach((t, ti) => { SURF[di][ti] = Math.sqrt(sse[t - 1] / n); });
  });
});

let refreshSearch = () => {};
{
  const gridHost = document.getElementById('gs-grid');
  const randHost = document.getElementById('gs-rand');
  const budgetEl = document.getElementById('gs-budget');
  const busy = document.getElementById('gs-busy');
  let draw = 0;

  const trials = () => {
    const n = +budgetEl.value;
    const pick = (len) => Array.from({ length: n }, (_, i) => Math.round((i + 0.5) * len / n - 0.5));
    const grid = gridPoints({ d: pick(DEPTHS.length), t: pick(TREES.length) });
    const rand = randomPoints({ d: DEPTHS.map((_, i) => i), t: TREES.map((_, i) => i) }, n * n, seeded(500 + draw));
    return { grid, rand };
  };
  const bestOf = (pts) => pts.reduce((b, p) => (SURF[p.d][p.t] < SURF[b.d][b.t] ? p : b));

  const heat = (host, pts, label, color) => {
    const w = 440, h = 330;
    const pad = { l: 56, r: 14, t: 26, b: 44 };
    const svg = svgRoot(host, w, h);
    el('text', { x: pad.l, y: 16, 'font-size': 12.5, 'font-weight': 660, fill: token(color) }, svg).textContent = label;
    if (!surfDone) {
      el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: token('--text-muted'), 'font-size': 13 }, svg)
        .textContent = 'Scoring every combination…';
      return;
    }
    const flat = SURF.flat();
    const lo = Math.min(...flat), hi = Math.max(...flat);
    const cw = (w - pad.l - pad.r) / DEPTHS.length, ch = (h - pad.t - pad.b) / TREES.length;
    const shade = (v) => 0.08 + 0.85 * (1 - (v - lo) / (hi - lo));
    DEPTHS.forEach((d, di) => TREES.forEach((t, ti) => {
      const v = SURF[di][ti];
      const r = el('rect', { x: pad.l + di * cw, y: h - pad.b - (ti + 1) * ch, width: cw - 1, height: ch - 1,
        fill: token('--series-1'), opacity: shade(v) }, svg);
      r.addEventListener('pointerenter', (ev) => tip.show(
        `depth <b>${d}</b>, <b>${t}</b> trees<br>cross-validated RMSE ${money(v)}`, ev.clientX, ev.clientY));
      r.addEventListener('pointerleave', () => tip.hide());
    }));
    DEPTHS.forEach((d, di) => {
      el('text', { class: 'tick', x: pad.l + di * cw + cw / 2, y: h - pad.b + 14, 'text-anchor': 'middle' }, svg).textContent = d;
    });
    TREES.forEach((t, ti) => {
      if (ti % 2 === 0) el('text', { class: 'tick', x: pad.l - 6, y: h - pad.b - ti * ch - ch / 2 + 4, 'text-anchor': 'end' }, svg).textContent = t;
    });
    el('text', { class: 'axis-label', x: (pad.l + w - pad.r) / 2, y: h - 4, 'text-anchor': 'middle' }, svg).textContent = 'Maximum depth';
    el('text', { class: 'axis-label', x: 12, y: (pad.t + h - pad.b) / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 12 ${(pad.t + h - pad.b) / 2})` }, svg).textContent = 'Number of trees';
    const best = bestOf(pts);
    for (const p of pts) {
      const cx = pad.l + p.d * cw + cw / 2, cy = h - pad.b - p.t * ch - ch / 2;
      el('circle', { cx, cy, r: p === best ? 7 : 4.6, fill: token('--surface-1'),
        stroke: token(color), 'stroke-width': p === best ? 3 : 2 }, svg);
    }
    /* the true best, for reference */
    let tb = { d: 0, t: 0 };
    DEPTHS.forEach((_, di) => TREES.forEach((_, ti) => { if (SURF[di][ti] < SURF[tb.d][tb.t]) tb = { d: di, t: ti }; }));
    el('rect', { x: pad.l + tb.d * cw + 1, y: h - pad.b - (tb.t + 1) * ch + 1, width: cw - 3, height: ch - 3,
      fill: 'none', stroke: token('--text-primary'), 'stroke-width': 2.4, 'stroke-dasharray': '4 2' }, svg);
  };

  const update = () => {
    const n = +budgetEl.value;
    document.getElementById('gs-budget-out').textContent = `${n * n} (${n} × ${n})`;
    const { grid, rand } = trials();
    heat(gridHost, grid, `Grid search: ${grid.length} trials`, '--series-7');
    heat(randHost, rand, `Random search: ${rand.length} trials`, '--series-6');
    if (!surfDone) return;
    let tb = { d: 0, t: 0 };
    DEPTHS.forEach((_, di) => TREES.forEach((_, ti) => { if (SURF[di][ti] < SURF[tb.d][tb.t]) tb = { d: di, t: ti }; }));
    const g = bestOf(grid), r = bestOf(rand);
    const distinct = (pts) => new Set(pts.map((p) => p.d)).size;
    document.getElementById('gs-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem;color:var(--series-7)">${money(SURF[g.d][g.t])}</div>
        <div class="stat__label">Best found by grid search<br><span class="muted">depth ${DEPTHS[g.d]}, ${TREES[g.t]} trees · ${distinct(grid)} depths tried</span></div></div>
      <div class="stat"><div class="stat__value" style="font-size:1.25rem;color:var(--series-6)">${money(SURF[r.d][r.t])}</div>
        <div class="stat__label">Best found by random search<br><span class="muted">depth ${DEPTHS[r.d]}, ${TREES[r.t]} trees · ${distinct(rand)} depths tried</span></div></div>
      <div class="stat stat--muted"><div class="stat__value" style="font-size:1.25rem">${money(SURF[tb.d][tb.t])}</div>
        <div class="stat__label">Best anywhere in the space<br><span class="muted">depth ${DEPTHS[tb.d]}, ${TREES[tb.t]} trees · dashed outline</span></div></div>`;
    document.getElementById('gs-note').innerHTML =
      `The stronger the blue, the lower the error. The shading changes a great deal from left to right and very little `
      + `from bottom to top: depth matters here and the number of trees barely does. `
      + `A ${n} × ${n} grid spends its ${n * n} trials on only <b>${distinct(grid)}</b> different depths, `
      + `repeating each at ${n} tree counts that make almost no difference. Random search, with the `
      + `same ${n * n} trials, tried <b>${distinct(rand)}</b> different depths. When one `
      + `hyperparameter matters far more than another, a grid spends most of its budget varying the `
      + `one that does not, and random search does not.`;
  };

  budgetEl.addEventListener('input', update);
  document.getElementById('gs-redraw').addEventListener('click', () => { draw++; update(); });
  responsive(gridHost, update);
  responsive(randHost, update);
  refreshSearch = (msg) => { busy.textContent = msg || ''; update(); };

  /* what a search costs */
  const costHost = document.getElementById('gs-cost');
  const RANDOM_BUDGET = 30;
  document.getElementById('gs-cost-note').innerHTML =
    `With eight values for each of three hyperparameters, a grid is ${(8 ** 3).toLocaleString('en-CA')} `
    + `combinations and ${(8 ** 3 * 5).toLocaleString('en-CA')} fitted models. A random search costs `
    + `whatever number of trials it is given, however large the space it samples from.`;
  responsive(costHost, () => {
    const w = 700, h = 260;
    const pad = { l: 70, r: 170, t: 16, b: 44 };
    const svg = svgRoot(costHost, w, h);
    const V = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    const sx = scale(2, 10, pad.l, w - pad.r);
    const sy = scale(1, 4, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Values tried per hyperparameter', yLabel: 'Models fitted (log scale)',
      xTicks: V, yTicks: [1, 2, 3, 4], yFmt: (v) => (10 ** v).toLocaleString('en-CA'),
    });
    const lines = [
      [(v) => v * v * 5, '--series-7', 'Grid, 2 hyperparameters'],
      [(v) => v * v * v * 5, '--series-8', 'Grid, 3 hyperparameters'],
      [() => RANDOM_BUDGET * 5, '--series-6', `Random, ${RANDOM_BUDGET} trials`],
    ];
    const labels = [];
    for (const [f, c, name] of lines) {
      const pts = V.map((v) => [sx(v), sy(Math.log10(f(v)))]);
      el('path', { class: 'series-line', d: linePath(pts), stroke: token(c), 'stroke-width': 2.4 }, svg);
      labels.push({ y: pts[pts.length - 1][1], text: name, color: token(c), size: 10.5 });
    }
    directLabels(svg, labels, w - pad.r + 8, pad.t + 6, h - pad.b);
  });
}

/* ===================================================================
   5. Three models, one hyperparameter each
   =================================================================== */
const TM = { lasso: null, tree: null, forest: null };
/* few homes for many predictors, so that a small λ has room to overfit */
const LASSO_N = 40;
let refreshModels = () => {};
{
  const legendHost = document.getElementById('tm-legend');
  legendHost.innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-train')}"></span>Error on the training data</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-test')}"></span>Cross-validated error</span>`;
  const P = linearDesign(ds, ds.sample(5, 1)).names.length;
  document.getElementById('tm-sub').textContent =
    `Lasso on ${LASSO_N} homes and all ${P} predictors; a decision tree and a random forest on ${N} `
    + `homes. Five-fold cross-validation throughout.`;

  const redraws = [];
  const panel = (hostId, data, opts) => {
    const host = document.getElementById(hostId);
    const draw = () => {
      const w = 300, h = 250;
      const pad = { l: 52, r: 12, t: 30, b: 42 };
      const svg = svgRoot(host, w, h);
      el('text', { x: pad.l, y: 16, 'font-size': 12.5, 'font-weight': 660, fill: token('--text-primary') }, svg).textContent = opts.title;
      const D = TM[data];
      if (!D) {
        el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: token('--text-muted'), 'font-size': 12 }, svg).textContent = 'Fitting…';
        return;
      }
      const all = [...D.train, ...D.cv];
      const sx = scale(D.x[0], D.x[D.x.length - 1], pad.l, w - pad.r);
      const sy = scale(0, Math.max(...all) * 1.08, h - pad.b, pad.t);
      frame(svg, w, h, pad, sx, sy, { xLabel: opts.xLabel, yFmt: k$, xTicks: opts.xTicks, xFmt: opts.xFmt });
      const best = D.cv.indexOf(Math.min(...D.cv));
      if (opts.zones) {
        const [left, right] = opts.zones;
        el('text', { class: 'annot', x: pad.l + 4, y: pad.t + 10, fill: token('--text-muted') }, svg).textContent = left;
        el('text', { class: 'annot', x: w - pad.r - 4, y: pad.t + 10, 'text-anchor': 'end', fill: token('--text-muted') }, svg).textContent = right;
      }
      const g = el('g', { 'clip-path': clipRect(svg, `${hostId}-clip`, pad, w, h) }, svg);
      for (const [key, c] of [['train', '--c-train'], ['cv', '--c-test']]) {
        el('path', { class: 'series-line', stroke: token(c), 'stroke-width': 2.2,
          d: linePath(D.x.map((x, i) => [sx(x), sy(D[key][i])])) }, g);
      }
      if (opts.best !== false) {
        el('line', { x1: sx(D.x[best]), x2: sx(D.x[best]), y1: pad.t, y2: h - pad.b,
          stroke: token('--good'), 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, svg);
        el('circle', { cx: sx(D.x[best]), cy: sy(D.cv[best]), r: 5, fill: token('--good') }, svg);
      }
    };
    redraws.push(draw);
    responsive(host, draw);
  };
  panel('tm-lasso', 'lasso', { title: 'Lasso: λ', xLabel: 'log λ', zones: ['overfitting', 'underfitting'] });
  panel('tm-tree', 'tree', { title: 'Decision tree: maximum depth', xLabel: 'Maximum depth',
    xTicks: [1, 4, 8, 12], zones: ['underfitting', 'overfitting'] });
  panel('tm-forest', 'forest', { title: 'Random forest: number of trees', xLabel: 'Number of trees',
    xTicks: [1, 25, 50, 75, 100], best: false });

  refreshModels = () => {
    redraws.forEach((f) => f());
    if (!TM.lasso || !TM.tree || !TM.forest) return;
    const bl = TM.lasso.cv.indexOf(Math.min(...TM.lasso.cv));
    const bt = TM.tree.cv.indexOf(Math.min(...TM.tree.cv));
    const f10 = TM.forest.cv[9], f100 = TM.forest.cv[TM.forest.cv.length - 1], f1 = TM.forest.cv[0];
    document.getElementById('tm-note').innerHTML =
      `<b>Lasso</b>: cross-validated error is lowest at log λ ≈ <b>${TM.lasso.x[bl].toFixed(1)}</b> `
      + `(${money(TM.lasso.cv[bl])}). Smaller values fit the training homes more closely and do worse on `
      + `held-out ones; larger values shrink every coefficient toward nothing. `
      + `<b>Tree</b>: lowest at depth <b>${TM.tree.x[bt]}</b> (${money(TM.tree.cv[bt])}). Deeper trees keep `
      + `driving training error toward zero while held-out error climbs. `
      + `<b>Forest</b>: one tree scores ${money(f1)}, ten trees ${money(f10)}, and a hundred ${money(f100)}. `
      + `Adding trees has no overfitting side: past a point it stops helping rather than starting to `
      + `hurt. Three models, three hyperparameters, and one method for choosing all of them.`;
  };
}

const modelJobs = () => {
  const jobs = [];
  jobs.push(() => {
    const L = linearDesign(ds, ds.sample(5, LASSO_N));
    const { Z } = standardize(L.X);
    const yb = mean(L.y), yc = L.y.map((v) => v - yb);
    const grid = logGrid(Math.exp(6), Math.exp(Math.log(lassoLambdaMax(Z, yc)) + 0.3), 30);
    const path = regPath(Z, yc, grid, 'lasso');
    const train = path.map((b) => Math.sqrt(mean(Z.map((z, i) => (yc[i] - z.reduce((s, v, j) => s + v * b[j], 0)) ** 2))));
    const cv = crossValidate(L.X, L.y, grid, 'lasso', 5, seeded(9)).meanMse.map(Math.sqrt);
    TM.lasso = { x: grid.map(Math.log), train, cv };
  });
  jobs.push(() => {
    const depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const train = depths.map((d) => {
      const t = growTree(HOMES.X, HOMES.y, { maxDepth: d, minLeaf: 1 });
      return rmse(HOMES.y, [...Array(N).keys()], (i) => predictTree(t, HOMES.X[i]));
    });
    const cv = depths.map((d) => cvScore(FOLDS, (tr, te) => {
      const t = growTree(tr.map((i) => HOMES.X[i]), tr.map((i) => HOMES.y[i]), { maxDepth: d, minLeaf: 1 });
      return rmse(HOMES.y, te, (i) => predictTree(t, HOMES.X[i]));
    }).mean);
    TM.tree = { x: depths, train, cv };
  });
  /* the forest, with the same one-forest-gives-every-count trick */
  const T_MAX = 100;
  const acc = { cvSse: new Float64Array(T_MAX), n: 0 };
  FOLDS.forEach((te) => jobs.push(() => {
    const held = new Set(te);
    const tr = FOLDS.flat().filter((i) => !held.has(i));
    const rf = randomForest(tr.map((i) => HOMES.X[i]), tr.map((i) => HOMES.y[i]), { nTrees: T_MAX, maxDepth: 12, minLeaf: 1, rand: seeded(3) });
    const s = new Float64Array(te.length);
    rf.trees.forEach((t, k) => te.forEach((i, j) => { s[j] += predictTree(t, HOMES.X[i]); acc.cvSse[k] += (HOMES.y[i] - s[j] / (k + 1)) ** 2; }));
    acc.n += te.length;
  }));
  jobs.push(() => {
    const rf = randomForest(HOMES.X, HOMES.y, { nTrees: T_MAX, maxDepth: 12, minLeaf: 1, rand: seeded(3) });
    const s = new Float64Array(N), trSse = new Float64Array(T_MAX);
    rf.trees.forEach((t, k) => HOMES.X.forEach((r, i) => { s[i] += predictTree(t, r); trSse[k] += (HOMES.y[i] - s[i] / (k + 1)) ** 2; }));
    TM.forest = {
      x: Array.from({ length: T_MAX }, (_, i) => i + 1),
      train: Array.from(trSse, (v) => Math.sqrt(v / N)),
      cv: Array.from(acc.cvSse, (v) => Math.sqrt(v / acc.n)),
    };
  });
  return jobs;
};

/* ===================================================================
   6. The grid search from the lecture's worked example
   =================================================================== */
const RG = { depths: [2, 4, 8, null], mtry: [1, 3, 5, 9], trees: 40, cells: null };
const gridJobs = () => {
  const out = RG.depths.map(() => RG.mtry.map(() => []));
  const jobs = [];
  RG.depths.forEach((d, di) => RG.mtry.forEach((m, mi) => FOLDS.forEach((te) => jobs.push(() => {
    const held = new Set(te);
    const tr = FOLDS.flat().filter((i) => !held.has(i));
    const rf = randomForest(tr.map((i) => HOMES.X[i]), tr.map((i) => HOMES.y[i]),
      { nTrees: RG.trees, maxDepth: d ?? 50, minLeaf: 1, mtry: m, rand: seeded(3) });
    out[di][mi].push(rmse(HOMES.y, te, (i) => rf.predict(HOMES.X[i])));
    if (out.every((row) => row.every((c) => c.length === FOLDS.length))) RG.cells = out.map((row) => row.map(mean));
  }))));
  return jobs;
};
let refreshGrid = () => {};
{
  const host = document.getElementById('rg-heat');
  document.getElementById('rg-sub').textContent =
    `A random forest of ${RG.trees} trees on ${N} homes, with ${HOMES.X[0].length} features. The stronger the blue, the lower the error.`;
  const draw = () => {
    const w = 560, h = 300;
    const svg = svgRoot(host, w, h);
    if (!RG.cells) {
      el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: token('--text-muted'), 'font-size': 13 }, svg)
        .textContent = 'Fitting eighty forests…';
      return;
    }
    const flat = RG.cells.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
    const x0 = 120, y0 = 40, cw = 105, ch = 58;
    RG.mtry.forEach((m, mi) => el('text', { class: 'axis-label', x: x0 + mi * cw + cw / 2, y: y0 - 12,
      'text-anchor': 'middle', 'font-weight': 620 }, svg).textContent = `${m} per split`);
    let best = [0, 0];
    RG.cells.forEach((row, di) => row.forEach((v, mi) => { if (v < RG.cells[best[0]][best[1]]) best = [di, mi]; }));
    RG.depths.forEach((d, di) => {
      el('text', { class: 'axis-label', x: x0 - 12, y: y0 + di * ch + ch / 2 + 4, 'text-anchor': 'end',
        'font-weight': 620 }, svg).textContent = d == null ? 'No limit' : `Depth ${d}`;
      RG.mtry.forEach((m, mi) => {
        const v = RG.cells[di][mi];
        const a = 0.1 + 0.85 * (1 - (v - lo) / (hi - lo));
        const isBest = di === best[0] && mi === best[1];
        el('rect', { x: x0 + mi * cw + 2, y: y0 + di * ch + 2, width: cw - 4, height: ch - 4, rx: 4,
          fill: token('--series-1'), opacity: a }, svg);
        if (isBest) el('rect', { x: x0 + mi * cw + 2, y: y0 + di * ch + 2, width: cw - 4, height: ch - 4, rx: 4,
          fill: 'none', stroke: token('--text-primary'), 'stroke-width': 2.5 }, svg);
        el('text', { x: x0 + mi * cw + cw / 2, y: y0 + di * ch + ch / 2 + (isBest ? 0 : 5), 'text-anchor': 'middle',
          'font-size': 13, 'font-weight': 660, fill: token(a > 0.62 ? '--surface-1' : '--text-primary') }, svg).textContent = k$(v);
        if (isBest) el('text', { x: x0 + mi * cw + cw / 2, y: y0 + di * ch + ch / 2 + 16, 'text-anchor': 'middle',
          'font-size': 10, 'font-weight': 700, fill: token(a > 0.62 ? '--surface-1' : '--text-primary') }, svg).textContent = 'best';
      });
    });
    const [bd, bm] = best;
    document.getElementById('rg-note').innerHTML =
      `The lowest cross-validated error is <b>${money(RG.cells[bd][bm])}</b>, with `
      + `${RG.depths[bd] == null ? 'no depth limit' : `a maximum depth of ${RG.depths[bd]}`} and `
      + `${RG.mtry[bm]} feature${RG.mtry[bm] === 1 ? '' : 's'} per split. The worst cell is `
      + `${money(hi)}. Sixteen combinations at five folds each is eighty forests; adding a third `
      + `hyperparameter with four values would make it three hundred and twenty.`;
  };
  refreshGrid = draw;
  responsive(host, draw);
}

/* ===================================================================
   7. Tuning on the test set, against tuning by cross-validation
   =================================================================== */
const WF = {
  tested: [], cv: [], repeats: 60, train: 200, test: 40, fresh: 1000,
  depths: [1, 2, 3, 4, 6, 8, 12], leaves: [1, 3, 6, 12, 25],
};
WF.cands = WF.depths.flatMap((d) => WF.leaves.map((m) => ({ maxDepth: d, minLeaf: m })));
/* One repetition is split into seven jobs so no single one holds the page:
   fit every candidate on the training rows, cross-validate them one fold at a
   time, then choose. The 1,000 fresh homes stand in for the true error. */
const workflowJobs = () => Array.from({ length: WF.repeats }, (_, r) => {
  const range = (from, n) => Array.from({ length: n }, (_, i) => from + i);
  const tr = range(0, WF.train), te = range(WF.train, WF.test), fresh = range(WF.train + WF.test, WF.fresh);
  const folds = kfold(WF.train, 5, seeded(r));
  let H, models, onTest, cvSum;
  const fit = (c, idx) => growTree(idx.map((i) => H.X[i]), idx.map((i) => H.y[i]), c);
  const score = (t, idx) => rmse(H.y, idx, (i) => predictTree(t, H.X[i]));
  return [
    () => {
      H = treeDesign(ds, ds.sample(600 + r, WF.train + WF.test + WF.fresh));
      models = WF.cands.map((c) => fit(c, tr));
      onTest = models.map((t) => score(t, te));
      cvSum = new Float64Array(WF.cands.length);
    },
    ...folds.map((held) => () => {
      const out = new Set(held);
      const inFold = tr.filter((i) => !out.has(i));
      WF.cands.forEach((c, k) => { cvSum[k] += score(fit(c, inFold), held); });
    }),
    () => {
      /* tuned on the test set: keep the combination with the lowest test
         error, and report that same test error */
      const bT = onTest.indexOf(Math.min(...onTest));
      WF.tested.push({ reported: onTest[bT], fresh: score(models[bT], fresh) });
      /* tuned by cross-validation, refitted on all training rows, tested once */
      const bC = cvSum.indexOf(Math.min(...cvSum));
      WF.cv.push({ reported: onTest[bC], fresh: score(models[bC], fresh) });
    },
  ];
}).flat();
let refreshWorkflow = () => {};
{
  const host = document.getElementById('wf-chart');
  document.getElementById('wf-sub').textContent =
    `${WF.repeats} repetitions, each on a new sample of homes: ${WF.train} to train on, ${WF.test} as the `
    + `test set, and ${WF.fresh.toLocaleString('en-CA')} more that neither method sees, whose error stands `
    + `in for the true one. The search is over ${WF.cands.length} combinations of a tree's maximum depth `
    + `and minimum leaf size.`;
  const gaps = (arr) => arr.map((o) => o.reported - o.fresh);
  const draw = () => {
    const w = 900, h = 250;
    const pad = { l: 190, r: 20, t: 26, b: 44 };
    const svg = svgRoot(host, w, h);
    const all = [...gaps(WF.tested), ...gaps(WF.cv)];
    if (!all.length) {
      el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: token('--text-muted'), 'font-size': 13 }, svg)
        .textContent = 'Running the tuning job…';
      return;
    }
    const m = Math.max(8000, ...all.map(Math.abs)) * 1.1;
    const sx = scale(-m, m, pad.l, w - pad.r);
    const sy = scale(0, 1, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, { xLabel: 'Reported error minus true error', yTicks: [], xFmt: k$ });
    el('line', { x1: sx(0), x2: sx(0), y1: pad.t, y2: h - pad.b, stroke: token('--text-muted'), 'stroke-dasharray': '4 3' }, svg);
    el('text', { class: 'annot', x: sx(0) - 6, y: pad.t - 8, 'text-anchor': 'end' }, svg).textContent = '← looks better than it is';
    el('text', { class: 'annot', x: sx(0) + 6, y: pad.t - 8 }, svg).textContent = 'looks worse than it is →';
    const rows = [['Tuned on the test set', gaps(WF.tested), '--c-test', pad.t + 48],
      ['Tuned by cross-validation', gaps(WF.cv), '--c-train', pad.t + 128]];
    for (const [label, vals, c, y] of rows) {
      el('text', { class: 'axis-label', x: pad.l - 12, y: y + 4, 'text-anchor': 'end', 'font-weight': 640 }, svg).textContent = label;
      const placed = [];
      vals.forEach((v) => {
        const x = sx(v);
        let dy = 0, k = 0;
        while (placed.some((q) => Math.abs(q.x - x) < 7 && Math.abs(q.y - dy) < 7) && k < 12) {
          k++; dy = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 6;
        }
        placed.push({ x, y: dy });
        el('circle', { cx: x, cy: y + dy, r: 3.2, fill: token(c), opacity: 0.7 }, svg);
      });
      const mv = mean(vals);
      el('line', { x1: sx(mv), x2: sx(mv), y1: y - 34, y2: y + 34, stroke: token(c), 'stroke-width': 2.6 }, svg);
      el('text', { class: 'annot', x: sx(mv), y: y + 46, 'text-anchor': 'middle', 'font-weight': 660, fill: token(c),
        stroke: token('--surface-1'), 'stroke-width': 4, 'paint-order': 'stroke' }, svg)
        .textContent = `mean ${mv < 0 ? '−' : '+'}${money(Math.abs(mv))}`;
    }
    const done = WF.tested.length;
    const gT = mean(gaps(WF.tested)), gC = mean(gaps(WF.cv));
    const seT = done > 1 ? sd(gaps(WF.tested)) / Math.sqrt(done) : 0;
    const flatT = WF.tested.filter((o) => o.reported < o.fresh).length;
    const flatC = WF.cv.filter((o) => o.reported < o.fresh).length;
    const signed = (v) => `${v < 0 ? '−' : '+'}${money(Math.abs(v))}`;
    document.getElementById('wf-stats').innerHTML = `
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.3rem">${signed(gT)}</div>
        <div class="stat__label">Tuned on the test set: reported minus true, on average<br><span class="muted">too flattering in ${flatT} of ${done}</span></div></div>
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.3rem">${signed(gC)}</div>
        <div class="stat__label">Tuned by cross-validation, tested once<br><span class="muted">too flattering in ${flatC} of ${done}</span></div></div>`;
    const note = document.getElementById('wf-note');
    if (done < WF.repeats) { note.textContent = ''; return; }
    const seC = sd(gaps(WF.cv)) / Math.sqrt(done);
    const biased = gT < -2 * seT;
    note.innerHTML = (biased
      ? `Choosing the combination by its test error and then reporting that same test error makes the `
        + `tree look <b>${money(Math.abs(gT))}</b> better, on average, than it is on homes it has never `
        + `met (standard error ${money(seT)}), and too flattering in <b>${flatT} of ${done}</b> repetitions. `
        + `With ${WF.cands.length} candidates and ${WF.test} test homes, some candidate is bound to suit `
        + `those particular homes by luck, and picking by test error picks that luck along with the `
        + `model. The test set has helped make a decision, so it is no longer unseen data. `
      : `Choosing the combination by its test error and reporting that same error is off by `
        + `${signed(gT)} on average here (standard error ${money(seT)}). `)
      + `Choosing by cross-validation on the training rows and testing once at the end is off by `
      + `${signed(gC)} on average (standard error ${money(seC)}), too flattering in ${flatC} of ${done}`
      + (Math.abs(gC) < 2 * seC
        ? `: noisy, as any ${WF.test}-home test set is, but not biased in either direction.`
        : '.');
    host.dataset.done = '';
  };
  refreshWorkflow = draw;
  responsive(host, draw);
}

/* ===================================================================
   Background work, in reading order
   =================================================================== */
(async () => {
  await runJobs(surfaceJobs(), (i, n) => refreshSearch(i < n ? `scoring the search space: ${i} of ${n} forests` : ''));
  surfDone = true;
  refreshSearch('');
  {
    /* section 8's evidence, from the same surface */
    const byTrees = DEPTHS.map((_, di) => Math.max(...SURF[di]) - Math.min(...SURF[di]));
    const byDepth = TREES.map((_, ti) => Math.max(...SURF.map((row) => row[ti])) - Math.min(...SURF.map((row) => row[ti])));
    document.getElementById('tradeoff-evidence').innerHTML =
      `Across the <a href="#search">depth-by-trees search space</a>, changing the number of trees `
      + `from ${TREES[0]} to ${TREES[TREES.length - 1]} moves the cross-validated error by at most <b>${money(Math.max(...byTrees))}</b> at any depth, and by `
      + `a median of ${money([...byTrees].sort((a, b) => a - b)[Math.floor(byTrees.length / 2)])}. `
      + `Changing the depth from ${DEPTHS[0]} to ${DEPTHS[DEPTHS.length - 1]} moves it by at least <b>${money(Math.min(...byDepth))}</b> at `
      + `any number of trees.`;
  }
  await runJobs(modelJobs(), () => refreshModels());
  refreshModels();
  await runJobs(gridJobs(), () => refreshGrid());
  refreshGrid();
  const wfBusy = document.getElementById('wf-busy');
  await runJobs(workflowJobs(), (i, n) => {
    wfBusy.textContent = i < n ? `${WF.tested.length} of ${WF.repeats} repetitions` : '';
    refreshWorkflow();
  });
})();
