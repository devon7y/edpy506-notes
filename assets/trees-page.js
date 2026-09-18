/* Everything interactive on trees.html. As on the regression page, every number
   in the prose beside a figure is computed here rather than typed in, so the
   text cannot drift away from what the figure shows. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  mean, money, money1k, fmt, rng, clipRect, directLabels, shuffle,
  tweenInput, tweenInputs,
} from './site.js';
import { ols, metrics } from './linreg.js';
import {
  growTree, predictTree, pathTree, countLeaves, treeDepth, pruneTree,
  randomForest, mseOf, shuffleColumn, permutationImportance,
} from './trees.js';
import { currentDataset, pairs, treeDesign, fmtValue, labelOf } from './datasets.js';

initChrome();

const ds = currentDataset();
const tip = tooltip();
const k$ = money1k;
const rmseOf = (predict, X, y) => Math.sqrt(mseOf(predict, X, y));

/* One-feature data (price against age) for the single-tree figures, and the
   full feature set for the forest and importance figures. */
const AGE_TRAIN = pairs(ds, 'curved', 21, 40);
const AGE_TEST = pairs(ds, 'curved', 22, 40);
const Xtr = AGE_TRAIN.xs.map((x) => [x]);
const Xte = AGE_TEST.xs.map((x) => [x]);
const AGE_FEATURE = [{ label: 'Age', unit: 'years', fmt: (v) => v.toFixed(0), type: 'numeric' }];

document.getElementById('curve-desc').textContent = ds.views.curved.describe;

/* ===================================================================
   Shared: drawing a tree
   ===================================================================
   Layout is the standard one for a small binary tree -- leaves are spread
   evenly left to right in the order they appear, and an internal node sits
   above the midpoint of its two children. Anything with more leaves than
   `detailMax` is drawn as structure only: at that size the labels overlap into
   illegibility, and a tree nobody can read is itself the point being made. */
function layoutTree(root) {
  const nodes = [];
  let leafIndex = 0;
  const walk = (n, depth, parent, side) => {
    const rec = { node: n, depth, parent, side, x: 0 };
    nodes.push(rec);
    if (n.left) {
      rec.kids = [walk(n.left, depth + 1, rec, 'left'), walk(n.right, depth + 1, rec, 'right')];
      rec.x = (rec.kids[0].x + rec.kids[1].x) / 2;
    } else {
      rec.x = leafIndex++;
    }
    return rec;
  };
  const rootRec = walk(root, 0, null, null);
  return { nodes, root: rootRec, leaves: leafIndex, depth: treeDepth(root) };
}

function drawTree(host, root, opts = {}) {
  const {
    features = AGE_FEATURE, highlight = null, detailMax = 12,
    title = null, onNode = null,
  } = opts;
  const L = layoutTree(root);
  const detailed = L.leaves <= detailMax;
  const w = 460;
  const rowH = detailed ? 62 : 40;
  const h = 34 + (L.depth + 1) * rowH;
  const svg = svgRoot(host, w, h, 'chart--tall');
  const pad = { l: 22, r: 22, t: title ? 26 : 12 };
  const sx = (x) => (L.leaves <= 1 ? w / 2
    : pad.l + (x * (w - pad.l - pad.r)) / (L.leaves - 1));
  const sy = (d) => pad.t + d * rowH;

  if (title) {
    el('text', {
      x: w / 2, y: 14, 'text-anchor': 'middle', 'font-size': 11,
      'font-weight': 620, fill: token('--text-secondary'),
    }, svg).textContent = title;
  }

  const onPath = new Set(highlight || []);
  const boxW = detailed ? 74 : 8;
  const boxH = detailed ? 30 : 8;

  /* edges first */
  for (const rec of L.nodes) {
    if (!rec.parent) continue;
    const hot = onPath.has(rec.node) && onPath.has(rec.parent.node);
    el('path', {
      d: `M${sx(rec.parent.x)} ${sy(rec.parent.depth) + boxH / 2} `
        + `V${sy(rec.depth) - rowH / 2 + boxH / 2} H${sx(rec.x)} V${sy(rec.depth) - boxH / 2}`,
      fill: 'none', stroke: token(hot ? '--accent' : '--baseline'),
      'stroke-width': hot ? 2.4 : 1.2,
    }, svg);
    if (detailed) {
      /* Offset outward from the parent: centred on the connector, these sit on
         top of the parent's split condition whenever the child is close in. */
      el('text', {
        x: sx(rec.x) + (rec.side === 'left' ? -6 : 6),
        y: sy(rec.depth) - boxH / 2 - 5,
        'text-anchor': rec.side === 'left' ? 'end' : 'start',
        'font-size': 9, 'font-weight': hot ? 700 : 500,
        fill: token(hot ? '--accent' : '--text-muted'),
      }, svg).textContent = rec.side === 'left' ? 'yes' : 'no';
    }
  }

  /* then nodes */
  for (const rec of L.nodes) {
    const n = rec.node;
    const hot = onPath.has(n);
    const isLeaf = !n.left;
    const x = sx(rec.x), y = sy(rec.depth);
    const g = el('g', {}, svg);
    el('rect', {
      x: x - boxW / 2, y: y - boxH / 2, width: boxW, height: boxH,
      rx: isLeaf ? 4 : 14,
      fill: hot ? token('--accent-wash') : token('--surface-1'),
      stroke: token(hot ? '--accent' : (isLeaf ? '--good' : '--border-strong')),
      'stroke-width': hot ? 2 : 1.2,
    }, g);
    if (detailed) {
      el('text', {
        x, y: y - 3, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 640,
        fill: token(hot ? '--accent' : '--text-primary'),
      }, g).textContent = k$(n.value);
      el('text', {
        x, y: y + 9, 'text-anchor': 'middle', 'font-size': 8.5, fill: token('--text-muted'),
      }, g).textContent = isLeaf ? `${n.n} home${n.n === 1 ? '' : 's'}` : `${n.n} homes`;
      if (!isLeaf) {
        const f = features[n.feature] || AGE_FEATURE[0];
        el('text', {
          x, y: y + boxH / 2 + 11, 'text-anchor': 'middle', 'font-size': 9,
          'font-weight': 600, fill: token(hot ? '--accent' : '--text-secondary'),
        }, g).textContent = `${f.label} ≤ ${f.type === 'categorical'
          ? f.levels[Math.floor(n.threshold)] : f.fmt(n.threshold)}`;
      }
    }
    if (onNode) onNode(g, n, rec);
  }
  return { detailed, leaves: L.leaves };
}

/** A tree's prediction as a step function across the age range, for plotting. */
function stepPath(tree, sx, sy, lo = 0, hi = 80, steps = 400) {
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const x = lo + ((hi - lo) * i) / steps;
    pts.push([sx(x), sy(predictTree(tree, [x]))]);
  }
  return linePath(pts);
}

/** The price-against-age scatter most of the single-tree figures sit on. */
function ageScatter(host, w, h, draw, opts = {}) {
  const { showTest = true, yPad = 40000 } = opts;
  const pad = { l: 62, r: 14, t: 14, b: 40 };
  const svg = svgRoot(host, w, h);
  const all = [...AGE_TRAIN.ys, ...(showTest ? AGE_TEST.ys : [])];
  const sx = scale(-2, 82, pad.l, w - pad.r);
  const sy = scale(Math.min(...all) - yPad, Math.max(...all) + yPad, h - pad.b, pad.t);
  frame(svg, w, h, pad, sx, sy, {
    xLabel: 'Age of the home (years)', yLabel: `${ds.target.label} ($)`,
    xTicks: [0, 20, 40, 60, 80], yFmt: ds.target.short,
  });
  const g = el('g', { 'clip-path': clipRect(svg, `clip-${host.id}`, pad, w, h) }, svg);
  draw(g, sx, sy, svg, pad);
  return { svg, g, sx, sy, pad };
}

const trainDots = (g, sx, sy) => AGE_TRAIN.xs.forEach((x, i) => el('circle', {
  cx: sx(x), cy: sy(AGE_TRAIN.ys[i]), r: 4.2, fill: token('--c-train'),
  stroke: token('--surface-1'), 'stroke-width': 1,
}, g));
const testDots = (g, sx, sy) => AGE_TEST.xs.forEach((x, i) => el('circle', {
  cx: sx(x), cy: sy(AGE_TEST.ys[i]), r: 4.2, fill: 'none',
  stroke: token('--c-test'), 'stroke-width': 1.7,
}, g));

/* ===================================================================
   1. A line and a tree, side by side
   =================================================================== */
{
  const fit = ols(AGE_TRAIN.xs, AGE_TRAIN.ys);
  const tree = growTree(Xtr, AGE_TRAIN.ys, { maxDepth: 4, minLeaf: 3 });
  const linePred = (r) => fit.b0 + fit.b1 * r[0];
  const treePred = (r) => predictTree(tree, r);
  const stats = [
    ['Straight line', linePred, '--c-fit'],
    ['Tree, 4 deep', treePred, '--series-5'],
  ];
  document.getElementById('vs-stats').innerHTML = stats.map(([n, p]) => `
    <div class="stat"><div class="stat__value" style="font-size:1.3rem">${money(rmseOf(p, Xte, AGE_TEST.ys))}</div>
      <div class="stat__label"><b>${n}</b> · test RMSE<br><span class="muted">training RMSE ${money(rmseOf(p, Xtr, AGE_TRAIN.ys))}</span></div></div>`).join('')
    + `<div class="stat stat--muted"><div class="stat__value" style="font-size:1.3rem">${fmt(metrics(AGE_TRAIN.ys, AGE_TRAIN.xs.map((x) => linePred([x]))).r2, 2)}</div>
      <div class="stat__label"><b>R²</b> of the line<br><span class="muted">on the training homes</span></div></div>`;
  document.getElementById('vs-note').innerHTML =
    `The line is not badly fitted — it is the best straight line there is. It is the wrong `
    + `shape. Price falls with age, flattens, and turns back up, and one slope has to average `
    + `those together into something close to flat, which is why its R² is only `
    + `${fmt(metrics(AGE_TRAIN.ys, AGE_TRAIN.xs.map((x) => linePred([x]))).r2, 2)}. `
    + `The tree makes no assumption about shape at all: it cuts the age range into `
    + `${countLeaves(tree)} pieces and predicts the average within each. That is why it follows `
    + `the turn, and it is also the entire trick.`;

  responsive(document.getElementById('vs-line'), () => {
    ageScatter(document.getElementById('vs-line'), 440, 330, (g, sx, sy) => {
      trainDots(g, sx, sy);
      el('path', {
        class: 'series-line', stroke: token('--c-fit'), 'stroke-width': 2.5,
        d: linePath([[sx(-2), sy(fit.b0 + fit.b1 * -2)], [sx(82), sy(fit.b0 + fit.b1 * 82)]]),
      }, g);
    });
  });
  responsive(document.getElementById('vs-tree'), () => {
    ageScatter(document.getElementById('vs-tree'), 440, 330, (g, sx, sy) => {
      trainDots(g, sx, sy);
      el('path', {
        class: 'series-line', d: stepPath(tree, sx, sy), stroke: token('--series-5'), 'stroke-width': 2.5,
      }, g);
    });
  });
}

/* ===================================================================
   2. Walking one home to its leaf
   =================================================================== */
{
  const tree = growTree(Xtr, AGE_TRAIN.ys, { maxDepth: 3, minLeaf: 3 });
  const sel = document.getElementById('walk-home');
  const host = document.getElementById('walk-tree');
  /* a spread of ages, so the menu covers the whole tree rather than one branch */
  const picks = [0, 6, 13, 19, 26, 32, 39].map((i) => Math.min(i, AGE_TRAIN.xs.length - 1));
  sel.innerHTML = picks.map((i) =>
    `<option value="${i}">${AGE_TRAIN.xs[i].toFixed(0)} years old · sold for ${money(AGE_TRAIN.ys[i])}</option>`).join('');

  const update = () => {
    const i = +sel.value;
    const row = [AGE_TRAIN.xs[i]];
    const path = pathTree(tree, row);
    const leaf = path[path.length - 1];
    drawTree(host, tree, { highlight: path });
    document.getElementById('walk-steps').innerHTML = path.map((n, s) => {
      if (!n.left) {
        return `<div class="small" style="margin-top:0.5rem"><span class="steplabel">${s + 1}</span>
          <b>Leaf node.</b> Predict ${money(n.value)} — the average of the ${n.n} training
          home${n.n === 1 ? '' : 's'} that ended up here.</div>`;
      }
      const yes = row[n.feature] <= n.threshold;
      return `<div class="small" style="margin-top:0.5rem"><span class="steplabel">${s + 1}</span>
        ${s === 0 ? '<b>Root node.</b> ' : '<b>Decision node.</b> '}
        Is age ≤ ${n.threshold.toFixed(0)}? This home is ${row[0].toFixed(0)},
        so <b>${yes ? 'yes' : 'no'}</b> — go ${yes ? 'left' : 'right'}.</div>`;
    }).join('');
    document.getElementById('walk-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.3rem">${money(leaf.value)}</div>
        <div class="stat__label">Predicted<br><span class="muted">actually sold for ${money(AGE_TRAIN.ys[i])}, so off by ${money(Math.abs(AGE_TRAIN.ys[i] - leaf.value))}</span></div></div>`;
  };
  sel.addEventListener('change', update);
  responsive(host, update);
}

/* ===================================================================
   3. The greedy search for one split
   =================================================================== */
{
  const host = document.getElementById('split-chart');
  const sseHost = document.getElementById('split-sse');
  const atEl = document.getElementById('split-at');
  const outEl = document.getElementById('split-out');

  /* total SSE of the two regions, for every place the split could go */
  const regions = (t) => {
    const L = [], R = [];
    AGE_TRAIN.xs.forEach((x, i) => (x <= t ? L : R).push(AGE_TRAIN.ys[i]));
    return [L, R];
  };
  const sseAt = (t) => {
    const [L, R] = regions(t);
    if (!L.length || !R.length) return NaN;
    const s = (a) => { const m = mean(a); return a.reduce((acc, v) => acc + (v - m) ** 2, 0); };
    return s(L) + s(R);
  };
  const CANDIDATES = [];
  for (let t = 2; t <= 78; t += 0.5) {
    const v = sseAt(t);
    if (Number.isFinite(v)) CANDIDATES.push({ t, sse: v });
  }
  const BEST = CANDIDATES.reduce((a, b) => (b.sse < a.sse ? b : a));
  const ROOT_SSE = (() => {
    const m = mean(AGE_TRAIN.ys);
    return AGE_TRAIN.ys.reduce((acc, v) => acc + (v - m) ** 2, 0);
  })();

  let draw, drawSse;
  const update = () => {
    const t = +atEl.value;
    outEl.textContent = `${t} years`;
    const [L, R] = regions(t);
    const cur = sseAt(t);
    /* Both candidate positions and the slider step in halves, so this is exact
       rather than a tolerance -- and it has to be, because the minimum is sharp
       enough that half a year either side is a visibly different number. */
    const atBest = t === BEST.t;
    document.getElementById('split-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${money(mean(L))}</div>
        <div class="stat__label"><em>c</em><sub>1</sub> · average of the ${L.length} younger homes</div></div>
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${money(mean(R))}</div>
        <div class="stat__label"><em>c</em><sub>2</sub> · average of the ${R.length} older homes</div></div>
      <div class="stat ${atBest ? '' : 'stat--muted'}">
        <div class="stat__value" style="font-size:1.25rem">${((1 - cur / ROOT_SSE) * 100).toFixed(1)}%</div>
        <div class="stat__label">of the error removed by this one split<br>
          <span class="muted">the best split removes ${((1 - BEST.sse / ROOT_SSE) * 100).toFixed(1)}%</span></div></div>`;
    document.getElementById('split-note').innerHTML = atBest
      ? `<b>That is the split the tree takes.</b> Splitting at ${BEST.t.toFixed(1)} years removes `
        + `${((1 - BEST.sse / ROOT_SSE) * 100).toFixed(1)}% of the error that predicting one flat `
        + `average leaves behind, and no other cut point does better. Note what the search did `
        + `<em>not</em> do: it did not consider where it might want to cut next. Greedy means `
        + `this choice is made once, on its own, and stands.`
      : `Splitting here leaves <b>${(cur / BEST.sse).toFixed(2)}×</b> as much error as the best `
        + `cut, which is at <b>${BEST.t.toFixed(1)} years</b>. Every point on the curve below is one `
        + `candidate split evaluated exactly this way, and searching them all is all the fitting `
        + `procedure is.`;
    draw();
    drawSse();
  };

  draw = () => {
    const t = +atEl.value;
    const [L, R] = regions(t);
    ageScatter(host, 900, 340, (g, sx, sy) => {
      trainDots(g, sx, sy);
      el('line', {
        x1: sx(t), x2: sx(t), y1: sy.range[1], y2: sy.range[0],
        stroke: token('--series-6'), 'stroke-width': 2, 'stroke-dasharray': '5 4',
      }, g);
      for (const [xs, xe, vals, label] of [[-2, t, L, 'c₁'], [t, 82, R, 'c₂']]) {
        if (!vals.length) continue;
        const m = mean(vals);
        el('line', {
          x1: sx(xs), x2: sx(xe), y1: sy(m), y2: sy(m),
          stroke: token('--series-5'), 'stroke-width': 3,
        }, g);
        el('text', {
          x: sx((xs + xe) / 2), y: sy(m) - 8, 'text-anchor': 'middle',
          'font-size': 11, 'font-weight': 640, fill: token('--series-5'),
        }, g).textContent = `${label} = ${k$(m)}`;
        vals.forEach(() => {});
      }
      AGE_TRAIN.xs.forEach((x, i) => {
        const m = mean(x <= t ? L : R);
        el('line', {
          x1: sx(x), x2: sx(x), y1: sy(AGE_TRAIN.ys[i]), y2: sy(m),
          stroke: token('--text-muted'), 'stroke-width': 1, opacity: 0.4,
        }, g);
      });
    }, { showTest: false });
  };

  drawSse = () => {
    const t = +atEl.value;
    const w = 900, h = 190;
    const pad = { l: 62, r: 14, t: 16, b: 40 };
    const svg = svgRoot(sseHost, w, h);
    const lo = Math.min(...CANDIDATES.map((c) => c.sse));
    const hi = Math.max(...CANDIDATES.map((c) => c.sse));
    const sx = scale(-2, 82, pad.l, w - pad.r);
    const sy = scale(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.06, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Where the split goes (years)', yLabel: 'Total SSE of the two regions',
      xTicks: [0, 20, 40, 60, 80], yTicks: [], 
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'sse-clip', pad, w, h) }, svg);
    el('path', {
      class: 'series-line', stroke: token('--series-8'), 'stroke-width': 2,
      d: linePath(CANDIDATES.map((c) => [sx(c.t), sy(c.sse)])),
    }, g);
    el('circle', { cx: sx(BEST.t), cy: sy(BEST.sse), r: 5, fill: token('--good') }, g);
    el('text', {
      x: sx(BEST.t) + 9, y: sy(BEST.sse) + 4, 'text-anchor': 'start', 'font-size': 10,
      'font-weight': 640, fill: token('--good'),
    }, g).textContent = `smallest: ${BEST.t.toFixed(1)} years`;
    const cur = sseAt(t);
    el('line', {
      x1: sx(t), x2: sx(t), y1: pad.t, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 2, 'stroke-dasharray': '5 4',
    }, svg);
    if (Number.isFinite(cur)) el('circle', { cx: sx(t), cy: sy(cur), r: 4.5, fill: token('--series-6') }, g);
  };

  atEl.addEventListener('input', update);
  /* Sweeping the split across the range traces the SSE curve underneath it,
     which is the search the tree performs. */
  document.getElementById('split-best').addEventListener('click', () =>
    tweenInput(atEl, BEST.t, update, { ms: 900 }));
  responsive(host, update);
  responsive(sseHost, drawSse);
}

/* ===================================================================
   4. Depth
   =================================================================== */
const DEPTH_MAX = 8;
const DEPTH_FITS = Array.from({ length: DEPTH_MAX }, (_, i) => {
  const t = growTree(Xtr, AGE_TRAIN.ys, { maxDepth: i + 1, minLeaf: 1, minSplit: 2 });
  return {
    d: i + 1, tree: t, leaves: countLeaves(t),
    train: rmseOf((r) => predictTree(t, r), Xtr, AGE_TRAIN.ys),
    test: rmseOf((r) => predictTree(t, r), Xte, AGE_TEST.ys),
  };
});
const DEPTH_BEST = DEPTH_FITS.reduce((a, b) => (b.test < a.test ? b : a));
{
  const chartHost = document.getElementById('depth-chart');
  const treeHost = document.getElementById('depth-tree');
  const curveHost = document.getElementById('depth-curve');
  const dEl = document.getElementById('tree-depth');
  document.getElementById('depth-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:${token('--c-train')}"></span>Training homes (${Xtr.length})</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--dot" style="background:transparent;border:2px solid ${token('--c-test')}"></span>Test homes (${Xte.length})</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--series-5')}"></span>What the tree predicts</span>`;

  let draw, drawCurve;
  const update = () => {
    const f = DEPTH_FITS[+dEl.value - 1];
    document.getElementById('depth-out').textContent = f.d;
    document.getElementById('depth-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${f.leaves}</div>
        <div class="stat__label">Leaf nodes<br><span class="muted">one prediction each</span></div></div>
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.25rem">${money(f.train)}</div>
        <div class="stat__label">Training RMSE</div></div>
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.25rem">${money(f.test)}</div>
        <div class="stat__label">Test RMSE</div></div>`;
    document.getElementById('depth-note').innerHTML =
      `Test error is lowest at depth <b>${DEPTH_BEST.d}</b> (${money(DEPTH_BEST.test)}, `
      + `${DEPTH_BEST.leaves} leaves) and rises after it. By depth ${DEPTH_MAX} the tree has `
      + `${DEPTH_FITS[DEPTH_MAX - 1].leaves} leaves for ${Xtr.length} training homes and its training `
      + `error is down to ${money(DEPTH_FITS[DEPTH_MAX - 1].train)}, while test error has gone `
      + `<b>up</b> to ${money(DEPTH_FITS[DEPTH_MAX - 1].test)}. `
      + `The staircase on the left is fitting individual homes by then — each of those narrow `
      + `steps is one or two houses' worth of noise being treated as a rule.`;
    draw();
    drawCurve();
  };

  draw = () => {
    const f = DEPTH_FITS[+dEl.value - 1];
    ageScatter(chartHost, 440, 330, (g, sx, sy) => {
      testDots(g, sx, sy);
      trainDots(g, sx, sy);
      el('path', {
        class: 'series-line', d: stepPath(f.tree, sx, sy), stroke: token('--series-5'), 'stroke-width': 2.5,
      }, g);
    });
    drawTree(treeHost, f.tree, { title: `${f.leaves} leaf nodes`, detailMax: 10 });
  };

  drawCurve = () => {
    const d = +dEl.value;
    const w = 900, h = 220;
    const pad = { l: 66, r: 84, t: 16, b: 40 };
    const svg = svgRoot(curveHost, w, h);
    const all = DEPTH_FITS.flatMap((f) => [f.train, f.test]);
    const sx = scale(1, DEPTH_MAX, pad.l, w - pad.r);
    const sy = scale(0, Math.max(...all) * 1.1, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Maximum depth', yLabel: 'RMSE',
      xTicks: DEPTH_FITS.map((f) => f.d), yFmt: k$,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'depth-clip', pad, w, h) }, svg);
    const labels = [];
    for (const [key, color, name] of [['train', '--c-train', 'Training'], ['test', '--c-test', 'Test']]) {
      const pts = DEPTH_FITS.map((f) => [sx(f.d), sy(f[key])]);
      el('path', { class: 'series-line', d: linePath(pts), stroke: token(color), 'stroke-width': 2.4 }, g);
      DEPTH_FITS.forEach((f, i) => el('circle', {
        cx: pts[i][0], cy: pts[i][1], r: f.d === d ? 5 : 3, fill: token(color),
        opacity: f.d === d ? 1 : 0.5,
      }, g));
      labels.push({ y: pts[pts.length - 1][1], text: name, color: token(color) });
    }
    directLabels(svg, labels, w - pad.r + 7, pad.t + 6, h - pad.b);
    el('line', {
      x1: sx(DEPTH_BEST.d), x2: sx(DEPTH_BEST.d), y1: pad.t, y2: h - pad.b,
      stroke: token('--good'), 'stroke-width': 1.5, 'stroke-dasharray': '4 3',
    }, svg);
    el('text', {
      class: 'annot', x: sx(DEPTH_BEST.d) + 5, y: pad.t + 12, fill: token('--good'), 'font-weight': 620,
    }, svg).textContent = 'lowest test error';
  };

  dEl.addEventListener('input', update);
  responsive(chartHost, update);
  responsive(curveHost, drawCurve);
}

/* ===================================================================
   5. Pruning
   =================================================================== */
{
  const FULL = growTree(Xtr, AGE_TRAIN.ys, { maxDepth: 12, minLeaf: 1, minSplit: 2 });
  document.getElementById('prune-full').textContent = countLeaves(FULL);
  /* α is a penalty per leaf in the units of SSE, so the slider runs over
     fractions of the root node's own SSE -- the only scale that means anything
     across datasets. */
  const ALPHAS = Array.from({ length: 101 }, (_, i) => FULL.sse * (i / 100) ** 2.2 * 0.5);
  const PRUNED = ALPHAS.map((a) => {
    const t = pruneTree(FULL, a).tree;
    return {
      a, tree: t, leaves: countLeaves(t),
      train: rmseOf((r) => predictTree(t, r), Xtr, AGE_TRAIN.ys),
      test: rmseOf((r) => predictTree(t, r), Xte, AGE_TEST.ys),
    };
  });
  let iBest = 0;
  PRUNED.forEach((p, i) => { if (p.test < PRUNED[iBest].test) iBest = i; });

  const chartHost = document.getElementById('prune-chart');
  const treeHost = document.getElementById('prune-tree');
  const aEl = document.getElementById('alpha');

  let draw;
  const update = () => {
    const p = PRUNED[+aEl.value];
    document.getElementById('alpha-out').textContent = p.a === 0 ? '0' : p.a.toExponential(1);
    document.getElementById('prune-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${p.leaves}</div>
        <div class="stat__label">Leaf nodes left<br><span class="muted">from ${PRUNED[0].leaves}</span></div></div>
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.25rem">${money(p.train)}</div>
        <div class="stat__label">Training RMSE</div></div>
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.25rem">${money(p.test)}</div>
        <div class="stat__label">Test RMSE</div></div>`;
    document.getElementById('prune-note').innerHTML =
      `The unpruned tree has ${PRUNED[0].leaves} leaves and a training RMSE of `
      + `${money(PRUNED[0].train)} — it has very nearly memorised the ${Xtr.length} training homes — `
      + `yet its test RMSE is ${money(PRUNED[0].test)}. The best α here cuts it back to `
      + `<b>${PRUNED[iBest].leaves} leaves</b> and a test RMSE of <b>${money(PRUNED[iBest].test)}</b>, `
      + `an improvement of ${money(PRUNED[0].test - PRUNED[iBest].test)} bought entirely by `
      + `<em>deleting</em> parts of the model. Training error gets worse the whole way: `
      + `${money(PRUNED[0].train)} to ${money(PRUNED[iBest].train)}. That is the bias pruning `
      + `introduces, and it is the price being paid on purpose.`;
    draw();
  };

  draw = () => {
    const p = PRUNED[+aEl.value];
    ageScatter(chartHost, 440, 330, (g, sx, sy) => {
      testDots(g, sx, sy);
      trainDots(g, sx, sy);
      el('path', {
        class: 'series-line', d: stepPath(p.tree, sx, sy), stroke: token('--series-5'), 'stroke-width': 2.5,
      }, g);
    });
    drawTree(treeHost, p.tree, { title: `${p.leaves} leaf nodes`, detailMax: 10 });
  };

  aEl.addEventListener('input', update);
  document.getElementById('alpha-best').addEventListener('click', () =>
    tweenInput(aEl, iBest, update, { ms: 900 }));
  responsive(chartHost, update);
}

/* ===================================================================
   6. All four knobs
   =================================================================== */
{
  const chartHost = document.getElementById('h-chart');
  const treeHost = document.getElementById('h-tree');
  const els = {
    depth: document.getElementById('h-depth'),
    split: document.getElementById('h-split'),
    leaf: document.getElementById('h-leaf'),
  };

  let draw;
  const update = () => {
    const maxDepth = +els.depth.value, minSplit = +els.split.value, minLeaf = +els.leaf.value;
    document.getElementById('h-depth-out').textContent = maxDepth;
    document.getElementById('h-split-out').textContent = minSplit;
    document.getElementById('h-leaf-out').textContent = minLeaf;
    const t = growTree(Xtr, AGE_TRAIN.ys, { maxDepth, minSplit, minLeaf });
    const train = rmseOf((r) => predictTree(t, r), Xtr, AGE_TRAIN.ys);
    const test = rmseOf((r) => predictTree(t, r), Xte, AGE_TEST.ys);
    document.getElementById('h-stats').innerHTML = `
      <div class="stat"><div class="stat__value" style="font-size:1.25rem">${countLeaves(t)}</div>
        <div class="stat__label">Leaf nodes · depth ${treeDepth(t)}</div></div>
      <div class="stat stat--train"><div class="stat__value" style="font-size:1.25rem">${money(train)}</div>
        <div class="stat__label">Training RMSE</div></div>
      <div class="stat stat--test"><div class="stat__value" style="font-size:1.25rem">${money(test)}</div>
        <div class="stat__label">Test RMSE</div></div>`;
    const defaults = maxDepth === 10 && minSplit === 2 && minLeaf === 1;
    document.getElementById('h-note').innerHTML = defaults
      ? `<b>These are the Python defaults</b>, with no depth limit — a tree left entirely to its `
        + `own devices. It has ${countLeaves(t)} leaves for ${Xtr.length} homes and a training RMSE of `
        + `${money(train)}. Raise any of the three sliders and training error gets worse while test `
        + `error, for a while, gets better.`
      : `Each slider stops the tree from a different direction: depth caps how many questions can be `
        + `asked in a row, minimum split stops a node too small to be worth dividing, and minimum leaf `
        + `refuses to create a region with too few homes in it. `
        + `${test < 40000 ? 'This combination is doing better on unseen homes than the defaults do.' : ''}`;
    draw(t);
  };

  draw = (t) => {
    ageScatter(chartHost, 440, 330, (g, sx, sy) => {
      testDots(g, sx, sy);
      trainDots(g, sx, sy);
      el('path', {
        class: 'series-line', d: stepPath(t, sx, sy), stroke: token('--series-5'), 'stroke-width': 2.5,
      }, g);
    });
    drawTree(treeHost, t, { title: `${countLeaves(t)} leaf nodes`, detailMax: 10 });
  };

  for (const e of Object.values(els)) e.addEventListener('input', update);
  document.getElementById('h-defaults').addEventListener('click', () => tweenInputs([
    { el: els.depth, to: 10 }, { el: els.split, to: 2 }, { el: els.leaf, to: 1 },
  ], update));
  responsive(chartHost, update);
}

/* ===================================================================
   7. High variance, shown
   =================================================================== */
{
  const host = document.getElementById('var-chart');
  const N = 30;
  document.getElementById('var-n').textContent = N;
  let seed = 400;
  let mode = 'all';
  let SAMPLES = [];

  const redraw = () => {
    SAMPLES = Array.from({ length: 8 }, (_, i) => {
      const s = pairs(ds, 'curved', seed + i * 13, N);
      return growTree(s.xs.map((x) => [x]), s.ys, { maxDepth: 4, minLeaf: 2 });
    });
  };
  redraw();

  let draw;
  const update = () => {
    document.querySelectorAll('#var-mode button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.m === mode)));
    /* how far apart the eight trees are, in dollars, averaged over the range */
    const grid = Array.from({ length: 81 }, (_, i) => i);
    const spread = mean(grid.map((x) => {
      const p = SAMPLES.map((t) => predictTree(t, [x]));
      const m = mean(p);
      return Math.sqrt(mean(p.map((v) => (v - m) ** 2)));
    }));
    document.getElementById('var-note').innerHTML =
      `At a typical age the eight trees disagree with each other by about `
      + `<b>${money(spread)}</b>, and every one of them was fitted to ${N} homes drawn from the `
      + `same relationship. That disagreement is variance, and it is error: a home can only have `
      + `one price, so at most one of these eight trees is close to right about it. `
      + `Switch to their average and the wobble largely cancels. `
      + `<b>That average is what a random forest is.</b>`;
    draw();
  };

  draw = () => {
    ageScatter(host, 900, 340, (g, sx, sy) => {
      const grid = [];
      for (let i = 0; i <= 320; i++) grid.push(-2 + (84 * i) / 320);
      if (mode === 'all') {
        SAMPLES.forEach((t, i) => el('path', {
          class: 'series-line', d: linePath(grid.map((x) => [sx(x), sy(predictTree(t, [x]))])),
          stroke: token('--series-5'), 'stroke-width': 1.6, opacity: 0.55,
        }, g));
      } else {
        SAMPLES.forEach((t) => el('path', {
          class: 'series-line', d: linePath(grid.map((x) => [sx(x), sy(predictTree(t, [x]))])),
          stroke: token('--baseline'), 'stroke-width': 1, opacity: 0.35,
        }, g));
        el('path', {
          class: 'series-line',
          d: linePath(grid.map((x) => [sx(x), sy(mean(SAMPLES.map((t) => predictTree(t, [x]))))])),
          stroke: token('--series-6'), 'stroke-width': 3.2,
        }, g);
      }
    }, { showTest: false });
  };

  document.getElementById('var-redraw').addEventListener('click', () => {
    seed += 113; redraw(); update();
  });
  document.querySelectorAll('#var-mode button').forEach((b) =>
    b.addEventListener('click', () => { mode = b.dataset.m; update(); }));
  responsive(host, update);
}

/* ===================================================================
   8-10. The full feature set: forests and variable importance
   =================================================================== */
const RF_TRAIN_N = 300, RF_TEST_N = 120, RF_MAX_TREES = 100;
const FULL_DATA = (() => {
  const rows = ds.sample(5, RF_TRAIN_N + RF_TEST_N);
  const d = treeDesign(ds, rows);
  return {
    ...d,
    rows,
    Xtr: d.X.slice(0, RF_TRAIN_N), ytr: d.y.slice(0, RF_TRAIN_N),
    Xte: d.X.slice(RF_TRAIN_N), yte: d.y.slice(RF_TRAIN_N),
    teRows: rows.slice(RF_TRAIN_N),
  };
})();
const P = FULL_DATA.features.length;
/* The single tree and every tree in the forest are grown identically -- fully,
   with no depth limit and no minimum leaf. The only difference between the two
   models is the random feature subset at each split and the resampled rows, so
   the comparison below isolates exactly that. */
const TREE_OPTS = { maxDepth: 30, minLeaf: 1, minSplit: 2 };
const SINGLE_TREE = growTree(FULL_DATA.Xtr, FULL_DATA.ytr, TREE_OPTS);
const SINGLE = {
  train: rmseOf((r) => predictTree(SINGLE_TREE, r), FULL_DATA.Xtr, FULL_DATA.ytr),
  test: rmseOf((r) => predictTree(SINGLE_TREE, r), FULL_DATA.Xte, FULL_DATA.yte),
};

/* Forests are memoized per m: changing the number of trees then only slices an
   array the page already has, which keeps that slider instant. */
const forestCache = new Map();
function forestFor(mtry) {
  if (!forestCache.has(mtry)) {
    const rf = randomForest(FULL_DATA.Xtr, FULL_DATA.ytr, {
      ...TREE_OPTS, nTrees: RF_MAX_TREES, mtry, rand: rng(2),
    });
    /* cumulative test predictions, so any prefix of the forest is free to score */
    const cum = { tr: null, te: null, curve: [] };
    const accTr = new Float64Array(FULL_DATA.Xtr.length);
    const accTe = new Float64Array(FULL_DATA.Xte.length);
    rf.trees.forEach((t, i) => {
      FULL_DATA.Xtr.forEach((r, j) => { accTr[j] += predictTree(t, r); });
      FULL_DATA.Xte.forEach((r, j) => { accTe[j] += predictTree(t, r); });
      const n = i + 1;
      const rm = (acc, X, y) => Math.sqrt(mean(y.map((v, j) => (v - acc[j] / n) ** 2)));
      cum.curve.push({
        n, train: rm(accTr, FULL_DATA.Xtr, FULL_DATA.ytr), test: rm(accTe, FULL_DATA.Xte, FULL_DATA.yte),
      });
    });
    forestCache.set(mtry, { rf, curve: cum.curve });
  }
  return forestCache.get(mtry);
}

const SQRT_P = Math.max(1, Math.round(Math.sqrt(P)));
document.getElementById('rf-p').textContent = P;
document.getElementById('rf-n').textContent = RF_TRAIN_N;
document.getElementById('rf-t').textContent = RF_TEST_N;
document.getElementById('rf-mtry').value = SQRT_P;

{
  const host = document.getElementById('rf-chart');
  const nEl = document.getElementById('rf-n-trees');
  const mEl = document.getElementById('rf-mtry');
  const busy = document.getElementById('rf-busy');
  document.getElementById('rf-legend').innerHTML = `
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-test')}"></span>Forest, on held-out homes</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--c-train')}"></span>Forest, on its training homes</span>
    <span class="legend__item"><span class="legend__swatch legend__swatch--line" style="background:${token('--text-muted')}"></span>One unpruned tree, on held-out homes</span>`;

  let draw;
  const update = () => {
    const m = +mEl.value, n = +nEl.value;
    document.getElementById('rf-mtry-out').textContent = m === SQRT_P ? `${m}  (√${P} ≈ ${m})` : m;
    document.getElementById('rf-n-trees-out').textContent = n;
    busy.textContent = forestCache.has(m) ? '' : 'growing the forest…';
    const go = () => {
      const F = forestFor(m);
      busy.textContent = '';
      const at = F.curve[n - 1];
      document.getElementById('rf-stats').innerHTML = `
        <div class="stat stat--test"><div class="stat__value" style="font-size:1.3rem">${money(at.test)}</div>
          <div class="stat__label">Forest · test RMSE<br><span class="muted">${n} tree${n === 1 ? '' : 's'}, m = ${m}</span></div></div>
        <div class="stat stat--train"><div class="stat__value" style="font-size:1.3rem">${money(at.train)}</div>
          <div class="stat__label">Forest · training RMSE</div></div>
        <div class="stat stat--muted"><div class="stat__value" style="font-size:1.3rem">${money(SINGLE.test)}</div>
          <div class="stat__label">One unpruned tree · test RMSE<br><span class="muted">training RMSE ${money(SINGLE.train)}</span></div></div>`;
      const best = F.curve[F.curve.length - 1];
      document.getElementById('rf-note').innerHTML =
        `A single unpruned tree memorises its training homes — ${money(SINGLE.train)} training error, `
        + `${money(SINGLE.test)} on homes it has not seen. The forest never gets near zero on its `
        + `training homes and is better on the ones that count. `
        + `${at.test < SINGLE.test
          ? `At ${n} tree${n === 1 ? '' : 's'} it is <b>${money(SINGLE.test - at.test)}</b> better than the single tree.`
          : `At ${n} tree${n === 1 ? '' : 's'} it is not yet better than the single tree — one tree is not a forest, and the averaging is the whole point.`}`
        + `<br><br>Most of the gain arrives in the first twenty trees or so and the curve flattens after that. `
        + `Adding more trees does not make a forest overfit; it just stops helping. `
        + `<b>The <em>m</em> slider is the one that matters.</b> At m = ${P} every tree sees every feature, `
        + `so they are all nearly the same tree and averaging them buys little. Lower m forces them apart. `
        + `The usual starting point is √p ≈ ${SQRT_P}.`;
      draw(F, n, m);
    };
    if (forestCache.has(m)) go(); else setTimeout(go, 16);
  };

  draw = (F, n, m) => {
    const w = 900, h = 330;
    const pad = { l: 70, r: 100, t: 16, b: 42 };
    const svg = svgRoot(host, w, h);
    const all = F.curve.flatMap((c) => [c.train, c.test]).concat([SINGLE.test, SINGLE.train]);
    const sx = scale(1, RF_MAX_TREES, pad.l, w - pad.r);
    const sy = scale(0, Math.max(...all) * 1.08, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Number of trees in the forest', yLabel: 'RMSE', yFmt: k$,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'rf-clip', pad, w, h) }, svg);
    el('line', {
      x1: pad.l, x2: w - pad.r, y1: sy(SINGLE.test), y2: sy(SINGLE.test),
      stroke: token('--text-muted'), 'stroke-width': 1.8, 'stroke-dasharray': '6 4',
    }, g);
    const labels = [{ y: sy(SINGLE.test), text: 'One tree', color: token('--text-muted') }];
    for (const [key, color, name] of [['train', '--c-train', 'Forest, training'], ['test', '--c-test', 'Forest, test']]) {
      const pts = F.curve.map((c) => [sx(c.n), sy(c[key])]);
      el('path', { class: 'series-line', d: linePath(pts), stroke: token(color), 'stroke-width': 2.2 }, g);
      labels.push({ y: pts[pts.length - 1][1], text: name, color: token(color), size: 10 });
    }
    directLabels(svg, labels, w - pad.r + 7, pad.t + 6, h - pad.b);
    el('line', {
      x1: sx(n), x2: sx(n), y1: pad.t, y2: h - pad.b,
      stroke: token('--series-6'), 'stroke-width': 1.6, 'stroke-dasharray': '4 3',
    }, svg);
  };

  nEl.addEventListener('input', update);
  mEl.addEventListener('input', update);
  responsive(host, update);

  /* the cost of the accuracy: readability */
  document.getElementById('forest-cost').innerHTML =
    `<strong>What the forest costs you.</strong> The single tree in section 5 fitted on one page `
    + `and every prediction it made could be read off as a short list of questions. This forest `
    + `is ${RF_MAX_TREES} trees, each grown to depth ${treeDepth(forestFor(SQRT_P).rf.trees[0])} or so on a different `
    + `resample of the homes, and a prediction is the average of ${RF_MAX_TREES} separate walks. `
    + `There is no diagram of it, which is why `
    + `<a href="#importance">variable importance</a> has to be measured rather than read off.`;
}

/* ===================================================================
   10. Permutation importance
   =================================================================== */
{
  const F = forestFor(SQRT_P);
  const predict = F.rf.predict;
  const feats = FULL_DATA.features;
  const BASE = rmseOf(predict, FULL_DATA.Xte, FULL_DATA.yte);
  const SHOWN = 8;                 // homes visible in the table
  const rand = rng(77);

  let shuffledCol = null;
  let current = FULL_DATA.Xte;
  const results = new Map();       // feature index -> rise in RMSE

  /* Short headers, and no address column: all eleven columns have to fit
     without a horizontal scroll, because the two that matter most -- what the
     forest predicted and what the home sold for -- are the last two. */
  const renderTable = () => {
    const hot = (j) => (shuffledCol === j ? ` style="color:${token('--series-6')};font-weight:640"` : '');
    const head = `<thead><tr>${feats.map((f, j) => `<th${hot(j)}>${f.short || f.label}</th>`).join('')}
      <th>Forest says</th><th>Sold for</th><th>Off by</th></tr></thead>`;
    const body = current.slice(0, SHOWN).map((row, i) => {
      const p = predict(row);
      const err = Math.abs(p - FULL_DATA.yte[i]);
      return `<tr>
        ${feats.map((f, j) => `<td${hot(j)}>${fmtValue(f, row[j])}</td>`).join('')}
        <td class="num">${money(p)}</td>
        <td class="num">${money(FULL_DATA.yte[i])}</td>
        <td class="num" style="color:${err > BASE ? token('--critical') : token('--good')}">${money(err)}</td>
      </tr>`;
    }).join('');
    document.getElementById('pi-table').innerHTML = head + `<tbody>${body}</tbody>`
      + `<tfoot><tr><td colspan="${feats.length}" class="muted small" style="text-align:left">`
      + `${SHOWN} of the ${RF_TEST_N} test homes</td>`
      + `<td colspan="3" class="muted small">typical error over all ${RF_TEST_N}: `
      + `<b>${money(rmseOf(predict, current, FULL_DATA.yte))}</b></td></tr></tfoot>`;
  };

  const renderStats = () => {
    const now = rmseOf(predict, current, FULL_DATA.yte);
    const rise = now - BASE;
    document.getElementById('pi-stats').innerHTML = `
      <div class="stat ${shuffledCol === null ? '' : (rise > BASE * 0.08 ? 'stat--test' : 'stat--muted')}">
        <div class="stat__value">${money(now)}</div>
        <div class="stat__label">Typical error on the test homes<br>
          <span class="muted">${shuffledCol === null ? `baseline, nothing shuffled`
            : `${rise > 500 ? `${money(rise)} worse` : 'essentially unchanged'} after shuffling ${feats[shuffledCol].label}`}</span></div></div>`;
  };

  const renderBars = () => {
    const max = Math.max(1, ...results.values());
    const order = feats.map((f, j) => j).sort((a, b) => (results.get(b) ?? -1) - (results.get(a) ?? -1));
    document.getElementById('pi-bars').innerHTML = `<div class="bars">${order.map((j) => {
      const r = results.get(j);
      const pct = r === undefined ? 0 : Math.max(1.5, (r / max) * 100);
      return `<div class="bars__row">
        <div class="bars__name">${feats[j].label}</div>
        <div class="bars__track"><div class="bars__fill ${r === undefined ? 'bars__fill--pending' : ''}" style="width:${pct}%"></div></div>
        <div class="bars__val">${r === undefined ? '<span class="muted">untested</span>' : `+${money(r)}`}</div>
      </div>`;
    }).join('')}</div>`;

    const tested = [...results.keys()];
    if (tested.length < 2) {
      document.getElementById('pi-note').innerHTML =
        `Baseline typical error is <b>${money(BASE)}</b> on ${RF_TEST_N} homes the forest has `
        + `never seen. Shuffle a column and the forest is handed the same homes with that one `
        + `fact scrambled; how much worse it does is that column's importance.`;
      return;
    }
    const top = tested.reduce((a, b) => (results.get(b) > results.get(a) ? b : a));
    const noiseTested = tested.filter((j) => feats[j].noise);
    document.getElementById('pi-note').innerHTML =
      `<b>${feats[top].label}</b> is the most important predictor tested so far: scrambling it `
      + `costs ${money(results.get(top))} of accuracy. `
      + (noiseTested.length
        ? `Scrambling ${noiseTested.map((j) => feats[j].label.toLowerCase()).join(' or ')} `
          + `costs ${noiseTested.map((j) => money(results.get(j))).join(' and ')} — `
          + `next to nothing, which is the correct answer, because ${noiseTested.length > 1 ? 'those columns are' : 'that column is'} noise.`
        : `Try one of the columns at the bottom of the list next.`);
  };

  const doShuffle = (j) => {
    shuffledCol = j;
    current = shuffleColumn(FULL_DATA.Xte, j, rand);
    const rise = rmseOf(predict, current, FULL_DATA.yte) - BASE;
    results.set(j, Math.max(0, rise));
    renderAll();
  };

  const renderButtons = () => {
    document.getElementById('pi-buttons').innerHTML = feats.map((f, j) =>
      `<button class="btn btn--small" data-j="${j}"${shuffledCol === j ? ' style="border-color:var(--series-6);color:var(--series-6)"' : ''}>${f.label}</button>`).join('');
    document.querySelectorAll('#pi-buttons button').forEach((b) =>
      b.addEventListener('click', () => doShuffle(+b.dataset.j)));
  };

  const renderAll = () => { renderTable(); renderStats(); renderBars(); renderButtons(); };

  document.getElementById('pi-all').addEventListener('click', () => {
    /* the real procedure: average several shuffles per column, so a single
       lucky permutation cannot decide a feature's rank */
    const { rise } = permutationImportance(predict, FULL_DATA.Xte, FULL_DATA.yte, rng(9), 5);
    rise.forEach((r, j) => results.set(j, Math.max(0, Math.sqrt(BASE ** 2 + r) - BASE)));
    shuffledCol = null;
    current = FULL_DATA.Xte;
    renderAll();
    const noise = feats.map((f, j) => (f.noise ? j : -1)).filter((j) => j >= 0);
    const real = feats.map((f, j) => (f.noise ? -1 : j)).filter((j) => j >= 0);
    document.getElementById('pi-note').innerHTML =
      `Every column, five shuffles each, averaged — which is how it is done in practice, `
      + `because one permutation of one column is a single noisy measurement. `
      + `The ${real.length} predictors that genuinely set the price separate cleanly from the `
      + `${noise.length} that do not: the weakest real one costs `
      + `${money(Math.min(...real.map((j) => results.get(j))))} and the worst offender among the `
      + `noise columns costs ${money(Math.max(...noise.map((j) => results.get(j))))}.`;
  });
  document.getElementById('pi-reset').addEventListener('click', () => {
    shuffledCol = null; current = FULL_DATA.Xte; results.clear(); renderAll();
  });

  document.getElementById('pi-keypoint').innerHTML =
    `<strong>Notice what permutation importance does not need.</strong> It never looks inside the `
    + `model. It does not care that this one is a forest, and it would give an answer for a linear `
    + `regression, a single tree, or anything else that takes the same columns and returns a number. `
    + `All it uses is the model's predictions before and after one column is scrambled — which is `
    + `why it stays available exactly when the model has become too complicated to read.`;

  renderAll();
}
