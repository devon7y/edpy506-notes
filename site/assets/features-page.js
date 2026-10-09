/* Every figure on features.html. The wrapper-method figure is the expensive
   one: every point on its curve is a cross-validated random forest. Its work
   runs through runJobs, so the page stays usable while it is computed, and
   forward selection is only run when it is asked for. */

import {
  initChrome, svgRoot, frame, scale, linePath, el, responsive, token, tooltip,
  mean, money, money1k, rng, gauss, clipRect, tweenInput, ticks,
} from './site.js';
import { randomForest, forestImportance } from './trees.js';
import { zscore, minmax, correlations, correlationFilter, rfe } from './features.js';
import { kfold, runJobs } from './tuning.js';
import { currentDataset, treeDesign, NEIGHBOURHOODS } from './datasets.js';

initChrome();

/* Look a feature up by its key and throw if it is not there. indexOf returns
   -1 for a missing name, list[-1] is undefined, and a sentence built on that
   reads perfectly well while saying something false -- which is how the first
   draft of the filter note came to call a kept feature dropped. */
const at = (keys, key) => {
  const i = keys.indexOf(key);
  if (i < 0) throw new Error(`features-page: no feature with key "${key}"`);
  return i;
};

const ds = currentDataset();
const tip = tooltip();
const k$ = money1k;

/* ===================================================================
   3. Scaling: the lecture's five ages, and the shape of a distribution
   =================================================================== */
{
  const box = document.getElementById('sc-outlier');
  const stripHost = document.getElementById('sc-strip');
  const BASE = [20, 30, 40, 50, 60];
  let ages = BASE;

  let draw;
  const update = () => {
    ages = box.checked ? [...BASE, 120] : BASE;
    const z = zscore(ages), n = minmax(ages);
    document.getElementById('sc-table').innerHTML = `
      <thead><tr><th>Age (x)</th><th>Standardised (z)</th><th>Normalised (x′)</th></tr></thead>
      <tbody>${ages.map((a, i) => `<tr${a === 120 ? ' style="background:var(--accent-wash)"' : ''}>
        <td class="num">${a}</td>
        <td class="num">${z.values[i] < 0 ? '−' : ''}${Math.abs(z.values[i]).toFixed(2)}</td>
        <td class="num">${n.values[i].toFixed(2)}</td></tr>`).join('')}</tbody>
      <tfoot><tr><td class="muted small">mean ${z.mean.toFixed(1)}, SD ${z.sd.toFixed(2)}</td>
        <td class="muted small">mean 0, SD 1</td>
        <td class="muted small">min ${n.min}, max ${n.max}</td></tr></tfoot>`;
    const z5 = zscore(BASE), n5 = minmax(BASE);
    document.getElementById('sc-note').innerHTML = box.checked
      ? `One more home, and the other five all move. The 60-year-old home was `
        + `<b>${n5.values[4].toFixed(2)}</b> after normalising and is now `
        + `<b>${n.values[4].toFixed(2)}</b>, because the maximum is now 120. After standardising it `
        + `was <b>${z5.values[4].toFixed(2)}</b> and is now <b>${z.values[4].toFixed(2)}</b>, because `
        + `the mean rose to ${z.mean.toFixed(1)} and the standard deviation to ${z.sd.toFixed(2)}. `
        + `Neither scaling knows the new home is extreme; each is simply computed from whatever `
        + `values it is given, which is why outliers are dealt with before a feature is scaled.`
      : `The lecture's example. Standardised, the ages have a mean of 0 and a standard deviation `
        + `of 1; normalised, they run from 0 to 1. The standard deviation here is the population `
        + `one, ${z.sd.toFixed(2)}, and the order and relative spacing of the homes is the same in `
        + `all three columns.`;
    draw();
  };

  draw = () => {
    const z = zscore(ages), n = minmax(ages);
    const w = 440, h = 200;
    const svg = svgRoot(stripHost, w, h);
    const rowsDef = [
      ['Age', ages, 0, 130, (v) => v],
      ['z', z.values, -2.2, 2.4, (v) => v],
      ['x′', n.values, -0.05, 1.05, (v) => v],
    ];
    rowsDef.forEach(([label, vals, lo, hi], r) => {
      const y = 36 + r * 58;
      const sx = scale(lo, hi, 60, w - 18);
      el('line', { x1: 60, x2: w - 18, y1: y, y2: y, stroke: token('--baseline') }, svg);
      el('text', { class: 'axis-label', x: 50, y: y + 4, 'text-anchor': 'end', 'font-weight': 640 }, svg).textContent = label;
      vals.forEach((v, i) => {
        const out = ages[i] === 120;
        el('circle', { cx: sx(v), cy: y, r: 6, fill: token(out ? '--series-6' : '--series-1'),
          stroke: token('--surface-1'), 'stroke-width': 1.5 }, svg);
        el('text', { class: 'tick', x: sx(v), y: y - 11, 'text-anchor': 'middle' }, svg)
          .textContent = label === 'Age' ? String(v) : v.toFixed(2);
      });
    });
  };

  box.addEventListener('change', update);
  responsive(stripHost, update);
}

{
  const host = document.getElementById('sh-chart');
  const prices = ds.sample(5, 1000).map((r) => r.price);
  const z = zscore(prices), n = minmax(prices);
  document.getElementById('sh-sub').textContent =
    `${prices.length.toLocaleString('en-CA')} homes. The same values, drawn three times, with the vertical axis the share of homes in each bin.`;
  document.getElementById('sh-note').innerHTML =
    `Raw prices run from ${money(n.min)} to ${money(n.max)}. Normalised they run from 0 to 1, and `
    + `standardised they are centred on 0 with a standard deviation of 1. The three histograms `
    + `are identical in shape, bin for bin, because both scalings only subtract one number and `
    + `divide by another.`;

  responsive(host, () => {
    const w = 900, h = 250;
    const svg = svgRoot(host, w, h);
    const panels = [
      { label: 'Raw values', sub: 'price in dollars', vals: prices, fmt: k$, color: '--series-7' },
      { label: 'Normalised', sub: 'x′ = (x − min) / (max − min)', vals: n.values, fmt: (v) => v.toFixed(2), color: '--series-8' },
      { label: 'Standardised', sub: 'z = (x − mean) / SD', vals: z.values, fmt: (v) => v.toFixed(1), color: '--series-2' },
    ];
    const BINS = 24;
    panels.forEach((P, k) => {
      const x0 = 10 + k * 300, pw = 270;
      const lo = Math.min(...P.vals), hi = Math.max(...P.vals);
      const counts = new Array(BINS).fill(0);
      P.vals.forEach((v) => { counts[Math.min(BINS - 1, Math.floor(((v - lo) / (hi - lo)) * BINS))]++; });
      const peak = Math.max(...counts);
      const sy = scale(0, peak * 1.1, h - 40, 52);
      el('text', { x: x0 + 8, y: 18, 'font-size': 13, 'font-weight': 660, fill: token(P.color) }, svg).textContent = P.label;
      el('text', { class: 'tick', x: x0 + 8, y: 34 }, svg).textContent = P.sub;
      const bw = (pw - 16) / BINS;
      counts.forEach((c, i) => el('rect', { x: x0 + 8 + i * bw + 0.8, y: sy(c), width: bw - 1.6,
        height: sy(0) - sy(c), rx: 1.5, fill: token(P.color), opacity: 0.72 }, svg));
      el('line', { class: 'axis', x1: x0 + 8, x2: x0 + pw - 8, y1: h - 40, y2: h - 40 }, svg);
      for (const f of [0, 0.5, 1]) {
        el('text', { class: 'tick', x: x0 + 8 + f * (pw - 16), y: h - 24, 'text-anchor': 'middle' }, svg)
          .textContent = P.fmt(lo + f * (hi - lo));
      }
    });
  });
}

/* ===================================================================
   3. Square root and log, applied to evenly spaced values
   =================================================================== */
{
  const host = document.getElementById('tr-chart');
  const minEl = document.getElementById('tr-min');
  const COUNT = 18;
  const ROWS = [
    { label: 'Raw data', formula: 'x', f: (x) => x, ok: () => true, color: '--series-6' },
    { label: 'Square root', formula: '√x', f: Math.sqrt, ok: (x) => x >= 0, color: '--series-1' },
    { label: 'Log', formula: 'log x', f: Math.log, ok: (x) => x > 0, color: '--series-7' },
  ];
  const values = () => Array.from({ length: COUNT }, (_, i) => +minEl.value + i);
  const num = (v, d = 2) => `${v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`;
  const pct = (v) => `${Math.round(100 * v)}%`;

  /* The gaps between the two smallest and the two largest values. Their ratio
     is unit-free, so it compares rows on different scales. It is taken over
     the values every row can transform, so all three compare the same range. */
  const common = (xs) => xs.filter((x) => ROWS.every((row) => row.ok(x)));
  const gaps = (row, xs) => {
    const v = common(xs).map(row.f);
    const first = v[1] - v[0], last = v[v.length - 1] - v[v.length - 2];
    return { first, last, ratio: last / first };
  };

  const draw = () => {
    const xs = values();
    /* On a phone the chart is drawn at the phone's width, with the row names
       above the rows rather than beside them, so its text stays legible. */
    const w = Math.round(Math.max(320, Math.min(900, host.clientWidth || 900)));
    const narrow = w < 600;
    const L = narrow ? 18 : 150, R = narrow ? 14 : 26;
    const rowY = narrow ? [48, 168, 288] : [34, 154, 274];
    const h = rowY[2] + 42;
    const svg = svgRoot(host, w, h);
    const sx = scale(xs[0], xs[COUNT - 1], L, w - R);

    /* Each transformed row is stretched so its smallest and largest values
       sit under the same two values in the raw row. The spacing in between
       is then the transformation's own. */
    const pos = ROWS.map((row) => {
      const valid = xs.filter(row.ok);
      const a = valid[0], b = valid[valid.length - 1];
      const map = scale(row.f(a), row.f(b), sx(a), sx(b));
      return { map, lo: row.f(a), hi: row.f(b), x: xs.map((x) => (row.ok(x) ? map(row.f(x)) : null)) };
    });

    const DOT = narrow ? 4.5 : 6;
    const marks = xs.map(() => ({ dots: [], links: [] }));
    const linkG = el('g', {}, svg);
    for (let r = 0; r + 1 < ROWS.length; r++) {
      xs.forEach((_, i) => {
        const x1 = pos[r].x[i], x2 = pos[r + 1].x[i];
        if (x1 == null || x2 == null) return;
        marks[i].links.push(el('line', { x1, y1: rowY[r] + 9, x2, y2: rowY[r + 1] - 9,
          stroke: token('--baseline'), 'stroke-width': 1.2 }, linkG));
      });
    }

    ROWS.forEach((row, r) => {
      const y = rowY[r], P = pos[r];
      el('line', { x1: L - 6, x2: w - R + 6, y1: y, y2: y, stroke: token('--grid'), 'stroke-width': 1 }, svg);
      if (narrow) {
        el('text', { x: 4, y: y - 20, 'font-size': 13, 'font-weight': 680, fill: token(row.color),
          stroke: token('--surface-1'), 'stroke-width': 5, 'paint-order': 'stroke' }, svg)
          .textContent = `${row.label}, ${row.formula}`;
      } else {
        el('text', { x: 12, y: y - 2, 'font-size': 14, 'font-weight': 680, fill: token(row.color) }, svg).textContent = row.label;
        el('text', { class: 'tick', x: 12, y: y + 15 }, svg).textContent = row.formula;
      }

      const tv = r === 0 ? ticks(xs[0], xs[COUNT - 1], narrow ? 4 : 8) : ticks(P.lo, P.hi, narrow ? 3 : 5);
      for (const t of tv) {
        const tx = r === 0 ? sx(t) : P.map(t);
        el('line', { x1: tx, x2: tx, y1: y + 10, y2: y + 15, stroke: token('--text-muted') }, svg);
        el('text', { class: 'tick', x: tx, y: y + 28, 'text-anchor': 'middle',
          stroke: token('--surface-1'), 'stroke-width': 5, 'paint-order': 'stroke' }, svg)
          .textContent = Number.isInteger(t) ? num(t, 0) : num(t, 1);
      }

      const bad = xs.filter((x) => !row.ok(x));
      if (bad.length) {
        for (const x of bad) {
          const cx = sx(x), d = 4.5;
          el('path', { d: `M${cx - d} ${y - d}L${cx + d} ${y + d}M${cx - d} ${y + d}L${cx + d} ${y - d}`,
            stroke: token('--text-muted'), 'stroke-width': 1.6 }, svg);
        }
        el('text', { class: 'annot', x: narrow ? w - 4 : sx(bad[0]) - 6, y: narrow ? y - 20 : y - 14,
          'text-anchor': narrow ? 'end' : 'start',
          stroke: token('--surface-1'), 'stroke-width': 5, 'paint-order': 'stroke' }, svg)
          .textContent = `not defined for ${bad.length === 1 ? num(bad[0], 0) : `${num(bad[0], 0)} to ${num(bad[bad.length - 1], 0)}`}`;
      }

      xs.forEach((x, i) => {
        if (P.x[i] == null) return;
        marks[i].dots.push(el('circle', { cx: P.x[i], cy: y, r: DOT, fill: token(row.color),
          stroke: token('--surface-1'), 'stroke-width': 1.5 }, svg));
      });
    });

    /* one hit target per value in each row, so a value can be traced through
       all three rows from wherever the pointer finds it */
    const light = (i, on) => {
      marks[i].dots.forEach((d) => { d.setAttribute('r', on ? DOT + 2.5 : DOT);
        d.setAttribute('stroke', token(on ? '--text-primary' : '--surface-1')); });
      marks[i].links.forEach((l) => { l.setAttribute('stroke', token(on ? '--text-primary' : '--baseline'));
        l.setAttribute('stroke-width', on ? 2.2 : 1.2); });
    };
    xs.forEach((x, i) => ROWS.forEach((row, r) => {
      if (pos[r].x[i] == null) return;
      const hit = el('circle', { cx: pos[r].x[i], cy: rowY[r], r: narrow ? 9 : 13, fill: 'transparent' }, svg);
      hit.addEventListener('pointerenter', (ev) => {
        light(i, true);
        tip.show(ROWS.map((q) => `${q.formula} = ${q.ok(x) ? `<b>${num(q.f(x), q === ROWS[0] ? 0 : 2)}</b>` : 'not defined'}`)
          .join('<br>'), ev.clientX, ev.clientY);
      });
      hit.addEventListener('pointerleave', () => { light(i, false); tip.hide(); });
    }));
  };

  const update = () => {
    const xs = values(), a = xs[0], b = xs[COUNT - 1];
    document.getElementById('tr-min-out').textContent = `${num(a, 0)} to ${num(b, 0)}`;
    const c = common(xs), whole = c[0] === a;
    const range = `${num(c[0], 0)} to ${num(b, 0)}`;
    document.getElementById('tr-sub').textContent =
      `${COUNT} values from ${num(a, 0)} to ${num(b, 0)}, one apart. Each transformed row is `
      + `stretched so its smallest and largest values sit under the same values in the raw row; `
      + `the numbers beneath a row are its own. The log is the natural log. Each card gives the gap `
      + `between the two largest values as a share of the gap between the two smallest`
      + `${whole ? '' : `, over ${range}, the values both transformations can take`}.`;
    const g = ROWS.map((row) => gaps(row, xs));
    document.getElementById('tr-stats').innerHTML = ROWS.map((row, r) => `
      <div class="stat"><div class="stat__value" style="font-size:1.3rem;color:var(${row.color})">${pct(g[r].ratio)}</div>
        <div class="stat__label">${row.label}, ${row.formula}<br>
        <span class="muted">top gap ${num(g[r].last)}, bottom gap ${num(g[r].first)}</span></div></div>`).join('');

    const [, sq, lg] = g;
    const nNeg = xs.filter((x) => x < 0).length;
    const lost = xs.filter((x) => !ROWS[2].ok(x)).length;
    let domain = '';
    if (nNeg) {
      domain = `The square root is defined for zero and above, so it has no value for the `
        + `${nNeg === 1 ? 'negative value' : `${nNeg} negative values`}. The log is defined only `
        + `above zero, so it loses ${lost}, zero among them. `;
    } else if (xs.includes(0)) {
      domain = `Zero has a square root, 0, but no log, so the log row has one value fewer. `;
    }
    document.getElementById('tr-note').innerHTML = domain
      + `Every raw gap is 1. ${whole ? 'After' : `Over ${range}, after`} a square root the gap `
      + `between the two largest values is <b>${num(sq.last)}</b>, ${pct(sq.ratio)} of the gap `
      + `between the two smallest; after a log it is <b>${num(lg.last)}</b>, ${pct(lg.ratio)}. `
      + (sq.ratio < 1 && lg.ratio < sq.ratio
        ? `Both push the high values together and leave the low ones spread out, and the log `
          + `pushes harder.`
        : '');
    draw();
  };

  minEl.addEventListener('input', update);
  responsive(host, update);
}

/* ===================================================================
   4. One-hot against dummy coding
   =================================================================== */
{
  const levels = NEIGHBOURHOODS.map((n) => n.name);
  /* one home from each neighbourhood, in the dataset's own order */
  const homes = ds.sample(5, 200);
  const pick = levels.map((_, k) => homes.find((r) => r.neigh === k));
  const oneHot = (k) => levels.map((_, j) => (j === k ? 1 : 0));
  const dummy = (k) => levels.slice(1).map((_, j) => (j + 1 === k ? 1 : 0));
  const short = (n) => n.split(' ').map((w) => w[0]).join('') + (n.split(' ').length === 1 ? n.slice(1, 3) : '');
  document.getElementById('enc-table').innerHTML = `
    <thead>
      <tr><th></th><th colspan="${levels.length + 1}" style="text-align:center">One-hot: a column per category</th>
        <th colspan="${levels.length - 1}" style="text-align:center">Dummy: a column per category but one</th></tr>
      <tr><th>Neighbourhood</th>${levels.map((n) => `<th title="${n}">${short(n)}</th>`).join('')}<th>Sum</th>
        ${levels.slice(1).map((n) => `<th title="${n}">${short(n)}</th>`).join('')}</tr>
    </thead>
    <tbody>${pick.map((r) => {
      const oh = oneHot(r.neigh), dm = dummy(r.neigh);
      return `<tr><td>${levels[r.neigh]}</td>
        ${oh.map((v) => `<td class="${v ? 'num' : 'zero'}">${v}</td>`).join('')}
        <td class="num" style="color:var(--critical)">${oh.reduce((a, b) => a + b, 0)}</td>
        ${dm.map((v) => `<td class="${v ? 'num' : 'zero'}">${v}</td>`).join('')}</tr>`;
    }).join('')}</tbody>`;
  document.getElementById('enc-note').innerHTML =
    `Across the one-hot columns, every row adds up to exactly 1. A linear model with an `
    + `intercept already has a column of 1s, so the one-hot columns reproduce it perfectly and one `
    + `of them carries no information the others do not: that is the <b>dummy variable trap</b>, and `
    + `it leaves the coefficients with no unique solution. Dummy coding drops one category — here `
    + `<b>${levels[0]}</b>, which becomes the baseline every other neighbourhood is compared with — `
    + `and the trap disappears. A tree never adds columns together, so the redundancy does not `
    + `hurt it, and one-hot is the simpler choice there. The column headings abbreviate the six `
    + `neighbourhoods; hover one for its name.`;
}

/* ===================================================================
   7. The correlation filter
   =================================================================== */
const HOMES = ds.sample(5, 250);
const T = treeDesign(ds, HOMES);
{
  const NUMERIC = ['size', 'beds', 'age', 'lrt', 'houseNum'];
  const idx = NUMERIC.map((k) => T.features.findIndex((f) => f.key === k));
  const names = idx.map((j) => T.names[j]);
  const isNoise = idx.map((j) => !!T.features[j].noise);
  const Xn = T.X.map((r) => idx.map((j) => r[j]));
  const heatHost = document.getElementById('f-heat');
  const minEl = document.getElementById('f-min'), pairEl = document.getElementById('f-pair');

  let draw;
  const update = () => {
    const minT = +minEl.value / 100, maxP = +pairEl.value / 100;
    minEl.nextElementSibling.textContent = `|r| ${minT.toFixed(2)}`;
    pairEl.nextElementSibling.textContent = `|r| ${maxP.toFixed(2)}`;
    const F = correlationFilter(Xn, T.y, { minTarget: minT, maxPair: maxP });
    document.getElementById('f-list').innerHTML = `<div class="bars">${names.map((n, j) => {
      const why = F.reasons[j];
      const r = F.withY[j];
      return `<div class="bars__row" style="grid-template-columns:8.5rem 1fr 5.8rem">
        <div class="bars__name" style="${F.keep[j] ? '' : 'text-decoration:line-through;color:var(--text-muted)'}">${n}</div>
        <div class="bars__track"><div class="bars__fill" style="width:${Math.abs(r) * 100}%;background:${token(F.keep[j] ? '--series-1' : '--baseline')}"></div></div>
        <div class="bars__val">${r < 0 ? '−' : ''}${Math.abs(r).toFixed(3)}</div></div>
        <div class="small muted" style="margin:-0.2rem 0 0.35rem 0">${F.keep[j] ? 'kept'
          : why === 'weak' ? 'dropped: too weakly correlated with price'
          : `dropped: too similar to ${names[+why.split(':')[1]].toLowerCase()}`}${isNoise[j] ? ' · <b>has no effect on price</b>' : ''}</div>`;
    }).join('')}</div>`;
    const kept = names.filter((_, j) => F.keep[j]);
    const noiseKept = names.filter((_, j) => F.keep[j] && isNoise[j]);
    const realDropped = names.filter((_, j) => !F.keep[j] && !isNoise[j]);
    const byR = names.map((n, j) => [n, F.withY[j]]).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const hn = at(NUMERIC, 'houseNum'), lrt = at(NUMERIC, 'lrt'), age = at(NUMERIC, 'age');
    const lower = (n) => n.charAt(0).toLowerCase() + n.slice(1);
    document.getElementById('f-note').innerHTML =
      `Correlation with price, largest first: ${byR.map(([n, r]) => `${lower(n)} ${Math.abs(r).toFixed(3)}`).join(', ')}. `
      + `<b>House number</b>, which has no effect on price at all, correlates with it `
      + `${Math.abs(F.withY[hn]) > Math.abs(F.withY[lrt]) ? 'more strongly' : 'about as strongly'} than `
      + `<b>distance to the LRT</b>, which does. Distance to the LRT moves price by a modest amount `
      + `that the far larger effect of size drowns out in a single correlation, and age moves it `
      + `along a curve, which correlation, measuring a straight line, cannot see: age correlates at `
      + `only ${Math.abs(F.withY[age]).toFixed(3)}.<br><br>`
      + `At these thresholds the filter keeps ${kept.length ? kept.map(lower).join(', ') : 'nothing'}. `
      + (noiseKept.length ? `That includes <b>${noiseKept.map(lower).join(' and ')}</b>, which has no effect on price. ` : '')
      + (realDropped.length ? `It throws away <b>${realDropped.map(lower).join(' and ')}</b>, which ${realDropped.length > 1 ? 'do' : 'does'} affect price. ` : '')
      + `A filter this simple is fast and cheap, and it judges every feature by one number.`;
    draw(F);
  };

  draw = (F) => {
    F = F || correlationFilter(Xn, T.y, { minTarget: +minEl.value / 100, maxPair: +pairEl.value / 100 });
    const w = 520, h = 400;
    const svg = svgRoot(heatHost, w, h);
    const labels = [...names.map((n) => n.replace(/ \(.*\)/, '')), 'Price'];
    const m = labels.length;
    const cell = 58, x0 = 122, y0 = 30;
    const colour = (r) => {
      const c = token(r >= 0 ? '--series-1' : '--series-8');
      return { c, a: Math.min(1, Math.abs(r)) * 0.85 + 0.04 };
    };
    for (let a = 0; a < m; a++) {
      el('text', { class: 'tick', x: x0 - 8, y: y0 + a * cell + cell / 2 + 4, 'text-anchor': 'end',
        'font-size': 11 }, svg).textContent = labels[a];
      el('text', { class: 'tick', x: x0 + a * cell + cell / 2, y: y0 - 8, 'text-anchor': 'middle',
        'font-size': 10 }, svg).textContent = labels[a].length > 9 ? `${labels[a].slice(0, 8)}…` : labels[a];
      for (let b = 0; b < m; b++) {
        const r = a === m - 1 && b === m - 1 ? 1
          : a === m - 1 ? F.withY[b] : b === m - 1 ? F.withY[a] : F.R[a][b];
        const { c, a: alpha } = colour(r);
        const rect = el('rect', { x: x0 + b * cell + 1, y: y0 + a * cell + 1, width: cell - 2, height: cell - 2,
          rx: 3, fill: c, opacity: alpha }, svg);
        el('text', { x: x0 + b * cell + cell / 2, y: y0 + a * cell + cell / 2 + 4, 'text-anchor': 'middle',
          'font-size': 11, 'font-weight': 620, fill: token(Math.abs(r) > 0.55 ? '--surface-1' : '--text-primary'),
          'pointer-events': 'none' }, svg).textContent = r.toFixed(2);
        rect.addEventListener('pointerenter', (ev) => tip.show(
          `<b>${labels[a]}</b> and <b>${labels[b]}</b><br>r = ${r.toFixed(3)}`, ev.clientX, ev.clientY));
        rect.addEventListener('pointerleave', () => tip.hide());
      }
    }
    /* the price column, where the filter's first rule is applied */
    el('rect', { x: x0 + (m - 1) * cell, y: y0, width: cell, height: m * cell, fill: 'none',
      stroke: token('--series-6'), 'stroke-width': 2, rx: 4 }, svg);
  };

  minEl.addEventListener('input', update);
  pairEl.addEventListener('input', update);
  responsive(heatHost, update);
}

/* ===================================================================
   8. Wrapper methods: RFE and forward selection, with planted variables
   =================================================================== */
/* The two planted variables, built as the lecture's example built them: one
   weakly and one moderately related to the target, both artificial. */
const PLANTED = (() => {
  const r = rng((77 * 2654435761) >>> 0);
  const mu = mean(T.y), sdY = Math.sqrt(mean(T.y.map((v) => (v - mu) ** 2)));
  const z = T.y.map((v) => (v - mu) / sdY);
  return {
    weak: z.map((v) => 0.25 * v + gauss(r)),
    moderate: z.map((v) => 0.9 * v + gauss(r)),
  };
})();
const FEATURESETS = {
  plain: {
    X: T.X, names: T.names, keys: T.features.map((f) => f.key),
    noise: T.features.map((f) => !!f.noise), planted: T.names.map(() => null),
  },
  planted: {
    X: T.X.map((r, i) => [...r, PLANTED.weak[i], PLANTED.moderate[i]]),
    names: [...T.names, 'Planted: weak link', 'Planted: moderate link'],
    keys: [...T.features.map((f) => f.key), 'plantedWeak', 'plantedModerate'],
    noise: [...T.features.map((f) => !!f.noise), false, false],
    planted: [...T.names.map(() => null), 'weak', 'moderate'],
  },
};
const RF = { nTrees: 20, maxDepth: 10, minLeaf: 2 };
const FOLDS = kfold(T.y.length, 5, rng((9 * 2654435761) >>> 0));
const fitScoreJob = (F, cols, fold) => () => {
  const held = new Set(FOLDS[fold]);
  const tr = FOLDS.flat().filter((i) => !held.has(i));
  const rf = randomForest(tr.map((i) => cols.map((c) => F.X[i][c])), tr.map((i) => T.y[i]), { ...RF, rand: rng(2) });
  return Math.sqrt(mean(FOLDS[fold].map((i) => (T.y[i] - rf.predict(cols.map((c) => F.X[i][c]))) ** 2)));
};
const summarise = (scores) => {
  const m = mean(scores);
  return { mean: m, sd: Math.sqrt(mean(scores.map((s) => (s - m) ** 2))), scores };
};

const cache = new Map();
async function compute(method, set, onProgress) {
  const key = `${method}-${set}`;
  if (cache.has(key)) return cache.get(key);
  const F = FEATURESETS[set];
  const p = F.names.length;
  let order, curve;
  if (method === 'rfe') {
    /* rank by forest importance, dropping one feature a round */
    const R = rfe(p, (cols) => forestImportance(
      randomForest(F.X.map((r) => cols.map((c) => r[c])), T.y, { ...RF, rand: rng(2) }), cols.length),
    { keep: 1, step: 1 });
    order = [...Array(p).keys()].sort((a, b) => R.ranking[a] - R.ranking[b]);
    const jobs = [];
    for (let m = 1; m <= p; m++) for (let f = 0; f < FOLDS.length; f++) jobs.push(fitScoreJob(F, order.slice(0, m), f));
    const out = await runJobs(jobs, (i, n) => onProgress(`scoring ${i} of ${n} fits`));
    curve = [];
    for (let m = 0; m < p; m++) curve.push(summarise(out.slice(m * FOLDS.length, (m + 1) * FOLDS.length)));
    const res = { order, curve, rounds: R.rounds, ranking: R.ranking };
    cache.set(key, res);
    return res;
  }
  /* forward selection: at every step, cross-validate every remaining feature
     added to the chosen set, and keep whichever scores best */
  order = []; curve = [];
  const left = new Set([...Array(p).keys()]);
  let done = 0;
  const total = (p * (p + 1)) / 2 * FOLDS.length;
  while (left.size) {
    const cands = [...left];
    const jobs = [];
    for (const c of cands) for (let f = 0; f < FOLDS.length; f++) jobs.push(fitScoreJob(F, [...order, c], f));
    const out = await runJobs(jobs, (i) => onProgress(`scoring ${done + i} of ${total} fits`));
    done += jobs.length;
    let best = -1, bestS = null;
    cands.forEach((c, k) => {
      const s = summarise(out.slice(k * FOLDS.length, (k + 1) * FOLDS.length));
      if (!bestS || s.mean < bestS.mean) { bestS = s; best = c; }
    });
    order.push(best); curve.push(bestS); left.delete(best);
  }
  const res = { order, curve };
  cache.set(key, res);
  return res;
}

{
  const host = document.getElementById('w-chart');
  const keepEl = document.getElementById('w-keep');
  const plantEl = document.getElementById('w-plant');
  const busy = document.getElementById('w-busy');
  let method = 'rfe';
  let result = null;
  const set = () => (plantEl.checked ? 'planted' : 'plain');

  document.getElementById('w-sub').textContent =
    `${T.y.length} homes. Every point on the curve is a random forest's error, cross-validated `
    + `over ${FOLDS.length} folds, using the first n features in the order the method chose them.`;

  const bestK = () => {
    if (!result) return 1;
    let b = 0;
    result.curve.forEach((c, i) => { if (c.mean < result.curve[b].mean) b = i; });
    return b + 1;
  };

  let draw;
  const render = () => {
    const F = FEATURESETS[set()];
    const p = F.names.length;
    keepEl.max = p;
    if (+keepEl.value > p) keepEl.value = p;
    const k = +keepEl.value;
    keepEl.nextElementSibling.textContent = k;
    document.querySelectorAll('#w-method button').forEach((b) =>
      b.setAttribute('aria-pressed', String(b.dataset.m === method)));
    if (!result) { draw(); return; }
    const kept = new Set(result.order.slice(0, k));
    const tag = (j) => (F.planted[j] ? `<span style="color:var(--series-6)"> · planted</span>`
      : F.noise[j] ? `<span class="muted"> · no effect on price</span>` : '');
    const listTitle = method === 'rfe'
      ? `Ranked by RFE: the last survivor first, the first casualty last`
      : 'In the order forward selection added them';
    document.getElementById('w-list').innerHTML =
      `<div class="small muted" style="margin-bottom:0.45rem">${listTitle}</div>`
      + result.order.map((j, i) => `<div class="small" style="padding:0.18rem 0.4rem;border-radius:4px;${kept.has(j)
        ? 'background:var(--accent-wash)' : 'color:var(--text-muted)'}">
        <b style="display:inline-block;width:1.4rem">${i + 1}</b>${F.names[j]}${tag(j)}</div>`).join('');
    const b = bestK();
    const cur = result.curve[k - 1], best = result.curve[b - 1];
    const noiseKept = result.order.slice(0, k).filter((j) => F.noise[j]).map((j) => F.names[j].toLowerCase());
    const plantedKept = result.order.slice(0, k).filter((j) => F.planted[j]).map((j) => F.planted[j]);
    document.getElementById('w-note').innerHTML =
      `Keeping <b>${k}</b> feature${k === 1 ? '' : 's'}: cross-validated error <b>${money(cur.mean)}</b> `
      + `± ${money(cur.sd)}. The lowest error is at <b>${b}</b> features, ${money(best.mean)}. `
      + `${noiseKept.length ? `The set kept includes ${noiseKept.join(' and ')}, which ${noiseKept.length > 1 ? 'have' : 'has'} no effect on price. ` : 'None of the columns with no effect on price are in the set kept. '}`
      + `${set() === 'planted' ? (plantedKept.length
        ? `Of the planted variables, the ${plantedKept.join(' and the ')} one ${plantedKept.length > 1 ? 'are' : 'is'} kept. `
        : 'Neither planted variable is kept. ') : ''}`
      + `The curve falls steeply while the features that carry the most information go in, then `
      + `rises gently as features that add little are added after them: past that point each `
      + `extra column gives the forest more ways to fit noise than to fit price.`;
    draw();
  };

  draw = () => {
    const F = FEATURESETS[set()];
    const p = F.names.length;
    const w = 560, h = 360;
    const pad = { l: 66, r: 16, t: 18, b: 46 };
    const svg = svgRoot(host, w, h);
    const sx = scale(0.6, p + 0.4, pad.l, w - pad.r);
    if (!result) {
      el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: token('--text-muted'),
        'font-size': 13 }, svg).textContent = 'Fitting the forests…';
      return;
    }
    const hi = Math.max(...result.curve.map((c) => c.mean + c.sd));
    const lo = Math.min(...result.curve.map((c) => c.mean - c.sd));
    const sy = scale(Math.max(0, lo - (hi - lo) * 0.08), hi * 1.04, h - pad.b, pad.t);
    frame(svg, w, h, pad, sx, sy, {
      xLabel: 'Number of features kept', yLabel: 'Cross-validated RMSE',
      xTicks: [...Array(p).keys()].map((i) => i + 1), yFmt: k$,
    });
    const g = el('g', { 'clip-path': clipRect(svg, 'w-clip', pad, w, h) }, svg);
    /* one standard deviation above and below the mean, across the folds */
    const up = result.curve.map((c, i) => [sx(i + 1), sy(c.mean + c.sd)]);
    const dn = result.curve.map((c, i) => [sx(i + 1), sy(c.mean - c.sd)]).reverse();
    el('path', { d: linePath(up) + linePath(dn).replace('M', 'L') + 'Z',
      fill: token('--series-1'), opacity: 0.16 }, g);
    const pts = result.curve.map((c, i) => [sx(i + 1), sy(c.mean)]);
    el('path', { class: 'series-line', d: linePath(pts), stroke: token('--series-1'), 'stroke-width': 2.4 }, g);
    const k = +keepEl.value, b = bestK();
    pts.forEach(([x, y], i) => {
      const j = result.order[i];
      const c = el('circle', { cx: x, cy: y, r: i + 1 === k ? 6.5 : 4,
        fill: token(F.planted[j] ? '--series-6' : F.noise[j] ? '--text-muted' : '--series-1'),
        stroke: token('--surface-1'), 'stroke-width': 1.5 }, g);
      c.addEventListener('pointerenter', (ev) => tip.show(
        `<b>${i + 1} feature${i ? 's' : ''}</b>, adding ${F.names[j]}<br>`
        + `RMSE ${money(result.curve[i].mean)} ± ${money(result.curve[i].sd)}`, ev.clientX, ev.clientY));
      c.addEventListener('pointerleave', () => tip.hide());
    });
    el('line', { x1: sx(b), x2: sx(b), y1: pad.t, y2: h - pad.b, stroke: token('--good'),
      'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
    el('text', { class: 'annot', x: sx(b) + 5, y: pad.t + 12, fill: token('--good'), 'font-weight': 620 }, svg)
      .textContent = `lowest error: ${b}`;
    el('line', { x1: sx(k), x2: sx(k), y1: pad.t, y2: h - pad.b, stroke: token('--series-6'), 'stroke-width': 2 }, svg);
  };

  const refresh = async () => {
    const want = `${method}-${set()}`;
    result = cache.get(want) || null;
    render();
    if (result) { busy.textContent = ''; return; }
    busy.textContent = method === 'forward' ? 'running forward selection…' : 'running RFE…';
    const r = await compute(method, set(), (msg) => {
      if (`${method}-${set()}` === want) busy.textContent = `${method === 'forward' ? 'forward selection' : 'RFE'}: ${msg}`;
    });
    if (`${method}-${set()}` !== want) return;      // the reader moved on meanwhile
    result = r;
    busy.textContent = '';
    keepEl.value = Math.min(+keepEl.max, bestK());
    render();
    if (method === 'rfe' && set() === 'planted') domainNote(r);
  };

  keepEl.addEventListener('input', render);
  plantEl.addEventListener('change', refresh);
  document.querySelectorAll('#w-method button').forEach((b) =>
    b.addEventListener('click', () => { method = b.dataset.m; refresh(); }));
  document.getElementById('w-best').addEventListener('click', () => {
    if (result) tweenInput(keepEl, bestK(), render);
  });
  responsive(host, () => render());
  refresh();
}

/* ===================================================================
   9. Domain knowledge, from what RFE actually did
   =================================================================== */
function domainNote(r) {
  const F = FEATURESETS.planted;
  const rank = (key) => r.order.indexOf(at(F.keys, key)) + 1;
  const hn = rank('houseNum'), garage = rank('garage'), lrt = rank('lrt');
  const mod = rank('plantedModerate'), weak = rank('plantedWeak');
  const noiseRanks = r.order.map((j, i) => (F.noise[j] ? i + 1 : null)).filter(Boolean);
  const outranked = [['garage', garage], ['distance to the LRT', lrt]].filter(([, v]) => v > hn).map(([n]) => n);
  document.getElementById('domain-keypoint').innerHTML =
    `<strong>RFE ranked house number ${hn}${hn === 1 ? 'st' : hn === 2 ? 'nd' : hn === 3 ? 'rd' : 'th'} of `
    + `${F.names.length}${outranked.length ? `, above ${outranked.join(' and ')}` : ''}.</strong> `
    + `Nothing about how a home is priced depends on its street number, and anyone who knows the `
    + `problem would say so before running anything. A feature with no plausible link to the `
    + `outcome surviving longer than one with an obvious link is a cue to look closer, not a result `
    + `to accept.`;
  document.getElementById('domain-planted').innerHTML =
    `The planted variables behave as they did in the heart-disease example. The moderate one `
    + `ranks <b>${mod}</b>${mod <= 3 ? ', among the strongest features in the data' : ''}, and the weak `
    + `one <b>${weak}</b>. The three columns with no effect on price rank ${noiseRanks.join(', ')}. `
    + `RFE was right to keep the moderate variable: its signal is real, even though the variable `
    + `was manufactured. Knowing that it was planted, and how strongly, is the domain knowledge that `
    + `lets that choice be checked rather than trusted.`;
}
