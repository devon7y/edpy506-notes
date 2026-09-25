import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, mean } from '../assets/site.js';
import { classes, undersample, oversample, smote, smoteUnder } from '../assets/imbalance.js';
import { zscore, minmax, pearson, correlationFilter, forwardSelect, rfe } from '../assets/features.js';
import { kfold, cvScore, holdout, gridPoints, randomPoints, runJobs } from '../assets/tuning.js';
import { growTree, randomForest, treeImportance, forestImportance } from '../assets/trees.js';
import { currentDataset, marketLabels, treeDesign } from '../assets/datasets.js';

const ds = currentDataset();
const count = (y, v) => y.filter((c) => c === v).length;

/* a 90/10 set with two columns on different scales */
function skewed(seed = 4, n = 200) {
  const r = rng(seed), X = [], y = [];
  for (let i = 0; i < n; i++) {
    const cls = i < n / 10 ? 1 : 0;
    X.push([cls * 3 + r(), cls * 300 + 100 * r()]);
    y.push(cls);
  }
  return { X, y };
}

/* ---------- imbalance ---------- */

test('classes names the rarer label the minority', () => {
  const c = classes([0, 0, 0, 1]);
  assert.equal(c.minority, 1);
  assert.deepEqual(c.min, [3]);
  assert.equal(classes([1, 1, 0]).minority, 0);
});

test('undersampling keeps every minority row and evens the classes', () => {
  const { X, y } = skewed();
  const u = undersample(X, y, rng(1));
  assert.equal(count(u.y, 1), 20);
  assert.equal(count(u.y, 0), 20);
  assert.equal(u.dropped, 160);
  /* every kept row is an original row, unchanged */
  u.source.forEach((i, k) => assert.deepEqual(u.X[k], X[i]));
});

test('oversampling adds only copies of minority rows', () => {
  const { X, y } = skewed();
  const o = oversample(X, y, rng(2));
  assert.equal(count(o.y, 1), count(o.y, 0));
  assert.equal(o.added, 160);
  o.source.slice(X.length).forEach((i) => assert.equal(y[i], 1));
});

test('SMOTE places each new point on the segment between a row and one of its neighbours', () => {
  const { X, y } = skewed();
  const s = smote(X, y, rng(3), { k: 5 });
  assert.equal(count(s.y, 1), count(s.y, 0));
  assert.equal(s.synthetic.length, 160);
  for (const t of s.trace) {
    assert.ok(t.neighbours.includes(t.b) && t.b !== t.a);
    assert.equal(y[t.a], 1);
    assert.equal(y[t.b], 1);
    assert.ok(t.t >= 0 && t.t < 1);
    t.point.forEach((v, j) => assert.ok(Math.abs(v - (X[t.a][j] + t.t * (X[t.b][j] - X[t.a][j]))) < 1e-9));
  }
});

test('SMOTE finds neighbours on standardised columns, not raw units', () => {
  /* In raw units the second column dominates and A's nearest neighbour is C.
     Standardised, the first column carries the same weight and B is nearer. */
  const X = [[0, 0], [0.1, 60], [5, 50], [9, 9], [9, 10], [9, 11], [9, 12]];
  const y = [1, 1, 1, 0, 0, 0, 0];
  const s = smote(X, y, rng(1), { k: 1 });
  const fromA = s.trace.find((t) => t.a === 0);
  assert.ok(fromA, 'row A was never chosen');
  assert.equal(fromA.b, 2);
});

test('the hybrid SMOTEs part of the way and undersamples the rest', () => {
  const { X, y } = skewed();
  const h = smoteUnder(X, y, rng(5), { lift: 0.5 });
  /* 20 minority against 180: SMOTE lifts the minority to half of 180, then
     undersampling brings the majority down to meet it */
  assert.equal(h.synthetic.length, 70);
  assert.equal(count(h.y, 1), 90);
  assert.equal(count(h.y, 0), 90);
});

test('marketLabels lowers the base rate without changing the homes or turning a slow sale fast', () => {
  const rows = ds.sample(5, 1600);
  const before = JSON.stringify(rows);
  const { y, share } = marketLabels(rows, 0.05);
  assert.ok(Math.abs(share - 0.05) < 0.002, `share ${share}`);
  assert.equal(JSON.stringify(rows), before);
  y.forEach((v, i) => { if (v === 1) assert.equal(rows[i].fast, 1); });
});

/* ---------- features ---------- */

test('z-scores have mean 0 and sd 1; min–max spans 0 to 1', () => {
  const a = [3, 7, 1, 9, 4];
  const z = zscore(a).values;
  assert.ok(Math.abs(mean(z)) < 1e-12);
  assert.ok(Math.abs(Math.sqrt(mean(z.map((v) => v * v))) - 1) < 1e-12);
  const m = minmax(a).values;
  assert.equal(Math.min(...m), 0);
  assert.equal(Math.max(...m), 1);
});

test('pearson is 1, -1 and 0 where it should be', () => {
  assert.ok(Math.abs(pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-12);
  assert.ok(Math.abs(pearson([1, 2, 3], [3, 2, 1]) + 1) < 1e-12);
  assert.ok(Math.abs(pearson([1, 2, 3, 4], [1, -1, -1, 1])) < 1e-12);
});

test('the correlation filter drops the weak feature and the weaker of a redundant pair', () => {
  const r = rng(8), X = [], y = [];
  for (let i = 0; i < 300; i++) {
    const a = r(), noise = r();
    X.push([a, a + 0.01 * r(), noise]);
    y.push(2 * a + 0.05 * r());
  }
  const f = correlationFilter(X, y, { minTarget: 0.2, maxPair: 0.9 });
  assert.equal(f.reasons[2], 'weak');
  assert.equal(f.keep.filter(Boolean).length, 1);
  assert.ok(f.reasons.some((s) => s && s.startsWith('redundant:')));
});

test('forward selection adds the most useful feature first', () => {
  const useful = [0.1, 0.9, 0.3];
  const f = forwardSelect(3, (cols) => cols.reduce((s, c) => s + useful[c], 0) - 0.2 * cols.length);
  assert.deepEqual(f.order, [1, 2, 0]);
  assert.equal(f.path.length, 3);
});

test('RFE drops the least important feature each round and ranks by round', () => {
  const imp = [0.5, 0.1, 0.3, 0.05];
  const e = rfe(4, (cols) => cols.map((c) => imp[c]), { keep: 1, step: 1 });
  assert.deepEqual(e.rounds.map((r) => r.dropped[0]), [3, 1, 2]);
  assert.deepEqual(e.survivors, [0]);
  assert.deepEqual(e.ranking, [1, 3, 2, 4]);
});

test('tree importance puts the noise columns last and sums to one', () => {
  const H = treeDesign(ds, ds.sample(5, 400));
  const t = growTree(H.X, H.y, { maxDepth: 6, minLeaf: 5 });
  const imp = treeImportance(t, H.X[0].length);
  assert.ok(Math.abs(imp.reduce((s, v) => s + v, 0) - 1) < 1e-9);
  const rf = randomForest(H.X, H.y, { nTrees: 20, maxDepth: 8, rand: rng(2) });
  const fi = forestImportance(rf, H.X[0].length);
  const noise = ds.features.map((f, j) => (f.noise ? j : -1)).filter((j) => j >= 0);
  const size = ds.features.findIndex((f) => /size/i.test(f.label));
  for (const j of noise) assert.ok(fi[j] < fi[size], `${ds.features[j].label} outranked size`);
});

/* ---------- tuning ---------- */

test('k folds cover every row exactly once', () => {
  const folds = kfold(23, 5, rng(1));
  assert.equal(folds.length, 5);
  const all = folds.flat().sort((a, b) => a - b);
  assert.deepEqual(all, [...Array(23).keys()]);
  folds.forEach((f) => assert.ok(f.length === 4 || f.length === 5));
});

test('cvScore never trains on the fold it tests on', () => {
  const folds = kfold(30, 5, rng(2));
  const cv = cvScore(folds, (tr, te) => {
    const held = new Set(te);
    assert.ok(tr.every((i) => !held.has(i)));
    assert.equal(tr.length + te.length, 30);
    return te.length;
  });
  assert.equal(cv.mean, 6);
  assert.equal(cv.sd, 0);
});

test('holdout splits by the fraction asked for', () => {
  const h = holdout(200, 0.8, rng(3));
  assert.equal(h.train.length, 160);
  assert.equal(h.test.length, 40);
  assert.equal(new Set([...h.train, ...h.test]).size, 200);
});

test('a grid is every combination; random points come from the same values', () => {
  const space = { a: [1, 2, 3], b: ['x', 'y'] };
  const g = gridPoints(space);
  assert.equal(g.length, 6);
  assert.equal(new Set(g.map((p) => `${p.a}${p.b}`)).size, 6);
  const r = randomPoints(space, 50, rng(4));
  assert.equal(r.length, 50);
  r.forEach((p) => { assert.ok(space.a.includes(p.a)); assert.ok(space.b.includes(p.b)); });
});

test('runJobs runs every job in order and reports finishing', async () => {
  const seen = [];
  let last = null;
  const out = await runJobs(Array.from({ length: 40 }, (_, i) => () => { seen.push(i); return i * i; }),
    (i, n) => { last = [i, n]; }, 1);
  assert.deepEqual(seen, [...Array(40).keys()]);
  assert.equal(out[7], 49);
  assert.deepEqual(last, [40, 40]);
});
