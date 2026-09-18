import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, mean } from '../assets/site.js';
import {
  growTree, predictTree, pathTree, countLeaves, treeDepth, treeSSE,
  pruneTree, randomForest, mseOf, shuffleColumn, permutationImportance,
} from '../assets/trees.js';
import { currentDataset, pairs, treeDesign } from '../assets/datasets.js';

const ds = currentDataset();
const curvedSample = (seed, n) => pairs(ds, 'curved', seed, n);
const houseData = (seed = 5, n = 240) => treeDesign(ds, ds.sample(seed, n));

test('a depth-1 tree on a step function finds the step', () => {
  const X = [], y = [];
  for (let i = 0; i < 20; i++) { X.push([i / 20]); y.push(i < 10 ? 0 : 1); }
  const t = growTree(X, y, { maxDepth: 1 });
  assert.equal(t.feature, 0);
  assert.ok(Math.abs(t.threshold - 0.475) < 1e-9, `threshold ${t.threshold}`);
  assert.equal(predictTree(t, [0.1]), 0);
  assert.equal(predictTree(t, [0.9]), 1);
  assert.equal(countLeaves(t), 2);
  assert.equal(treeDepth(t), 1);
  assert.ok(treeSSE(t) < 1e-12);
});

test('deeper trees never have more training SSE, and respect maxDepth and minLeaf', () => {
  const { xs, ys } = curvedSample(21, 40);
  const X = xs.map((x) => [x]);
  let prev = Infinity;
  for (let d = 1; d <= 6; d++) {
    const t = growTree(X, ys, { maxDepth: d, minLeaf: 1 });
    const s = treeSSE(t);
    assert.ok(s <= prev + 1e-12);
    assert.ok(treeDepth(t) <= d);
    prev = s;
  }
  const t = growTree(X, ys, { maxDepth: 8, minLeaf: 5 });
  const leaves = [];
  const walk = (n) => (n.left ? (walk(n.left), walk(n.right)) : leaves.push(n.n));
  walk(t);
  assert.ok(leaves.every((n) => n >= 5), `smallest leaf ${Math.min(...leaves)}`);
});

test('minSplit stops nodes that are too small from splitting', () => {
  const { xs, ys } = curvedSample(21, 40);
  const X = xs.map((x) => [x]);
  const t = growTree(X, ys, { maxDepth: 8, minSplit: 12 });
  const walk = (n) => { if (n.left) { assert.ok(n.n >= 12); walk(n.left); walk(n.right); } };
  walk(t);
});

test('pathTree ends at the leaf that predictTree returns', () => {
  const { xs, ys } = curvedSample(21, 40);
  const X = xs.map((x) => [x]);
  const t = growTree(X, ys, { maxDepth: 4 });
  const p = pathTree(t, [0.42]);
  assert.equal(p[0], t);
  assert.equal(p[p.length - 1].value, predictTree(t, [0.42]));
  assert.ok(!p[p.length - 1].left);
});

/* α is a penalty per leaf measured in the same units as SSE, so on a target in
   dollars its useful range is set by the data. Every α below is a fraction of
   the root node's own SSE, which makes these assertions scale-free. */
test('pruning: α = 0 leaves the tree alone, a huge α collapses it to the root', () => {
  const { xs, ys } = curvedSample(21, 40);
  const X = xs.map((x) => [x]);
  const t = growTree(X, ys, { maxDepth: 6 });
  const p0 = pruneTree(t, 0).tree;
  assert.equal(countLeaves(p0), countLeaves(t));
  const pBig = pruneTree(t, t.sse * 10).tree;
  assert.equal(countLeaves(pBig), 1);
  assert.ok(Math.abs(pBig.value - mean(ys)) < 1e-6 * Math.abs(mean(ys)));
  // the number of leaves never goes up as alpha grows
  let prev = Infinity;
  for (const f of [0, 0.002, 0.01, 0.05, 0.2, 1]) {
    const k = countLeaves(pruneTree(t, t.sse * f).tree);
    assert.ok(k <= prev, `α = ${f}·SSE gave ${k} leaves after ${prev}`);
    prev = k;
  }
  assert.ok(prev === 1);
});

test('pruned subtree minimises SSE + α|T| against every prefix of leaves it could keep', () => {
  const { xs, ys } = curvedSample(21, 40);
  const X = xs.map((x) => [x]);
  const t = growTree(X, ys, { maxDepth: 3 });
  const alpha = t.sse * 0.02;
  const { tree, cost } = pruneTree(t, alpha);
  const rel = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
  assert.ok(rel(cost, treeSSE(tree) + alpha * countLeaves(tree)));
  // brute force: enumerate all pruned subtrees of a depth-3 tree
  const subtrees = (n) => {
    const leaf = { n: n.n, value: n.value, sse: n.sse, depth: n.depth };
    if (!n.left) return [leaf];
    const out = [leaf];
    for (const l of subtrees(n.left)) for (const r of subtrees(n.right)) out.push({ ...n, left: l, right: r });
    return out;
  };
  const best = Math.min(...subtrees(t).map((s) => treeSSE(s) + alpha * countLeaves(s)));
  assert.ok(rel(best, cost), `${best} vs ${cost}`);
});

test('random forest averages its trees and uses mtry ≈ √p', () => {
  const { X, y } = houseData();
  const rf = randomForest(X, y, { nTrees: 10, maxDepth: 4, minLeaf: 3, rand: rng(2) });
  assert.equal(rf.mtry, 3);
  const row = X[5];
  const avg = mean(rf.trees.map((t) => predictTree(t, row)));
  assert.ok(Math.abs(avg - rf.predict(row)) < 1e-9);
  // trees differ because each split saw a different feature subset
  const roots = new Set(rf.trees.map((t) => t.feature));
  assert.ok(roots.size > 1, 'root splits should not all use the same feature');
});

test('permutation importance ranks the real signal above the noise columns', () => {
  const { X, y, features } = houseData();
  const train = X.slice(0, 160), ytr = y.slice(0, 160);
  const test_ = X.slice(160), yte = y.slice(160);
  const rf = randomForest(train, ytr, { nTrees: 60, maxDepth: 8, minLeaf: 3, rand: rng(4) });
  const { base, rise } = permutationImportance(rf.predict, test_, yte, rng(8), 5);
  assert.ok(base > 0);
  const noise = features.map((f, j) => (f.noise ? j : -1)).filter((j) => j >= 0);
  const signal = features.map((f, j) => (f.noise ? -1 : j)).filter((j) => j >= 0);
  const worstSignal = Math.max(...noise.map((j) => rise[j]));
  assert.ok(rise[0] === Math.max(...rise), 'size is the most important feature');
  for (const j of noise) {
    assert.ok(rise[j] < rise[0] / 10, `noise feature ${features[j].label} should barely matter`);
  }
  assert.ok(signal.filter((j) => rise[j] > worstSignal).length >= 3,
    'most real predictors outrank every noise column');
  const shuffled = shuffleColumn(test_, 0, rng(1));
  assert.equal(shuffled.length, test_.length);
  assert.notDeepEqual(shuffled.map((r) => r[0]), test_.map((r) => r[0]));
  assert.deepEqual(shuffled.map((r) => r[1]), test_.map((r) => r[1]));
  assert.ok(mseOf(rf.predict, test_, yte) === base);
});
