/* Tree maths for the Trees & forests page: a CART regression tree grown by
   recursive partitioning on the SSE criterion, cost-complexity pruning, a
   random forest that draws a random subset of features at every split, and
   permutation importance. Plain arrays only, so it is unit-testable from node. */

import { shuffle } from './site.js';

/* ---------- growing a tree ---------- */

function pick(arr, m, rand) {
  return shuffle(arr, rand).slice(0, m).sort((a, b) => a - b);
}

/**
 * Grow a regression tree. X is an array of rows (each an array of feature
 * values), y the numeric target.
 *
 *   maxDepth  - deepest internal node allowed below the root (root is depth 0)
 *   minSplit  - fewest samples a node needs before it may be split
 *   minLeaf   - fewest samples allowed in a leaf
 *   mtry      - number of features considered at each split (null = all)
 *
 * Every split is chosen greedily: over the candidate features and every
 * distinct value of each, take the split whose two regions have the smallest
 * total SSE around their own means. Earlier splits are never revisited.
 */
export function growTree(X, y, opts = {}) {
  const { maxDepth = 6, minSplit = 2, minLeaf = 1, mtry = null, rand = Math.random } = opts;
  const p = X[0].length;
  const all = [...Array(p).keys()];

  const build = (idx, depth) => {
    const n = idx.length;
    let sum = 0, sq = 0;
    for (const i of idx) { sum += y[i]; sq += y[i] * y[i]; }
    const value = sum / n;
    const sse = Math.max(0, sq - (sum * sum) / n);
    const node = { n, value, sse, depth };
    if (depth >= maxDepth || n < minSplit || n < 2 * minLeaf || sse <= 1e-12) return node;

    const feats = mtry && mtry < p ? pick(all, mtry, rand) : all;
    let best = null;
    for (const j of feats) {
      const sorted = idx.slice().sort((a, b) => X[a][j] - X[b][j]);
      let sumL = 0, sqL = 0;
      for (let k = 0; k < n - 1; k++) {
        const i = sorted[k];
        sumL += y[i]; sqL += y[i] * y[i];
        const nl = k + 1, nr = n - nl;
        if (X[sorted[k]][j] === X[sorted[k + 1]][j]) continue;
        if (nl < minLeaf || nr < minLeaf) continue;
        const sumR = sum - sumL, sqR = sq - sqL;
        const s = (sqL - (sumL * sumL) / nl) + (sqR - (sumR * sumR) / nr);
        if (!best || s < best.sse - 1e-12) {
          best = { j, thr: (X[sorted[k]][j] + X[sorted[k + 1]][j]) / 2, sse: s, nl, sorted };
        }
      }
    }
    if (!best || best.sse >= sse - 1e-12) return node;
    node.feature = best.j;
    node.threshold = best.thr;
    node.left = build(best.sorted.slice(0, best.nl), depth + 1);
    node.right = build(best.sorted.slice(best.nl), depth + 1);
    return node;
  };

  return build([...Array(y.length).keys()], 0);
}

/** Walk the splits from the root to a leaf and return that leaf's value. */
export function predictTree(node, row) {
  while (node.left) node = row[node.feature] <= node.threshold ? node.left : node.right;
  return node.value;
}

/** The sequence of nodes visited for one row, root first, leaf last. */
export function pathTree(node, row) {
  const out = [node];
  while (node.left) {
    node = row[node.feature] <= node.threshold ? node.left : node.right;
    out.push(node);
  }
  return out;
}

export function countLeaves(node) {
  return node.left ? countLeaves(node.left) + countLeaves(node.right) : 1;
}

export function treeDepth(node) {
  return node.left ? 1 + Math.max(treeDepth(node.left), treeDepth(node.right)) : 0;
}

/** Training SSE of the whole tree: the sum over its leaves. */
export function treeSSE(node) {
  return node.left ? treeSSE(node.left) + treeSSE(node.right) : node.sse;
}

/* ---------- pruning ---------- */

/**
 * Cost-complexity pruning. For a given alpha, return the subtree of `node`
 * that minimises SSE + alpha × (number of leaves). Computed bottom-up: a
 * subtree is collapsed into a leaf whenever the leaf alone is no worse than
 * the best its children can do. alpha = 0 returns the tree unchanged.
 */
export function pruneTree(node, alpha) {
  if (!node.left) return { tree: node, cost: node.sse + alpha };
  const L = pruneTree(node.left, alpha);
  const R = pruneTree(node.right, alpha);
  const asLeaf = node.sse + alpha;
  if (asLeaf <= L.cost + R.cost + 1e-12) {
    const leaf = { n: node.n, value: node.value, sse: node.sse, depth: node.depth };
    return { tree: leaf, cost: asLeaf };
  }
  return { tree: { ...node, left: L.tree, right: R.tree }, cost: L.cost + R.cost };
}

/* ---------- random forest ---------- */

/**
 * A forest of trees that differ because each split considers only a random
 * subset of the features. `mtry` defaults to round(√p). Each tree is also
 * grown on its own resample of the rows, drawn with replacement, which is the
 * standard construction and the other half of what makes the trees differ.
 * Prediction is the average of the trees' predictions.
 */
export function randomForest(X, y, opts = {}) {
  const p = X[0].length;
  const { nTrees = 50, mtry = Math.max(1, Math.round(Math.sqrt(p))), rand = Math.random,
    bootstrap = true, ...treeOpts } = opts;
  const trees = [];
  const n = X.length;
  for (let t = 0; t < nTrees; t++) {
    let Xt = X, yt = y;
    if (bootstrap) {
      Xt = []; yt = [];
      for (let i = 0; i < n; i++) { const k = Math.floor(rand() * n); Xt.push(X[k]); yt.push(y[k]); }
    }
    trees.push(growTree(Xt, yt, { ...treeOpts, mtry, rand }));
  }
  const predict = (row) => {
    let s = 0;
    for (const t of trees) s += predictTree(t, row);
    return s / trees.length;
  };
  return { trees, mtry, predict };
}

/* ---------- permutation importance ---------- */

export function mseOf(predict, X, y) {
  let s = 0;
  for (let i = 0; i < X.length; i++) s += (y[i] - predict(X[i])) ** 2;
  return s / X.length;
}

/** Copy X with column j shuffled across rows and every other column untouched. */
export function shuffleColumn(X, j, rand = Math.random) {
  const col = shuffle(X.map((r) => r[j]), rand);
  return X.map((r, i) => { const c = r.slice(); c[j] = col[i]; return c; });
}

/**
 * Permutation importance of every feature: the rise in MSE on (X, y) when
 * that one column is shuffled, averaged over `repeats` shuffles.
 */
export function permutationImportance(predict, X, y, rand = Math.random, repeats = 1) {
  const base = mseOf(predict, X, y);
  const p = X[0].length;
  const rise = [];
  for (let j = 0; j < p; j++) {
    let acc = 0;
    for (let r = 0; r < repeats; r++) acc += mseOf(predict, shuffleColumn(X, j, rand), y);
    rise.push(acc / repeats - base);
  }
  return { base, rise };
}
