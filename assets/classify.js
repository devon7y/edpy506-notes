/* Classification maths for the classification page: logistic regression,
   k-nearest neighbours with the three distance metrics, a linear support
   vector machine, the confusion matrix and the metrics computed from it, and
   the ROC curve with its area. Plain arrays only, so every function here is
   unit-testable from node. */

import { mean, shuffle } from './site.js';
import { standardize } from './linreg.js';

export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

/* ---------- logistic regression ---------- */

/**
 * Binary logistic regression by Newton's method on the log-likelihood, which
 * is the standard fit (iteratively reweighted least squares). Returns the
 * intercept b0, the slopes b, and a predictor giving P(Y = 1 | X).
 *
 * `ridge` is a small penalty on the slopes. Without one the fit runs away to
 * infinite coefficients whenever the classes happen to be perfectly separable,
 * which on a small sample they sometimes are.
 */
export function logisticFit(X, y, opts = {}) {
  const { iters = 60, ridge = 1e-6, tol = 1e-10 } = opts;
  const n = X.length, p = X[0].length;
  const w = new Array(p + 1).fill(0);        // w[0] is the intercept
  const row = (i) => [1, ...X[i]];

  for (let it = 0; it < iters; it++) {
    /* gradient and Hessian of the penalised log-likelihood */
    const g = new Array(p + 1).fill(0);
    const H = Array.from({ length: p + 1 }, () => new Array(p + 1).fill(0));
    for (let i = 0; i < n; i++) {
      const x = row(i);
      let z = 0;
      for (let j = 0; j <= p; j++) z += w[j] * x[j];
      const mu = sigmoid(z);
      const s = mu * (1 - mu);
      for (let j = 0; j <= p; j++) {
        g[j] += (y[i] - mu) * x[j];
        for (let k = j; k <= p; k++) H[j][k] += s * x[j] * x[k];
      }
    }
    for (let j = 0; j <= p; j++) {
      for (let k = 0; k < j; k++) H[j][k] = H[k][j];
      if (j > 0) { g[j] -= ridge * w[j]; H[j][j] += ridge; }
      H[j][j] += 1e-9;
    }
    /* Newton step: solve H d = g */
    const d = solve(H, g);
    let step = 0;
    for (let j = 0; j <= p; j++) { w[j] += d[j]; step += Math.abs(d[j]); }
    if (!Number.isFinite(step)) break;
    if (step < tol) break;
  }

  const prob = (r) => {
    let z = w[0];
    for (let j = 0; j < p; j++) z += w[j + 1] * r[j];
    return sigmoid(z);
  };
  return { b0: w[0], b: w.slice(1), prob, logit: (r) => Math.log(prob(r) / (1 - prob(r))) };
}

/** Gaussian elimination with partial pivoting. */
function solve(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-300) continue;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = M[r][r] === 0 ? 0 : s / M[r][r];
  }
  return x;
}

/* ---------- k-nearest neighbours ---------- */

/**
 * Minkowski distance. p = 1 is Manhattan, p = 2 is Euclidean, and Python's
 * default is 2.
 */
export function minkowski(a, b, p = 2) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p;
  return s ** (1 / p);
}

/**
 * k-nearest neighbours. There is no fitting step: the training rows are the
 * model, which is why it is called a lazy learner. A prediction finds the k
 * closest training rows and lets them vote.
 *
 * Distances treat every feature as if it were on the same scale, so the
 * features are standardised here using the training rows' own mean and spread.
 * Skipping that lets whichever feature happens to have the largest units
 * decide every vote.
 */
export function knn(Xtrain, ytrain, opts = {}) {
  const { p = 2, standardise = true } = opts;
  const { Z, mu, sd } = standardise
    ? standardize(Xtrain)
    : { Z: Xtrain, mu: Xtrain[0].map(() => 0), sd: Xtrain[0].map(() => 1) };
  const enc = (r) => r.map((v, j) => (v - mu[j]) / sd[j]);

  /** The k nearest training rows to `row`, nearest first. */
  const neighbours = (row, k) => {
    const z = enc(row);
    return Z
      .map((t, i) => ({ i, d: minkowski(z, t, p) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
  };

  /** Share of the k nearest that are class 1, which is the vote. */
  const score = (row, k) => {
    const near = neighbours(row, k);
    return mean(near.map((nb) => ytrain[nb.i]));
  };

  return {
    mu,
    sd,
    neighbours,
    score,
    /* A tie at an even k is broken toward class 1, which has to be decided
       somehow; odd k avoids the question. */
    predict: (row, k) => (score(row, k) >= 0.5 ? 1 : 0),
  };
}

/* ---------- linear support vector machine ---------- */

/**
 * A linear support vector machine, fitted by sequential minimal optimisation
 * on the dual problem. SMO is the standard algorithm and it converges to the
 * actual maximal margin, which matters here because the figure's whole claim
 * is that no other separating line has a wider margin. A subgradient method
 * gets close and stops, and "close" is visible when the margin is drawn.
 *
 * Labels are 0/1 on the way in and treated as −1/+1 inside. `C` is the price
 * of letting a point sit inside the margin: a large C buys a narrow margin
 * that few points violate, a small C a wide one that tolerates violations. The
 * support vectors come out of the fit rather than being inferred from it, as
 * the points with a non-zero dual weight.
 */
export function svmFit(X, y, opts = {}) {
  /* 200 sweeps at a 1e-5 tolerance stops short of the optimum on overlapping
     data, and a boundary that has not converged is visibly not the widest one
     once the margin is drawn. These defaults reach it: about 300 sweeps on a
     couple of hundred points, which is a few milliseconds. */
  const { C = 1, maxSweeps = 2000, tol = 1e-3, standardise = true } = opts;
  const n = X.length, p = X[0].length;
  const { Z, mu, sd } = standardise
    ? standardize(X)
    : { Z: X, mu: X[0].map(() => 0), sd: X[0].map(() => 1) };
  const t = y.map((v) => (v === 1 ? 1 : -1));

  const dot = (u, v) => { let s = 0; for (let j = 0; j < p; j++) s += u[j] * v[j]; return s; };
  const K = Z.map((zi) => Float64Array.from(Z, (zj) => dot(zi, zj)));

  const a = new Float64Array(n);
  let b = 0;
  /* E[i] is the current error on point i. Keeping it up to date rather than
     recomputing it is what makes the pair selection below affordable. */
  const E = Float64Array.from(t, (ti) => -ti);

  /* One optimisation step on the pair (i, j), returning whether it moved. */
  const takeStep = (i, j) => {
    if (i === j) return false;
    const ai = a[i], aj = a[j];
    const [L, H] = t[i] !== t[j]
      ? [Math.max(0, aj - ai), Math.min(C, C + aj - ai)]
      : [Math.max(0, ai + aj - C), Math.min(C, ai + aj)];
    if (H - L < 1e-12) return false;
    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
    if (eta > -1e-12) return false;
    let ajNew = aj - (t[j] * (E[i] - E[j])) / eta;
    ajNew = Math.min(H, Math.max(L, ajNew));
    if (Math.abs(ajNew - aj) < 1e-10 * (ajNew + aj + 1e-10)) return false;
    const aiNew = ai + t[i] * t[j] * (aj - ajNew);

    const b1 = b - E[i] - t[i] * (aiNew - ai) * K[i][i] - t[j] * (ajNew - aj) * K[i][j];
    const b2 = b - E[j] - t[i] * (aiNew - ai) * K[i][j] - t[j] * (ajNew - aj) * K[j][j];
    const bNew = aiNew > 1e-12 && aiNew < C - 1e-12 ? b1
      : ajNew > 1e-12 && ajNew < C - 1e-12 ? b2 : (b1 + b2) / 2;

    const di = t[i] * (aiNew - ai), dj = t[j] * (ajNew - aj), db = bNew - b;
    for (let k = 0; k < n; k++) E[k] += di * K[i][k] + dj * K[j][k] + db;
    a[i] = aiNew; a[j] = ajNew; b = bNew;
    return true;
  };

  /* A point violates the Karush-Kuhn-Tucker conditions when its error has the
     wrong sign for where its dual weight sits. Only those can improve the fit. */
  const violates = (i) => (t[i] * E[i] < -tol && a[i] < C - 1e-12)
    || (t[i] * E[i] > tol && a[i] > 1e-12);

  /* For a violating i, the textbook second choice is the j whose error is
     furthest from i's, because that is the step likeliest to move. If it makes
     no progress, try the unbound points and then everything else, and only
     give up on i once every partner has failed. This is the part the
     "simplified" version of the algorithm skips, and skipping it leaves the
     solution short of the maximal margin -- visibly so, once the margin is
     drawn on a chart. */
  const partner = (i) => {
    let best = -1, bestGap = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const gap = Math.abs(E[i] - E[j]);
      if (gap > bestGap) { bestGap = gap; best = j; }
    }
    return best;
  };
  const examine = (i) => {
    if (!violates(i)) return false;
    const j0 = partner(i);
    if (j0 >= 0 && takeStep(i, j0)) return true;
    for (let j = 0; j < n; j++) if (a[j] > 1e-12 && a[j] < C - 1e-12 && takeStep(i, j)) return true;
    for (let j = 0; j < n; j++) if (takeStep(i, j)) return true;
    return false;
  };

  let sweeps = 0, examineAll = true, changed = 0;
  while (sweeps < maxSweeps && (examineAll || changed > 0)) {
    sweeps++;
    changed = 0;
    if (examineAll) {
      for (let i = 0; i < n; i++) if (examine(i)) changed++;
    } else {
      for (let i = 0; i < n; i++) {
        if (a[i] > 1e-12 && a[i] < C - 1e-12 && examine(i)) changed++;
      }
    }
    /* Alternate between sweeping everything and sweeping only the points on
       the margin, which is where the remaining work almost always is. */
    examineAll = examineAll ? false : changed === 0;
  }

  const w = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    if (a[i] <= 1e-12) continue;
    for (let j = 0; j < p; j++) w[j] += a[i] * t[i] * Z[i][j];
  }

  const enc = (r) => r.map((v, j) => (v - mu[j]) / sd[j]);
  const margin = (r) => {
    const z = enc(r);
    let m = b;
    for (let j = 0; j < p; j++) m += w[j] * z[j];
    return m;
  };
  const norm = Math.sqrt(w.reduce((s, v) => s + v * v, 0)) || 1;
  const support = [];
  for (let i = 0; i < n; i++) if (a[i] > 1e-8) support.push(i);

  return {
    w, b, mu, sd, margin, support, alpha: Array.from(a), sweeps,
    /* The margin's width in standardised units, which is what the fit
       maximises. */
    width: 2 / norm,
    predict: (r) => (margin(r) >= 0 ? 1 : 0),
    score: (r) => margin(r),
  };
}

/* ---------- the confusion matrix and what comes out of it ---------- */

/**
 * Counts, with 1 as the positive class.
 *   tp  actual 1, predicted 1
 *   fn  actual 1, predicted 0   — a Type II error
 *   fp  actual 0, predicted 1   — a Type I error
 *   tn  actual 0, predicted 0
 */
export function confusion(yTrue, yPred) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1) (yPred[i] === 1 ? tp++ : fn++);
    else (yPred[i] === 1 ? fp++ : tn++);
  }
  return { tp, fp, tn, fn, n: yTrue.length };
}

/** Accuracy, precision, recall and F1. Each is 0 to 1, higher is better. */
export function classMetrics(cm) {
  const { tp, fp, tn, fn } = cm;
  const accuracy = (tp + tn) / (tp + tn + fp + fn || 1);
  const precision = tp + fp === 0 ? NaN : tp / (tp + fp);
  const recall = tp + fn === 0 ? NaN : tp / (tp + fn);
  const f1 = Number.isFinite(precision) && Number.isFinite(recall) && precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : NaN;
  return { accuracy, precision, recall, f1 };
}

/** Predictions at a chosen classification threshold. */
export const atThreshold = (scores, t) => scores.map((s) => (s >= t ? 1 : 0));

/**
 * The ROC curve: every threshold's false positive rate against its true
 * positive rate, ordered by descending score, starting at (0,0).
 */
export function roc(labels, scores) {
  const n = labels.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const P = labels.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  const N = n - P;
  const pts = [{ fpr: 0, tpr: 0, t: Infinity }];
  let tp = 0, fp = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    if (labels[i] === 1) tp++; else fp++;
    /* Only emit a point once every tie at this score has been consumed. */
    if (k + 1 < n && scores[idx[k + 1]] === scores[i]) continue;
    pts.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0, t: scores[i] });
  }
  return pts;
}

/**
 * Area under the ROC curve, by rank. A perfect classifier scores 1, a random
 * one 0.5. Equivalently: take one observation of each class at random, and this
 * is the chance the positive one is scored higher.
 */
export function auc(labels, scores) {
  const n = labels.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const r = (i + j + 2) / 2;                 // mean of the 1-based ranks in the tie
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    i = j + 1;
  }
  let n1 = 0, n0 = 0, sum = 0;
  for (let k = 0; k < n; k++) {
    if (labels[k] === 1) { n1++; sum += ranks[k]; } else n0++;
  }
  if (!n1 || !n0) return NaN;
  return (sum - (n1 * (n1 + 1)) / 2) / (n1 * n0);
}

/** Accuracy of a 0/1 prediction. */
export const accuracyOf = (yTrue, yPred) =>
  mean(yTrue.map((v, i) => (v === yPred[i] ? 1 : 0)));

/** A stratified split, so both classes appear in both halves. */
export function split(rows, y, frac, rand = Math.random) {
  const byClass = { 0: [], 1: [] };
  y.forEach((v, i) => byClass[v].push(i));
  const tr = [], te = [];
  for (const c of [0, 1]) {
    const ids = shuffle(byClass[c], rand);
    const cut = Math.round(ids.length * frac);
    tr.push(...ids.slice(0, cut));
    te.push(...ids.slice(cut));
  }
  return { train: tr.sort((a, b) => a - b), test: te.sort((a, b) => a - b) };
}
