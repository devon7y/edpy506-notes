/* Regression maths for the Regression page: ordinary least squares, the
   evaluation measures, polynomial least squares, ridge and lasso, and k-fold
   cross-validation over a lambda grid. Plain arrays and plain numbers only, so
   every function here can be unit-tested from node (see test/). */

import { mean, sd, shuffle } from './site.js';

/* ---------- simple linear regression ---------- */

/** Ordinary least squares for one feature: the closed-form slope and intercept. */
export function ols(xs, ys) {
  const xb = mean(xs), yb = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (ys[i] - yb) * (xs[i] - xb);
    den += (xs[i] - xb) ** 2;
  }
  const b1 = den === 0 ? 0 : num / den;
  return { b0: yb - b1 * xb, b1 };
}

/** SSE, R², MAE, MSE and RMSE of a set of predictions against the truth.
    R² is the ratio of explained to total variance, Σ(ŷ−ȳ)² / Σ(y−ȳ)². */
export function metrics(ys, yhat) {
  const n = ys.length, yb = mean(ys);
  let sse = 0, sae = 0, ssr = 0, sst = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - yhat[i];
    sse += e * e;
    sae += Math.abs(e);
    ssr += (yhat[i] - yb) ** 2;
    sst += (ys[i] - yb) ** 2;
  }
  return {
    sse,
    r2: sst === 0 ? NaN : ssr / sst,
    mae: sae / n,
    mse: sse / n,
    rmse: Math.sqrt(sse / n),
  };
}

/* ---------- linear algebra ---------- */

/** Solve A x = b by Gaussian elimination with partial pivoting. A is square. */
export function linsolve(A, b) {
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

/** Least-squares solution of A x ≈ y by Householder QR. A is an array of rows
    with more rows than columns. Stable enough for a degree-14 polynomial. */
export function lstsq(A, y) {
  const m = A.length, n = A[0].length;
  const R = A.map((r) => r.slice());
  const b = y.slice();
  for (let k = 0; k < n && k < m; k++) {
    let norm = 0;
    for (let i = k; i < m; i++) norm += R[i][k] ** 2;
    norm = Math.sqrt(norm);
    if (norm === 0) continue;
    const alpha = R[k][k] > 0 ? -norm : norm;
    const v = new Float64Array(m);
    for (let i = k; i < m; i++) v[i] = R[i][k];
    v[k] -= alpha;
    let vv = 0;
    for (let i = k; i < m; i++) vv += v[i] ** 2;
    if (vv === 0) continue;
    for (let j = k; j < n; j++) {
      let s = 0;
      for (let i = k; i < m; i++) s += v[i] * R[i][j];
      s = (2 * s) / vv;
      for (let i = k; i < m; i++) R[i][j] -= s * v[i];
    }
    let s = 0;
    for (let i = k; i < m; i++) s += v[i] * b[i];
    s = (2 * s) / vv;
    for (let i = k; i < m; i++) b[i] -= s * v[i];
  }
  const x = new Array(n).fill(0);
  for (let i = Math.min(n, m) - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= R[i][j] * x[j];
    x[i] = Math.abs(R[i][i]) < 1e-300 ? 0 : s / R[i][i];
  }
  return x;
}

/* ---------- polynomial least squares (for the under/overfitting demo) ---------- */

/** Fit a polynomial of the given degree by least squares. x is rescaled to
    [-1, 1] internally so high degrees stay numerically sane. */
export function polyfit(xs, ys, degree) {
  const lo = Math.min(...xs), hi = Math.max(...xs);
  const t = (x) => (hi === lo ? 0 : (2 * (x - lo)) / (hi - lo) - 1);
  const A = xs.map((x) => {
    const u = t(x);
    const row = [];
    let p = 1;
    for (let d = 0; d <= degree; d++) { row.push(p); p *= u; }
    return row;
  });
  const coef = lstsq(A, ys);
  const predict = (x) => {
    const u = t(x);
    let s = 0, p = 1;
    for (let d = 0; d <= degree; d++) { s += coef[d] * p; p *= u; }
    return s;
  };
  return { coef, predict };
}

/* ---------- standardisation ---------- */

/** Column-standardise a matrix of rows: mean 0, standard deviation 1. */
export function standardize(X) {
  const n = X.length, p = X[0].length;
  const mu = new Array(p).fill(0), s = new Array(p).fill(0);
  for (const r of X) for (let j = 0; j < p; j++) mu[j] += r[j] / n;
  for (const r of X) for (let j = 0; j < p; j++) s[j] += (r[j] - mu[j]) ** 2 / (n - 1);
  for (let j = 0; j < p; j++) s[j] = Math.sqrt(s[j]) || 1;
  const Z = X.map((r) => r.map((v, j) => (v - mu[j]) / s[j]));
  return { Z, mu, sd: s };
}

/* ---------- ridge and lasso ---------- */

/** Ridge regression on standardised features and a centred target:
    minimise Σ(y − ŷ)² + λ Σ b_j². Closed form: (ZᵀZ + λI) b = Zᵀy. */
export function ridge(Z, yc, lambda) {
  const n = Z.length, p = Z[0].length;
  const G = Array.from({ length: p }, () => new Array(p).fill(0));
  const r = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const z = Z[i];
    for (let j = 0; j < p; j++) {
      r[j] += z[j] * yc[i];
      for (let k = j; k < p; k++) G[j][k] += z[j] * z[k];
    }
  }
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) G[j][k] = G[k][j];
    G[j][j] += lambda;
  }
  return linsolve(G, r);
}

const soft = (z, g) => (z > g ? z - g : z < -g ? z + g : 0);

/** Lasso on standardised features and a centred target:
    minimise Σ(y − ŷ)² + λ Σ |b_j|, by coordinate descent. `init` warm-starts
    from a neighbouring lambda, which is what makes a whole path cheap. */
export function lasso(Z, yc, lambda, init = null, maxIter = 1000, tol = 1e-8) {
  const n = Z.length, p = Z[0].length;
  const b = init ? init.slice() : new Array(p).fill(0);
  const colSq = new Array(p).fill(0);
  for (const z of Z) for (let j = 0; j < p; j++) colSq[j] += z[j] ** 2;
  const resid = yc.map((y, i) => {
    let s = y;
    for (let j = 0; j < p; j++) s -= Z[i][j] * b[j];
    return s;
  });
  for (let it = 0; it < maxIter; it++) {
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      if (colSq[j] === 0) continue;
      let rho = 0;
      for (let i = 0; i < n; i++) rho += Z[i][j] * (resid[i] + Z[i][j] * b[j]);
      const bj = soft(rho, lambda / 2) / colSq[j];
      const d = bj - b[j];
      if (d !== 0) {
        for (let i = 0; i < n; i++) resid[i] -= Z[i][j] * d;
        b[j] = bj;
        maxDelta = Math.max(maxDelta, Math.abs(d));
      }
    }
    if (maxDelta < tol) break;
  }
  return b;
}

/** The smallest lambda at which the lasso sets every coefficient to zero. */
export function lassoLambdaMax(Z, yc) {
  const p = Z[0].length;
  let m = 0;
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < Z.length; i++) s += Z[i][j] * yc[i];
    m = Math.max(m, Math.abs(s));
  }
  return 2 * m;
}

/** Coefficient path over a lambda grid. Lambdas are visited from largest to
    smallest so each lasso fit warm-starts from the previous one. Returns one
    coefficient vector per lambda, in the grid's original order. */
export function regPath(Z, yc, lambdas, method) {
  const order = lambdas.map((l, i) => [l, i]).sort((a, b) => b[0] - a[0]);
  const out = new Array(lambdas.length);
  let prev = null;
  for (const [lam, i] of order) {
    const b = method === 'ridge' ? ridge(Z, yc, lam) : lasso(Z, yc, lam, prev);
    prev = b;
    out[i] = b;
  }
  return out;
}

/** Fit on raw X and y, predict on raw rows. Standardises internally. */
export function fitRegularized(X, y, lambda, method) {
  const { Z, mu, sd: s } = standardize(X);
  const yb = mean(y);
  const yc = y.map((v) => v - yb);
  const b = method === 'ridge' ? ridge(Z, yc, lambda) : lasso(Z, yc, lambda);
  const predict = (row) => yb + row.reduce((acc, v, j) => acc + ((v - mu[j]) / s[j]) * b[j], 0);
  return { b, yb, mu, sd: s, predict };
}

/* ---------- cross-validation over a lambda grid ---------- */

/** k-fold cross-validation of ridge or lasso over a lambda grid. Returns the
    mean and standard error of the test-fold MSE at every lambda, the index of
    the lambda with the smallest mean MSE, and the index of the largest lambda
    whose mean MSE is within one standard error of that minimum. */
export function crossValidate(X, y, lambdas, method, k = 5, rand = Math.random) {
  const n = X.length;
  const idx = shuffle([...Array(n).keys()], rand);
  const folds = Array.from({ length: k }, (_, f) => idx.filter((_, i) => i % k === f));
  const errs = lambdas.map(() => []);
  for (let f = 0; f < k; f++) {
    const test = new Set(folds[f]);
    const trI = idx.filter((i) => !test.has(i));
    const teI = folds[f];
    const Xtr = trI.map((i) => X[i]);
    const ytr = trI.map((i) => y[i]);
    const { Z, mu, sd: s } = standardize(Xtr);
    const yb = mean(ytr);
    const yc = ytr.map((v) => v - yb);
    const path = regPath(Z, yc, lambdas, method);
    path.forEach((b, li) => {
      let sse = 0;
      for (const i of teI) {
        const pred = yb + X[i].reduce((acc, v, j) => acc + ((v - mu[j]) / s[j]) * b[j], 0);
        sse += (y[i] - pred) ** 2;
      }
      errs[li].push(sse / teI.length);
    });
  }
  const meanMse = errs.map(mean);
  const seMse = errs.map((e) => (e.length > 1 ? sd(e) / Math.sqrt(e.length) : 0));
  let iMin = 0;
  meanMse.forEach((v, i) => { if (v < meanMse[iMin]) iMin = i; });
  const thresh = meanMse[iMin] + seMse[iMin];
  let i1se = iMin;
  lambdas.forEach((lam, i) => { if (meanMse[i] <= thresh && lam > lambdas[i1se]) i1se = i; });
  return { meanMse, seMse, iMin, i1se };
}

/** Log-spaced grid from lo to hi inclusive. */
export function logGrid(lo, hi, count) {
  const a = Math.log(lo), b = Math.log(hi);
  return Array.from({ length: count }, (_, i) => Math.exp(a + ((b - a) * i) / (count - 1)));
}
