import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, gauss, mean } from '../assets/site.js';
import {
  ols, metrics, linsolve, lstsq, polyfit, standardize, ridge, lasso,
  lassoLambdaMax, regPath, fitRegularized, crossValidate, logGrid,
} from '../assets/linreg.js';
import { currentDataset, pairs, linearDesign } from '../assets/datasets.js';

const ds = currentDataset();
/* The three views every test below uses: price against size (roughly a
   straight line), price against age with the other features pinned (curved),
   and the full design matrix for the regularization tests. */
const houses1D = () => pairs(ds, 'linear', 7, 30);
const curvedSample = (seed, n) => pairs(ds, 'curved', seed, n);
const featureData = (seed = 5, n = 120) => linearDesign(ds, ds.sample(seed, n));

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('ols recovers an exact line', () => {
  const xs = [1, 2, 3, 4, 5];
  const ys = xs.map((x) => 3 + 2 * x);
  const { b0, b1 } = ols(xs, ys);
  close(b0, 3); close(b1, 2);
});

test('ols matches the closed form on noisy data and R² agrees with 1 − SSE/SST', () => {
  const { xs, ys } = houses1D();
  const { b0, b1 } = ols(xs, ys);
  const yhat = xs.map((x) => b0 + b1 * x);
  const m = metrics(ys, yhat);
  const yb = mean(ys);
  const sst = ys.reduce((s, v) => s + (v - yb) ** 2, 0);
  close(m.r2, 1 - m.sse / sst, 1e-9);
  assert.ok(m.r2 > 0 && m.r2 < 1);
  close(m.rmse, Math.sqrt(m.mse));
});

test('mean-only model has R² = 0 and a hand-set line has higher SSE than OLS', () => {
  const { xs, ys } = houses1D();
  const yb = mean(ys);
  const m0 = metrics(ys, xs.map(() => yb));
  close(m0.r2, 0);
  const { b0, b1 } = ols(xs, ys);
  const best = metrics(ys, xs.map((x) => b0 + b1 * x)).sse;
  const worse = metrics(ys, xs.map((x) => b0 + 1.3 * b1 * x)).sse;
  assert.ok(worse > best);
});

test('linsolve and lstsq solve a small system', () => {
  const A = [[2, 1], [1, 3]];
  const b = [3, 5];
  const x = linsolve(A, b);
  close(x[0], 0.8); close(x[1], 1.4);
  const x2 = lstsq(A, b);
  close(x2[0], 0.8, 1e-9); close(x2[1], 1.4, 1e-9);
});

test('polyfit of degree n−1 interpolates every training point', () => {
  const { xs, ys } = curvedSample(3, 12);
  const span = Math.max(...ys) - Math.min(...ys);
  const { predict } = polyfit(xs, ys, 11);
  const worst = Math.max(...xs.map((x, i) => Math.abs(predict(x) - ys[i])));
  assert.ok(worst < span * 1e-6, `interpolation error ${worst}`);
});

test('polyfit degree 1 equals ols', () => {
  const { xs, ys } = curvedSample(3, 12);
  const { predict } = polyfit(xs, ys, 1);
  const { b0, b1 } = ols(xs, ys);
  close(predict(0.3), b0 + b1 * 0.3, 1e-9);
});

test('ridge at λ = 0 equals ordinary least squares; larger λ shrinks the coefficients', () => {
  const { X, y } = featureData();
  const { Z } = standardize(X);
  const yb = mean(y);
  const yc = y.map((v) => v - yb);
  const b0 = ridge(Z, yc, 0);
  const ls = lstsq(Z, yc);
  b0.forEach((v, j) => close(v, ls[j], 1e-8));
  const norm = (b) => Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  const b1 = ridge(Z, yc, 50);
  const b2 = ridge(Z, yc, 5000);
  assert.ok(norm(b1) < norm(b0));
  assert.ok(norm(b2) < norm(b1));
  assert.ok(b2.every((v) => Math.abs(v) > 0), 'ridge never reaches exactly zero');
});

test('lasso is all zeros at λmax and drops features as λ grows', () => {
  const { X, y } = featureData();
  const { Z } = standardize(X);
  const yb = mean(y);
  const yc = y.map((v) => v - yb);
  const lmax = lassoLambdaMax(Z, yc);
  const bMax = lasso(Z, yc, lmax * 1.0001);
  assert.ok(bMax.every((v) => v === 0));
  const bSmall = lasso(Z, yc, 1e-3);
  const ls = lstsq(Z, yc);
  bSmall.forEach((v, j) => close(v, ls[j], 1e-3));
  const bMid = lasso(Z, yc, lmax / 4);
  const zeros = bMid.filter((v) => v === 0).length;
  assert.ok(zeros > 0 && zeros < bMid.length, `expected some but not all zeros, got ${zeros}`);
  assert.ok(bMid[0] !== 0, 'size, the strongest predictor, survives a moderate λ');
});

test('regPath returns coefficients in grid order and lasso zero-count is monotone', () => {
  const { X, y } = featureData();
  const { Z } = standardize(X);
  const yb = mean(y);
  const yc = y.map((v) => v - yb);
  const grid = logGrid(Math.exp(2), Math.exp(16), 25);
  const path = regPath(Z, yc, grid, 'lasso');
  assert.equal(path.length, grid.length);
  const zeros = path.map((b) => b.filter((v) => v === 0).length);
  for (let i = 1; i < zeros.length; i++) assert.ok(zeros[i] >= zeros[i - 1] - 1);
  const r = regPath(Z, yc, grid, 'ridge');
  assert.equal(r.length, grid.length);
});

test('fitRegularized predicts on raw rows', () => {
  const { X, y } = featureData();
  const f = fitRegularized(X, y, 10, 'ridge');
  const sse = X.reduce((s, r, i) => s + (y[i] - f.predict(r)) ** 2, 0);
  const sst = y.reduce((s, v) => s + (v - mean(y)) ** 2, 0);
  assert.ok(sse / sst < 0.3, 'fit explains most of the variance');
});

test('crossValidate picks a λ and a one-SE λ that is at least as large', () => {
  const { X, y } = featureData();
  const grid = logGrid(Math.exp(2), Math.exp(16), 30);
  const cv = crossValidate(X, y, grid, 'lasso', 5, rng(5));
  assert.equal(cv.meanMse.length, grid.length);
  assert.ok(cv.meanMse.every((v) => Number.isFinite(v) && v > 0));
  assert.ok(grid[cv.i1se] >= grid[cv.iMin]);
  assert.ok(cv.meanMse[cv.i1se] <= cv.meanMse[cv.iMin] + cv.seMse[cv.iMin] + 1e-12);
  const cvr = crossValidate(X, y, logGrid(Math.exp(-2), Math.exp(12), 30), 'ridge', 5, rng(5));
  assert.ok(cvr.meanMse.every((v) => Number.isFinite(v) && v > 0));
});

test('gauss draws have roughly unit variance', () => {
  const rand = rng(9);
  const v = Array.from({ length: 4000 }, () => gauss(rand));
  const m = mean(v);
  const s2 = v.reduce((s, x) => s + (x - m) ** 2, 0) / v.length;
  assert.ok(Math.abs(m) < 0.06 && Math.abs(s2 - 1) < 0.08);
});
