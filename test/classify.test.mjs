import test from 'node:test';
import assert from 'node:assert/strict';
import { rng, gauss, mean } from '../assets/site.js';
import {
  sigmoid, logisticFit, minkowski, knn, svmFit, confusion, classMetrics,
  atThreshold, roc, auc, accuracyOf, split,
} from '../assets/classify.js';
import { standardize } from '../assets/linreg.js';
import { currentDataset, labelled, classDesign } from '../assets/datasets.js';

const ds = currentDataset();
const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('sigmoid maps the whole line into 0 to 1', () => {
  close(sigmoid(0), 0.5);
  assert.ok(sigmoid(-40) > 0 && sigmoid(-40) < 1e-6);
  /* At 40 the result rounds to exactly 1 in double precision, which is the
     correct answer to the question the page asks: the curve never leaves the
     interval, whatever you feed it. */
  assert.ok(sigmoid(40) <= 1 && sigmoid(40) >= 1 - 1e-12);
  assert.ok(sigmoid(10) < 1, 'still strictly inside at a moderate value');
  assert.ok(sigmoid(-1e6) >= 0 && sigmoid(1e6) <= 1);
  for (const z of [-9, -1, 0, 1, 9]) close(sigmoid(z) + sigmoid(-z), 1, 1e-12);
});

test('logistic regression recovers the coefficients it was generated with', () => {
  /* The dataset builds the label from -0.26*over - 0.42*(lrt-4), so a fit on
     those two features has to come back with those numbers. */
  const { rows } = labelled(ds, 5, 4000);
  const X = rows.map((r) => [r.over, r.lrt]);
  const y = rows.map((r) => r.fast);
  const f = logisticFit(X, y);
  close(f.b[0], -0.26, 0.04);
  close(f.b[1], -0.42, 0.12);
  close(f.b0, 0.42 * 4, 0.3);
});

test('logistic probabilities stay inside 0 and 1 however extreme the input', () => {
  const { rows } = labelled(ds, 7, 400);
  const f = logisticFit(rows.map((r) => [r.over]), rows.map((r) => r.fast));
  for (const over of [-1e4, -100, 0, 100, 1e4]) {
    const p = f.prob([over]);
    assert.ok(p >= 0 && p <= 1, `p = ${p} at over = ${over}`);
  }
  assert.ok(f.prob([-60]) > 0.99, 'a big discount sells');
  assert.ok(f.prob([60]) < 0.01, 'a big markup does not');
});

test('the logit is linear in the feature even though the probability is not', () => {
  const { rows } = labelled(ds, 5, 1500);
  const f = logisticFit(rows.map((r) => [r.over]), rows.map((r) => r.fast));
  const d1 = f.logit([10]) - f.logit([0]);
  const d2 = f.logit([30]) - f.logit([20]);
  close(d1, d2, 1e-6);
  const p1 = f.prob([10]) - f.prob([0]);
  const p2 = f.prob([30]) - f.prob([20]);
  assert.ok(Math.abs(p1 - p2) > 1e-3, 'the probability change is not constant');
});

test('minkowski gives Manhattan at p = 1 and Euclidean at p = 2', () => {
  const a = [0, 0], b = [3, 4];
  close(minkowski(a, b, 1), 7);
  close(minkowski(a, b, 2), 5);
  assert.ok(minkowski(a, b, 10) > 4 && minkowski(a, b, 10) < 4.2);
  close(minkowski(a, a, 2), 0);
});

test('knn at k = 1 reproduces its training labels exactly', () => {
  const { rows } = labelled(ds, 5, 200);
  const X = rows.map((r) => [r.over, r.lrt]);
  const y = rows.map((r) => r.fast);
  const m = knn(X, y);
  for (let i = 0; i < X.length; i++) assert.equal(m.predict(X[i], 1), y[i]);
});

test('knn at k = n predicts the majority class for every row', () => {
  const { rows } = labelled(ds, 11, 120);
  const X = rows.map((r) => [r.over, r.lrt]);
  const y = rows.map((r) => r.fast);
  const majority = mean(y) >= 0.5 ? 1 : 0;
  const m = knn(X, y);
  const preds = new Set(X.map((r) => m.predict(r, X.length)));
  assert.equal(preds.size, 1);
  assert.equal([...preds][0], majority);
});

test('knn standardises, so a feature in larger units cannot dominate the vote', () => {
  /* Same rows, but distance to the LRT rescaled from km to metres. Without
     standardisation that one column would decide every neighbour. */
  const { rows } = labelled(ds, 5, 300);
  const y = rows.map((r) => r.fast);
  const km = rows.map((r) => [r.over, r.lrt]);
  const m = rows.map((r) => [r.over, r.lrt * 1000]);
  const a = knn(km, y), b = knn(m, y);
  const qa = km.map((r) => a.predict(r, 7));
  const qb = m.map((r) => b.predict(r, 7));
  assert.deepEqual(qa, qb, 'rescaling a column must not change the predictions');

  const raw = knn(m, y, { standardise: false });
  const qraw = m.map((r) => raw.predict(r, 7));
  assert.notDeepEqual(qraw, qb, 'without standardisation the rescaling does change them');
});

test('knn neighbours come back nearest first and there are k of them', () => {
  const { rows } = labelled(ds, 5, 150);
  const X = rows.map((r) => [r.over, r.lrt]);
  const m = knn(X, rows.map((r) => r.fast));
  const near = m.neighbours([0, 3], 6);
  assert.equal(near.length, 6);
  for (let i = 1; i < near.length; i++) assert.ok(near[i].d >= near[i - 1].d);
});

test('a linear SVM separates separable classes, and no other separator has a wider margin', () => {
  /* Two well-spread blobs. Parallel line segments will not do as a test case:
     every point is then the same distance from the boundary, so every point is
     a support vector and "widest margin" says nothing. */
  const rand = rng(12);
  const X = [], y = [];
  for (let i = 0; i < 45; i++) {
    X.push([2.2 * gauss(rand) - 4, 2.2 * gauss(rand) - 3]); y.push(0);
    X.push([2.2 * gauss(rand) + 4, 2.2 * gauss(rand) + 3]); y.push(1);
  }
  /* A large C is the hard-margin case: violating the margin is not worth it,
     so the fit is the maximal margin classifier from the lecture. The tolerance
     is tighter than the page needs, because this test is about whether the
     algorithm reaches the optimum rather than whether a chart looks right. */
  const m = svmFit(X, y, { C: 50, tol: 1e-8 });
  assert.equal(accuracyOf(y, X.map((r) => m.predict(r))), 1, 'separates them');

  const enc = (r) => r.map((v, j) => (v - m.mu[j]) / m.sd[j]);
  const geo = (w, b) => {
    const norm = Math.hypot(w[0], w[1]);
    return Math.min(...X.map((r, i) => {
      const z = enc(r);
      return ((y[i] ? 1 : -1) * (w[0] * z[0] + w[1] * z[1] + b)) / norm;
    }));
  };
  const best = geo(m.w, m.b);
  assert.ok(best > 0, 'every point is on its own side');

  /* Rotate and shift the boundary. Nothing else that still separates the data
     may have a wider margin -- that is what "maximal margin" claims. */
  let beaten = 0, tried = 0;
  for (let a = -0.8; a <= 0.8001; a += 0.04) {
    const w = [m.w[0] * Math.cos(a) - m.w[1] * Math.sin(a),
               m.w[0] * Math.sin(a) + m.w[1] * Math.cos(a)];
    for (let db = -0.5; db <= 0.5001; db += 0.04) {
      const g = geo(w, m.b + db);
      if (g <= 0) continue;
      tried++;
      if (g > best + 0.005) beaten++;
    }
  }
  assert.ok(tried > 100, `expected many alternative separators, tried ${tried}`);
  assert.equal(beaten, 0, `${beaten} of ${tried} alternatives had a wider margin`);

  /* A support vector with a dual weight strictly inside (0, C) sits exactly on
     the margin, which is the optimality condition the solver has to satisfy. */
  const t = y.map((v) => (v ? 1 : -1));
  const unbound = m.support.filter((i) => m.alpha[i] < 50 - 1e-9);
  assert.ok(unbound.length >= 2, `expected points on the margin, got ${unbound.length}`);
  for (const i of unbound) close(t[i] * m.margin(X[i]), 1, 1e-4);

  assert.ok(m.support.length <= X.length / 4,
    `only a few points should define the boundary, got ${m.support.length} of ${X.length}`);
  /* Every other point is strictly further out than every support vector. */
  const dist = X.map((r, i) => t[i] * m.margin(r));
  const others = dist.filter((_, i) => !m.support.includes(i));
  assert.ok(Math.min(...others) >= Math.max(...m.support.map((i) => dist[i])) - 1e-4);
});

test('moving a point that is not a support vector leaves the boundary alone', () => {
  /* Fitted on already-standardised columns with standardise off. Otherwise
     moving one point shifts that column's own mean and spread, every point's
     encoded position moves with it, and the boundary changes for a reason that
     has nothing to do with support vectors. */
  const { rows } = labelled(ds, 5, 120);
  const y = rows.map((r) => r.fast);
  const { Z } = standardize(rows.map((r) => [r.over, r.lrt]));
  const base = svmFit(Z, y, { C: 1, standardise: false });
  const sup = new Set(base.support);

  /* A point on the correct side and comfortably outside the margin. */
  const t = y.map((v) => (v ? 1 : -1));
  const far = Z.findIndex((r, i) => !sup.has(i) && t[i] * base.margin(r) > 1.3);
  assert.ok(far >= 0, 'there is a point well clear of the margin');

  /* Push it further out, along the direction the boundary points. */
  const dir = Math.hypot(base.w[0], base.w[1]);
  const moved = Z.map((r, i) => (i === far
    ? [r[0] + (2 * t[i] * base.w[0]) / dir, r[1] + (2 * t[i] * base.w[1]) / dir]
    : r));
  assert.ok(t[far] * base.margin(moved[far]) > t[far] * base.margin(Z[far]),
    'the move is away from the boundary, not toward it');

  const after = svmFit(moved, y, { C: 1, standardise: false });
  /* Agreement to the solver's own tolerance is the most that can be claimed. */
  close(after.w[0], base.w[0], 2e-3);
  close(after.w[1], base.w[1], 2e-3);
  close(after.b, base.b, 2e-3);
});

test('the confusion matrix matches the cat and dog example from the lecture', () => {
  const actual = ['dog','cat','dog','cat','dog','dog','cat','dog','cat','dog',
                  'dog','dog','dog','cat','dog','dog','cat','dog','dog','cat'];
  const pred   = ['dog','dog','dog','cat','dog','dog','cat','cat','cat','cat',
                  'dog','dog','dog','cat','dog','dog','cat','dog','dog','cat'];
  const y = actual.map((v) => (v === 'cat' ? 1 : 0));
  const p = pred.map((v) => (v === 'cat' ? 1 : 0));
  const cm = confusion(y, p);
  assert.deepEqual({ tp: cm.tp, fn: cm.fn, fp: cm.fp, tn: cm.tn },
    { tp: 6, fn: 1, fp: 2, tn: 11 });
  const m = classMetrics(cm);
  close(m.accuracy, 17 / 20, 1e-12);
  close(m.precision, 6 / 8, 1e-12);
  close(m.recall, 6 / 7, 1e-12);
  close(m.f1, (2 * (6 / 8) * (6 / 7)) / ((6 / 8) + (6 / 7)), 1e-12);
});

test('the metrics match the credit card fraud example from the lecture', () => {
  const cm = { tp: 20, tn: 70, fp: 5, fn: 5, n: 100 };
  const m = classMetrics(cm);
  close(m.accuracy, 0.90, 1e-12);
  close(m.precision, 0.80, 1e-12);
  close(m.recall, 0.80, 1e-12);
  close(m.f1, 0.80, 1e-12);
});

test('accuracy is high and useless when the classes are imbalanced', () => {
  /* 95 negatives, 5 positives, and a model that always answers negative. */
  const y = [...Array(95).fill(0), ...Array(5).fill(1)];
  const p = y.map(() => 0);
  const m = classMetrics(confusion(y, p));
  close(m.accuracy, 0.95, 1e-12);
  close(m.recall, 0, 1e-12);
  assert.ok(Number.isNaN(m.precision), 'precision is undefined with no positive predictions');
});

test('predicting every case positive gives perfect recall and poor precision', () => {
  const y = [...Array(90).fill(0), ...Array(10).fill(1)];
  const m = classMetrics(confusion(y, y.map(() => 1)));
  close(m.recall, 1, 1e-12);
  close(m.precision, 0.1, 1e-12);
});

test('lowering the threshold trades precision for recall, monotonically', () => {
  const { rows } = labelled(ds, 5, 600);
  const f = logisticFit(rows.map((r) => [r.over, r.lrt]), rows.map((r) => r.fast));
  const y = rows.map((r) => r.fast);
  const s = rows.map((r) => f.prob([r.over, r.lrt]));
  let prevRecall = -1;
  for (const t of [0.9, 0.7, 0.5, 0.3, 0.1]) {
    const m = classMetrics(confusion(y, atThreshold(s, t)));
    assert.ok(m.recall >= prevRecall - 1e-12, `recall fell as the threshold dropped to ${t}`);
    prevRecall = m.recall;
  }
});

test('AUC is 1 for a perfect ranking, 0.5 for a constant one, and 0 when reversed', () => {
  const y = [0, 0, 1, 1];
  close(auc(y, [0.1, 0.2, 0.8, 0.9]), 1);
  close(auc(y, [0.5, 0.5, 0.5, 0.5]), 0.5);
  close(auc(y, [0.9, 0.8, 0.2, 0.1]), 0);
});

test('AUC ignores class balance, which is the whole reason for using it', () => {
  const rand = rng(3);
  const y = [], s = [];
  for (let i = 0; i < 400; i++) {
    const cls = rand() < 0.05 ? 1 : 0;          // 5% positive
    y.push(cls);
    s.push(cls ? 0.55 + 0.45 * rand() : 0.45 * rand());
  }
  const a = auc(y, s);
  assert.ok(a > 0.97, `AUC ${a}`);
  /* The always-negative model is 95% accurate on the same data. */
  close(classMetrics(confusion(y, y.map(() => 0))).accuracy, 1 - mean(y), 1e-12);
});

test('the ROC curve starts at the origin, ends at (1,1), and never descends', () => {
  const { rows } = labelled(ds, 9, 400);
  const f = logisticFit(rows.map((r) => [r.over, r.lrt]), rows.map((r) => r.fast));
  const y = rows.map((r) => r.fast);
  const s = rows.map((r) => f.prob([r.over, r.lrt]));
  const pts = roc(y, s);
  close(pts[0].fpr, 0); close(pts[0].tpr, 0);
  close(pts[pts.length - 1].fpr, 1, 1e-9);
  close(pts[pts.length - 1].tpr, 1, 1e-9);
  for (let i = 1; i < pts.length; i++) {
    assert.ok(pts[i].fpr >= pts[i - 1].fpr - 1e-12);
    assert.ok(pts[i].tpr >= pts[i - 1].tpr - 1e-12);
  }
  /* The area under the drawn curve must equal the rank-based AUC. */
  let area = 0;
  for (let i = 1; i < pts.length; i++) {
    area += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
  }
  close(area, auc(y, s), 1e-9);
});

test('a stratified split keeps both classes on both sides', () => {
  const { rows } = labelled(ds, 5, 300);
  const y = rows.map((r) => r.fast);
  const { train, test: te } = split(rows, y, 0.7, rng(4));
  assert.equal(train.length + te.length, rows.length);
  assert.equal(new Set([...train, ...te]).size, rows.length);
  for (const part of [train, te]) {
    const ys = part.map((i) => y[i]);
    assert.ok(ys.includes(0) && ys.includes(1));
  }
  assert.ok(Math.abs(mean(train.map((i) => y[i])) - mean(te.map((i) => y[i]))) < 0.06);
});

test('knn accuracy on held-out homes peaks somewhere in the middle of k', () => {
  const { rows } = labelled(ds, 5, 400);
  const d = classDesign(ds, rows, ['over', 'lrt']);
  const { train, test: te } = split(rows, d.y, 0.6, rng(2));
  const m = knn(train.map((i) => d.X[i]), train.map((i) => d.y[i]));
  const acc = [];
  for (const k of [1, 3, 5, 9, 15, 25, 45]) {
    acc.push({ k, a: accuracyOf(te.map((i) => d.y[i]), te.map((i) => m.predict(d.X[i], k))) });
  }
  const best = acc.reduce((a, b) => (b.a > a.a ? b : a));
  assert.ok(best.k > 1, `k = 1 should not win outright, got ${JSON.stringify(acc)}`);
  const trainAcc = accuracyOf(train.map((i) => d.y[i]), train.map((i) => m.predict(d.X[i], 1)));
  assert.equal(trainAcc, 1, 'k = 1 is perfect on its own training rows');
});
