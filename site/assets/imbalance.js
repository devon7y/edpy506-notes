/* Resampling for imbalanced classes: random undersampling, random
   oversampling, SMOTE, and SMOTE followed by random undersampling. Every
   function takes a training set and returns a new one; nothing here touches a
   test set, because resampling the data a model is scored on would change the
   question being asked of it. Plain arrays, so node can test it directly. */

import { shuffle } from './site.js';
import { standardize } from './linreg.js';

const indicesOf = (y, cls) => y.reduce((a, v, i) => (v === cls ? (a.push(i), a) : a), []);

/** Which label is the minority, and the indices of each class. */
export function classes(y) {
  const ones = indicesOf(y, 1), zeros = indicesOf(y, 0);
  const minority = ones.length <= zeros.length ? 1 : 0;
  return {
    minority,
    majority: 1 - minority,
    min: minority === 1 ? ones : zeros,
    maj: minority === 1 ? zeros : ones,
  };
}

/**
 * Random undersampling: drop majority rows at random, without replacement,
 * until the two classes are the same size. `ratio` is the minority's size as a
 * share of the majority's afterwards; 1 means an even split.
 */
export function undersample(X, y, rand = Math.random, ratio = 1) {
  const c = classes(y);
  const keepMaj = Math.min(c.maj.length, Math.round(c.min.length / ratio));
  const kept = [...c.min, ...shuffle(c.maj, rand).slice(0, keepMaj)].sort((a, b) => a - b);
  return {
    X: kept.map((i) => X[i]), y: kept.map((i) => y[i]),
    source: kept, dropped: c.maj.length - keepMaj,
  };
}

/**
 * Random oversampling: draw minority rows with replacement until the classes
 * match. Nothing new is added; the extra rows are copies, which is why the
 * model can end up memorising them.
 */
export function oversample(X, y, rand = Math.random, ratio = 1) {
  const c = classes(y);
  const want = Math.round(c.maj.length * ratio) - c.min.length;
  const copies = [];
  for (let i = 0; i < want; i++) copies.push(c.min[Math.floor(rand() * c.min.length)]);
  const idx = [...Array(X.length).keys(), ...copies];
  return { X: idx.map((i) => X[i]), y: idx.map((i) => y[i]), source: idx, added: copies.length };
}

/**
 * SMOTE. For each synthetic row: pick a minority row A at random, find its k
 * nearest minority neighbours, pick one of them B at random, and place a new
 * row at a random point on the segment from A to B.
 *
 * Neighbours are found on standardised columns, for the reason distance needs
 * standardising anywhere. The new point is placed in the original units:
 * a point a fraction t of the way from A to B is the same point whichever
 * units it is measured in, so only the neighbour search depends on scale.
 *
 * Returns the augmented set, and a `trace` of every synthetic point built --
 * A, its neighbours, B, t and the result -- so a figure can show the steps.
 */
export function smote(X, y, rand = Math.random, opts = {}) {
  const { k = 5, ratio = 1 } = opts;
  const c = classes(y);
  const want = Math.max(0, Math.round(c.maj.length * ratio) - c.min.length);
  if (!want || c.min.length < 2) return { X: X.slice(), y: y.slice(), synthetic: [], trace: [] };

  const { Z } = standardize(X);
  const d2 = (a, b) => a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0);
  const kk = Math.min(k, c.min.length - 1);
  /* each minority row's k nearest minority neighbours, computed once */
  const near = new Map(c.min.map((i) => [i, c.min
    .filter((j) => j !== i)
    .map((j) => ({ j, d: d2(Z[i], Z[j]) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, kk)
    .map((o) => o.j)]));

  const synthetic = [], trace = [];
  for (let s = 0; s < want; s++) {
    const a = c.min[Math.floor(rand() * c.min.length)];
    const nb = near.get(a);
    const b = nb[Math.floor(rand() * nb.length)];
    const t = rand();
    const point = X[a].map((v, j) => v + t * (X[b][j] - v));
    synthetic.push(point);
    trace.push({ a, b, t, neighbours: nb, point });
  }
  return {
    X: [...X, ...synthetic],
    y: [...y, ...synthetic.map(() => c.minority)],
    synthetic, trace,
  };
}

/**
 * The hybrid: SMOTE the minority part of the way up, then randomly
 * undersample the majority down to meet it. `lift` is the minority's size as a
 * share of the majority's after the SMOTE step.
 */
export function smoteUnder(X, y, rand = Math.random, opts = {}) {
  const { lift = 0.5, k = 5 } = opts;
  const up = smote(X, y, rand, { k, ratio: lift });
  const down = undersample(up.X, up.y, rand, 1);
  return { X: down.X, y: down.y, synthetic: up.synthetic, dropped: down.dropped, trace: up.trace };
}
