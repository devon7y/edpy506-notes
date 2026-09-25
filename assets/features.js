/* Feature engineering: the two scalings, correlation, and the two wrapper
   methods of feature selection (forward selection and recursive feature
   elimination). The wrappers take the scoring and importance functions as
   arguments, so any model can sit inside them. */

import { mean } from './site.js';

/* ---------- scaling ---------- */

/** Population standard deviation, which is what the lecture's example uses. */
export const sdPop = (a) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
};

/** z = (x − mean) / SD. Mean 0 and standard deviation 1 afterwards. */
export function zscore(a) {
  const m = mean(a), s = sdPop(a) || 1;
  return { values: a.map((v) => (v - m) / s), mean: m, sd: s };
}

/** x′ = (x − min) / (max − min). Range 0 to 1 afterwards. */
export function minmax(a) {
  const lo = Math.min(...a), hi = Math.max(...a);
  const span = hi - lo || 1;
  return { values: a.map((v) => (v - lo) / span), min: lo, max: hi };
}

/* ---------- correlation ---------- */

export function pearson(a, b) {
  const ma = mean(a), mb = mean(b);
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < a.length; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

/** Correlation of every column with every other, and with the target. */
export function correlations(X, y) {
  const p = X[0].length;
  const col = (j) => X.map((r) => r[j]);
  const cols = Array.from({ length: p }, (_, j) => col(j));
  const R = cols.map((a) => cols.map((b) => pearson(a, b)));
  const withY = cols.map((a) => pearson(a, y));
  return { R, withY };
}

/**
 * The correlation filter from the lecture: keep features correlated with the
 * target, and of any pair highly correlated with each other keep only the one
 * more correlated with the target.
 */
export function correlationFilter(X, y, { minTarget = 0.1, maxPair = 0.8 } = {}) {
  const { R, withY } = correlations(X, y);
  const p = withY.length;
  const reasons = new Array(p).fill(null);
  withY.forEach((r, j) => { if (Math.abs(r) < minTarget) reasons[j] = 'weak'; });
  for (let a = 0; a < p; a++) {
    for (let b = a + 1; b < p; b++) {
      if (Math.abs(R[a][b]) < maxPair || reasons[a] || reasons[b]) continue;
      const loser = Math.abs(withY[a]) >= Math.abs(withY[b]) ? b : a;
      reasons[loser] = `redundant:${loser === a ? b : a}`;
    }
  }
  return { R, withY, keep: reasons.map((r) => r === null), reasons };
}

/* ---------- wrapper methods ---------- */

/**
 * Forward selection: start with nothing and add, one at a time, whichever
 * remaining feature improves the score most. `score(cols)` returns a number
 * where higher is better. Returns the order features were added in and the
 * score after each addition.
 */
export function forwardSelect(p, score) {
  const chosen = [], path = [];
  const left = new Set(Array.from({ length: p }, (_, j) => j));
  while (left.size) {
    let best = -1, bestScore = -Infinity;
    for (const j of left) {
      const s = score([...chosen, j]);
      if (s > bestScore) { bestScore = s; best = j; }
    }
    chosen.push(best);
    left.delete(best);
    path.push({ added: best, cols: chosen.slice(), score: bestScore });
  }
  return { order: chosen, path };
}

/**
 * Recursive feature elimination: fit on every feature, rank them by
 * importance, drop the least important `step`, and repeat on what is left
 * until `keep` remain. `importance(cols)` returns one importance per entry of
 * cols. The lecture's point that the researcher chooses `keep`, `step` and the
 * model is why all three are arguments.
 *
 * Returns the rounds in order and a ranking, 1 for the features that survive,
 * then increasing for each earlier round's casualties.
 */
export function rfe(p, importance, { keep = 1, step = 1 } = {}) {
  let cols = Array.from({ length: p }, (_, j) => j);
  const rounds = [];
  const ranking = new Array(p).fill(0);
  while (cols.length > keep) {
    const imp = importance(cols);
    const order = cols.map((c, i) => ({ c, v: imp[i] })).sort((a, b) => a.v - b.v);
    const drop = order.slice(0, Math.min(step, cols.length - keep)).map((o) => o.c);
    rounds.push({ cols: cols.slice(), importance: imp.slice(), dropped: drop });
    cols = cols.filter((c) => !drop.includes(c));
  }
  cols.forEach((c) => { ranking[c] = 1; });
  rounds.slice().reverse().forEach((r, i) => r.dropped.forEach((c) => { ranking[c] = i + 2; }));
  return { rounds, survivors: cols, ranking };
}
