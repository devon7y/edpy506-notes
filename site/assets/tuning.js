/* Model tuning: k-fold splits, cross-validated scoring, grid and random
   search, and a runner that spreads expensive work across animation frames so
   a page stays responsive while a search space is being scored. */

import { mean, sd, shuffle } from './site.js';

/** k folds of row indices, shuffled once, as equal in size as they can be. */
export function kfold(n, k, rand = Math.random) {
  const idx = shuffle([...Array(n).keys()], rand);
  return Array.from({ length: k }, (_, f) => idx.filter((_, i) => i % k === f));
}

/**
 * Cross-validate one setting. `fitScore(trainIdx, testIdx)` fits on the first
 * set of rows and returns a score on the second. Returns every fold's score,
 * their mean, and their standard deviation.
 */
export function cvScore(folds, fitScore) {
  const all = folds.flat();
  const scores = folds.map((test) => {
    const held = new Set(test);
    return fitScore(all.filter((i) => !held.has(i)), test);
  });
  return { scores, mean: mean(scores), sd: scores.length > 1 ? sd(scores) : 0 };
}

/** One train/test split: `frac` of the rows for training, the rest for testing. */
export function holdout(n, frac, rand = Math.random) {
  const idx = shuffle([...Array(n).keys()], rand);
  const cut = Math.round(n * frac);
  return { train: idx.slice(0, cut), test: idx.slice(cut) };
}

/** Every combination of the listed values: the points a grid search tries. */
export function gridPoints(space) {
  const keys = Object.keys(space);
  return keys.reduce((acc, key) => acc.flatMap((pt) => space[key].map((v) => ({ ...pt, [key]: v }))), [{}]);
}

/** `n` combinations drawn at random from the same space. */
export function randomPoints(space, n, rand = Math.random) {
  const keys = Object.keys(space);
  return Array.from({ length: n }, () =>
    Object.fromEntries(keys.map((key) => [key, space[key][Math.floor(rand() * space[key].length)]])));
}

/**
 * Run a list of jobs a few at a time, yielding to the browser between
 * batches. Scoring a search space can take a couple of seconds; done in one
 * go it freezes every slider on the page for that long.
 */
export async function runJobs(jobs, onProgress = () => {}, budgetMs = 24) {
  const out = new Array(jobs.length);
  let i = 0;
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  while (i < jobs.length) {
    const start = now();
    while (i < jobs.length && now() - start < budgetMs) { out[i] = jobs[i](); i++; }
    onProgress(i, jobs.length);
    if (i < jobs.length) await new Promise((r) => setTimeout(r, 0));
  }
  return out;
}
