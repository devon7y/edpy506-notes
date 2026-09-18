/* Datasets for the interactive figures.
 *
 * Every page draws on the same example the course uses: Edmonton home prices,
 * with the features from Dr. Bulut's in-class demos. The data are synthetic
 * and seeded, so the site draws the same houses on every visit and the numbers
 * quoted in the text next to a figure stay true. None of this is real market
 * data.
 *
 * Adding a dataset later
 * ----------------------
 * Add an entry to DATASETS with the same shape as `edmonton` below:
 *   id, name, blurb          - how the dataset is described on the page
 *   target                   - the value being predicted, with a formatter
 *   features                 - one descriptor per column (see FEATURES)
 *   sample(seed, n, fixed)   - draws n rows; `fixed` pins some features
 *   views.linear             - which feature has a roughly straight-line
 *                              relationship with the target (line-fitting demo)
 *   views.curved             - which feature has a curved relationship, and the
 *                              values the other features are held at so that
 *                              the curve is visible (under/overfitting and
 *                              tree demos)
 * Then pass ?dataset=<id> in the URL, or change DEFAULT_DATASET.
 */

import { rng, gauss, money } from './site.js';

/* ---------- the Edmonton housing example ---------- */

/* Neighbourhoods and their price multipliers, from the class demo. Listed in
   price order so a tree can treat the index as an ordered level. */
export const NEIGHBOURHOODS = [
  { name: 'Castle Downs', mult: 0.80 },
  { name: 'Mill Woods', mult: 0.85 },
  { name: 'Terwillegar', mult: 1.05 },
  { name: 'Strathcona', mult: 1.15 },
  { name: 'Glenora', mult: 1.30 },
  { name: 'Windermere', mult: 1.35 },
];
export const GARAGES = ['None', 'Detached', 'Attached'];
export const DOORS = ['Red', 'Blue', 'Green', 'Black', 'Yellow', 'White'];
const STREETS = [
  '142 Ave', 'Windsor Park Rd', 'Whyte Ave', 'Terwillegar Dr', 'Mill Woods Rd',
  '167 St', 'Rabbit Hill Rd', '87 Ave', 'Castle Downs Rd', 'Saskatchewan Dr',
];

const int = (v) => String(Math.round(v));
/* `short` is the header a wide table uses; `label` is the name everywhere else.
   `noise` marks a feature that does not enter the price at all. */
const FEATURES = [
  { key: 'size', label: 'Size', short: 'Sq ft', unit: 'sq ft', type: 'numeric', fmt: (v) => Math.round(v).toLocaleString('en-CA') },
  { key: 'beds', label: 'Bedrooms', short: 'Beds', type: 'numeric', fmt: int },
  { key: 'neigh', label: 'Neighbourhood', short: 'Neighbourhood', type: 'categorical', levels: NEIGHBOURHOODS.map((n) => n.name) },
  { key: 'age', label: 'Age', short: 'Age', unit: 'years', type: 'numeric', fmt: int },
  { key: 'garage', label: 'Garage', short: 'Garage', type: 'categorical', levels: GARAGES },
  { key: 'lrt', label: 'Distance to LRT', short: 'LRT km', unit: 'km', type: 'numeric', fmt: (v) => v.toFixed(1) },
  { key: 'door', label: 'Front door colour', short: 'Door', type: 'categorical', levels: DOORS, noise: true },
  { key: 'houseNum', label: 'House number', short: 'House no.', type: 'numeric', fmt: int, noise: true },
  { key: 'odd', label: 'Odd or even number', short: 'Odd/even', type: 'categorical', levels: ['Even', 'Odd'], noise: true },
];

/* What a home is worth, feature by feature. Every term is a fixed number of
   dollars added on, apart from age: prices fall as a home ages, bottom out for
   mid-century homes, and recover for the oldest ones. That one curve is what
   the non-linear demos rely on, and it is the reason a straight line cannot
   fit the whole picture. The last three features do not enter the price at
   all, so anything a model appears to learn from them is a coincidence. */

export const NEIGH_PREMIUM = [-55000, -35000, 20000, 65000, 130000, 155000];
export const GARAGE_PREMIUM = [0, 12000, 26000];
const ageEffect = (age) => -2600 * age + 31 * age * age;

function priceOf(r, noise) {
  return 60000
    + 205 * r.size
    + 11000 * r.beds
    + NEIGH_PREMIUM[r.neigh]
    + ageEffect(r.age)
    + GARAGE_PREMIUM[r.garage]
    - 7000 * r.lrt
    + noise;
}

/**
 * Draw n homes. `fixed` pins any feature to one value (a level name for a
 * categorical feature), so the relationship between the remaining feature and
 * the price is not buried under everything else varying at once.
 */
function sampleEdmonton(seed = 1, n = 120, fixed = {}) {
  const rand = rng(seed);
  const level = (key, levels) => (fixed[key] === undefined ? Math.floor(rand() * levels.length)
    : levels.indexOf(fixed[key]));
  const num = (key, draw) => (fixed[key] === undefined ? draw() : fixed[key]);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const r = {};
    r.size = num('size', () => 900 + 1800 * rand());
    r.beds = num('beds', () => Math.max(1, Math.min(5, Math.round(r.size / 600 + 0.8 * gauss(rand)))));
    r.neigh = level('neigh', FEATURES[2].levels);
    r.age = num('age', () => 80 * rand());
    r.garage = level('garage', GARAGES);
    r.lrt = num('lrt', () => 0.3 + 7.7 * rand());
    r.door = level('door', DOORS);
    r.houseNum = num('houseNum', () => 1000 + Math.floor(rand() * 9000));
    r.odd = r.houseNum % 2;
    r.addr = `${r.houseNum} ${STREETS[Math.floor(rand() * STREETS.length)]}`;
    r.price = priceOf(r, 20000 * gauss(rand));
    rows.push(r);
  }
  return rows;
}

export const DATASETS = {
  edmonton: {
    id: 'edmonton',
    name: 'Edmonton home prices',
    blurb: 'Synthetic homes modelled on the example used throughout the course: '
      + 'the same features as the in-class demos, with prices that follow them and two columns that are pure noise.',
    target: { key: 'price', label: 'Price', unit: '$', fmt: money, short: (v) => `$${Math.round(v / 1000)}k` },
    features: FEATURES,
    sample: sampleEdmonton,
    views: {
      linear: { x: 'size' },
      curved: {
        x: 'age',
        fixed: { size: 1600, beds: 3, neigh: 'Terwillegar', garage: 'Attached', lrt: 3.0 },
        describe: 'three-bedroom, 1,600 sq ft homes in Terwillegar with an attached garage, 3 km from the LRT',
      },
    },
  },
};

export const DEFAULT_DATASET = 'edmonton';

/** The dataset the page should use: ?dataset=<id> in the URL, else the default. */
export function currentDataset() {
  let id = DEFAULT_DATASET;
  try {
    const q = new URLSearchParams(globalThis.location?.search || '').get('dataset');
    if (q && DATASETS[q]) id = q;
  } catch { /* not in a browser */ }
  return DATASETS[id];
}

/* ---------- encodings ---------- */

export const feature = (ds, key) => ds.features.find((f) => f.key === key);

/** Format one feature value for display. */
export function fmtValue(f, v) {
  if (f.type === 'categorical') return f.levels[v];
  return f.fmt ? f.fmt(v) : String(v);
}

/** Feature name with unit, for axis labels and table headers. */
export const labelOf = (f) => (f.unit ? `${f.label} (${f.unit})` : f.label);

/**
 * Design matrix for a linear model: numeric features as they are, each
 * categorical feature as one 0/1 column per level except the first. Returns
 * the rows, the target, and the column names.
 */
export function linearDesign(ds, rows) {
  const names = [];
  const cols = [];
  for (const f of ds.features) {
    if (f.type === 'categorical') {
      f.levels.slice(1).forEach((lv, k) => { names.push(`${f.label}: ${lv}`); cols.push((r) => (r[f.key] === k + 1 ? 1 : 0)); });
    } else {
      names.push(labelOf(f));
      cols.push((r) => r[f.key]);
    }
  }
  return {
    X: rows.map((r) => cols.map((c) => c(r))),
    y: rows.map((r) => r[ds.target.key]),
    names,
    noise: names.map((nm) => ds.features.some((f) => f.noise && nm.startsWith(f.label))),
  };
}

/**
 * Design matrix for a tree: one column per feature, with categorical levels
 * as their index. A tree only ever asks "is this value at or below a
 * threshold", so an ordered index is all it needs.
 */
export function treeDesign(ds, rows) {
  return {
    X: rows.map((r) => ds.features.map((f) => r[f.key])),
    y: rows.map((r) => r[ds.target.key]),
    names: ds.features.map((f) => f.label),
    features: ds.features,
  };
}

/** x/y pairs for a one-feature view of the dataset. */
export function pairs(ds, view, seed, n) {
  const v = ds.views[view];
  const rows = ds.sample(seed, n, v.fixed || {});
  rows.sort((a, b) => a[v.x] - b[v.x]);
  return { xs: rows.map((r) => r[v.x]), ys: rows.map((r) => r[ds.target.key]), rows, feature: feature(ds, v.x) };
}
