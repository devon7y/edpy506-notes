# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Interactive course notes for **EDPY 506 Machine Learning: Theory and
Applications** (University of Alberta, Dr. Okan Bulut). It is a static site with
no build step, no dependencies and no framework: plain HTML, CSS and ES modules,
served as files.

It grows one topic at a time. Each week the lectures cover new material and a
page is added or an existing one is extended.

## The three rules

**1. Nothing goes on the site that was not in the lectures.** This is the
hardest rule to keep and the one that matters most. The site is a study aid for
a specific course, and material the course did not cover is worse than useless
in it — it will be studied and it will not be examined, and the reader has no
way to tell which is which. When a figure needs machinery the lectures did not
cover (bootstrap resampling inside the random forest, a log axis, the way a
demo's data are generated), say so in a `<details class="aside">` or a
`figure__note`, so implementation detail and course content stay visibly
separate. If something genuinely seems missing, ask; do not fill it in.

**2. Pages are organised by topic, not by lecture.** A lecture that covers two
topics becomes two pages. A topic revisited in a later lecture extends its
existing page rather than starting a second one. `CHAPTERS` in
`assets/site.js` is the single list that drives the top bar, the index cards
and the prev/next pager — add a page there and it appears in all three.

**3. Every number in the prose is computed, never typed.** If a paragraph says
the best degree is 2, or that lasso dropped four predictors, that sentence is
built in JavaScript from the same computation the figure beside it ran. This is
what stops the text drifting away from the figures when a seed or a sample size
changes. Look at how `figure__note` and `keypoint` contents are assembled in
`assets/regression-page.js` — none of those sentences are static HTML.

## The running example

Every page uses the same dataset: **synthetic Edmonton home prices**, with the
features from Dr. Bulut's in-class demos (size, bedrooms, neighbourhood, age,
garage, distance to LRT, front door colour, house number, odd/even). It is
generated in `assets/datasets.js` from a seeded RNG, so the site draws the same
homes on every visit.

Three properties of it are load-bearing, and changing any of them breaks pages:

- **Price is additive in the features, except age.** Age is quadratic: prices
  fall, bottom out around 42 years, and recover. That single curve is what the
  under/overfitting demo, the whole trees page and the "a line cannot fit this"
  argument all rest on.
- **Front door colour, house number and odd/even do not affect price at all.**
  They are the noise columns. The regression page shows least squares giving
  them non-zero coefficients; the trees page shows permutation importance
  correctly finding them worthless. That contrast is the point of both.
- **Noise has a standard deviation of $20,000.** That is the floor on test
  error, and several notes quote it.

### Adding another dataset later

`assets/datasets.js` is built for this. `DATASETS` maps an id to a descriptor
with `name`, `target`, `features`, a seeded `sample(seed, n, fixed)`, and
`views.linear` / `views.curved` naming which feature to plot for the
one-feature figures. Add an entry and the pages pick it up from
`?dataset=<id>` in the URL; `DEFAULT_DATASET` sets the one used otherwise.
A new dataset needs the same three properties above, or the prose beside the
figures stops being true.

## Layout

```text
index.html            landing page and the running example
regression.html       topic 1
trees.html            topic 2
assets/
  site.css            design tokens, layout, components — light and dark
  site.js             chrome (nav, theme, pager), SVG helpers, small stats, RNG
  datasets.js         the homes, and the encodings each model needs
  linreg.js           OLS, metrics, polyfit, ridge, lasso, cross-validation
  trees.js            CART, pruning, random forest, permutation importance
  regression-page.js  every figure on regression.html
  trees-page.js       every figure on trees.html
test/                 node unit tests for linreg.js and trees.js
tools/check_pages.py  browser check: errors, empty figures, overflow, id clashes
```

The maths modules (`linreg.js`, `trees.js`) import nothing but `site.js` and
use plain arrays, which is what lets `node --test` run them directly.

## Publishing

The site is served straight from `main` by GitHub Pages at
<https://devon7y.github.io/edpy506-notes/>. There is no build and no deploy
step: pushing to `main` republishes it a minute or so later. Run the checks
below before pushing, because a broken push is a broken public page.

The repository is public, which is what makes Pages free on this account. That
is the reason the slides, the syllabus and the instructor's demos are
gitignored rather than merely untracked.

## Checks

```bash
npm test          # node unit tests for the maths
npm run check     # unit tests, then every page in a real browser
npm run serve     # http://localhost:8000
```

`tools/check_pages.py` needs playwright (`pip install playwright && playwright
install chromium`). It loads each page in light and dark, drives every slider
and button, and fails on a script error, a figure that rendered empty, a page
wider than the viewport at 390/768/1280px, or an id used by a script that is
also a `<section>` anchor. Run it after any page change — several real bugs on
this site were only visible through it.

The site must be **served**, not opened as `file://`: the pages are ES modules.

## Conventions

- **Charts are hand-drawn SVG.** No charting library. Use the helpers in
  `site.js`: `svgRoot`, `frame`, `scale`, `linePath`, `el`, `clipRect`,
  `directLabels`, `responsive`, `token`. Colours come from CSS custom
  properties via `token()` so charts follow the theme; never hard-code a hex.
- **`responsive(host, draw)` for anything that draws**, so it redraws on resize
  and on a theme change.
- **Clip the plot area** with `clipRect` for any curve that can leave it. A
  degree-11 polynomial goes to nine figures.
- **`directLabels` rather than a legend** where series can be labelled at their
  ends; it pushes labels apart so two series ending close together stay
  readable.
- **A log axis when a quantity spans orders of magnitude.** Both the
  bias–variance chart and the degree/RMSE chart need one; on a linear axis the
  entire interesting region is a smear along the bottom.
- **Slider labels wrap their text in `<span class="field__name">`.** `.field`
  is a flex container, and without the span a `<sub>` or `<sup>` in the label
  drops onto the baseline.
- **Prefer a `<section id>` name that no control uses.** See rule in
  `check_pages.py`.

## Style

Prose is written for someone taking the course, not for someone who already
knows the answer. Explain the mechanism, then show it, then say what the figure
showed. Say what a model cannot do as plainly as what it can.

Course terms appear in bold on first use and match the lectures' wording, so
the site and the slides can be read together. Sentences are short and carry one
idea. An em dash is the house punctuation for an aside; the pages use them
freely and consistently, and should keep doing so.

## What is not in the repository

The lecture slides, the syllabus and Dr. Bulut's own HTML demos are his
intellectual property and are **gitignored** (`*.pdf`, `*Demo*.html`). They sit
in the working directory and are read locally while a page is being written, but
they are never committed: the repository is public so GitHub Pages can serve it.

**They are reference, not source material.** Read a demo to understand what idea
it teaches and what example it uses, then build our own figure for that idea.
Do not copy its markup, its styling, its data or its code into the site, do not
link to it, and do not reproduce a slide's figures. Never remove `*.pdf` or
`*Demo*.html` from `.gitignore`.

The Edmonton housing example is deliberately shared with the demos, because the
course uses it throughout and a reader should not have to learn a second
dataset. The implementation is ours.
