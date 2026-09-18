# EDPY 506 — interactive course notes

Study notes for **EDPY 506 Machine Learning: Theory and Applications**
(University of Alberta), rebuilt as pages you can interact with. Every model
the course covers gets a page, a figure whose parameter you can drag, and an
account of what happens when you do.

**Topics so far**

| Page | Covers |
|---|---|
| [Regression](regression.html) | Least squares, R² and RMSE, multiple regression, bias and variance, under- and overfitting, ridge, lasso, choosing λ by cross-validation |
| [Decision trees and random forests](trees.html) | Where a split comes from, growing and overgrowing a tree, cost-complexity pruning, hyperparameters, random forests, permutation importance |

## Running it

No build step and no dependencies. It has to be *served* rather than opened as
a `file://` path, because the pages are ES modules.

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Checks

```bash
npm test        # unit tests for the regression and tree implementations
npm run check   # those, then every page loaded and driven in a real browser
```

The browser check needs playwright:

```bash
pip install playwright && playwright install chromium
```

## About the data

The homes are **simulated**, not real Edmonton sales. They are generated from a
seeded random number generator in `assets/datasets.js`, so the site draws the
same homes every time, and because the true relationship is known the pages can
say honestly when a model has found it and when it has not. Three of the nine
features have no effect on price at all; several figures exist to show what
different models do with them.

## Scope

Content follows the EDPY 506 lectures and adds nothing beyond them. Where a
figure relies on machinery the lectures did not cover, the page says so in a
note or a collapsed aside.

The lecture slides, the syllabus and the instructor's own in-class demos are
Dr. Bulut's intellectual property and are not included in this repository. The
figures here were built from scratch; where they explain the same idea as a
demo shown in class, they are a separate implementation of it, not a copy.
