# purgedcv

> Purged, embargoed walk-forward backtesting that quantifies exactly how much of your reported skill was leakage.

> **Implementation note.** The runnable core is pure standard-library Python:
> no NumPy/pandas/scikit-learn, DuckDB, Typer or pytest. Splitters are plain
> functions returning `Fold(train, test)` index lists (not the scikit-learn
> splitter class API), the model is a hand-written least-squares linear score,
> tests use `unittest`, and all data comes from the synthetic generator in
> `src/generator.py` (no FRED/EIA/M4 loaders). The layout is `src/*.py` +
> `tests/`, not the directory tree sketched below. Fit-before-split pipeline
> checks and the deflated metric are not implemented.


`COMPACT` · **AI / ML Engineering** · Advanced · ~5-6 days · Fintech - systematic strategy and risk model validation

**Primary language:** Python
**Tags:** `time-series`, `backtesting`, `data-leakage`, `cross-validation`, `evaluation`

---

## The problem

A time-series model backtests beautifully and loses money live. The causes are structural and repeat across the industry: k-fold cross-validation on ordered data, overlapping label windows leaking across the train/test boundary, and features engineered on the full series before splitting. Each produces an optimistic result that looks entirely credible right up until capital is committed.

## ⭐ The differentiator

Implements **purged, embargoed walk-forward cross-validation** - removing training samples whose label windows overlap the test period and embargoing a gap after it - and ships a **leakage-quantification mode that runs the same model under naive k-fold to report exactly how much of the performance was leakage**. Reporting that delta is the point: it converts an abstract methodological argument into a number an interviewer can look at. A generic backtest uses `TimeSeriesSplit` and still leaks through overlapping labels.

This is the sentence to lead with when someone asks you to walk through the
project. Everything else in this repo exists to make it true and to prove it.

## Data

Public time series - FRED macroeconomic series, EIA energy data, openly licensed M4/M5 competition data - plus a synthetic generator producing a **known-zero-signal series** where any reported skill is, by definition, leakage.

> No paid API key is required to run or demo this project. Where a paid
> service would add value it is wired as an optional enhancement behind an
> interface with an offline mock as the default implementation.

## Stack

- Python: NumPy, pandas, scikit-learn-compatible splitters
- DuckDB, Typer CLI
- pytest

## Core capabilities

- Purged and embargoed walk-forward splitter conforming to the scikit-learn splitter API
- Overlapping-label detection given a label-horizon specification
- Feature-engineering leakage checks, detecting fit-before-split on pipeline objects
- Leakage-quantification mode reporting naive against purged performance side by side
- Deflated performance metric accounting for the number of configurations tried

## Repository layout

```
src/splitters/
src/leakage/
src/metrics/
generator/
test/null/
```

## Build plan

1. Generate the zero-signal series first. It is the instrument that makes everything else measurable.
2. Run naive k-fold on it and record the spurious Sharpe. Keep that number.
3. Implement purging and embargo, re-run, and show the two side by side.
4. Deflated metrics last - they address the multiple-configurations problem the split does not.

## Testing strategy

The critical test: on the **known-zero-signal series, assert naive k-fold reports spurious skill while purged walk-forward reports none**. Plus assert no training index ever overlaps a test label window across thousands of generated configurations - a property test, because the overlap bug appears only at specific horizon/window combinations.

Tests assert **correctness**, not merely that the code runs. A green suite on
this repo is a claim about behaviour under adversarial conditions; treat any
test that would pass against a deliberately broken implementation as a bug in
the test.

## Quality & safety layer

The splitter refuses to run without an explicit label horizon. Defaulting it is how overlapping-label leakage gets reintroduced.

## Measurable outcome

> Design target was "Sharpe 1.9 naive vs 0.1 purged". Measured instead (see
> below): **Sharpe +0.66 under naive k-fold and -0.15 under purged +
> embargoed CV - on data with provably zero signal.** The size of the naive
> number depends on horizon, feature count and sample size; 1.9 was not
> reproduced at this configuration.

State it in these terms — business units, not technical ones — in your CV
bullet and in the first thirty seconds of describing the project.

## Measured results

From `python -m src.demo` (deterministic seeds, ~30 s). Series: 1500 daily
bars, 15-bar forward-return labels, 5 folds, 8 seeds; the "TRUE
out-of-sample" row fits on the same purged training sets and scores on an
independently generated future series.

**Zero-signal series (10 pure-noise, horizon-smoothed features):**

| protocol | accuracy | Sharpe (ann.) | leaking train rows |
|---|---|---|---|
| naive shuffled k-fold | 0.5721 +- 0.0113 | +0.659 +- 0.115 | 5940 |
| walk-forward, no purge | 0.4836 +- 0.0132 | -0.125 +- 0.130 | 120 |
| purged + embargoed | 0.4822 +- 0.0144 | -0.154 +- 0.139 | 0 |
| TRUE out-of-sample | 0.4887 +- 0.0058 | -0.096 +- 0.047 | 0 |

Paired inflation over truth: naive k-fold **+0.755 Sharpe (t=5.1), +8.35 pp
accuracy (t=5.4)**; purged + embargoed -0.057 Sharpe (t=-0.3).

**Planted edge (correlation 0.15, Bayes accuracy 0.5479):** truth Sharpe
+0.14, purged + embargoed +0.08 (0.4 SE off, indistinguishable), naive
k-fold +0.84 (6.2x the real edge, t=5.1).

**Purge isolated** (expanding-window walk-forward, 20 folds, 40-bar labels,
20 seeds): not purging adds +2.09 +- 0.53 pp accuracy (t=4.0) and +0.101 +-
0.031 Sharpe (t=3.2) on zero-signal data.

Test suite: 28 `unittest` tests, including a 2500-configuration property
test (>5000 folds) checking purged training sets against an independent
set-based overlap oracle, and a minimality check that the purge drops
nothing it does not have to.

### Limitations

- At the main 5-fold/15-bar setting, unpurged contiguous walk-forward is
  statistically indistinguishable from purged (t=-0.2): the shuffle is the
  big leak, and the boundary purge matters only when horizon is large
  relative to training size (Experiment 3). The spec's implied "TimeSeriesSplit
  still leaks visibly" is only shown in that regime.
- The headline Sharpe is +0.66, not the spec's 1.9; the magnitude is
  configuration-dependent.
- Synthetic Gaussian data only; no real market series, no fat tails or
  regime changes. 8 seeds give wide error bars (Sharpe SE ~0.1-0.15).
- The purge is conservative: it treats a training window touching the first
  test bar as overlapping.
- Not implemented: fit-before-split detection on pipeline objects, deflated
  Sharpe, CLI, DuckDB, scikit-learn API compatibility.

## Interview questions this project answers

- **Why can you not use k-fold on time series?**
- **What is purging, and what is an embargo?**
- **Your backtest shows Sharpe 2. What do you check first?**

## What this deliberately is *not*

- Not a trading strategy. It is the validation layer a strategy needs before anyone believes it.
- Not a backtesting engine - it is the splitter and the leakage audit.


## Run it now

```bash
python -m unittest discover -s tests -v   # the suite
python -m src.demo                        # the 60-second artefact
```

Requires Python 3.11+. The runnable core uses **only the standard
library** (including `sqlite3`), so there is nothing to install.

## Getting started

```bash
git clone <your-fork-url> purgedcv
cd purgedcv
python -m unittest tests.test_leakage -v   # spurious skill vs none
python -m src.demo                         # full leakage report
```

Docker is supported but optional — every path above works on a plain
Windows/macOS/Linux laptop without a cloud account.

## Definition of done

- [ ] The differentiator above is implemented, and a test proves it
- [ ] The measurable outcome is produced by a command anyone can run
- [ ] `README` explains the one decision a generic version gets wrong
- [ ] CI runs the full suite on every push and is green on `main`
- [ ] A recruiter can see the headline artefact in under 60 seconds

## Licence

MIT — see [LICENSE](LICENSE).
