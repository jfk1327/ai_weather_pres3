# Talk results: can a VAE generate extreme Irish wind storms?

Two identical conv VAEs (d = 8, β = 1, 3 seeds each) on ERA5 daily-max 10 m gust, 32×32 around Ireland.
**(a) z-score**: one global mean/std. **(b) EVT margins**: per pixel, empirical CDF below the 95th percentile,
Generalised Pareto above, then Φ⁻¹ → N(0, 1). Tuned on 2015–19; all numbers below are on **2020–24** or on
generated samples vs the **1985–2014** training climate. "Ireland" = max over land points of the island.

Learning notebook with the maths and these results: `talk/explainer.ipynb`. Rebuild everything:
`talk/run_all.sh` (~1 h on an RTX 5060).

## Headline numbers

| | ERA5 1985–2014 | z-score VAE | EVT VAE |
|---|---|---|---|
| Reconstruction RMSE, test (m/s) | — | 1.13 | 1.14 |
| Peak over Ireland, windiest 1% of test days (bias, m/s) | — | −1.7 ± 0.2 | −1.8 ± 0.1 |
| 1-in-100-day Ireland peak at σ = 1 (m/s) | 34.4 | 28.5 ± 0.5 | 29.3 ± 0.7 |
| 1-in-1000-day Ireland peak at σ = 1 (m/s) | 40.9 | 32.2 | 33.8 |
| σ needed to match the 1-in-100-day peak | — | **1.47** | **1.32** |
| Median day at that σ (real 16.8 m/s) | — | 18.3 (+1.4) | 17.3 (+0.45) |
| Maps with negative gusts at σ = 1.3 / 1.5 / 2 | 0% | 2.3 / 6.3 / 21% | **0 / 0 / 0%** |
| Max Ireland peak in 400 generated years, σ = 1.3 (m/s) | 47.4 (30 yrs) | 44.3 | 51.6 |

± = 1 s.d. over 3 seeds.

## Talking points (suggested slide order: figures in `talk/figures/`)

1. **Hypothesis confirmed at σ = 1** (`05_return_levels.png`): both VAEs sit 4–6 m/s below ERA5 at every
   return period over Ireland.
2. **Why, part 1: MSE flattens peaks** (`03_reconstruction_peaks.png`, `04_storms.png`): the windiest days come
   back ≈ 1.8 m/s too weak (Barra: 41.7 → 36.7 m/s). Averaging real storms over a ~50 km positional uncertainty
   gives the same −1.8 m/s (notebook Section 3). The transform does **not** fix this, since it's the loss.
3. **Why, part 2: storms are rare codes** (`08_latent_norms.png`): storm days encode at ‖μ‖ ≈ 5.1; under the
   N(0, I) prior that radius is a 1-in-1000 draw, but storms are 1 in 100 days.
4. **σ-scaling is one knob for everything** (`06_sigma_quantiles.png`): matching the tail inflates the typical
   day. The EVT model needs less σ (1.32 vs 1.47) and distorts the median ~3× less.
5. **The z-score model makes impossible weather**: negative gusts in 2–21% of maps for σ = 1.3–2. EVT: never.
6. **The EVT model at σ ≈ 1.3 tracks the ERA5 return-level curve over Ireland** to ~10-year return periods and
   extrapolates beyond the record (`05_return_levels.png`, right).
7. **But it's spatially too smooth** (`07_extreme_samples.png`): extreme samples have the right magnitude and
   location (W/NW) but broad footprints, unlike the narrow bands of real storms.
8. **Where the baseline fails most** (`09_q99_maps.png`): the z-score deficit is concentrated on the west coast
   and Atlantic margin (−5 to −6 m/s at σ = 1), where storms arrive.
9. Tuning (`01_tuning.png`): reconstruction and generation want different settings (bigger d reconstructs
   better, but samples are most realistic at small d, β ≈ 1). Selection used only the *bulk* (10–90%) so
   the tails stayed an honest test. One shared configuration for both models.

## Robustness

- **Seeds:** all model numbers are means over 3 seeds. At σ = 1 the two models' tails are **not**
  distinguishable (28.5 ± 0.5 vs 29.3 ± 0.7 m/s); the differences that are clear of seed noise are the
  matched σ, the median distortion, the negative gusts and the return-level curves at σ = 1.3.
- **Latent size** (`results/sensitivity_d32.log`, d = 32, 1 seed): same direction, smaller gap. Matched σ 1.35
  vs 1.32, median +0.9 vs +0.4 m/s, negative gusts 2.8% vs 0% at σ = 1.3. The *direction* is robust; the *size*
  of the EVT advantage depends on the configuration.
- **Selection noise** (`results/seed_noise.csv`): the score's seed s.d. is 0.02–0.3 m/s; d = 8, β = 1 stays
  best on a 3-seed average.

## Caveats (say them before someone asks)

- ERA5 underestimates peak gusts (~25% vs stations): these are ERA5-world extremes.
- GPD fits treat days as independent (no declustering); the threshold (95th percentile) wasn't varied.
- Over the open ocean ξ ≈ 0, so the GPD barely bounds the tail (median end point 76 m/s vs 47 m/s over
  Ireland): the EVT model overshoots the domain-wide max beyond ~2-year return periods (up to 67 m/s).
- Generated return periods assume independent days; σ itself has no calibrated return-period meaning.
- Only 5 test years: a handful of strong storms.

## Files

`wind.py` data/transforms/VAE/training · `tune.py` sweep → `results/tuning.csv` · `final.py` 3 seeds →
`results/models/` · `evaluate.py` → `figures/`, `results/tail_table.csv`, `results/summary.json`,
`results/reconstruction_test.csv` · `plots.py` style · `explainer.ipynb` learning notebook.
