# Progress

## Phase 0 – Setup and data ✅ (checkpoint passed 2026-10-14)

### Done
- [Claude] Local env checked: `.venv` with torch 2.14.1+cu132; RTX 5060 Laptop GPU (sm_120) found and used for training (smoke test).
- [Claude] wandb login verified with a test run (project `aiweather-smoke`, safe to delete).
- [Claude] CDS key in `~/.cdsapirc` verified.
- [Claude] Repo switched to a flat layout on `main`; README rewritten with setup steps.
- [Claude] `download_era5.py`: reads hourly ERA5 from ECMWF's ARCO Zarr store, computes daily max gust and
  speed, runs sanity checks, writes `data/era5_ireland_wind_daily_max.nc` (14,610 days × 32 × 32, ~75 s).
  Stored as int16 at 0.01 m/s + zlib: 33 MB, committed to git (max rounding error 0.005 m/s).
  All 5 sanity checks pass.

### Findings
- **Data source:** the CDS API queue made 40 years take hours (max 1 year per request for both products;
  2 years rejected; the web form has the same limits). ECMWF's ARCO store (`geoChunked.zarr`, chunks
  67,584 h × 4 × 4) returns 40 years for our box in ~25 s per variable.
- **Gust day boundary:** `fg10` stamped T is the max over (T−1h, T]. Shifting back 1 h before the daily max
  reproduces the CDS daily-statistics product exactly (0.000 m/s diff, Jan 2024). Without it: up to 9.3 m/s off.
- **Speed:** daily max speed is computed from hourly speed `hypot(u10, v10)`, not from daily max u, v.
- Latitude in the saved file runs south → north: use `origin="lower"` in `imshow`.
- Windiest days (domain max gust): 2015-01-08 (56.6 m/s), 2005-01-11, 2015-01-09, 1998-01-04, 1989-02-13.
- The original box (lat 51–55.5, lon −11 to −5) is only 19 × 25 points at 0.25°.

### Decisions
- [James] Grid: larger box that is natively 32×32 (lat 49.75–57.5, lon −12.75 to −5.0) instead of interpolating; tail metrics on an Ireland mask.
- [James] Everyone on `main`, each person commits only in `people/<name>/`.
- [James] 2026-10-15: teammates (Kalpana, Lindah; Colab, no wandb) share ONE working VAE notebook
  (`notebooks/vae_minimal.ipynb`) to play with and break in their own Colab copies; no per-person copies, no
  collaborator access needed (repo is public). James keeps learning from skeletons in `people/james/`.

### Open questions
- Gust vs speed: decide after Phase 1 exploration.

## Phase 2 – VAE pipeline (in progress)

### Teammate version ✅ (Claude, at James's request)
`notebooks/vae_minimal.ipynb`: one self-contained Colab notebook for running, understanding the losses
and hyperparameter tuning (year split, global z-score, conv VAE 32→16→8→4, MSE-sum + β·KL).
- Runs saved to Drive/`runs/` (`runs_log.csv` + per-run history/figures/weights); a repeated `RUN_NAME` is
  refused; Section 10 overlays runs on the validation years.
- Tuning uses train + val only; test years are shown only when `EVALUATE_ON_TEST=True` (final run, once).
- Generation shown at σ = 1 only; "things to try" + 7 environmental extensions written as steps (not coded).
- Deliberately **excluded** (James's analysis): σ-scaled sampling, domain-max/extremes comparison, latent
  norms, tail metrics.
Defaults on the RTX 5060: 429,409 params, 0.6 s/epoch, val error 0.99 m/s per pixel after 30 epochs.
Earlier σ run (2026-10-15, now removed from the notebook; useful for James): windiest test day 48 m/s real vs
44 m/s reconstructed; samples' domain max ~33 m/s at σ=1 and ~37 m/s at σ=1.3 vs ~42 m/s for real test days.

### James's version (learning)
| Step | File | Status |
|---|---|---|
| Dataset/DataLoader | `people/james/dataset.py` | Skeleton + self-test by Claude; **James implementing** (step 1: `load_split`) |
| Model | `people/james/model.py` | Not started |
| Loss | (in `model.py` or `train.py`) | Not started |
| Training loop | `people/james/train.py` | Not started |
| Tail metrics | `people/james/metrics.py` | Not started |

Reference point (PCA + Gaussian sampling, sigma=1, train years): domain-max gust q99 34.1 vs real 39.4 m/s
(5.3 m/s too low); produces negative gusts (min -13 m/s). PC1 skewness 0.73; PC2 spread 2x larger on windy days.

## Who wrote what
| File | Written by |
|---|---|
| README.md, requirements.txt, PROGRESS.md | Claude |
| download_era5.py | Claude |
| notebooks/vae_minimal.ipynb | Claude |
| people/james/00_data_basics.ipynb, dataset.py skeleton | Claude (skeleton); James implements |
