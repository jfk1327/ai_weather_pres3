# AI for Weather – Extreme Irish wind fields with a VAE

Train a VAE on daily-maximum 10 m wind over Ireland (ERA5, 32 × 32 grid, 1985–2024), then sample
z ~ N(0, σ²I) with σ > 1 to generate more extreme fields (after Oliveira et al. 2021, arXiv 2108.00048).
We test how well the generated tails match the real ones.

## Start here: train the VAE and try hyperparameters

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfk1327/ai_weather_pres3/blob/main/notebooks/vae_minimal.ipynb)

1. Click the badge. No GitHub account is needed.
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.** It downloads the code and data, then trains in about a minute.
4. Change the numbers in **Section 1 (settings)** and run all again. **Section 10** lists things to try
   (and to break on purpose). Change one thing at a time and screenshot the plots for the presentation.
5. **File → Save a copy in Drive** keeps your edited version. It doesn't change the repo or anyone
   else's copy, so experiment freely. Click the badge again for a fresh copy.

The notebook on GitHub is saved with its outputs, so you can see the expected results before running it.

## Layout

```
notebooks/vae_minimal.ipynb           the shared VAE notebook (start here)
data/era5_ireland_wind_daily_max.nc   daily-max gust + speed, 14,610 days × 32 × 32 (33 MB, in git)
download_era5.py                      rebuilds that file from ERA5 (~2 min, needs a CDS key)
people/james/                         James's from-scratch version (work in progress)
PROGRESS.md                           what's done, decisions, open questions
```

Running the notebook needs no access to the repo. To push files (only if you want to), ask James to
add you as a collaborator, and only commit inside your own `people/<your-name>/` folder.

## Run on a laptop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # True if your GPU is usable
```
Then open your notebook in Jupyter or VS Code (choose the `.venv` kernel).

## Rebuilding the data (optional)

Not needed: the data file is already in the repo. To rebuild it, you need a CDS account:
1. Make an account at cds.climate.copernicus.eu, accept the ERA5 licence, copy your API token.
2. Create `~/.cdsapirc` on your own machine (**never** in this repo):
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-token>
   ```
3. `python download_era5.py`. It reads ERA5 from ECMWF's ARCO store and runs sanity checks.

## Git cheat sheet

```bash
git pull                              # get everyone's latest work
git add people/<your-name>
git commit -m "what you did"
git push                              # if rejected: git pull, then git push again
```

**This repo is public.** Never commit passwords, API keys or tokens.

## Data credit

Generated using Copernicus Climate Change Service information [2026]. ERA5 hourly data on single
levels (Hersbach et al. 2020), accessed via the ECMWF ARCO data store. Neither the European Commission
nor ECMWF is responsible for any use that may be made of this information.
