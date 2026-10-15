# AI for Weather – Extreme Irish wind fields with a VAE

Train a VAE on daily-maximum 10 m wind over Ireland (ERA5, 32 × 32 grid, 1985–2024), then sample
z ~ N(0, σ²I) with σ > 1 to generate more extreme fields (after Oliveira et al. 2021, arXiv 2108.00048).
We test how well the generated tails match the real ones.

## Start here

| Who | Notebook | Colab |
|---|---|---|
| Kalpana | [`people/kalpana/vae_minimal.ipynb`](people/kalpana/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfk1327/ai_weather_pres3/blob/main/people/kalpana/vae_minimal.ipynb) |
| Lindah | [`people/lindah/vae_minimal.ipynb`](people/lindah/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jfk1327/ai_weather_pres3/blob/main/people/lindah/vae_minimal.ipynb) |

On Colab: **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all** (about a minute).
Each person's folder has a README with what to change and how to save.

## Layout

```
data/era5_ireland_wind_daily_max.nc   daily-max gust + speed, 14,610 days × 32 × 32 (33 MB, in git)
download_era5.py                      rebuilds that file from ERA5 (~2 min, needs a CDS key)
people/kalpana/  people/lindah/       full working VAE notebook each – your own copy to edit freely
people/james/                         James's from-scratch version (work in progress)
PROGRESS.md                           what's done, decisions, open questions
```

**Only commit inside your own `people/<your-name>/` folder.** Ask in the group chat before changing
anything else.

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
