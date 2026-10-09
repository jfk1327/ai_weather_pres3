# AI for Weather – Group Project

Train a **transformer or VAE** on a weather/climate task.

> **Task:** _TBD_ · **Dataset:** _TBD_ · **Model:** _TBD_

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd AI_weather
uv sync                       # creates .venv with Python 3.12 + all deps
uv run nbstripout --install   # strips notebook outputs on commit (run once per clone)
uv run wandb login            # experiment tracking (team project)
```

GPU check: `uv run python -c "import torch; print(torch.cuda.is_available())"`

## Layout

| Path | Purpose |
|---|---|
| `src/aiweather/data/` | Dataset loading & preprocessing |
| `src/aiweather/models/` | Model architectures |
| `src/aiweather/training/` | Training loops |
| `src/aiweather/eval/` | Metrics & evaluation |
| `configs/` | YAML experiment configs |
| `scripts/` | Entry points (download data, train, evaluate) |
| `notebooks/` | Exploration – prefix with your initials, e.g. `jk_01_explore.ipynb` |
| `tests/` | Tests |
| `data/`, `outputs/` | Local only – gitignored |

## Workflow

- Never push directly to `main`; branch (`feature/<short-name>`) and open a PR.
- Get one review before merging.
- Add dependencies with `uv add <pkg>` and commit `pyproject.toml` + `uv.lock` together.
- Don't commit data, checkpoints or secrets (`.env`, `.cdsapirc`).
