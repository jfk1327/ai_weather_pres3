"""Dataset and DataLoaders for daily-max wind maps.

Data file: data/era5_ireland_wind_daily_max.nc (from download_era5.py)
  variables "gust", "speed": (time, latitude, longitude) = (14610, 32, 32), m/s, float32
Split by YEAR (consecutive days are correlated, so a random split would leak):
  train <= 2014, val 2015-2019, test 2020-2024

The whole dataset is ~60 MB per variable in memory (stored compressed on disk), so it fits in memory: no need to read from disk per item.

Self-test (run after implementing):
    python people/james/dataset.py   (from the repo root, or python dataset.py from people/james/)

Docs:
  - PyTorch tutorial "Datasets & DataLoaders": sections "Creating a Custom Dataset for your files"
    and "Preparing your data for training with DataLoaders"
    https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
  - torch.utils.data API, section "Map-style datasets" and the DataLoader parameters
    (batch_size, shuffle, drop_last, num_workers, pin_memory)
    https://docs.pytorch.org/docs/stable/data.html
  - xarray "Indexing and selecting data", section on .sel with slices of dates
    https://docs.xarray.dev/en/stable/user-guide/indexing.html
"""
import time
from pathlib import Path

import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

# repo root = the first parent folder containing data/, so this runs from any working directory
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
DATA_PATH = ROOT / "data" / "era5_ireland_wind_daily_max.nc"
SPLITS = {"train": (1985, 2014), "val": (2015, 2019), "test": (2020, 2024)}


# TODO(human): implement the three pieces below (marked TODO). Decide the normalisation
# (what statistic, global vs per-pixel, any transform first?) and justify it in a comment.
# Weigh: gust >= 0 and right-skewed; storms are the rare pixels you care most about;
# whatever you choose must be invertible so samples can be mapped back to m/s.

def load_split(variable, split, path=DATA_PATH):
    """Return the raw maps for one split as a float32 tensor of shape (N, 32, 32), in m/s.

    TODO: open the file, pick `variable`, select the years in SPLITS[split], convert to a tensor.
    """
    raise NotImplementedError


def compute_norm_stats(train_maps):
    """Compute normalisation statistics from the TRAINING maps only.

    Args:
        train_maps: (N_train, 32, 32) tensor in m/s.
    Returns:
        dict of tensors (your choice of keys) used by WindDataset to normalise and
        by `denormalise` to invert it.

    TODO: decide what to compute. Why must val/test never be used here?
    """
    raise NotImplementedError


def denormalise(x, stats):
    """Invert the normalisation: model space -> m/s. Shape-preserving.

    TODO: needed later to turn decoded samples back into m/s for the tail metrics.
    """
    raise NotImplementedError


class WindDataset(Dataset):
    """Map-style dataset: item i is one day's normalised map.

    __getitem__ must return a float32 tensor of shape (1, 32, 32): (channels, H, W).
    Think about why the channel dimension is needed for Conv2d.

    TODO: implement __init__ (store normalised data), __len__, __getitem__.
    """

    def __init__(self, maps, stats):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError


def make_loaders(variable="gust", batch_size=64):
    """Build train/val/test DataLoaders that share the TRAINING statistics.

    Returns:
        (train_loader, val_loader, test_loader, stats)

    TODO: which loaders should shuffle, and why? Does drop_last matter here?
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Self-test: checks the contract the model and training loop rely on. Don't edit; make it pass.
    train_dl, val_dl, test_dl, stats = make_loaders("gust", batch_size=64)

    sizes = {name: len(dl.dataset) for name, dl in zip(SPLITS, (train_dl, val_dl, test_dl))}
    print("days per split:", sizes)
    assert sizes == {"train": 10957, "val": 1826, "test": 1827}, "split by year is wrong"

    x = next(iter(train_dl))
    assert isinstance(x, torch.Tensor), f"a batch should be a tensor, got {type(x)}"
    assert x.shape == (64, 1, 32, 32), f"batch shape {tuple(x.shape)}, expected (64, 1, 32, 32)"
    assert x.dtype == torch.float32, f"dtype {x.dtype}, expected float32"
    assert torch.isfinite(x).all(), "NaN or inf in batch"

    raw = load_split("gust", "train")[:64]
    back = denormalise(WindDataset(raw, stats).__getitem__(0), stats).squeeze(0)
    assert torch.allclose(back, raw[0], atol=1e-3), "denormalise(normalise(x)) != x"

    xs = torch.cat([b for b in train_dl])
    print(f"normalised train data: mean {xs.mean():.3f}, std {xs.std():.3f}, "
          f"min {xs.min():.2f}, max {xs.max():.2f}")

    t = time.time()
    for _ in train_dl:
        pass
    print(f"one pass over train loader: {time.time() - t:.2f}s ({len(train_dl)} batches)")
    print("all dataset checks passed")
