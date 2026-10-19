"""Shared code for the talk: data, the two marginal transforms, the VAE, training, sampling, metrics.

Two models, identical except for the space the VAE is trained in:
  zscore: x -> (x - mean) / std                      (one global mean/std from train; linear)
  evt:    x -> Phi^-1(F(x)), F per pixel = empirical CDF below the 95th percentile,
          Generalised Pareto (GPD) above it          (non-linear; tail extrapolates past the training max)
Both are exactly invertible, so samples are always compared in m/s.

Splits by year: train 1985-2014, val 2015-2019 (all tuning), test 2020-2024 (final numbers only).
"""
import copy
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.special import ndtr, ndtri
from scipy.stats import genpareto
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "era5_ireland_wind_daily_max.nc"
OUT = ROOT / "talk"
SPLITS = {"train": ("1985", "2014"), "val": ("2015", "2019"), "test": ("2020", "2024")}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def load(variable="gust"):
    """Return {split: (maps (N, 32, 32) float32 m/s, dates)} plus lat, lon."""
    with xr.open_dataset(DATA_PATH) as ds:
        da = ds[variable].load()
    out = {s: (da.sel(time=slice(a, b)).values.astype(np.float32),
               da.sel(time=slice(a, b)).time.values) for s, (a, b) in SPLITS.items()}
    return out, da.latitude.values, da.longitude.values


def ireland_mask(lat, lon):
    """(32, 32) bool: land points of the island of Ireland (Great Britain points removed)."""
    from global_land_mask import globe
    LO, LA = np.meshgrid(lon, lat)
    land = globe.is_land(LA, LO)
    return land & (LO < -5.4) & (LA > 51.3) & (LA < 55.45) & ~((LA > 54.9) & (LO > -6.2))


# ---------------------------------------------------------------- marginal transforms

class ZScore:
    name = "zscore"

    def fit(self, train):
        self.mean, self.std = float(train.mean()), float(train.std())
        return self

    def forward(self, x):
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        return (z * self.std + self.mean).astype(np.float32)


class EVTMargins:
    """Per-pixel semi-parametric CDF: empirical below quantile q, GPD above; then Phi^-1 to N(0, 1).

    Above the threshold u_p (pixel p's q-quantile) the excess Y = X - u_p is modelled as
        P(Y > y) = (1 + xi * y / sigma)^(-1/xi)          (Pickands-Balkema-de Haan)
    so F(x) = 1 - (1 - q) * (1 + xi (x - u_p) / sigma)^(-1/xi) for x > u_p.
    Days are treated as independent (no declustering): a known simplification.
    """
    name = "evt"

    def __init__(self, q=0.95, n_grid=2000):
        self.q, self.n_grid = q, n_grid

    def fit(self, train):
        X = train.reshape(len(train), -1).astype(np.float64)
        n, P = X.shape
        self.shape = train.shape[1:]
        probs = np.linspace(0.5 / n, self.q, self.n_grid)
        grid = np.quantile(X, probs, axis=0)                          # (n_grid, P)
        # anchor F(0) = 0 so calm values below the grid stay invertible (gust > 0; linear in between)
        self.probs = np.concatenate([[0.0], probs])
        grid = np.concatenate([np.zeros((1, P)), grid])
        self.grid = grid + np.arange(len(grid))[:, None] * 1e-7       # strictly increasing for interp
        self.thr = grid[-1]
        self.xi, self.sigma = np.zeros(P), np.zeros(P)
        for p in range(P):
            exc = X[X[:, p] > self.thr[p], p] - self.thr[p]
            self.xi[p], _, self.sigma[p] = genpareto.fit(exc, floc=0)
        return self

    def _tail_sf(self, x):        # P(X > x) for x above threshold, broadcast over pixels
        y = np.maximum(x - self.thr, 0)
        xi = np.where(np.abs(self.xi) < 1e-6, 1e-6, self.xi)
        base = np.maximum(1 + xi * y / self.sigma, 1e-300)
        return (1 - self.q) * base ** (-1 / xi)

    def forward(self, x):
        X = x.reshape(len(x), -1).astype(np.float64)
        u = np.empty_like(X)
        for p in range(X.shape[1]):
            u[:, p] = np.interp(X[:, p], self.grid[:, p], self.probs)
        tail = X > self.thr
        sf = self._tail_sf(X)
        z = np.where(tail, -ndtri(np.clip(sf, 1e-300, 1)), ndtri(np.clip(u, 1e-12, 1 - 1e-12)))
        return z.reshape(x.shape).astype(np.float32)

    def inverse(self, z):
        Z = z.reshape(len(z), -1).astype(np.float64)
        u, sf = ndtr(Z), ndtr(-Z)                       # sf computed directly: no 1 - u underflow
        X = np.empty_like(Z)
        for p in range(Z.shape[1]):
            X[:, p] = np.interp(u[:, p], self.probs, self.grid[:, p])
        r = np.clip(sf / (1 - self.q), 1e-300, 1)       # tail: invert sf = (1-q) * (1 + xi y / sigma)^(-1/xi)
        xi = np.where(np.abs(self.xi) < 1e-6, 1e-6, self.xi)
        x_tail = self.thr + self.sigma / xi * (r ** (-xi) - 1)
        X = np.where(u > self.q, x_tail, X)
        return X.reshape(z.shape).astype(np.float32)

    def upper_bound(self):
        """Per-pixel upper end point (finite only where xi < 0)."""
        return np.where(self.xi < 0, self.thr - self.sigma / self.xi, np.inf).reshape(self.shape)


TRANSFORMS = {"zscore": ZScore, "evt": EVTMargins}


# ---------------------------------------------------------------- model

class VAE(nn.Module):
    """Same architecture as notebooks/vae_minimal.ipynb: 32 -> 16 -> 8 -> 4, mirrored decoder."""

    def __init__(self, latent_dim=16, c=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.ReLU(),
            nn.Flatten())
        n = 4 * c * 16
        self.to_mu, self.to_logvar = nn.Linear(n, latent_dim), nn.Linear(n, latent_dim)
        self.from_z = nn.Linear(latent_dim, n)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (4 * c, 4, 4)), nn.ReLU(),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(2 * c, c, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(c, 1, 4, 2, 1))           # no output activation: both spaces are unbounded

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode(self, z):
        return self.decoder(self.from_z(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta):
    """Sum over pixels / latent dims, mean over the batch: beta = 1 is the (unit-variance Gaussian) ELBO."""
    n = x.shape[0]
    recon = ((x_hat - x) ** 2).sum() / n
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum() / n
    return recon + beta * kl, recon, kl


# ---------------------------------------------------------------- training

def prepare(transform_name, data):
    """Fit the transform on train; return (transform, {split: tensor (N, 1, 32, 32) on DEVICE})."""
    tf = TRANSFORMS[transform_name]().fit(data["train"][0])
    t = {s: torch.from_numpy(tf.forward(data[s][0]))[:, None].to(DEVICE) for s in data}
    return tf, t


def train(t, latent_dim=16, beta=1.0, lr=1e-3, batch_size=128, max_epochs=200, patience=20, seed=0,
          verbose=False):
    """Adam + early stopping on validation total loss. Returns (model at best epoch, history list)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VAE(latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xtr, xva = t["train"], t["val"]
    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr), device=DEVICE)
        sums = torch.zeros(3, device=DEVICE)
        for i in range(0, len(xtr), batch_size):
            xb = xtr[perm[i:i + batch_size]]
            loss, rec, kl = vae_loss(xb, *model(xb), beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += torch.stack([loss, rec, kl]).detach() * len(xb)
        tr = (sums / len(xtr)).tolist()
        va = evaluate_loss(model, xva, beta)
        hist.append(dict(epoch=epoch, train_total=tr[0], train_recon=tr[1], train_kl=tr[2],
                         val_total=va[0], val_recon=va[1], val_kl=va[2], seconds=time.time() - t0))
        if va[0] < best - 1e-4:
            best, best_state, wait = va[0], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}: train {tr[0]:.1f}  val {va[0]:.1f} (recon {va[1]:.1f}, KL {va[2]:.1f})")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def evaluate_loss(model, x, beta, batch=2048):
    model.eval()
    s = torch.zeros(3, device=DEVICE)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        loss, rec, kl = vae_loss(xb, *model(xb), beta)
        s += torch.stack([loss, rec, kl]) * len(xb)
    return (s / len(x)).tolist()


# ---------------------------------------------------------------- using a trained model

@torch.no_grad()
def encode_mu(model, x, batch=4096):
    out = [model.encode(x[i:i + batch]) for i in range(0, len(x), batch)]
    return torch.cat([m for m, _ in out]).cpu().numpy(), torch.cat([lv for _, lv in out]).cpu().numpy()


@torch.no_grad()
def reconstruct(model, tf, x, batch=4096):
    """Deterministic reconstruction decode(mu), mapped back to m/s: (N, 32, 32)."""
    out = [model.decode(model.encode(x[i:i + batch])[0]) for i in range(0, len(x), batch)]
    return tf.inverse(torch.cat(out)[:, 0].cpu().numpy())


@torch.no_grad()
def sample(model, tf, n, sigma=1.0, seed=0, batch=4096):
    """Decode z ~ N(0, sigma^2 I) and map back to m/s: (n, 32, 32)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    out = []
    for i in range(0, n, batch):
        z = sigma * torch.randn(min(batch, n - i), model.latent_dim, generator=g, device=DEVICE)
        out.append(model.decode(z)[:, 0].cpu().numpy())
    return tf.inverse(np.concatenate(out))


def save(path, model, tf, config, hist):
    """Weights + config + history in `path`; the fitted transform once per transform in the same folder
    (<name>_transform.npz), because it depends only on the training data, not on the seed."""
    path = Path(path)
    np.savez_compressed(path.parent / f"{tf.name}_transform.npz", **tf.__dict__)
    torch.save(dict(state=model.state_dict(), tf_name=tf.name, config=config, hist=hist), path)


def load_model(path):
    path = Path(path)
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = VAE(ck["config"]["latent_dim"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    tf = TRANSFORMS[ck["tf_name"]].__new__(TRANSFORMS[ck["tf_name"]])
    with np.load(path.parent / f"{ck['tf_name']}_transform.npz") as f:
        tf.__dict__.update({k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files})
    return model, tf, ck["config"], ck["hist"]


# ---------------------------------------------------------------- metrics

def domain_max(maps, mask=None):
    m = maps if mask is None else maps[:, mask]
    return m.reshape(len(maps), -1).max(1)


def bulk_score(gen, real, probs=np.arange(0.1, 0.91, 0.1)):
    """Mean |quantile difference| in m/s over the BULK (10-90%) of pixel values and of daily domain max.
    Deliberately excludes the tails, so tuning on it leaves the extremes as an honest evaluation."""
    pix = np.abs(np.quantile(gen, probs) - np.quantile(real, probs)).mean()
    dmx = np.abs(np.quantile(domain_max(gen), probs) - np.quantile(domain_max(real), probs)).mean()
    return float(pix), float(dmx)


def rmse_ms(a, b, mask=None):
    d = (a - b) if mask is None else (a - b)[:, mask]
    return float(np.sqrt((d ** 2).mean()))
