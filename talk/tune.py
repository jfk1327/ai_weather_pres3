"""Hyperparameter sweep on the VALIDATION years: latent size x beta, for both transforms.

Selection metric = bulk realism of sigma=1 samples (10-90% quantiles of pixel values and of the daily
domain max vs the validation years). Tails are excluded on purpose, so the extremes stay an honest test.

    python talk/tune.py            (~30-40 min on the RTX 5060)  ->  talk/results/tuning.csv
"""
import itertools
import time

import numpy as np
import pandas as pd

import wind as W

LATENTS, BETAS = [8, 16, 32, 64], [0.25, 0.5, 1.0, 2.0, 4.0]
N_SAMPLES = 14610


def main():
    data, lat, lon = W.load("gust")
    mask = W.ireland_mask(lat, lon)
    out = W.OUT / "results" / "tuning.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["zscore", "evt"]:
        tf, t = W.prepare(name, data)
        for L, beta in itertools.product(LATENTS, BETAS):
            t0 = time.time()
            model, hist = W.train(t, latent_dim=L, beta=beta, patience=15, seed=0)
            rec = W.reconstruct(model, tf, t["val"])
            real = data["val"][0]
            gen = W.sample(model, tf, N_SAMPLES, sigma=1.0, seed=1)
            pix, dmx = W.bulk_score(gen, real)
            mu, logvar = W.encode_mu(model, t["val"])
            kl_dim = (-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar))).mean(0)
            best = min(hist, key=lambda r: r["val_total"])
            rows.append(dict(
                transform=name, latent_dim=L, beta=beta, epochs=len(hist), best_epoch=best["epoch"],
                val_rmse_ms=W.rmse_ms(rec, real), val_rmse_ireland_ms=W.rmse_ms(rec, real, mask),
                val_kl=best["val_kl"], active_dims=int((kl_dim > 0.05).sum()),
                bulk_pixel_ms=pix, bulk_dmax_ms=dmx, bulk_score=pix + dmx,
                gen_dmax_q99=float(np.quantile(W.domain_max(gen), 0.99)),
                real_dmax_q99=float(np.quantile(W.domain_max(real), 0.99)),
                gen_min_ms=float(gen.min()), seconds=time.time() - t0))
            r = rows[-1]
            print(f"{name:6s} L={L:2d} beta={beta:4.2f}: rmse {r['val_rmse_ms']:.2f} m/s, bulk {r['bulk_score']:.2f}, "
                  f"active {r['active_dims']:2d}/{L}, q99 {r['gen_dmax_q99']:.1f}/{r['real_dmax_q99']:.1f}, "
                  f"{r['epochs']} ep, {r['seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
