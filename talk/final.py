"""Train the chosen configuration for each transform with 3 seeds and save the models.

One SHARED configuration for both transforms (so any difference is due to the transform, not the size):
the (latent_dim, beta) with the lowest bulk_score summed over the two transforms in talk/results/tuning.csv
(validation years only), among runs with no collapsed latent space (>= half the latent dims active).
results/seed_noise.csv checks the top 3 candidates with 2 more seeds: the choice (d=8, beta=1) holds.

    python talk/final.py     ->  talk/results/models/{zscore,evt}_s{0,1,2}.pt, talk/results/chosen.csv
"""
import pandas as pd

import wind as W

SEEDS = [0, 1, 2]


def choose(tuning):
    ok = tuning[tuning.active_dims >= tuning.latent_dim / 2]
    both = ok.groupby(["latent_dim", "beta"]).filter(lambda g: len(g) == 2)
    L, beta = both.groupby(["latent_dim", "beta"]).bulk_score.sum().idxmin()
    return ok[(ok.latent_dim == L) & (ok.beta == beta)].set_index("transform")


def main():
    res = W.OUT / "results"
    chosen = choose(pd.read_csv(res / "tuning.csv"))
    chosen.to_csv(res / "chosen.csv")
    print(chosen[["latent_dim", "beta", "val_rmse_ms", "bulk_score", "active_dims"]])
    (res / "models").mkdir(exist_ok=True)
    data, _, _ = W.load("gust")
    for name, row in chosen.iterrows():
        tf, t = W.prepare(name, data)
        cfg = dict(latent_dim=int(row.latent_dim), beta=float(row.beta))
        for seed in SEEDS:
            model, hist = W.train(t, **cfg, patience=15, seed=seed)
            W.save(res / "models" / f"{name}_s{seed}.pt", model, tf, dict(cfg, seed=seed), hist)
            print(f"{name} seed {seed}: {len(hist)} epochs, best val total "
                  f"{min(h['val_total'] for h in hist):.1f}", flush=True)


if __name__ == "__main__":
    main()
