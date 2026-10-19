"""Final comparison on the TEST years (2020-2024) + generation: figures for the talk and the tail table.

    python talk/evaluate.py    ->  talk/figures/*.png, talk/results/tail_table.csv, talk/results/summary.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import genpareto, norm

import plots as P
import wind as W

NAMES, SEEDS = ["zscore", "evt"], [0, 1, 2]
SIGMAS = [0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
N_GEN = 14610                      # 40 "years" of generated days per (model, sigma, seed)
N_LONG = 10 * N_GEN                # 400 years for the return-level curves (seed 0, sigma 1 and 1.3)
STORMS = {"Ciara": "2020-02-09", "Barra": "2021-12-07", "Eunice": "2022-02-18",
          "Agnes": "2023-09-27", "Isha": "2024-01-21"}
FIG, RES = W.OUT / "figures", W.OUT / "results"


def gen_stats(model, tf, n, sigma, seed, mask, chunk=N_GEN):
    """Domain max, Ireland max and min for n generated maps (in chunks, to bound memory)."""
    dmax, imax, mins = [], [], []
    for k, i in enumerate(range(0, n, chunk)):
        g = W.sample(model, tf, min(chunk, n - i), sigma, seed=1000 * seed + k)
        dmax.append(W.domain_max(g)); imax.append(W.domain_max(g, mask)); mins.append(g.min(axis=(1, 2)))
    return np.concatenate(dmax), np.concatenate(imax), np.concatenate(mins)


def main():
    P.style()
    FIG.mkdir(exist_ok=True)
    data, lat, lon = W.load("gust")
    mask = W.ireland_mask(lat, lon)
    train, test = data["train"][0], data["test"][0]
    dates = pd.to_datetime(data["test"][1])
    models = {(n, s): W.load_model(RES / "models" / f"{n}_s{s}.pt") for n in NAMES for s in SEEDS}
    xtest = {n: torch.from_numpy(models[(n, 0)][1].forward(test))[:, None].to(W.DEVICE) for n in NAMES}
    summary = {"config": {n: models[(n, 0)][2] for n in NAMES}}

    # ------------------------------------------------ 1. tuning landscape (validation)
    tun = pd.read_csv(RES / "tuning.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for r, n in enumerate(NAMES):
        sub = tun[tun["transform"] == n]
        for c, (col, lab) in enumerate([("bulk_score", "sample realism error (m/s, bulk 10–90%; lower = better)"),
                                        ("val_rmse_ms", "reconstruction RMSE (m/s; lower = better)")]):
            piv = sub.pivot(index="latent_dim", columns="beta", values=col)
            ax = axes[r, c]
            ax.imshow(piv.values, cmap=P.WIND, origin="lower", aspect="auto")
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    dark = v > np.nanpercentile(piv.values, 60)
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="#fcfcfb" if dark else P.INK)
            ax.set_xticks(range(piv.shape[1]), [f"{b:g}" for b in piv.columns])
            ax.set_yticks(range(piv.shape[0]), piv.index)
            ax.set_xlabel("β (KL weight)"); ax.set_ylabel("latent size")
            ax.set_title(f"{P.LABELS[n]}\n{lab}", fontsize=10)
            ax.grid(False)
            ch = models[(n, 0)][2]
            ax.add_patch(plt.Rectangle((list(piv.columns).index(ch["beta"]) - .5,
                                        list(piv.index).index(ch["latent_dim"]) - .5), 1, 1,
                                       fill=False, ec=P.COLORS["evt"] if n == "evt" else P.COLORS["zscore"], lw=2.5))
    fig.suptitle("Tuning on validation years (2015–2019): box = chosen configuration", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "01_tuning.png"); plt.close(fig)

    # ------------------------------------------------ 2. the GPD tail fit
    ev = models[("evt", 0)][1]
    xi = ev.xi.reshape(32, 32)
    p = int(np.argmax(train.reshape(len(train), -1).max(0) * mask.ravel()))   # windiest Ireland land pixel
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    im = P.wind_map(a, xi, lat, lon, vmin=-0.25, vmax=0.25, mask=mask, cmap=P.DIVERGE, peak=False,
                    title="GPD shape ξ per pixel (ξ < 0: bounded tail)")
    a.plot(lon[p % 32], lat[p // 32], "o", ms=8, mfc="none", mec=P.INK, mew=2)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    xs = train.reshape(len(train), -1)[:, p]
    v, T = P.exceedance(xs)
    b.plot(T, v, ".", color=P.INK, ms=4, label="ERA5 1985–2014 (one pixel, circled)")
    Tg = np.logspace(-1.5, 3, 200)
    sf = 1 / (Tg * 365.25)
    keep = sf < 1 - ev.q
    gpd = ev.thr[p] + genpareto.isf(sf[keep] / (1 - ev.q), ev.xi[p], 0, ev.sigma[p])
    b.plot(Tg[keep], gpd, color=P.COLORS["evt"], label=f"GPD tail fit (ξ = {ev.xi[p]:.2f})")
    b.plot(Tg, norm.isf(sf, xs.mean(), xs.std()), color=P.MUTED, ls="--", label="Gaussian fit")
    b.axhline(ev.thr[p], color=P.AXIS, lw=1); b.text(Tg[-1], ev.thr[p] - 0.4, "threshold (95th percentile)", ha="right", va="top",
                                                    fontsize=8.5, color=P.INK2)
    b.set_xscale("log"); b.set_xlabel("return period (years)"); b.set_ylabel("daily max gust (m/s)")
    b.set_title("Tail of one pixel: GPD vs Gaussian"); b.legend(loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "02_gpd_fit.png"); plt.close(fig)

    # ------------------------------------------------ 3. reconstruction: MSE blurs the peaks
    recon = {n: W.reconstruct(models[(n, 0)][0], models[(n, 0)][1], xtest[n]) for n in NAMES}
    real_peak = W.domain_max(test, mask)
    top = real_peak >= np.quantile(real_peak, 0.99)
    rec_rows = []
    for n in NAMES:
        for s in SEEDS:
            m, tf = models[(n, s)][:2]
            r = recon[n] if s == 0 else W.reconstruct(m, tf, torch.from_numpy(tf.forward(test))[:, None].to(W.DEVICE))
            pk = W.domain_max(r, mask)
            rec_rows.append(dict(model=n, seed=s, rmse_ms=W.rmse_ms(r, test), rmse_ireland_ms=W.rmse_ms(r, test, mask),
                                 peak_bias_all_ms=float((pk - real_peak).mean()),
                                 peak_bias_top1pct_ms=float((pk - real_peak)[top].mean())))
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(RES / "reconstruction_test.csv", index=False)
    ra = rec_df.drop(columns="seed").groupby("model").agg(["mean", "std"]).round(3)
    ra.columns = [f"{a}_{b}" for a, b in ra.columns]
    summary["reconstruction_test"] = ra.to_dict(orient="index")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    lim = [10, real_peak.max() + 3]
    for ax, n in zip(axes, NAMES):
        pk = W.domain_max(recon[n], mask)
        ax.plot(lim, lim, color=P.AXIS, lw=1)
        ax.plot(real_peak, pk, ".", color=P.COLORS[n], ms=3, alpha=0.5)
        ax.plot(real_peak[top], pk[top], "o", color=P.COLORS[n], ms=5, mec=P.SURFACE, mew=0.8)
        bias = (pk - real_peak)[top].mean()
        ax.set_title(f"{P.LABELS[n]}\nwindiest 1% of days: peak {bias:+.1f} m/s", fontsize=10)
        ax.set_xlabel("real peak gust over Ireland (m/s)"); ax.set_xlim(lim); ax.set_ylim(lim)
    axes[0].set_ylabel("reconstructed peak (m/s)")
    fig.suptitle("Test years 2020–2024: reconstructions flatten the peaks", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "03_reconstruction_peaks.png"); plt.close(fig)

    # ------------------------------------------------ 4. named storms (test years)
    storm_idx = {}
    for name, d in STORMS.items():
        win = np.where(np.abs((dates - pd.Timestamp(d)).days) <= 1)[0]
        storm_idx[name] = int(win[np.argmax(real_peak[win])])
    fig, axes = plt.subplots(3, len(STORMS), figsize=(2.6 * len(STORMS), 8.2))
    vmax = 45
    for j, (name, i) in enumerate(storm_idx.items()):
        im = P.wind_map(axes[0, j], test[i], lat, lon, vmin=10, vmax=vmax, mask=mask,
                        title=f"{name}\n{dates[i]:%d %b %Y}")
        for r, n in enumerate(NAMES, start=1):
            P.wind_map(axes[r, j], recon[n][i], lat, lon, vmin=10, vmax=vmax, mask=mask)
    for ax, lab in zip(axes[:, 0], ["ERA5 (real)", "z-score VAE", "EVT VAE"]):
        ax.text(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="daily max gust (m/s)")
    fig.suptitle("Named storms in the test years: real vs reconstructed", fontweight="bold")
    fig.savefig(FIG / "04_storms.png"); plt.close(fig)
    summary["storms"] = {k: dict(date=str(dates[i].date()), real_ireland_max=float(real_peak[i]),
                                 **{n: float(W.domain_max(recon[n][i:i + 1], mask)[0]) for n in NAMES})
                         for k, i in storm_idx.items()}

    # ------------------------------------------------ 5. generation statistics over sigma and seeds
    rows = []
    for src, x in [("ERA5 train 1985–2014", train), ("ERA5 test 2020–2024", test)]:
        dm, im_ = W.domain_max(x), W.domain_max(x, mask)
        rows.append(dict(source=src, sigma=np.nan, seed=np.nan, dmax=dm, imax=im_, mins=x.min(axis=(1, 2))))
    for (n, s), (m, tf, _, _) in models.items():
        for sg in SIGMAS:
            dm, im_, mn = gen_stats(m, tf, N_GEN, sg, s, mask)
            rows.append(dict(source=n, sigma=sg, seed=s, dmax=dm, imax=im_, mins=mn))
    q = [0.5, 0.9, 0.99, 0.999]
    tab = pd.DataFrame([dict(source=r["source"], sigma=r["sigma"], seed=r["seed"],
                             **{f"dmax_q{int(qq*1000)/10:g}": np.quantile(r["dmax"], qq) for qq in q},
                             dmax_max=r["dmax"].max(),
                             **{f"ireland_q{int(qq*1000)/10:g}": np.quantile(r["imax"], qq) for qq in q},
                             ireland_max=r["imax"].max(),
                             pct_maps_negative=100 * (r["mins"] < 0).mean(),
                             pct_maps_above_train_max=100 * (r["dmax"] > train.max()).mean()) for r in rows])
    agg = tab.groupby(["source", "sigma"], dropna=False).agg(["mean", "std"]).drop(columns="seed")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.round(2).to_csv(RES / "tail_table.csv", index=False)

    # sigma at which the generated q99 of Ireland max matches the training climate, and what that costs the median
    ref = tab[tab.source.str.startswith("ERA5 train")].iloc[0]
    summary["matched_sigma"] = {}
    for n in NAMES:
        a = agg[agg.source == n].sort_values("sigma")
        for key in ["ireland_q99", "ireland_q99.9"]:
            s_star = float(np.interp(ref[key], a[f"{key}_mean"], a.sigma))
            med = float(np.interp(s_star, a.sigma, a["ireland_q50_mean"]))
            summary["matched_sigma"][f"{n}_{key}"] = dict(sigma=round(s_star, 3), median_at_sigma=round(med, 2),
                                                          real_median=round(float(ref["ireland_q50"]), 2))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True)
    for ax, key, lab in zip(axes, ["ireland_q50", "ireland_q99", "ireland_q99.9"],
                            ["median day", "1-in-100 days", "1-in-1000 days"]):
        ax.axhline(ref[key], color=P.INK, lw=1.5)
        ax.text(SIGMAS[0], ref[key], " ERA5 1985–2014", va="bottom", fontsize=8.5, color=P.INK)
        for n in NAMES:
            a = agg[agg.source == n].sort_values("sigma")
            ax.fill_between(a.sigma, a[f"{key}_mean"] - a[f"{key}_std"], a[f"{key}_mean"] + a[f"{key}_std"],
                            color=P.COLORS[n], alpha=0.18, lw=0)
            ax.plot(a.sigma, a[f"{key}_mean"], "o-", color=P.COLORS[n], ms=4, label=P.LABELS[n])
        ax.set_title(f"peak gust over Ireland: {lab}"); ax.set_xlabel("sampling σ  (z ~ N(0, σ²I))")
    axes[0].set_ylabel("m/s"); axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("σ-scaling raises the extremes, but it raises the typical day too", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "06_sigma_quantiles.png"); plt.close(fig)

    # ------------------------------------------------ 6. return-level curves (400 generated years, seed 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    long = {}
    for n in NAMES:
        m, tf = models[(n, 0)][:2]
        for sg in [1.0, 1.3]:
            long[(n, sg)] = gen_stats(m, tf, N_LONG, sg, 7, mask)
    for ax, k, lab in zip(axes, [0, 1], ["domain max (incl. ocean)", "max over Ireland (land)"]):
        for src, x, col, ls in [("ERA5 1985–2014", train, P.INK, "-"), ("ERA5 2020–2024", test, P.MUTED, "-")]:
            vals = W.domain_max(x) if k == 0 else W.domain_max(x, mask)
            v, T = P.exceedance(vals)
            ax.plot(T, v, color=col, lw=2.5 if col == P.INK else 1.5, ls=ls, label=src)
        for n in NAMES:
            for sg in [1.0, 1.3]:
                v, T = P.exceedance(long[(n, sg)][k])
                ax.plot(T, v, color=P.COLORS[n], ls=P.SIGMA_STYLE[sg], label=f"{P.LABELS[n]}, σ={sg:g}")
        ax.set_xscale("log"); ax.set_xlim(0.05, 400)
        ax.set_xlabel("return period (years; 1 year = 365 generated days)")
        ax.set_ylabel("daily max gust (m/s)"); ax.set_title(lab)
        ax.set_ylim(np.quantile(W.domain_max(train) if k == 0 else W.domain_max(train, mask), 0.8) - 2, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Return levels: how often does each model produce a given peak?", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "05_return_levels.png"); plt.close(fig)
    summary["long_run_max"] = {f"{n}_sigma{sg:g}": dict(domain=float(long[(n, sg)][0].max()),
                                                         ireland=float(long[(n, sg)][1].max()))
                               for (n, sg) in long}

    # ------------------------------------------------ 7. the most extreme generated maps vs the most extreme real days
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 8.2))
    ti = np.argsort(W.domain_max(train, mask))[::-1][:4]
    tdates = pd.to_datetime(data["train"][1])
    for j, i in enumerate(ti):
        im = P.wind_map(axes[0, j], train[i], lat, lon, vmin=10, vmax=55, mask=mask, title=f"{tdates[i]:%d %b %Y}")
    for r, n in enumerate(NAMES, start=1):
        m, tf = models[(n, 0)][:2]
        g = W.sample(m, tf, N_GEN, 1.3, seed=11)
        gi = np.argsort(W.domain_max(g, mask))[::-1][:4]
        for j, i in enumerate(gi):
            P.wind_map(axes[r, j], g[i], lat, lon, vmin=10, vmax=55, mask=mask)
    for ax, lab in zip(axes[:, 0], ["ERA5 top 4 days\n(1985–2014)", "z-score VAE\ntop 4 of 40 yrs, σ=1.3",
                                    "EVT VAE\ntop 4 of 40 yrs, σ=1.3"]):
        ax.text(-0.08, 0.5, lab, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="daily max gust (m/s)")
    fig.suptitle("The most extreme maps: real storms vs generated", fontweight="bold")
    fig.savefig(FIG / "07_extreme_samples.png"); plt.close(fig)

    # ------------------------------------------------ 8. where storms live in latent space
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for n in NAMES:
        m = models[(n, 0)][0]
        mu, _ = W.encode_mu(m, xtest[n])
        r = np.linalg.norm(mu, axis=1)
        axes[0].plot(real_peak, r, ".", color=P.COLORS[n], ms=3, alpha=0.45, label=P.LABELS[n])
        L = m.latent_dim
        summary.setdefault("latent", {})[n] = dict(corr_norm_peak=float(np.corrcoef(real_peak, r)[0, 1]),
                                                    norm_median=float(np.median(r)),
                                                    norm_top1pct=float(np.median(r[top])), latent_dim=L)
        axes[1].hist(r[top], bins=12, density=True, color=P.COLORS[n], alpha=0.35, lw=0,
                     label=f"‖μ‖ of the windiest 1% of test days ({'z-score' if n == 'zscore' else 'EVT'}, d={L})")
    for L in sorted({models[(n, 0)][0].latent_dim for n in NAMES}):
        for sg in [1.0, 1.3]:
            zz = sg * np.sqrt(np.random.default_rng(0).chisquare(L, 50000))
            axes[1].hist(zz, bins=80, histtype="step", color=P.INK2, ls=P.SIGMA_STYLE[sg], lw=1.6, density=True,
                         label=f"‖z‖ drawn at σ={sg:g} (d={L})")
    axes[0].set_xlabel("real peak gust over Ireland (m/s)"); axes[0].set_ylabel("‖μ‖ (encoder mean)")
    axes[0].set_title("Test days: windier days sit further from the origin"); axes[0].legend(fontsize=9)
    axes[1].set_xlabel("distance from origin"); axes[1].set_title("Where σ-sampling draws vs where storms live")
    axes[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "08_latent_norms.png"); plt.close(fig)

    # ------------------------------------------------ 9. per-pixel 99th percentile: real vs generated (sigma = 1)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4))
    q99_real = np.quantile(train, 0.99, axis=0)
    P.wind_map(axes[0, 0], q99_real, lat, lon, vmin=15, vmax=40, mask=mask, title="ERA5 1985–2014", peak=False)
    axes[1, 0].axis("off")
    for j, n in enumerate(NAMES, start=1):
        m, tf = models[(n, 0)][:2]
        g = W.sample(m, tf, N_GEN, 1.0, seed=5)
        q99 = np.quantile(g, 0.99, axis=0)
        im1 = P.wind_map(axes[0, j], q99, lat, lon, vmin=15, vmax=40, mask=mask, title=f"{P.LABELS[n]}, σ=1",
                         peak=False)
        im2 = P.wind_map(axes[1, j], q99 - q99_real, lat, lon, vmin=-6, vmax=6, mask=mask, cmap=P.DIVERGE,
                         title=f"difference (mean {np.mean(q99 - q99_real):+.1f} m/s)", peak=False)
    fig.colorbar(im1, ax=axes[0], fraction=0.02, pad=0.01, label="99th percentile gust (m/s)")
    fig.colorbar(im2, ax=axes[1], fraction=0.02, pad=0.01, label="generated − real (m/s)")
    fig.suptitle("Per-pixel 1-in-100-day gust: where does each model fall short?", fontweight="bold")
    fig.savefig(FIG / "09_q99_maps.png"); plt.close(fig)

    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))
    cols = ["source", "sigma", "ireland_q50_mean", "ireland_q99_mean", "ireland_q99.9_mean", "ireland_max_mean",
            "pct_maps_negative_mean", "pct_maps_above_train_max_mean"]
    print(agg[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
