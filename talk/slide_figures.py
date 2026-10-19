"""Simple, large-font figures for the Google Slides deck (talk/figures/slides/).

    python talk/slide_figures.py
Also re-renders the evaluate.py figures without their in-figure titles (the slide title carries the message).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import plots as P
import wind as W

OUT = W.OUT / "figures" / "slides"
BIG = {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
       "legend.fontsize": 13}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def main():
    P.style()
    plt.rcParams.update(BIG)
    OUT.mkdir(parents=True, exist_ok=True)
    data, lat, lon = W.load("gust")
    mask = W.ireland_mask(lat, lon)
    train = data["train"][0]
    allmaps = np.concatenate([data[s][0] for s in data])
    alldates = pd.to_datetime(np.concatenate([data[s][1] for s in data]))

    # ---- speaker 1: the data
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = P.wind_map(axes[0], train.mean(0), lat, lon, vmin=10, vmax=20, mask=mask, peak=False,
                    title="average daily max gust, 1985–2014")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03, label="m/s")
    i = int(np.argmax(W.domain_max(allmaps)))
    im = P.wind_map(axes[1], allmaps[i], lat, lon, vmin=10, vmax=55, mask=mask,
                    title=f"windiest day: {alldates[i]:%d %b %Y}")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03, label="m/s")
    for ax in axes:
        ax.title.set_fontsize(15)
    fig.tight_layout()
    save(fig, "s_data_maps.png")

    years = alldates.year
    ann = pd.Series(W.domain_max(allmaps, mask)).groupby(years).max()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    spans = [("training\n1985–2014", 1985, 2014, "#e8eef7"), ("validation\n2015–19", 2015, 2019, "#d5e3f5"),
             ("test\n2020–24", 2020, 2024, "#f0efec")]
    for lab, a, b, col in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=col, lw=0)
        ax.text((a + b) / 2, ann.max() + 1.5, lab, ha="center", va="bottom", fontsize=14, color=P.INK)
    ax.bar(ann.index, ann.values, color=P.INK2, width=0.7)
    ax.set_ylim(25, ann.max() + 9)
    ax.set_ylabel("strongest gust over Ireland (m/s)")
    ax.set_xlim(1984.4, 2024.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "s_data_split.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = train.ravel()[::7]
    axes[0].hist(x, bins=70, color=P.INK2)
    axes[0].set_xlabel("gust (m/s)"); axes[0].set_title("before: metres per second")
    z = (x - train.mean()) / train.std()
    axes[1].hist(z, bins=70, color=P.COLORS["zscore"])
    axes[1].set_xlabel("standardised gust"); axes[1].set_title("after: (gust − 14.8) / 5.8")
    for ax in axes:
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
    axes[1].axvline(0, color=P.INK, lw=1)
    fig.tight_layout()
    save(fig, "s_data_norm.png")

    # ---- speaker 2: training
    m, tf, cfg, hist = W.load_model(W.OUT / "results" / "models" / "zscore_s0.pt")
    xt = W.torch.from_numpy(tf.forward(data["test"][0]))[:, None].to(W.DEVICE)
    rec = W.reconstruct(m, tf, xt)
    k = int(np.argmax(W.domain_max(data["test"][0], mask)))
    fig = plt.figure(figsize=(12, 4.2))
    a1 = fig.add_axes([0.0, 0.08, 0.26, 0.8]); a3 = fig.add_axes([0.74, 0.08, 0.26, 0.8])
    P.wind_map(a1, data["test"][0][k], lat, lon, vmin=10, vmax=45, mask=mask, peak=False, title="real map (1,024 numbers)")
    P.wind_map(a3, rec[k], lat, lon, vmin=10, vmax=45, mask=mask, peak=False, title="rebuilt map")
    a2 = fig.add_axes([0.26, 0.0, 0.48, 1.0]); a2.axis("off"); a2.set_xlim(0, 10); a2.set_ylim(0, 5)
    for xc, lab in [(1.6, "encoder"), (8.4, "decoder")]:
        a2.add_patch(FancyBboxPatch((xc - 1.1, 1.7), 2.2, 1.4, boxstyle="round,pad=0.05", fc="#e8eef7", ec="none"))
        a2.text(xc, 2.4, lab, ha="center", va="center", fontsize=16, color=P.INK)
    for j in range(8):
        a2.add_patch(plt.Rectangle((4.0 + j * 0.25, 2.0), 0.2, 0.8, fc=P.COLORS["zscore"], ec="none"))
    a2.text(5.0, 1.4, "8 numbers", ha="center", fontsize=15, color=P.INK)
    a2.text(5.0, 3.3, "the code", ha="center", fontsize=13, color=P.INK2)
    for x0, x1 in [(0.1, 0.45), (2.75, 3.85), (6.15, 7.25), (9.55, 9.95)]:
        a2.add_patch(FancyArrowPatch((x0, 2.4), (x1, 2.4), arrowstyle="-|>", mutation_scale=18, color=P.INK2, lw=1.8))
    save(fig, "s_vae_diagram.png")

    h = pd.DataFrame(hist)
    to_ms = lambda r: np.sqrt(r / 1024) * tf.std
    best = int(h.val_total.idxmin())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(h.epoch, to_ms(h.train_recon), color=P.INK2, label="training years")
    axes[0].plot(h.epoch, to_ms(h.val_recon), color=P.COLORS["zscore"], label="validation years")
    axes[0].set_ylim(0.9, 2.0); axes[0].set_ylabel("rebuild error (m/s per pixel)")
    axes[0].set_title("rebuild error")
    axes[1].plot(h.epoch, h.train_kl, color=P.INK2, label="training years")
    axes[1].plot(h.epoch, h.val_kl, color=P.COLORS["zscore"], label="validation years")
    axes[1].set_title("KL: how far codes stray from the bell curve")
    for ax in axes:
        ax.axvline(h.epoch[best], color=P.INK, lw=1, ls="--")
        ax.set_xlabel("epoch (one pass over 30 years)")
    axes[0].text(h.epoch[best] + 1, 1.9, f"kept: epoch {h.epoch[best]}", fontsize=13, color=P.INK)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1, 0.85))
    fig.tight_layout()
    save(fig, "s_loss_curves.png")

    tun = pd.read_csv(W.OUT / "results" / "tuning.csv")
    sub = tun[tun["transform"] == "zscore"]
    ramp = {8: "#86b6ef", 16: "#3987e5", 32: "#1c5cab", 64: "#0d366b"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for L, g in sub.groupby("latent_dim"):
        g = g.sort_values("beta")
        axes[0].plot(g.beta, 100 * g.active_dims / L, "o-", color=ramp[L], ms=7)
        axes[0].text(g.beta.iloc[-1] * 1.08, 100 * g.active_dims.iloc[-1] / L, f"{L} numbers", va="center",
                     fontsize=13, color=P.INK)
        axes[1].plot(g.beta, g.val_rmse_ms, "o-", color=ramp[L], ms=7)
        off = {8: 0, 16: 0, 32: 0.025, 64: -0.025}[L]
        axes[1].text(0.25 / 1.1, g.val_rmse_ms.iloc[0] + off, f"{L}", ha="right", va="center", fontsize=13, color=P.INK)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.set_xlabel("β: weight on the KL term")
    axes[0].set_xlim(0.2, 7); axes[1].set_xlim(0.15, 5)
    axes[0].set_ylabel("code numbers in use (%)"); axes[0].set_title("too much KL switches code numbers off")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("validation rebuild error (m/s)"); axes[1].set_title("…and makes rebuilds blurrier")
    axes[0].plot(1, 100, "o", ms=16, mfc="none", mec=P.INK, mew=2.5)
    axes[0].text(1, 92, "chosen", ha="center", va="top", fontsize=13, color=P.INK)
    fig.tight_layout()
    save(fig, "s_kl_collapse.png")

    # ---- speaker 3: why extremes (real return levels with named storms)
    test_dates = pd.to_datetime(data["test"][1])
    peak_all = W.domain_max(train, mask)
    v, T = P.exceedance(peak_all)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(T, v, color=P.INK, lw=2.5)
    ax.set_xscale("log"); ax.set_xlabel("happens about once every …"); ax.set_ylabel("strongest gust over Ireland (m/s)")
    ax.set_xlim(0.003, 40); ax.set_ylim(10, 50)
    ax.set_xticks([1 / 365.25, 10 / 365.25, 100 / 365.25, 1, 10, 30], ["day", "10 days", "100 days", "year", "10 years", "30 years"])
    ax.minorticks_off()
    for q, lab in [(0.5, "a typical day"), (0.99, "1 day in 100"), (0.999, "1 day in 1000")]:
        val = np.quantile(peak_all, q)
        Tq = 1 / ((1 - q) * 365.25) if q < 1 else None
        ax.plot(Tq if q != 0.5 else 1 / (0.5 * 365.25), val, "o", color=P.INK, ms=10, mec=P.SURFACE, mew=1.5)
        ax.text((Tq if q != 0.5 else 1 / (0.5 * 365.25)) * 1.25, val - 1.2, f"{lab}: {val:.0f} m/s", fontsize=14, color=P.INK)
    fig.tight_layout()
    save(fig, "s_extremes_real.png")


def return_levels_ireland():
    """The key result as one simple panel: Ireland land max, real vs both models at sigma 1 and 1.3."""
    import evaluate as E
    data, lat, lon = W.load("gust")
    mask = W.ireland_mask(lat, lon)
    fig, ax = plt.subplots(figsize=(11, 5))
    v, T = P.exceedance(W.domain_max(data["train"][0], mask))
    ax.plot(T, v, color=P.INK, lw=3.5, label="real (ERA5 1985–2014)")
    for n in ["zscore", "evt"]:
        m, tf, _, _ = W.load_model(W.OUT / "results" / "models" / f"{n}_s0.pt")
        for sg in [1.0, 1.3]:
            v, T = P.exceedance(E.gen_stats(m, tf, E.N_LONG, sg, 7, mask)[1])
            name = "z-score VAE" if n == "zscore" else "EVT VAE"
            ax.plot(T, v, color=P.COLORS[n], ls=P.SIGMA_STYLE[sg], lw=2.5, label=f"{name}, σ = {sg:g}")
    ax.set_xscale("log"); ax.set_xlim(0.05, 400); ax.set_ylim(24, 54)
    ax.set_xticks([0.1, 1, 10, 100], ["month", "year", "10 years", "100 years"]); ax.minorticks_off()
    ax.set_xlabel("happens about once every …"); ax.set_ylabel("strongest gust over Ireland (m/s)")
    ax.legend(loc="upper left", fontsize=13)
    fig.tight_layout()
    save(fig, "s_return_levels_ireland.png")


def untitled_results():
    """Re-render evaluate.py's figures into figures/slides/ without in-figure titles (results are deterministic)."""
    import matplotlib.figure
    import evaluate as E
    matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
    E.FIG = OUT
    E.main()


if __name__ == "__main__":
    main()
    return_levels_ireland()
    untitled_results()
