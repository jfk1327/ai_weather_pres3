"""Plot style shared by the slide figures and the notebook (light theme, validated categorical palette)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"real": INK, "real_test": MUTED, "zscore": "#2a78d6", "evt": "#eb6834"}
LABELS = {"zscore": "VAE, z-score (baseline)", "evt": "VAE, EVT margins (GPD tail)"}
SIGMA_STYLE = {0.5: ":", 1.0: "-", 1.3: "--", 1.6: "-."}

# sequential (wind speed): one hue, light -> dark; diverging (differences): blue <-> grey <-> red
WIND = LinearSegmentedColormap.from_list("wind", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#256abf",
                                                  "#184f95", "#0d366b", "#06203f"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#cde2fb", "#f0efec",
                                                    "#f7cccb", "#e34948", "#8f1d1d"])


def style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.family": "sans-serif"})


def wind_map(ax, field, lat, lon, vmin=0, vmax=50, mask=None, title=None, cmap=WIND, peak=True):
    """One 32x32 map, south at the bottom, with the Ireland outline and the peak value."""
    ext = [lon[0] - 0.125, lon[-1] + 0.125, lat[0] - 0.125, lat[-1] + 0.125]
    im = ax.imshow(field, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if mask is not None:
        from scipy.ndimage import binary_fill_holes          # outline the coast, not the loughs
        ax.contour(lon, lat, binary_fill_holes(mask).astype(float), levels=[0.5], colors=[INK2], linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="normal")
    if peak:
        ax.text(0.03, 0.04, f"max {np.nanmax(field):.0f} m/s", transform=ax.transAxes, fontsize=8.5,
                color=INK, bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1.5))
    return im


def exceedance(values, per_year=365.25):
    """Empirical return periods: sorted values (descending) and return period in years (Weibull positions)."""
    v = np.sort(values)[::-1]
    p = np.arange(1, len(v) + 1) / (len(v) + 1)      # P(X >= v_k)
    return v, 1 / (p * per_year)
