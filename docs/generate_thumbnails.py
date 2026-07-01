#!/usr/bin/env python3
"""Generate dedicated thumbnail images for all SFI gallery and benchmark demos.

Each thumbnail is a 400×280 single-panel visual with a dark transparent
background, designed to look good at small size on the Furo dark theme.

Usage:
    python docs/generate_thumbnails.py

Outputs into: docs/source/_static/thumbs/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── SFI palette ──────────────────────────────────────────────────────────
C = dict(
    blue="#3B9EFF",
    gold="#FFC20A",
    orange="#FF7A1A",
    purple="#5D3A9B",
    teal="#40B0A6",
    pink="#FF2D6F",
    blue2="#1A85FF",
    magenta="#D35FB7",
    grey="#808080",
    text="#D0D0D0",
    dim="#B0B0B0",
    bg="#131416",
)

OUTDIR = Path(__file__).resolve().parent / "source" / "_static" / "thumbs"
DPI = 150
SIZE = (400 / DPI, 280 / DPI)  # inches → 400×280 px


def _setup_ax(ax, spines=True):
    """Style a single axes for thumbnail use."""
    ax.set_facecolor("none")
    if not spines:
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for sp in ax.spines.values():
            sp.set_color(C["grey"])
        ax.tick_params(colors=C["dim"], labelsize=6)
        ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])


def _save(fig, name):
    fig.savefig(
        OUTDIR / f"{name}.png",
        dpi=DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"  ✓ {name}.png")


# ═══════════════════════════════════════════════════════════════════════════
#  GALLERY THUMBNAILS
# ═══════════════════════════════════════════════════════════════════════════

def thumb_ou_demo():
    """OU: 2D trajectory spiralling toward origin with arrow field."""
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    # Generate OU trajectory
    dt, N = 0.02, 800
    x = np.zeros((N, 2))
    for i in range(1, N):
        x[i] = x[i-1] - 1.0 * x[i-1] * dt + np.sqrt(2 * dt) * rng.standard_normal(2)
    ax.scatter(x[:, 0], x[:, 1], c=np.linspace(0, 1, N), cmap="cool",
               s=2, alpha=0.7, zorder=2)
    # Arrow field
    gx = np.linspace(-3, 3, 8)
    gy = np.linspace(-2, 2, 6)
    X, Y = np.meshgrid(gx, gy)
    ax.quiver(X, Y, -X, -Y, color=C["dim"], alpha=0.3, scale=30, width=0.005, zorder=1)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    _save(fig, "ou_demo")


def thumb_doublewell_demo():
    """Double-well: potential curve + particle cloud."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    x = np.linspace(-2, 2, 300)
    V = (x**2 - 1)**2
    ax.fill_between(x, V, alpha=0.15, color=C["blue"])
    ax.plot(x, V, color=C["blue"], lw=2.5)
    # Scatter particles in wells
    rng = np.random.default_rng(7)
    xp = np.concatenate([rng.normal(-1, 0.25, 80), rng.normal(1, 0.25, 80)])
    yp = (xp**2 - 1)**2 + rng.uniform(-0.1, 0.1, len(xp))
    ax.scatter(xp, yp, color=C["gold"], s=4, alpha=0.6, zorder=3)
    ax.set_ylim(-0.3, 2.5)
    _save(fig, "doublewell_demo")


def thumb_limitcycle_demo():
    """Limit cycle: spiral trajectory converging to circle."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    # Spiral out from center → limit cycle
    t = np.linspace(0, 25, 2000)
    r = 1.0 - 0.8 * np.exp(-0.3 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    ax.plot(x, y, color=C["teal"], lw=1.2, alpha=0.8)
    # Limit cycle circle
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=C["gold"], lw=2, ls="--", alpha=0.6)
    # Quiver arrows on the cycle
    n_arr = 8
    for i in range(n_arr):
        a = 2 * np.pi * i / n_arr
        ax.annotate("", xy=(1.05 * np.cos(a + 0.15), 1.05 * np.sin(a + 0.15)),
                     xytext=(1.05 * np.cos(a), 1.05 * np.sin(a)),
                     arrowprops=dict(arrowstyle="->", color=C["gold"], lw=1.5))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    _save(fig, "limitcycle_demo")


def thumb_lorenz_demo():
    """Lorenz attractor: butterfly projection."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    # Integrate Lorenz
    dt = 0.005
    N = 15000
    xyz = np.zeros((N, 3))
    xyz[0] = [1, 1, 1]
    sigma, rho, beta = 10, 28, 8/3
    for i in range(1, N):
        x, y, z = xyz[i-1]
        xyz[i] = xyz[i-1] + dt * np.array([sigma*(y-x), x*(rho-z)-y, x*y - beta*z])
    colors = np.linspace(0, 1, N)
    ax.scatter(xyz[:, 0], xyz[:, 2], c=colors, cmap="plasma", s=0.3, alpha=0.6, rasterized=True)
    ax.set_aspect(0.5)
    _save(fig, "lorenz_demo")


def thumb_sparsity_demo():
    """Sparsity: bar chart of coefficients (sparse selection)."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=True)
    rng = np.random.default_rng(12)
    n = 20
    coeffs = np.zeros(n)
    active = [2, 5, 7, 13]
    coeffs[active] = rng.uniform(0.3, 1.5, len(active)) * rng.choice([-1, 1], len(active))
    colors = [C["teal"] if i in active else C["grey"] for i in range(n)]
    alphas = [0.9 if i in active else 0.2 for i in range(n)]
    for i in range(n):
        ax.bar(i, coeffs[i], color=colors[i], alpha=alphas[i], width=0.8)
    ax.axhline(0, color=C["dim"], lw=0.5, alpha=0.5)
    ax.set_xlabel("basis index", fontsize=7, color=C["text"])
    ax.set_ylabel("coefficient", fontsize=7, color=C["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "sparsity_demo")


def thumb_experimental_workflow_demo():
    """Experimental workflow: noisy trajectory scatter."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(99)
    # Simulated "experimental" 2D trajectory with noise
    dt, N = 0.05, 600
    x = np.zeros((N, 2))
    for i in range(1, N):
        x[i] = x[i-1] - 0.5 * x[i-1] * dt + np.sqrt(2 * 0.4 * dt) * rng.standard_normal(2)
    noise = 0.15 * rng.standard_normal((N, 2))
    xn = x + noise
    ax.plot(xn[:, 0], xn[:, 1], color=C["blue"], lw=0.4, alpha=0.4)
    ax.scatter(xn[::3, 0], xn[::3, 1], color=C["blue"], s=3, alpha=0.7, zorder=2)
    # "Inferred" arrows
    gx = np.linspace(-2, 2, 5)
    gy = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(gx, gy)
    ax.quiver(X, Y, -0.5*X, -0.5*Y, color=C["gold"], alpha=0.6, scale=8, width=0.008, zorder=3)
    ax.set_aspect("equal")
    _save(fig, "experimental_workflow_demo")


def thumb_custom_basis_demo():
    """Custom basis: trajectory clouds in different trap geometries."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(55)
    colors = [C["blue"], C["orange"], C["teal"]]
    centers = [(-1.5, 0), (1.5, 0.5), (0, -1.2)]
    for c, center in zip(colors, centers):
        pts = rng.normal(0, 0.5, (150, 2)) + np.array(center)
        ax.scatter(pts[:, 0], pts[:, 1], color=c, s=3, alpha=0.5)
        ax.scatter(*center, color=c, s=40, marker="x", linewidths=2, zorder=3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    _save(fig, "custom_basis_demo")


def thumb_van_der_pol_demo():
    """Van der Pol: phase portrait with limit cycle."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    # Integrate VdP
    dt, mu = 0.01, 2.0
    N = 8000
    xy = np.zeros((N, 2))
    xy[0] = [0.1, 0.1]
    for i in range(1, N):
        x, y = xy[i-1]
        xy[i, 0] = x + y * dt
        xy[i, 1] = y + (mu * (1 - x**2) * y - x) * dt
    # Color by time
    ax.scatter(xy[:, 0], xy[:, 1], c=np.linspace(0, 1, N),
               cmap="cool", s=0.5, alpha=0.7)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-6, 6)
    ax.set_aspect(0.4)
    _save(fig, "van_der_pol_demo")


def thumb_benchmark_demo():
    """Underdamped benchmark: oscillating time series."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=True)
    t = np.linspace(0, 10, 500)
    x_true = np.exp(-0.3 * t) * np.cos(4 * t)
    rng = np.random.default_rng(11)
    x_noisy = x_true + 0.08 * rng.standard_normal(len(t))
    ax.plot(t, x_noisy, color=C["blue"], lw=0.8, alpha=0.6, label="data")
    ax.plot(t, x_true, color=C["gold"], lw=1.5, label="inferred")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("time", fontsize=7, color=C["text"])
    ax.legend(fontsize=6, frameon=False, labelcolor=C["text"])
    _save(fig, "benchmark_demo")


def thumb_nn_force_demo():
    """NN force: Müller-Brown potential contour landscape."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    # Simplified Müller-Brown-like potential
    x = np.linspace(-1.5, 1.2, 200)
    y = np.linspace(-0.5, 2.0, 200)
    X, Y = np.meshgrid(x, y)
    # Multi-well surface
    V = (3 * np.exp(-(X - 0.2)**2 - (Y - 0.5)**2)
         - 4 * np.exp(-0.5 * (X + 0.5)**2 - 2.0 * (Y - 1.5)**2)
         + 5 * np.exp(-(X + 0.8)**2 - (Y - 0.2)**2)
         - 3 * np.exp(-2 * (X - 0.8)**2 - 0.5 * (Y - 1.2)**2))
    ax.contourf(X, Y, V, levels=20, cmap="inferno", alpha=0.85)
    ax.contour(X, Y, V, levels=10, colors=C["dim"], linewidths=0.3, alpha=0.4)
    # Sample trajectory
    rng = np.random.default_rng(77)
    traj_x = np.cumsum(rng.normal(0, 0.03, 200)) - 0.5
    traj_y = np.cumsum(rng.normal(0, 0.03, 200)) + 0.8
    ax.plot(traj_x, traj_y, color=C["teal"], lw=1, alpha=0.8)
    ax.set_aspect("equal")
    _save(fig, "nn_force_demo")


def thumb_abp_align_demo():
    """Aligning ABPs: colored oriented particles."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(33)
    N = 60
    pos = rng.uniform(-2, 2, (N, 2))
    theta = rng.uniform(0, 2 * np.pi, N)
    # Color by angle
    colors = plt.cm.hsv(theta / (2 * np.pi))
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=25, alpha=0.8, edgecolors="none")
    # Arrows showing orientation
    dx = 0.3 * np.cos(theta)
    dy = 0.3 * np.sin(theta)
    ax.quiver(pos[:, 0], pos[:, 1], dx, dy, color=colors, scale=5,
              width=0.006, alpha=0.7, headwidth=3)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    _save(fig, "abp_align_demo")


def thumb_multi_experiment_demo():
    """Multi-experiment: colored particle clusters from different runs."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(44)
    exp_colors = [C["blue"], C["orange"], C["teal"], C["purple"]]
    for i, col in enumerate(exp_colors):
        cx = 2.5 * np.cos(2 * np.pi * i / len(exp_colors))
        cy = 2.5 * np.sin(2 * np.pi * i / len(exp_colors))
        N = 30
        pos = rng.normal(0, 0.6, (N, 2)) + np.array([cx, cy])
        theta = rng.uniform(0, 2 * np.pi, N)
        ax.scatter(pos[:, 0], pos[:, 1], color=col, s=15, alpha=0.7, edgecolors="none")
        dx = 0.25 * np.cos(theta)
        dy = 0.25 * np.sin(theta)
        ax.quiver(pos[:, 0], pos[:, 1], dx, dy, color=col, scale=8,
                  width=0.005, alpha=0.5, headwidth=3)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    _save(fig, "multi_experiment_demo")


def thumb_gray_scott_demo():
    """Gray-Scott: Turing-like pattern."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(88)
    # Generate a Turing-pattern-like image
    n = 120
    # Random Fourier modes
    kx = np.fft.fftfreq(n)[:, None]
    ky = np.fft.fftfreq(n)[None, :]
    k2 = kx**2 + ky**2
    # Band-pass filter for Turing-like spots
    k_target = 0.15
    envelope = np.exp(-((np.sqrt(k2) - k_target)**2) / (2 * 0.02**2))
    phases = rng.uniform(0, 2 * np.pi, (n, n))
    field = np.real(np.fft.ifft2(envelope * np.exp(1j * phases)))
    field = (field - field.min()) / (field.max() - field.min())
    ax.imshow(field, cmap="inferno", aspect="auto", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, "gray_scott_demo")


def thumb_wet_active_nematic_demo():
    """Wet active nematic: HSV director field texture."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    rng = np.random.default_rng(66)
    n = 100
    # Smooth random angle field ∈ [0, π)
    raw = rng.standard_normal((n, n))
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(raw, sigma=8)
    angle = (np.arctan2(np.sin(2 * smooth), np.cos(2 * smooth)) + np.pi) / (2 * np.pi)
    # HSV: H = angle, S = 0.8, V = 0.9
    hsv = np.zeros((n, n, 3))
    hsv[:, :, 0] = angle
    hsv[:, :, 1] = 0.8
    hsv[:, :, 2] = 0.85
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    ax.imshow(rgb, aspect="auto", interpolation="bilinear")
    # Overlay defect markers
    ax.scatter([25, 70], [40, 65], marker="+", s=60, color="white", linewidths=1.5, zorder=3)
    ax.scatter([50, 20], [25, 75], marker="_", s=60, color="white", linewidths=1.5, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, "wet_active_nematic_demo")


# ═══════════════════════════════════════════════════════════════════════════
#  BENCHMARK THUMBNAILS
# ═══════════════════════════════════════════════════════════════════════════

def _bench_thumbnail(name, system_func, title_short):
    """Create a benchmark thumbnail: system visual + convergence inset."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=False)
    system_func(ax)
    # Add convergence inset in bottom-right
    inset = fig.add_axes([0.62, 0.08, 0.34, 0.35])
    inset.set_facecolor("none")
    rng = np.random.default_rng(42)
    lengths = np.array([100, 300, 1000, 3000, 10000])
    nmse = 0.5 / np.sqrt(lengths / 100) + 0.01 * rng.standard_normal(len(lengths))
    nmse = np.clip(nmse, 0.01, 1)
    inset.loglog(lengths, nmse, "o-", color=C["gold"], ms=3, lw=1.2)
    inset.set_xlabel("N", fontsize=5, color=C["dim"])
    inset.set_ylabel("NMSE", fontsize=5, color=C["dim"])
    inset.tick_params(labelsize=4, colors=C["dim"])
    for sp in inset.spines.values():
        sp.set_color(C["grey"])
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    _save(fig, name)


def thumb_lorenz_bench():
    def draw(ax):
        dt, N = 0.005, 10000
        xyz = np.zeros((N, 3)); xyz[0] = [1, 1, 1]
        s, r, b = 10, 28, 8/3
        for i in range(1, N):
            x, y, z = xyz[i-1]
            xyz[i] = xyz[i-1] + dt * np.array([s*(y-x), x*(r-z)-y, x*y - b*z])
        ax.scatter(xyz[:, 0], xyz[:, 2], c=np.linspace(0,1,N), cmap="plasma", s=0.2, alpha=0.5, rasterized=True)
        ax.set_aspect(0.5)
    _bench_thumbnail("lorenz_bench", draw, "Lorenz")


def thumb_doublewell_bench():
    def draw(ax):
        x = np.linspace(-2, 2, 300)
        V = (x**2 - 1)**2
        ax.fill_between(x, V, alpha=0.15, color=C["blue"])
        ax.plot(x, V, color=C["blue"], lw=2)
        ax.set_ylim(-0.3, 2.5)
    _bench_thumbnail("doublewell_bench", draw, "DW")


def thumb_harmonic_bench():
    def draw(ax):
        t = np.linspace(0, 10, 500)
        x = np.exp(-0.3 * t) * np.cos(5 * t)
        ax.plot(t, x, color=C["teal"], lw=1.5)
        ax.axhline(0, color=C["grey"], lw=0.5, alpha=0.3)
    _bench_thumbnail("harmonic_bench", draw, "Harmonic")


def thumb_vdp_bench():
    def draw(ax):
        dt, mu = 0.01, 2.0
        N = 6000
        xy = np.zeros((N, 2)); xy[0] = [0.1, 0.1]
        for i in range(1, N):
            x, y = xy[i-1]
            xy[i, 0] = x + y * dt
            xy[i, 1] = y + (mu * (1 - x**2) * y - x) * dt
        ax.scatter(xy[:, 0], xy[:, 1], c=np.linspace(0,1,N), cmap="cool", s=0.3, alpha=0.6)
        ax.set_aspect(0.4)
    _bench_thumbnail("vdp_bench", draw, "VdP")


def thumb_abp_bench():
    def draw(ax):
        rng = np.random.default_rng(33)
        N = 40
        pos = rng.uniform(-2, 2, (N, 2))
        theta = rng.uniform(0, 2*np.pi, N)
        colors = plt.cm.hsv(theta / (2*np.pi))
        ax.scatter(pos[:,0], pos[:,1], c=colors, s=20, alpha=0.8, edgecolors="none")
        dx = 0.3 * np.cos(theta); dy = 0.3 * np.sin(theta)
        ax.quiver(pos[:,0], pos[:,1], dx, dy, color=colors, scale=5, width=0.006, alpha=0.7)
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.3, 2.3)
        ax.set_aspect("equal")
    _bench_thumbnail("abp_bench", draw, "ABP")


def thumb_nn_muller_brown_bench():
    def draw(ax):
        x = np.linspace(-1.5, 1.2, 150)
        y = np.linspace(-0.5, 2.0, 150)
        X, Y = np.meshgrid(x, y)
        V = (3*np.exp(-(X-0.2)**2-(Y-0.5)**2)
             -4*np.exp(-0.5*(X+0.5)**2-2*(Y-1.5)**2)
             +5*np.exp(-(X+0.8)**2-(Y-0.2)**2)
             -3*np.exp(-2*(X-0.8)**2-0.5*(Y-1.2)**2))
        ax.contourf(X, Y, V, levels=15, cmap="inferno", alpha=0.85)
        ax.set_aspect("equal")
    _bench_thumbnail("nn_muller_brown_bench", draw, "NN-MB")


def thumb_nn_optimizer_comparison():
    """Optimizer comparison: grouped bar chart."""
    fig, ax = plt.subplots(figsize=SIZE)
    _setup_ax(ax, spines=True)
    rng = np.random.default_rng(55)
    opts = ["Adam", "SGD", "L-BFGS", "RMSprop"]
    vals = [0.92, 0.75, 0.88, 0.82]
    cols = [C["blue"], C["orange"], C["teal"], C["purple"]]
    bars = ax.bar(range(len(opts)), vals, color=cols, alpha=0.85, width=0.7)
    ax.set_xticks(range(len(opts)))
    ax.set_xticklabels(opts, fontsize=6, color=C["text"])
    ax.set_ylabel("accuracy", fontsize=7, color=C["text"])
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "nn_optimizer_comparison")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating thumbnails → {OUTDIR}/")

    # Gallery
    print("\nGallery thumbnails:")
    thumb_ou_demo()
    thumb_doublewell_demo()
    thumb_limitcycle_demo()
    thumb_lorenz_demo()
    thumb_sparsity_demo()
    thumb_experimental_workflow_demo()
    thumb_custom_basis_demo()
    thumb_van_der_pol_demo()
    thumb_benchmark_demo()
    thumb_nn_force_demo()
    thumb_abp_align_demo()
    thumb_multi_experiment_demo()
    thumb_gray_scott_demo()
    thumb_wet_active_nematic_demo()

    # Benchmarks
    print("\nBenchmark thumbnails:")
    thumb_lorenz_bench()
    thumb_doublewell_bench()
    thumb_harmonic_bench()
    thumb_vdp_bench()
    thumb_abp_bench()
    thumb_nn_muller_brown_bench()
    thumb_nn_optimizer_comparison()

    print(f"\nDone — {14 + 7} thumbnails generated.")


if __name__ == "__main__":
    main()
