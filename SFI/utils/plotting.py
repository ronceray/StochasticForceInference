from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.stats import pearsonr

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset

""" Utility classes for plotting the results of SFI, used only to
display the results of the example files.

"""

# ---------------------------------------------------------------------------
# Gallery colour palette
# ---------------------------------------------------------------------------

#: Named colour palette for consistent plotting across gallery examples
#: and user notebooks.  Adjusted for readability on both dark (#131416)
#: and light backgrounds.
SFI_COLORS = dict(
    data="#3B9EFF",  # bright blue
    inferred="#FFC20A",  # gold
    exact="#FF7A1A",  # bright orange
    bootstrap="#5D3A9B",  # purple
    highlight="#40B0A6",  # teal
    error="#FF2D6F",  # bright pink
    secondary="#1A85FF",  # lighter blue
    tertiary="#D35FB7",  # pink/magenta
)


# ---------------------------------------------------------------------------
# Figure timestamp
# ---------------------------------------------------------------------------


def stamp_output():
    """Print a discreet generation timestamp to stdout.

    Call once per script so the timestamp appears in terminal output
    blocks on gallery pages and in notebook cells.
    """
    print(f"[Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}]")


def stamp_fig(fig=None):
    """Add a discreet generation timestamp to the bottom-right of a figure.

    Useful for tracking when a cached or gallery figure was last rendered.
    Idempotent: a figure is only stamped once.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to stamp.  Defaults to ``plt.gcf()``.
    """
    if fig is None:
        fig = plt.gcf()
    if getattr(fig, "_sfi_stamped", False):
        return
    fig._sfi_stamped = True
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.005,
        now,
        fontsize=5,
        color="#808080",
        alpha=0.4,
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )


# ---------------------------------------------------------------------------
# Dark-theme figure helpers (reusable across gallery demos & notebooks)
# ---------------------------------------------------------------------------


def dark_ax(ax):
    """Style a matplotlib Axes for a dark background.

    Sets black face colour, white ticks and labels, and dark-grey spines.
    """
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("0.3")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def dark_fig(nrows=1, ncols=1, **kw):
    """Create a figure and axes with a black background.

    All arguments are forwarded to ``plt.subplots``.
    Returns ``(fig, axes)`` like ``plt.subplots``.
    """
    fig, axes = plt.subplots(nrows, ncols, **kw)
    fig.patch.set_facecolor("black")
    for ax in np.atleast_1d(axes).flat if hasattr(axes, "flat") else [axes]:
        dark_ax(ax)
    return fig, axes


def wrap_positions(X, box):
    """Wrap positions into a periodic box ``[0, box_i)`` per axis.

    Parameters
    ----------
    X : array_like, shape (..., d)
        Position array.
    box : array_like, shape (d,) or (2,)
        Box dimensions.

    Returns
    -------
    X_wrapped : ndarray
        Copy of *X* with each column wrapped modulo the corresponding
        box dimension.
    """
    X = np.array(X, copy=True, dtype=float)
    box = np.asarray(box)
    for i in range(len(box)):
        X[..., i] = X[..., i] % box[i]
    return X


def _equal_aspect(ax):
    """Set an equal aspect ratio, choosing an adjustable that is legal here.

    ``adjustable="datalim"`` (and ``axis("equal")``) raise at draw time when
    the axes shares its x or y with a sibling (e.g. ``sharex=True`` subplots),
    so fall back to ``adjustable="box"`` in that case.
    """
    shared = (
        len(ax.get_shared_x_axes().get_siblings(ax)) > 1
        or len(ax.get_shared_y_axes().get_siblings(ax)) > 1
    )
    ax.set_aspect("equal", adjustable="box" if shared else "datalim")


# ---------------------------------------------------------------------------
# Sparse-model diagnostics
# ---------------------------------------------------------------------------


def plot_pareto_front(result, *, criteria=("PASTIS", "BIC", "AIC"), ax=None):
    """Plot the Pareto front: information gain vs model size, with IC optima.

    Parameters
    ----------
    result : SparsityResult
        As returned by ``inf.sparsify_force()``.
    criteria : tuple of str
        Information-criterion names to mark on the plot.
    ax : matplotlib Axes, optional
        If *None*, a new figure is created.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    infos = result.best_info_by_k
    ks = list(range(len(infos)))
    valid = [(k, float(info)) for k, info in zip(ks, infos) if float(info) > -1e30]
    if valid:
        ks_v, infos_v = zip(*valid)
    else:
        ks_v, infos_v = [], []

    ax.plot(ks_v, infos_v, ".-", color="#B0B0B0", lw=1.5, label="Pareto front")

    colors = [SFI_COLORS["inferred"], SFI_COLORS["exact"], SFI_COLORS["highlight"]]
    for ic_name, c in zip(criteria, colors):
        try:
            k_sel, _, score, _ = result.select_by_ic(ic_name, p_param=1e-3)
            info_at_k = float(infos[k_sel]) if k_sel < len(infos) else None
            if info_at_k is not None and info_at_k > -1e30:
                ax.axvline(k_sel, color=c, ls="--", alpha=0.7, label=f"{ic_name} (k={k_sel})")
                ax.plot(k_sel, info_at_k, "o", color=c, ms=8, zorder=5)
        except Exception:  # ic_name absent or result schema mismatch
            pass

    ax.set_xlabel("Model size k")
    ax.set_ylabel("Information gain")
    ax.legend(fontsize=8)
    ax.set_title("Sparse model selection")
    return ax


def plot_recovery_bar(
    coeffs_inferred,
    support_inferred,
    *,
    coeffs_true=None,
    support_true=None,
    labels=None,
    stderr=None,
    yscale: str = "linear",
    sort: bool = False,
    show_pruned: bool = False,
    ax=None,
):
    """Bar chart of inferred sparse coefficients vs ground truth.

    Parameters
    ----------
    coeffs_inferred : array_like
        Coefficient values for the inferred support.
    support_inferred : array_like
        Indices of the selected basis functions.
    coeffs_true, support_true : array_like, optional
        Ground-truth coefficients / support (paired comparison).
    labels : list of str, optional
        Tick labels for basis functions.
    stderr : array_like, optional
        Standard errors for the inferred coefficients (drawn as error caps).
    yscale : str
        Matplotlib y-scale (``"linear"`` or ``"log"``; use ``"log"`` for
        magnitude bars with widely varying scales).
    sort : bool
        If True, order bars by descending ``|coefficient|``.
    show_pruned : bool
        If True (and ``labels`` given), append faded zero-bars for basis
        functions outside the inferred support.
    ax : matplotlib Axes, optional
        If *None*, a new figure is created.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    coeffs_inferred = np.asarray(coeffs_inferred, dtype=float)
    support_inferred = list(support_inferred)
    stderr = np.asarray(stderr, dtype=float) if stderr is not None else None

    if sort:
        order = np.argsort(-np.abs(coeffs_inferred))
        coeffs_inferred = coeffs_inferred[order]
        support_inferred = [support_inferred[k] for k in order]
        if stderr is not None:
            stderr = stderr[order]

    n = len(support_inferred)
    x = np.arange(n)
    bar_width = min(0.6, 6.0 / n) if n >= 5 else 0.8
    paired = coeffs_true is not None and support_true is not None
    offset = bar_width / 2 if paired else 0.0

    ax.bar(
        x - offset,
        coeffs_inferred,
        width=bar_width,
        color=SFI_COLORS["inferred"],
        alpha=0.8,
        yerr=stderr,
        capsize=4 if stderr is not None else 0,
        ecolor="#B0B0B0",
        label="inferred",
    )

    if paired:
        support_true_l = list(support_true)
        true_mapped = np.zeros(n)
        for i, s in enumerate(support_inferred):
            if s in support_true_l:
                true_mapped[i] = float(np.asarray(coeffs_true)[support_true_l.index(s)])
        ax.bar(
            x + offset,
            true_mapped,
            width=bar_width,
            fill=False,
            edgecolor=SFI_COLORS["exact"],
            lw=2,
            label="exact",
        )

    extra_ticks, extra_labels = [], []
    if show_pruned and labels is not None:
        pruned = [k for k in range(len(labels)) if k not in set(support_inferred)]
        for off, s in enumerate(pruned):
            xp = n + off
            ax.bar(xp, 0.0, width=bar_width, color="#808080", alpha=0.3)
            extra_ticks.append(xp)
            extra_labels.append(labels[s])

    if labels is not None:
        tick_labels = [labels[s] if s < len(labels) else str(s) for s in support_inferred] + extra_labels
        ax.set_xticks(list(x) + extra_ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8 if n < 8 else 7)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in support_inferred])

    if n < 5 and not extra_ticks:
        ax.set_xlim(-0.8, n - 0.2)

    ax.set_yscale(yscale)
    ax.set_ylabel("Coefficient value")
    ax.legend()
    ax.set_title("Sparse model coefficients")
    return ax


def plot_recovery_bar_multi(coeffs_list, labels, *, coeffs_true=None, group_names=None, ax=None):
    """Grouped bar chart comparing coefficients across several models.

    ``coeffs_list`` is a list of coefficient vectors (one per regime /
    solver), each aligned to ``labels``.  An optional ``coeffs_true`` is
    overlaid as a step reference.
    """
    if ax is None:
        _, ax = plt.subplots()
    coeffs_list = [np.asarray(c, dtype=float) for c in coeffs_list]
    G = len(coeffs_list)
    m = len(labels)
    x = np.arange(m)
    group_names = group_names if group_names is not None else [f"model {i + 1}" for i in range(G)]
    width = 0.8 / max(G, 1)
    palette = list(SFI_COLORS.values())
    for gi, (c, name) in enumerate(zip(coeffs_list, group_names)):
        ax.bar(x + (gi - (G - 1) / 2) * width, c, width=width, color=palette[gi % len(palette)], alpha=0.85, label=name)
    if coeffs_true is not None:
        ct = np.asarray(coeffs_true, dtype=float)
        ax.step(np.concatenate([x - 0.5, [x[-1] + 0.5]]), np.concatenate([ct, ct[-1:]]),
                where="post", color=SFI_COLORS["exact"], lw=1.5, label="exact")
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=45, ha="right", fontsize=8 if m < 8 else 7)
    ax.set_ylabel("Coefficient value")
    ax.axhline(0, color="#808080", lw=0.5)
    ax.legend(fontsize=8)
    return ax


def plot_recovery_matrix(true, inferred, *, row_labels=None, col_labels=None, cmap="RdBu_r", vmax=None, axes=None):
    """Side-by-side ``imshow`` of a true vs inferred parameter matrix."""
    true = np.asarray(true, dtype=float)
    inferred = np.asarray(inferred, dtype=float)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8, 4))
    if vmax is None:
        vmax = float(max(np.abs(true).max(), np.abs(inferred).max()))
    im = None
    for ax, mat, title in zip(axes, [true, inferred], ["True", "Inferred"]):
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        if col_labels is not None:
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        if row_labels is not None:
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels(row_labels, fontsize=8)
    if im is not None:
        plt.colorbar(im, ax=list(axes), shrink=0.8)
    return axes


def _collection_to_arrays(
    coll: TrajectoryCollection,
    *,
    dataset: int = 0,
):
    """
    Extract (t, X, mask) from a TrajectoryCollection as NumPy arrays.

    Parameters
    ----------
    coll :
        TrajectoryCollection object.
    dataset :
        Dataset index inside the collection.

    Returns
    -------
    t : ndarray, shape (T,)
        Absolute times.
    X : ndarray, shape (T, N, d)
        State array.
    mask : ndarray, shape (T, N)
        Boolean validity mask.
    """
    if not isinstance(coll, TrajectoryCollection):
        raise TypeError(f"Expected TrajectoryCollection, got {type(coll)!r}")
    if not coll.datasets:
        raise ValueError("Empty TrajectoryCollection")

    if not (0 <= dataset < len(coll.datasets)):
        raise IndexError(f"dataset index {dataset} out of range for D={len(coll.datasets)}")

    ds: TrajectoryDataset = coll.datasets[dataset]

    # Positions, always as (T, N, d)
    X = np.asarray(ds._X3d())

    # Mask
    try:
        M = np.asarray(ds._M2d())
    except Exception:
        M = np.ones(X.shape[:2], dtype=bool)

    T = X.shape[0]

    # Time axis
    if ds.t is not None:
        t = np.asarray(ds.t)
    else:
        dt = ds.dt
        if dt is None:
            t = np.arange(T, dtype=float)
        else:
            dt_arr = np.asarray(dt)
            if dt_arr.ndim == 0:
                t = np.arange(T, dtype=float) * float(dt_arr)
            elif dt_arr.ndim == 1:
                if T == 0:
                    t = np.zeros((0,), dtype=float)
                else:
                    # t[0] = 0; t[1:] = cumsum(dt[:-1])
                    t = np.concatenate([[0.0], np.cumsum(dt_arr[:-1])])
            else:
                raise ValueError("dt must be scalar or (T,) to build a time axis.")

    return t, X, M


def axisvector(index, dim):
    """d-dimensional unit vector pointing in direction `index`."""
    e = np.zeros(dim, dtype=float)
    e[index] = 1.0
    return e


def comparison_scatter(
    Xexact,
    Xinferred,
    error=None,
    maxpoints=10000,
    vmax=None,
    color=None,
    alpha=0.05,
    y=0.8,
    mode="both",
    fontsize=9,
):
    """This method is used to compare inferred components to the
    exact ones along the trajectory, in a graphical way.

    Xexact, Xinferred: jnp arrays (Nsteps,...); must have the same shape.

    error: predicted standard deviation for X_inferred.

    maxpoints: if Nsteps / 2 * maxpoints, data will be subsampled.

    """

    subsample = max(1, Xexact.shape[0] // maxpoints)
    # Flatten the data:
    Xe = np.array(Xexact)[::subsample].reshape(-1)
    Xi = np.array(Xinferred)[::subsample].reshape(-1)

    MSE = sum((Xe - Xi) ** 2) / sum(Xe**2 + Xi**2)

    if vmax is None:
        vmax = max(abs(Xe).max(), abs(Xi).max())
    plt.scatter(Xe, Xi, alpha=alpha, linewidth=0, c=color)

    if error is not None:
        xvals = np.array([-vmax, vmax])
        confidence_interval = 2 * error**0.5 * Xi.std()
        plt.plot(xvals, xvals + confidence_interval, ls=":", color="#808080")
        plt.plot(xvals, xvals - confidence_interval, ls=":", color="#808080")
    (r, p) = pearsonr(Xe, Xi)
    plt.plot([-1e10, 1e10], [-1e10, 1e10], ls="-", color="#808080")
    plt.grid(True)
    _equal_aspect(plt.gca())
    plt.xlabel("exact")
    plt.ylabel("inferred")

    titlestring = ""
    if mode == "r" or mode == "both":
        titlestring += r"$r=" + str(round(r, 2 if r < 0.98 else 3 if r < 0.999 else 4 if r < 0.9999 else 5)) + "$"
    if mode == "both":
        titlestring += "\n"
    if mode == "MSE" or mode == "both":
        titlestring += "MSE=" + str(round(MSE, 3))
    plt.title(titlestring, loc="left", y=y, x=0.05, fontsize=fontsize)
    plt.xticks([0.0])
    plt.yticks([0.0])
    plt.xlim(-vmax, vmax)
    plt.ylim(-vmax, vmax)


def timeseries(
    coll: TrajectoryCollection,
    *,
    dims=None,
    dataset: int = 0,
    particles=None,
    transform=None,
    ax=None,
    **plot_kw,
):
    """
    Plot x[dim](t) for one or many particles from a TrajectoryCollection.

    By default, plot all dimensions for the selected particles.

    Parameters
    ----------
    coll :
        TrajectoryCollection containing the data.
    dims :
        Iterable of state dimensions to plot. If None, plot all dims.
    dataset :
        Dataset index inside the collection (default 0).
    particles :
        Iterable of particle indices to include. If None, include all particles.
    transform :
        Optional callable applied elementwise to the plotted values (e.g.
        ``np.exp`` to map a log-space coordinate back to population space).
    ax :
        Optional Matplotlib Axes. If None, use ``plt.gca()``.
    **plot_kw :
        Forwarded to ``ax.plot``.

    Returns
    -------
    ax :
        Matplotlib Axes used.
    """
    if ax is None:
        ax = plt.gca()

    t, X, M = _collection_to_arrays(coll, dataset=dataset)  # X: (T, N, D)
    T, N, D = X.shape

    # Select particles
    if particles is None:
        Xp = X
        Mp = M
        Nsel = N
    else:
        idx = np.asarray(particles, dtype=int)
        Xp = X[:, idx, :]
        Mp = M[:, idx]
        Nsel = Xp.shape[1]

    # Select dims
    if dims is None:
        dims = range(D)
    dims = list(dims)

    # Mask-aware plotting: one line per particle per dimension
    for n in range(Nsel):
        mask_n = Mp[:, n].astype(bool)
        for d in dims:
            vals = Xp[mask_n, n, d]
            if transform is not None:
                vals = transform(vals)
            ax.plot(t[mask_n], vals, **plot_kw)

    ax.set_xlabel("t")
    if len(dims) == 1:
        ax.set_ylabel(f"x[{dims[0]}]")
    else:
        ax.set_ylabel("x[d]")
    return ax


def phase2d(
    coll: TrajectoryCollection,
    *,
    dataset: int = 0,
    dims=None,
    dir1=None,
    dir2=None,
    shift=(0.0, 0.0),
    tmin=None,
    tmax=None,
    particles=None,
    cmap="viridis",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    plot_colorbar: bool = False,
    box=None,
    transform=None,
    color=None,
    ax=None,
    drop_masked: bool = True,
) -> LineCollection:
    """
    2D phase-space plot with connected line segments colored along the trajectory.

    Parameters
    ----------
    coll :
        TrajectoryCollection containing the data.
    dataset :
        Dataset index inside the collection.
    dims :
        Pair of coordinate indices (i, j) to plot. Ignored if ``dir1``/``dir2``
        are provided. If None and no directions are given, defaults to (0, 1).
    dir1, dir2 :
        Optional projection directions in R^d. If given, positions are projected
        onto these directions instead of using coordinate axes.
    shift :
        (xshift, yshift) added to all positions.
    tmin, tmax :
        Integer time-index bounds. If None, full range is used.
        Negative ``tmax`` is interpreted as from the end.
    particles :
        Sequence of particle indices to include. If None, include all.
    cmap :
        Colormap for the time-coloring.
    linewidth :
        Line width of the trajectory segments.
    alpha :
        Global alpha for the line collection.
    plot_colorbar :
        If True, add a colorbar for the time coloring.
    box :
        Optional periodic box ``(Lx, Ly)``. If given, positions are wrapped
        modulo the box and segments that cross a boundary are dropped, so
        trajectories render correctly in a periodic domain.
    transform :
        Optional callable applied elementwise to the projected ``x`` and
        ``y`` (e.g. ``np.exp`` to map a log-space coordinate back to
        population space). Applied after projection/shift/box-wrap.
    color :
        Optional single color. If given, draw the trajectory in this solid
        color instead of the time-colored gradient (incompatible with
        ``plot_colorbar``).
    ax :
        Optional Matplotlib Axes. If None, use ``plt.gca()``.
    drop_masked :
        If True, drop segments where either endpoint is masked.

    Returns
    -------
    lc :
        The created :class:`matplotlib.collections.LineCollection`.
    """
    if ax is None:
        ax = plt.gca()

    t, X, M = _collection_to_arrays(coll, dataset=dataset)  # X: (T, N, d)
    T, N, d = X.shape

    # Time window
    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = T
    if tmax < 0:
        tmax = T + tmax
    tmin = max(0, int(tmin))
    tmax = min(T, int(tmax))
    if tmax <= tmin:
        raise ValueError("Empty time window in phase2d")

    X = X[tmin:tmax]  # (T', N, d)
    M = M[tmin:tmax]  # (T', N)
    t_slice = t[tmin:tmax]  # (T',)

    # Particle selection
    if particles is None:
        X_sel = X
        M_sel = M
    else:
        idx = np.asarray(particles, dtype=int)
        X_sel = X[:, idx, :]
        M_sel = M[:, idx]

    # Projection
    if dir1 is not None or dir2 is not None:
        if dir1 is None or dir2 is None:
            raise ValueError("Either provide both dir1 and dir2, or neither.")
        dir1 = np.asarray(dir1, dtype=float)
        dir2 = np.asarray(dir2, dtype=float)
        if dir1.shape != (d,) or dir2.shape != (d,):
            raise ValueError(f"dir1/dir2 must have shape ({d},)")
        x = np.tensordot(X_sel, dir1, axes=([-1], [0]))  # (T', N_sel)
        y = np.tensordot(X_sel, dir2, axes=([-1], [0]))  # (T', N_sel)
    else:
        if dims is None:
            dims = (0, 1)
        i, j = dims
        if i >= d or j >= d:
            raise IndexError(f"dims {dims} out of range for state dimension d={d}")
        x = X_sel[..., i]
        y = X_sel[..., j]

    x = x + float(shift[0])
    y = y + float(shift[1])

    # Periodic wrap: fold into [0, L) and flag boundary-crossing segments.
    jump = None
    if box is not None:
        box = np.asarray(box, dtype=float)
        x = x % box[0]
        y = y % box[1]
        jump = (np.abs(np.diff(x, axis=0)) > 0.5 * box[0]) | (
            np.abs(np.diff(y, axis=0)) > 0.5 * box[1]
        )  # (T'-1, N_sel)

    if transform is not None:
        x = transform(x)
        y = transform(y)

    # Build segments
    XY = np.stack([x, y], axis=-1)  # (T', N_sel, 2)
    XY_start = XY[:-1]  # (T'-1, N_sel, 2)
    XY_end = XY[1:]  # (T'-1, N_sel, 2)
    seg_valid = M_sel[:-1] & M_sel[1:]  # (T'-1, N_sel)
    if jump is not None:
        seg_valid = seg_valid & ~jump

    segs = np.stack([XY_start, XY_end], axis=2)  # (T'-1, N_sel, 2, 2)
    segs_flat = segs.reshape(-1, 2, 2)
    valid_flat = seg_valid.reshape(-1)
    if drop_masked:
        segs_flat = segs_flat[valid_flat]

    if color is not None:
        if plot_colorbar:
            raise ValueError("phase2d: `color` and `plot_colorbar` are mutually exclusive.")
        lc = LineCollection(segs_flat, colors=color, linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
    else:
        # Color by mid-time
        t_mid = 0.5 * (t_slice[:-1] + t_slice[1:])  # (T'-1,)
        t_mid_2d = np.broadcast_to(t_mid[:, None], seg_valid.shape)  # (T'-1, N_sel)
        c_flat = t_mid_2d.reshape(-1)
        if drop_masked:
            c_flat = c_flat[valid_flat]

        norm = Normalize(vmin=float(c_flat.min()), vmax=float(c_flat.max()))
        lc = LineCollection(segs_flat, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        lc.set_array(c_flat)
        ax.add_collection(lc)

    # Limits
    xy_valid = XY.reshape(-1, 2)
    if drop_masked:
        mask_flat = M_sel.reshape(-1)
        xy_valid = xy_valid[mask_flat]
    if xy_valid.size:
        ax.set_xlim(xy_valid[:, 0].min(), xy_valid[:, 0].max())
        ax.set_ylim(xy_valid[:, 1].min(), xy_valid[:, 1].max())

    _equal_aspect(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if plot_colorbar:
        plt.colorbar(lc, ax=ax, label="t")

    return lc


def _time_window(T, tmin, tmax):
    """Resolve integer (tmin, tmax) bounds, with negative tmax from the end."""
    lo = 0 if tmin is None else max(0, int(tmin))
    if tmax is None:
        hi = T
    elif tmax < 0:
        hi = T + int(tmax)
    else:
        hi = min(T, int(tmax))
    if hi <= lo:
        raise ValueError("Empty time window.")
    return lo, hi


def phase2d_scalar(
    coll: TrajectoryCollection,
    *,
    color_fn,
    dataset: int = 0,
    dims=(0, 1),
    tmin=None,
    tmax=None,
    particles=None,
    cmap="plasma",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    plot_colorbar: bool = True,
    colorbar_label: str = "",
    ax=None,
    drop_masked: bool = True,
) -> LineCollection:
    """2D phase-space plot colored by a scalar field of the coordinate.

    Like :func:`phase2d`, but each segment is colored by
    ``color_fn(midpoint)`` rather than by time — e.g. the local diffusivity
    ``D(x)``, the speed, or a potential.  ``color_fn`` receives the
    **full d-dimensional** segment midpoints, shape ``(n_segments, d)``,
    and must return a scalar per segment, shape ``(n_segments,)``.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, d = X.shape
    lo, hi = _time_window(T, tmin, tmax)
    X, M = X[lo:hi], M[lo:hi]
    if particles is not None:
        idx = np.asarray(particles, dtype=int)
        X, M = X[:, idx], M[:, idx]
    i, j = dims
    XY = np.stack([X[..., i], X[..., j]], axis=-1)  # (T', N_sel, 2)
    Xmid = 0.5 * (X[:-1] + X[1:])  # (T'-1, N_sel, d)
    seg_valid = (M[:-1] & M[1:]).reshape(-1)
    segs = np.stack([XY[:-1], XY[1:]], axis=2).reshape(-1, 2, 2)
    cvals = np.asarray(color_fn(Xmid.reshape(-1, d))).reshape(-1)
    if drop_masked:
        segs, cvals = segs[seg_valid], cvals[seg_valid]
    norm = Normalize(vmin=float(cvals.min()), vmax=float(cvals.max()))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(cvals)
    ax.add_collection(lc)
    xy = XY.reshape(-1, 2)
    if drop_masked:
        xy = xy[M.reshape(-1)]
    if xy.size:
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    _equal_aspect(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if plot_colorbar:
        plt.colorbar(lc, ax=ax, label=colorbar_label)
    return lc


def timeseries_colored(
    coll: TrajectoryCollection,
    *,
    color_fn,
    dataset: int = 0,
    dims=None,
    particles=None,
    cmap="plasma",
    colorbar_label: str = "",
    plot_colorbar: bool = True,
    ax=None,
    s: float = 4.0,
    alpha: float = 1.0,
    rasterized: bool = True,
    **scatter_kw,
):
    """Plot ``x[dim](t)`` as a scatter colored by a scalar field.

    Mask-aware time series in which each point is colored by
    ``color_fn(X)`` — e.g. the local diffusivity along the trajectory.
    ``color_fn`` receives points of shape ``(n_points, d)`` and returns a
    scalar per point.  A single shared colorbar spans all series.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, D = X.shape
    if particles is not None:
        idx = np.asarray(particles, dtype=int)
        X, M = X[:, idx], M[:, idx]
    Nsel = X.shape[1]
    dims = range(D) if dims is None else list(dims)
    cvals = np.asarray(color_fn(X.reshape(-1, D))).reshape(T, Nsel)
    norm = Normalize(vmin=float(np.nanmin(cvals)), vmax=float(np.nanmax(cvals)))
    sm = None
    for n in range(Nsel):
        mask_n = M[:, n].astype(bool)
        for dd in dims:
            sm = ax.scatter(
                t[mask_n], X[mask_n, n, dd], c=cvals[mask_n, n], cmap=cmap, norm=norm,
                s=s, alpha=alpha, rasterized=rasterized, **scatter_kw,
            )
    ax.set_xlabel("t")
    ax.set_ylabel(f"x[{list(dims)[0]}]" if len(list(dims)) == 1 else "x[d]")
    if plot_colorbar and sm is not None:
        plt.colorbar(sm, ax=ax, label=colorbar_label)
    return ax


def phase3d(
    coll: TrajectoryCollection,
    *,
    dataset: int = 0,
    dims=(0, 1, 2),
    tmin=None,
    tmax=None,
    particles=None,
    cmap="viridis",
    linewidth: float = 1.5,
    alpha: float = 1.0,
    scatter_endpoints: bool = True,
    scatter_size: float = 30.0,
    ax=None,
    drop_masked: bool = True,
):
    """3D trajectory plot with segments colored along time.

    The 3D analog of :func:`phase2d`: draws each particle's path as a
    time-colored :class:`Line3DCollection`.  Pass a 3D axes via ``ax`` or a
    new one is created (``projection="3d"``).
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: WPS433

    if ax is None:
        ax = plt.gcf().add_subplot(111, projection="3d")
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, d = X.shape
    lo, hi = _time_window(T, tmin, tmax)
    X, M, tt = X[lo:hi], M[lo:hi], t[lo:hi]
    if particles is not None:
        idx = np.asarray(particles, dtype=int)
        X, M = X[:, idx], M[:, idx]
    i, j, k = dims
    norm = Normalize(vmin=float(tt.min()), vmax=float(tt.max()))
    allpts = []
    for n in range(X.shape[1]):
        m = M[:, n].astype(bool)
        pts = np.stack([X[m, n, i], X[m, n, j], X[m, n, k]], axis=-1)  # (Tm, 3)
        if len(pts) < 2:
            continue
        allpts.append(pts)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (Tm-1, 2, 3)
        t_mid = 0.5 * (tt[m][:-1] + tt[m][1:])
        lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        lc.set_array(t_mid)
        ax.add_collection3d(lc)
        if scatter_endpoints:
            ax.scatter(*pts[-1], s=scatter_size, color=plt.get_cmap(cmap)(1.0))
    if allpts:
        P = np.concatenate(allpts, axis=0)
        ax.set_xlim(P[:, 0].min(), P[:, 0].max())
        ax.set_ylim(P[:, 1].min(), P[:, 1].max())
        ax.set_zlim(P[:, 2].min(), P[:, 2].max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def trajectory_scatter(
    coll: TrajectoryCollection,
    *,
    dataset: int = 0,
    dims=(0, 1),
    particles=None,
    tmin=None,
    tmax=None,
    cmap=None,
    s: float = 2.0,
    alpha: float = 0.1,
    ax=None,
    drop_masked: bool = True,
    **scatter_kw,
):
    """All-frames density scatter cloud of a 2D projection.

    Unlike :func:`phase2d` (connected lines) or :func:`plot_particles`
    (one frame), this scatters every valid ``(particle, frame)`` position —
    useful for occupancy / home-range visualisations.  With ``cmap`` set,
    points are colored by time.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, d = X.shape
    lo, hi = _time_window(T, tmin, tmax)
    X, M, tt = X[lo:hi], M[lo:hi], t[lo:hi]
    if particles is not None:
        idx = np.asarray(particles, dtype=int)
        X, M = X[:, idx], M[:, idx]
    i, j = dims
    xf = X[..., i].reshape(-1)
    yf = X[..., j].reshape(-1)
    maskf = M.reshape(-1).astype(bool)
    cf = np.broadcast_to(tt[:, None], X.shape[:2]).reshape(-1) if cmap is not None else None
    if drop_masked:
        xf, yf = xf[maskf], yf[maskf]
        if cf is not None:
            cf = cf[maskf]
    if cf is not None:
        ax.scatter(xf, yf, c=cf, cmap=cmap, s=s, alpha=alpha, **scatter_kw)
    else:
        ax.scatter(xf, yf, color=SFI_COLORS["data"], s=s, alpha=alpha, **scatter_kw)
    _equal_aspect(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_field(
    coll: TrajectoryCollection,
    field,
    *,
    dataset: int = 0,
    dir1=None,
    dir2=None,
    center=None,
    N: int = 10,
    scale: float = 1.0,
    autoscale: bool = False,
    color="g",
    radius=None,
    positions=None,
    powernorm: float = 0.0,
    mask_unvisited: bool = False,
    clip_magnitude=None,
    **kwargs,
):
    """Plot a 2D vector field (or a 2D slice of a higher-dimensional field).

    Parameters
    ----------
    coll :
        TrajectoryCollection providing typical positions for scaling/centering.
    field :
        Callable ``field(X) -> F`` with X of shape (n_points, d) and F of same shape.
    dataset :
        Dataset index inside the collection used to estimate center/radius.
    dir1, dir2 :
        Projection directions in R^d. If None, use coordinate axes 0 and 1.
    center, N, scale, autoscale, color, radius, positions, powernorm, **kwargs :
        Grid / scaling controls.
    mask_unvisited :
        If True, drop arrows in grid cells with no trajectory data within one
        grid spacing (keeps the quiver legible in sparsely-sampled regions).
    clip_magnitude :
        If set, clip each arrow's magnitude to this value (long arrows in
        high-force regions no longer dominate the plot).
    """
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    d = X.shape[-1]

    if dir1 is None:
        dir1 = axisvector(0, d)
    if dir2 is None:
        dir2 = axisvector(1, d)
    if center is None:
        center = X.mean(axis=(0, 1))
    if radius is None:
        radius = 0.5 * (X.max(axis=(0, 1)) - X.min(axis=(0, 1))).max()

    if positions is None:
        positions = []
        for a in np.linspace(-radius, radius, N):
            for b in np.linspace(-radius, radius, N):
                positions.append(center + a * dir1 + b * dir2)

    gridX, gridY = [], []
    vX, vY = [], []
    for pos in positions:
        x = dir1.dot(pos)
        y = dir2.dot(pos)
        gridX.append(x)
        gridY.append(y)
        v = field(pos.reshape((1, d)))
        if powernorm != 0:
            v /= np.linalg.norm(v) ** powernorm
        vX.append(dir1.dot(v[0, :]))
        vY.append(dir2.dot(v[0, :]))

    vX = np.array(vX)
    vY = np.array(vY)

    if clip_magnitude is not None:
        mag = np.sqrt(vX**2 + vY**2)
        factor = np.where(mag > clip_magnitude, clip_magnitude / np.maximum(mag, 1e-12), 1.0)
        vX = vX * factor
        vY = vY * factor

    if autoscale:
        scale /= float(np.nanmax(np.sqrt(vX**2 + vY**2)))

    if mask_unvisited:
        data = X.reshape(-1, d)
        if data.shape[0] > 5000:
            data = data[:: max(1, data.shape[0] // 5000)]
        pos_arr = np.asarray([np.asarray(p) for p in positions])
        spacing = (2.0 * radius) / max(N - 1, 1)
        covered = np.zeros(len(pos_arr), dtype=bool)
        for ci in range(0, len(pos_arr), 256):
            chunk = pos_arr[ci : ci + 256]
            dmin = np.sqrt(((chunk[:, None, :] - data[None, :, :]) ** 2).sum(-1)).min(axis=1)
            covered[ci : ci + 256] = dmin <= spacing
        vX = np.where(covered, vX, np.nan)
        vY = np.where(covered, vY, np.nan)

    plt.quiver(
        gridX,
        gridY,
        scale * vX,
        scale * vY,
        scale=1.0,
        units="xy",
        color=color,
        minlength=0.0,
        **kwargs,
    )
    plt.ylim(-radius + dir2.dot(center), radius + dir2.dot(center))
    plt.xlim(-radius + dir1.dot(center), radius + dir1.dot(center))
    _equal_aspect(plt.gca())
    plt.xticks([])
    plt.yticks([])


def plot_tensor_field(
    coll: TrajectoryCollection,
    field,
    *,
    dataset: int = 0,
    center=None,
    N: int = 10,
    scale: float = 1.0,
    autoscale: bool = False,
    color="g",
    radius=None,
    positions=None,
    mode: str = "eigencross",
    **kwargs,
):
    """Plot a tensor field for 2D processes from a TrajectoryCollection.

    ``mode="eigencross"`` (default) draws each tensor as a pair of
    eigen-axis arrows; ``mode="ellipse"`` draws an eigen-aligned ellipse
    glyph (axis lengths ``∝ sqrt(eigenvalue)``), a clearer rendering for
    anisotropic diffusion fields.
    """
    if mode not in ("eigencross", "ellipse"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'eigencross' or 'ellipse'.")
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    d = X.shape[-1]
    if d != 2:
        raise ValueError(f"plot_tensor_field expects d=2, got d={d}")

    if center is None:
        center = X.mean(axis=(0, 1))
    if radius is None:
        radius = 0.5 * (X.max(axis=(0, 1)) - X.min(axis=(0, 1))).max()

    if positions is None:
        positions = []
        for a in np.linspace(-radius, radius, N):
            for b in np.linspace(-radius, radius, N):
                positions.append(center + np.array([a, b]))

    if mode == "ellipse":
        from matplotlib.patches import Ellipse

        ax = plt.gca()
        for pos in positions:
            tensor = np.asarray(field(pos.reshape((1, d)))).reshape(2, 2)
            w, Vv = np.linalg.eigh(0.5 * (tensor + tensor.T))
            w = np.clip(w, 1e-4, None)
            ang = np.degrees(np.arctan2(Vv[1, 1], Vv[0, 1]))
            ax.add_patch(
                Ellipse(
                    (pos[0], pos[1]),
                    width=scale * float(np.sqrt(w[1])),
                    height=scale * float(np.sqrt(w[0])),
                    angle=ang,
                    fill=False,
                    lw=1.2,
                    color=color,
                    **kwargs,
                )
            )
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        _equal_aspect(plt.gca())
        plt.xticks([])
        plt.yticks([])
        return

    Xp, Yp, U, V = [], [], [], []
    for pos in positions:
        posr = pos.reshape((1, d))
        tensor = field(posr)
        eigvals, eigvecs = np.linalg.eigh(tensor.reshape((2, 2)))
        for j in range(2):
            Xp.append(pos[0])
            Yp.append(pos[1])
            U.append(eigvals[j] * eigvecs[0, j])
            V.append(eigvals[j] * eigvecs[1, j])

    if autoscale:
        scale /= max(np.array(U) ** 2 + np.array(V) ** 2) ** 0.5

    Xp = np.array(Xp)
    Yp = np.array(Yp)
    dX = 0.5 * scale * np.array(U)
    dY = 0.5 * scale * np.array(V)

    plt.quiver(
        Xp - dX,
        Yp - dY,
        2 * dX,
        2 * dY,
        scale=1.0,
        units="xy",
        color=color,
        minlength=0.0,
        headwidth=1.0,
        headlength=0.0,
        **kwargs,
    )
    _equal_aspect(plt.gca())
    plt.xticks([])
    plt.yticks([])


def plot_profile_1d(
    coll: TrajectoryCollection,
    field,
    *,
    exact_field=None,
    dataset: int = 0,
    dim: int = 0,
    component=None,
    N: int = 200,
    ci=None,
    samples: bool = False,
    ax=None,
    margin: float = 0.05,
    label_exact: str = "Exact",
    label_inferred: str = "Inferred",
):
    """1D profile of an inferred field, optionally overlaid on the exact one.

    Evaluates ``field`` on a grid spanning the data range along ``dim`` and
    plots the inferred profile (gold), an optional exact overlay (orange
    dashes), an optional confidence band, and an optional sample histogram
    backdrop.  The 1D analog of :func:`plot_field` for forces ``F(x)`` and
    scalar diffusion profiles ``D(x)`` / ``D(v)``.

    Parameters
    ----------
    field, exact_field :
        Callables ``f(X) -> array`` (vector ``(N, d)`` or tensor
        ``(N, d, d)``).  ``exact_field`` is optional.
    dim :
        Coordinate axis to sweep.
    component :
        Which output component to plot.  Defaults to ``dim`` for a vector
        field and ``(dim, dim)`` for a tensor field.
    ci :
        Optional dict with ``"lower"``/``"upper"`` arrays (same length as the
        grid) for a confidence band.
    samples :
        If True, draw a faint histogram of the data along ``dim`` behind the
        curves.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    d = X.shape[-1]
    xmin = float(X[..., dim].min())
    xmax = float(X[..., dim].max())
    pad = margin * (xmax - xmin)
    grid = np.linspace(xmin - pad, xmax + pad, N)
    pts = np.zeros((N, d))
    pts[:, dim] = grid

    def _val(f):
        out = np.asarray(f(jnp.asarray(pts)))
        if out.ndim == 1:
            return out
        if out.ndim == 2:
            return out[:, component if component is not None else dim]
        if out.ndim == 3:
            c = component if component is not None else dim
            return out[:, c, c]
        return out.reshape(N, -1)[:, 0]

    if samples:
        data1 = X[..., dim].reshape(-1)[M.reshape(-1).astype(bool)]
        axh = ax.twinx()
        axh.hist(data1, bins=60, density=True, alpha=0.18, color=SFI_COLORS["data"])
        axh.set_yticks([])

    if exact_field is not None:
        ax.plot(grid, _val(exact_field), "--", lw=2, color=SFI_COLORS["exact"], label=label_exact)
    ax.plot(grid, _val(field), lw=2, color=SFI_COLORS["inferred"], label=label_inferred)
    if ci is not None:
        ax.fill_between(
            grid, np.asarray(ci["lower"]), np.asarray(ci["upper"]),
            color=SFI_COLORS["inferred"], alpha=0.2,
        )
    ax.axhline(0, color="#808080", lw=0.5)
    ax.set_xlabel(f"x[{dim}]")
    ax.legend()
    return ax


def plot_field_error(
    coll: TrajectoryCollection,
    field_inferred,
    field_exact,
    *,
    dataset: int = 0,
    N: int = 60,
    norm: str = "l2",
    cmap: str = "inferno",
    vmax=None,
    ax=None,
):
    """2D heatmap of the pointwise force-reconstruction error.

    Evaluates both fields on a grid spanning the (masked) data bounding box
    and renders ``||F_exact - F_inferred||`` as a ``pcolormesh``.  The
    spatial complement to :func:`comparison_scatter`.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    d = X.shape[-1]
    if d != 2:
        raise ValueError(f"plot_field_error requires 2D data, got d={d}")
    flat = X.reshape(-1, d)[M.reshape(-1).astype(bool)]
    xmin, ymin = flat.min(axis=0)
    xmax, ymax = flat.max(axis=0)
    GX, GY = np.meshgrid(np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, N))
    pts = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    Fi = np.asarray(field_inferred(jnp.asarray(pts)))
    Fe = np.asarray(field_exact(jnp.asarray(pts)))
    ordmap = {"l2": 2, "l1": 1, "linf": np.inf}
    err = np.linalg.norm(Fe - Fi, ord=ordmap.get(norm, 2), axis=-1).reshape(GX.shape)
    im = ax.pcolormesh(GX, GY, err, cmap=cmap, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=ax, label=f"||F_exact - F_inferred|| ({norm})")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def stream_field(
    coll: TrajectoryCollection,
    field,
    *,
    dataset: int = 0,
    dir1=None,
    dir2=None,
    center=None,
    radius=None,
    N: int = 20,
    density: float = 1.0,
    color: str = "#808080",
    ax=None,
    **streamplot_kw,
):
    """Streamplot of a callable vector field over the data domain.

    The streamline counterpart to :func:`plot_field` (quiver): integrates
    ``field(X) -> F`` into flow lines on a grid spanning the data bounding
    box.  Useful for visualising the topology of an inferred or analytic
    drift field.
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    d = X.shape[-1]
    if dir1 is None:
        dir1 = axisvector(0, d)
    if dir2 is None:
        dir2 = axisvector(1, d)
    if center is None:
        center = X.mean(axis=(0, 1))
    if radius is None:
        radius = 0.5 * (X.max(axis=(0, 1)) - X.min(axis=(0, 1))).max()
    g = np.linspace(-radius, radius, N)
    GX, GY = np.meshgrid(g, g)
    pts = (
        center[None, :]
        + GX.ravel()[:, None] * dir1[None, :]
        + GY.ravel()[:, None] * dir2[None, :]
    )
    F = np.asarray(field(jnp.asarray(pts)))
    U = F.dot(dir1).reshape(N, N)
    V = F.dot(dir2).reshape(N, N)
    ax.streamplot(g + float(dir1.dot(center)), g + float(dir2.dot(center)), U, V, density=density, color=color, **streamplot_kw)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_particles(
    coll: TrajectoryCollection,
    a: int = 0,
    b: int = 1,
    t_index: int = -1,
    colored: bool = True,
    active: bool = False,
    u: float = 0.35,
    *,
    dataset: int = 0,
    color_dim=None,
    cmap=None,
    vmin=None,
    vmax=None,
    quiver: bool = False,
    heading_dim=None,
    box=None,
    s: float = 100.0,
    ax=None,
    quiver_kw=None,
    **kwargs,
):
    """Display all particles at time index ``t_index`` from a collection.

    Parameters
    ----------
    a, b :
        State dimensions for the x/y axes of the snapshot.
    t_index :
        Frame index (negative counts from the end).
    colored :
        If True (and no ``color_dim``), color particles by index (magma).
    color_dim :
        Color particles by the value of this state dimension instead of by
        index — e.g. heading angle (default colormap ``"hsv"``).
    cmap, vmin, vmax :
        Colormap / normalisation for the coloring.
    quiver, heading_dim :
        If ``quiver=True``, overlay heading arrows from
        ``cos/sin(X[:, heading_dim])`` (``heading_dim`` defaults to 2, the
        orientation channel of an active particle).
    box :
        Optional periodic box ``(Lx, Ly)``; positions are wrapped into it.
    active, u :
        Legacy orientation marker (a dot at ``u·(cosθ, sinθ)``); superseded
        by ``quiver`` for active-matter snapshots.
    ax :
        Target axes (default: current axes).
    """
    if ax is None:
        ax = plt.gca()
    t, X, M = _collection_to_arrays(coll, dataset=dataset)  # X: (T, N, d)
    T = X.shape[0]
    if t_index < 0:
        t_index = T + t_index
    if not (0 <= t_index < T):
        raise IndexError(f"t_index {t_index} out of range for T={T}")

    X_t = X[t_index]  # (N, d)
    xy = X_t[:, [a, b]].astype(float)
    if box is not None:
        xy = wrap_positions(xy, np.asarray(box)[:2])
    x, y = xy[:, 0], xy[:, 1]

    if color_dim is not None:
        ax.scatter(x, y, c=X_t[:, color_dim], cmap=cmap or "hsv", vmin=vmin, vmax=vmax, s=s, **kwargs)
    elif colored:
        ax.scatter(x, y, cmap=cmap or "magma", s=s, c=np.linspace(0, 1, len(X_t)), vmin=vmin, vmax=vmax, **kwargs)
    else:
        ax.scatter(x, y, s=s, c="w", edgecolor="#808080", **kwargs)

    hd = heading_dim if heading_dim is not None else (2 if X_t.shape[1] >= 3 else None)
    if quiver and hd is not None:
        qkw = dict(color="#B0B0B0", pivot="mid", units="xy")
        qkw.update(quiver_kw or {})
        ax.quiver(x, y, np.cos(X_t[:, hd]), np.sin(X_t[:, hd]), **qkw)
    elif active and X_t.shape[1] >= 3:
        xa = x + u * np.cos(X_t[:, 2])
        ya = y + u * np.sin(X_t[:, 2])
        ax.scatter(xa, ya, c="#B0B0B0", s=20)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_particles_field(
    coll: TrajectoryCollection,
    field,
    *,
    dataset: int = 0,
    t_index: int = -1,
    dir1=None,
    dir2=None,
    center=None,
    radius=None,
    scale: float = 1.0,
    autoscale: bool = False,
    color="g",
    **kwargs,
):
    """
    Plot a 2D vector field evaluated at particle positions at a given time.
    """
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, d = X.shape

    if t_index < 0:
        t_index = T + t_index
    if not (0 <= t_index < T):
        raise IndexError(f"t_index {t_index} out of range for T={T}")

    X_t = X[t_index]

    if dir1 is None:
        dir1 = axisvector(0, d)
    if dir2 is None:
        dir2 = axisvector(1, d)

    F = field(X_t)
    if center is None:
        center = X_t.mean(axis=0)
    if radius is None:
        radius = 0.5 * (X_t.max(axis=0) - X_t.min(axis=0)).max()

    gridX, gridY, vX, vY = [], [], [], []
    for ind, pos in enumerate(X_t):
        x = dir1.dot(pos)
        y = dir2.dot(pos)
        gridX.append(x)
        gridY.append(y)
        vX.append(dir1.dot(F[ind, :]))
        vY.append(dir2.dot(F[ind, :]))

    if autoscale:
        scale /= max(np.array(vX) ** 2 + np.array(vY) ** 2) ** 0.5

    plt.quiver(
        gridX,
        gridY,
        scale * np.array(vX),
        scale * np.array(vY),
        scale=1.0,
        units="xy",
        color=color,
        minlength=0.0,
        **kwargs,
    )
    plt.ylim(-radius + dir2.dot(center), radius + dir2.dot(center))
    plt.xlim(-radius + dir1.dot(center), radius + dir1.dot(center))
    _equal_aspect(plt.gca())
    plt.xticks([])
    plt.yticks([])


def plot_nematic_director(
    ax,
    Qxx,
    Qxy,
    rho,
    *,
    skip: int = 2,
    scale: float = 2.5,
    color: str = "white",
    alpha: float = 0.55,
    linewidth: float = 0.45,
    **quiver_kw,
):
    """Overlay the nematic director field of a Q-tensor on an image axes.

    Given the order-parameter fields ``Qxx``, ``Qxy`` and density ``rho``
    (each a 2D grid), draws a *headless* quiver of the director
    ``ψ = ½·atan2(Qxy, Qxx)`` on a subsampled grid — the canonical
    active-nematic overlay.

    Parameters
    ----------
    ax :
        Target axes (typically holding an ``imshow`` of the density).
    Qxx, Qxy, rho :
        2D arrays of equal shape.
    skip :
        Subsampling stride for the director glyphs.
    scale :
        Glyph length (passed through to ``quiver`` ``scale``).
    """
    Qxx = np.asarray(Qxx)
    Qxy = np.asarray(Qxy)
    rho = np.maximum(np.asarray(rho), 1e-3)
    psi = 0.5 * np.arctan2(Qxy / rho, Qxx / rho)
    ny, nx = psi.shape
    iy, ix = np.meshgrid(
        np.arange(skip // 2, ny, skip), np.arange(skip // 2, nx, skip), indexing="ij"
    )
    th = psi[iy, ix]
    qkw = dict(
        scale=scale, color=color, alpha=alpha, linewidth=linewidth,
        headwidth=0, headlength=0, headaxislength=0, pivot="mid", units="xy",
    )
    qkw.update(quiver_kw)
    # Return the Quiver artist so callers can update it per animation frame
    # (set_offsets / set_UVC); for static use just ignore the return value.
    return ax.quiver(ix, iy, np.cos(th), np.sin(th), **qkw)


def plot_rods(
    ax,
    X_frame,
    *,
    angle_index: int = 2,
    length: float = 0.85,
    color: str = "#d4a96a",
    linewidth: float = 2.8,
    capstyle: str = "round",
    **kwargs,
):
    """Draw oriented rods (active-matter particles) as a ``LineCollection``.

    Each particle in ``X_frame`` (rows ``(x, y, …, θ, …)``) becomes a short
    segment of length ``length`` centred on ``(x, y)`` and oriented at
    ``θ = X_frame[:, angle_index]``.
    """
    X_frame = np.asarray(X_frame)
    x, y = X_frame[:, 0], X_frame[:, 1]
    th = X_frame[:, angle_index]
    h = 0.5 * length
    dx, dy = h * np.cos(th), h * np.sin(th)
    starts = np.stack([x - dx, y - dy], axis=-1)
    ends = np.stack([x + dx, y + dy], axis=-1)
    segs = np.stack([starts, ends], axis=1)  # (N, 2, 2)
    lc = LineCollection(segs, colors=color, linewidths=linewidth, capstyle=capstyle, **kwargs)
    ax.add_collection(lc)
    ax.set_aspect("equal")
    return lc


def plot_spde_snapshot(
    coll: TrajectoryCollection,
    t_indices,
    *,
    dataset: int = 0,
    scalar_channel: int = 0,
    vector_channels=None,
    grid_shape=None,
    dx=None,
    render: str = "imshow",
    streamplot_kw=None,
    quiver_kw=None,
    axes=None,
    vmin=None,
    vmax=None,
    cmap: str = "magma",
):
    """Render SPDE field snapshots from a gridded TrajectoryCollection.

    Reshapes each requested frame ``X[t]`` of shape ``(N, n_channels)`` to
    ``(*grid_shape, n_channels)`` and draws ``scalar_channel`` as an
    ``imshow`` image, optionally overlaying ``vector_channels`` as
    streamlines (``render="streamplot"``) or arrows (``render="quiver"``).

    Parameters
    ----------
    t_indices :
        A single frame index, or a sequence of indices (one panel each).
    grid_shape :
        ``(nx, ny)`` of the field grid.  Inferred as a square grid when
        omitted.
    dx :
        Physical grid spacing (default 1.0).
    vmin, vmax :
        Color limits; default to the 0.5/99.5 percentiles of the field.
    """
    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, C = X.shape
    if grid_shape is None:
        s = int(round(np.sqrt(N)))
        if s * s != N:
            raise ValueError("Provide grid_shape; cannot infer a non-square grid.")
        grid_shape = (s, s)
    dx = 1.0 if dx is None else float(dx)

    single = np.isscalar(t_indices)
    tis = [int(t_indices)] if single else [int(i) for i in t_indices]
    if axes is None:
        _, axes = plt.subplots(1, len(tis), figsize=(4 * len(tis), 4), squeeze=False)
        axes = axes[0]
    axes = np.atleast_1d(axes)

    gx = (np.arange(grid_shape[0]) + 0.5) * dx
    gy = (np.arange(grid_shape[1]) + 0.5) * dx
    for ax, ti in zip(axes, tis):
        field = X[ti].reshape(*grid_shape, C)
        scal = field[..., scalar_channel]
        vlo = float(np.percentile(scal, 0.5)) if vmin is None else vmin
        vhi = float(np.percentile(scal, 99.5)) if vmax is None else vmax
        ax.imshow(
            scal.T, origin="lower", cmap=cmap, vmin=vlo, vmax=vhi,
            extent=[0, grid_shape[0] * dx, 0, grid_shape[1] * dx],
        )
        if vector_channels is not None:
            vx = field[..., vector_channels[0]]
            vy = field[..., vector_channels[1]]
            if render == "streamplot":
                ax.streamplot(gx, gy, vx.T, vy.T, **(streamplot_kw or {}))
            else:
                GX, GY = np.meshgrid(gx, gy, indexing="ij")
                qkw = dict(color="#B0B0B0")
                qkw.update(quiver_kw or {})
                ax.quiver(GX, GY, vx, vy, **qkw)
        ax.set_xticks([])
        ax.set_yticks([])
    return axes[0] if single else axes


def spatial_acorr2d(field_2d, *, dx: float = 1.0, n_bins=None, normalize: bool = True):
    """Radially-averaged 2D spatial autocorrelation via FFT (periodic).

    Returns ``(r, C)`` where ``C(r)`` is the angle-averaged autocorrelation
    of ``field_2d`` (mean removed) at radial separation ``r``.  Assumes a
    periodic grid.
    """
    f = np.asarray(field_2d, dtype=float)
    f = f - f.mean()
    F = np.fft.rfft2(f)
    C = np.fft.irfft2(F * np.conj(F), s=f.shape) / f.size
    if normalize and C[0, 0] != 0:
        C = C / C[0, 0]
    nx, ny = f.shape
    ix = (np.arange(nx) + nx // 2) % nx - nx // 2
    iy = (np.arange(ny) + ny // 2) % ny - ny // 2
    GX, GY = np.meshgrid(ix, iy, indexing="ij")
    r = np.sqrt(GX**2 + GY**2) * dx
    if n_bins is None:
        n_bins = min(nx, ny) // 2
    edges = np.linspace(0.0, float(r.max()), n_bins + 1)
    which = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    cvals = C.ravel()
    radial = np.array(
        [cvals[which == b].mean() if np.any(which == b) else np.nan for b in range(n_bins)]
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, radial


def animate_particles(
    coll: TrajectoryCollection,
    *,
    dataset: int = 0,
    dims=(0, 1),
    trail: int = 0,
    overlay_fn=None,
    skip: int = 1,
    cmap: str = "magma",
    s: float = 100.0,
    color_dim=None,
    vmin=None,
    vmax=None,
    quiver: bool = False,
    heading_dim=None,
    box=None,
    interval: int = 50,
    ax=None,
    fig=None,
    blit: bool = False,
    **anim_kw,
):
    """Animate particle positions over time (frames read via the collection).

    Returns a :class:`matplotlib.animation.FuncAnimation`.  With ``trail>0``
    each particle leaves a fading tail; ``overlay_fn(ax, t_index, X_t)`` is
    called per frame for custom overlays.  For active matter, color points by
    a state dimension (``color_dim``, e.g. heading angle), overlay heading
    arrows (``quiver=True`` + ``heading_dim``), and wrap into a periodic box
    (``box=(Lx, Ly)``) — mirroring :func:`plot_particles`.
    """
    from matplotlib.animation import FuncAnimation

    t, X, M = _collection_to_arrays(coll, dataset=dataset)
    T, N, d = X.shape
    frames = list(range(0, T, skip))
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif ax is None:
        ax = fig.gca()
    elif fig is None:
        fig = ax.figure
    i, j = dims
    box = np.asarray(box, dtype=float) if box is not None else None

    def _xy(ti):
        xi, yi = X[ti, :, i].copy(), X[ti, :, j].copy()
        if box is not None:
            xi, yi = xi % box[0], yi % box[1]
        return xi, yi

    if box is not None:
        ax.set_xlim(0.0, float(box[0]))
        ax.set_ylim(0.0, float(box[1]))
    else:
        ax.set_xlim(float(X[..., i].min()), float(X[..., i].max()))
        ax.set_ylim(float(X[..., j].min()), float(X[..., j].max()))
    ax.set_aspect("equal")

    x0, y0 = _xy(frames[0])
    if color_dim is not None:
        cvals = X[:, :, color_dim]
        vmin = float(np.nanmin(cvals)) if vmin is None else vmin
        vmax = float(np.nanmax(cvals)) if vmax is None else vmax
        scat = ax.scatter(x0, y0, c=X[frames[0], :, color_dim], cmap=cmap or "hsv", vmin=vmin, vmax=vmax, s=s)
    else:
        scat = ax.scatter(x0, y0, c=np.linspace(0, 1, N), cmap=cmap, s=s)

    hd = heading_dim if heading_dim is not None else (2 if d >= 3 else None)
    quiv = None
    if quiver and hd is not None:
        th0 = X[frames[0], :, hd]
        quiv = ax.quiver(x0, y0, np.cos(th0), np.sin(th0), color="#B0B0B0", pivot="mid", units="xy")

    trails = []
    if trail > 0:
        for _ in range(N):
            (ln,) = ax.plot([], [], lw=0.8, alpha=0.5, color="#B0B0B0")
            trails.append(ln)

    def _update(fr):
        ti = frames[fr]
        xi, yi = _xy(ti)
        off = np.stack([xi, yi], axis=-1)
        scat.set_offsets(off)
        if color_dim is not None:
            scat.set_array(X[ti, :, color_dim])
        artists = [scat]
        if quiv is not None:
            quiv.set_offsets(off)
            th = X[ti, :, hd]
            quiv.set_UVC(np.cos(th), np.sin(th))
            artists.append(quiv)
        if trail > 0:
            lo = max(0, ti - trail)
            for n in range(N):
                trails[n].set_data(X[lo : ti + 1, n, i], X[lo : ti + 1, n, j])
            artists += trails
        if overlay_fn is not None:
            overlay_fn(ax, ti, X[ti])
        return artists

    return FuncAnimation(fig, _update, frames=len(frames), interval=interval, blit=blit, **anim_kw)


def animate_spde_comparison(
    coll_a: TrajectoryCollection,
    coll_b: TrajectoryCollection,
    *,
    dataset: int = 0,
    field_component: int = 0,
    grid_shape=None,
    skip: int = 1,
    vmin=None,
    vmax=None,
    cmap: str = "magma",
    titles=("A", "B"),
    interval: int = 50,
    plot_colorbar: bool = True,
    blit: bool = False,
    **anim_kw,
):
    """Side-by-side animation of one channel of two gridded collections.

    Returns a :class:`matplotlib.animation.FuncAnimation` with two image
    panels sharing color limits (e.g. data vs bootstrap SPDE fields).
    """
    from matplotlib.animation import FuncAnimation

    _, Xa, _ = _collection_to_arrays(coll_a, dataset=dataset)
    _, Xb, _ = _collection_to_arrays(coll_b, dataset=dataset)
    Tn = min(Xa.shape[0], Xb.shape[0])
    N, C = Xa.shape[1], Xa.shape[2]
    if grid_shape is None:
        s = int(round(np.sqrt(N)))
        if s * s != N:
            raise ValueError("Provide grid_shape; cannot infer a non-square grid.")
        grid_shape = (s, s)
    frames = list(range(0, Tn, skip))

    def _slab(Xarr, ti):
        return Xarr[ti].reshape(*grid_shape, C)[..., field_component].T

    if vmin is None:
        vmin = float(min(Xa[..., field_component].min(), Xb[..., field_component].min()))
    if vmax is None:
        vmax = float(max(Xa[..., field_component].max(), Xb[..., field_component].max()))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    im_a = axes[0].imshow(_slab(Xa, 0), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    im_b = axes[1].imshow(_slab(Xb, 0), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    if plot_colorbar:
        fig.colorbar(im_b, ax=axes.tolist(), shrink=0.8)

    def _update(fr):
        ti = frames[fr]
        im_a.set_data(_slab(Xa, ti))
        im_b.set_data(_slab(Xb, ti))
        return [im_a, im_b]

    return FuncAnimation(fig, _update, frames=len(frames), interval=interval, blit=blit, **anim_kw)


def plot_time_profile_comparison(t, true_profiles, inferred_profiles, *, labels=None, axes=None):
    """Plot true vs inferred time-dependent profiles (e.g. k(t), a(t)).

    ``true_profiles`` and ``inferred_profiles`` are sequences of 1D arrays
    (one per panel); ``true_profiles`` entries may be ``None`` to skip the
    reference.  ``labels`` gives a title per panel.
    """
    n = len(inferred_profiles)
    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
        axes = axes[0]
    axes = np.atleast_1d(axes)
    for k in range(n):
        ax = axes[k]
        if true_profiles is not None and true_profiles[k] is not None:
            ax.plot(t, true_profiles[k], color=SFI_COLORS["exact"], lw=2.4, label="true")
        ax.plot(t, inferred_profiles[k], color=SFI_COLORS["inferred"], lw=1.6, ls="--", label="inferred")
        if labels is not None:
            ax.set_title(labels[k])
        ax.set_xlabel("time")
        ax.legend(loc="upper right")
    return axes
