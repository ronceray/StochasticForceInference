import jax.numpy as jnp
from jax import jit
from typing import List, Dict, Tuple, Optional, Callable
import jax
import yaml

# ---------------------------------------------------------------------
#  Mathematical utilities
# ---------------------------------------------------------------------

def stable_pinv(G):
    """A robust pseudo-inverse for Gram matrices: perform
    pseudo-inversion on the normalized matrix (diagonal values =
    1). This allows numerically precise inversion even when the scales
    of the different basis functions are very different.

    """
    # Normalize the matrix before computing the pseudo-inverse, for numerical stability
    G_norm = jnp.sqrt(jnp.diag(G))  # Extract diagonal elements as normalization factors
    # Avoid division by zero: if G_norm[i] is zero, keep it zero in scaling
    safe_G_norm = jnp.where(G_norm > 0, G_norm, 1.0)  # Replace 0 with 1 to avoid NaN in division
    return jnp.linalg.pinv(G / jnp.outer(safe_G_norm, safe_G_norm)) * jnp.outer(1 / safe_G_norm, 1 / safe_G_norm)

@jit
def sqrtm_psd(mat):
    """Matrix square-root for symmetric positive-definite matrix
    (scipy.linalg.sqrtm not supported by Jax for now)."""
    eigvals, eigvecs = jnp.linalg.eigh(mat)
    return (eigvecs * jnp.sqrt(jnp.clip(eigvals, 0.0))) @ eigvecs.T


def solve_or_pinv(A: jnp.array, b: jnp.array, tol: float = 1e-15) -> jnp.array:
    """
    Solve A ⋅ x = b for x, with a fallback to the Moore–Penrose pseudo-inverse
    if A is singular or not square.  To improve numerical stability, we first
    normalize A by its diagonal: A_norm = D^{-1} A D^{-1}, b_norm = D^{-1} b,
    solve A_norm ⋅ x_norm = b_norm, and then recover x = D^{-1} x_norm.

    This ensures that the diagonal entries of A_norm are 1 (assuming A has
    positive diagonal), which often makes the linear solve or pseudo-inverse
    more robust when A has widely varying scales on its diagonal.

    Parameters
    ----------
    A : jnp.array, shape (k, k)
        The matrix to solve against.  We assume that A has nonnegative diagonal
        entries; if any diagonal entry is zero, we clip it to a small floor
        to avoid division by zero.
    b : jnp.array, shape (k,)
        The right-hand side vector.
    tol : float, default=1e-8
        The tolerance for the pseudo-inverse.  If A_norm is effectively singular,
        we compute x_norm = pinv(A_norm, rcond=tol) @ b_norm.

    Returns
    -------
    x : jnp.array, shape (k,)
        The solution vector to A ⋅ x = b, computed as follows:
          1) d_i = sqrt(max(A_{ii}, tol))
             (we floor each diagonal entry to tol > 0 to avoid zero divides)
          2) A_norm = D_inv @ A @ D_inv  where D_inv = diag(1 / d_i)
             b_norm = b / d
          3) Solve A_norm ⋅ x_norm = b_norm:
             • if A_norm is non-singular, use a direct solver
             • otherwise, fall back to x_norm = pinv(A_norm) @ b_norm
          4) Recover x = x_norm / d
    """
    # 1) Extract and “floor” the diagonal of A to avoid zeros or negatives.
    #    If A_{ii} is very small or negative (due to numerical noise), we floor it.
    diag_A = jnp.diag(A)                        # shape (k,)
    # Clip to tol to avoid sqrt of zero or negative
    diag_clipped = jnp.clip(diag_A, a_min=tol)   # shape (k,)
    d = jnp.sqrt(diag_clipped)                   # shape (k,)

    # 2) Build the inverse scaling matrix D_inv = diag(1 / d_i)
    #    We do this by dividing each row of A by d_i and each column by d_j.
    #    More precisely: A_norm[i,j] = A[i,j] / (d[i] * d[j]).
    D_inv = 1.0 / d                               # shape (k,)
    # Use broadcasting to normalize A: first divide each row by d,
    # then each column by d.
    A_norm = (A * D_inv[:, None]) * D_inv[None, :]  


    # 3) Normalize the RHS vector b as well: b_norm = b / d.
    b_norm = b / d                                # shape (k,)

    # 4) Attempt a direct solve of A_norm ⋅ x_norm = b_norm.
    #    If A_norm is non-singular, this will succeed.
    try:
        # We do not assume positive-definite here, so use 'gen' unless we detect
        # symmetry and positive definiteness.  For simplicity, assume 'gen'.
        x_norm = jsp_la.solve(A_norm, b_norm, assume_a='gen')
    except Exception:
        # 4a) Fallback: if A_norm is singular or not square, use pseudo-inverse.
        x_norm = jnp.linalg.pinv(A_norm, rcond=tol) @ b_norm

    # 5) Scale back to obtain the final solution: x_i = x_norm[i] / d[i].
    x = x_norm / d                                # shape (k,)

    return x


# ---------------------------------------------------------------------
# Trajectory <-> columns conversion helpers
#
#  Allows smooth conversion between csv format (particles index,time
#  step index, state vector) and structured arrays "X" for SFI (Nsteps
#  x Nparticles x dim) array, plus (Nsteps x Nparticles) mask.
#  ---------------------------------------------------------------------
import numpy as np
from typing import Any, Union

def flatten_X_to_columns(
    X: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (T × N × d) tensor → tracking columns.

    Returns 1-D arrays (particle_idx, time_idx, state_vectors).  Points
    whose mask is False *or* whose coordinates contain NaNs are thrown
    away.
    """
    T, N, d = X.shape
    time_idx      = np.repeat(np.arange(T), N)
    particle_idx  = np.tile  (np.arange(N), T)
    state_vectors = X.reshape(T * N, d)

    valid = ~np.isnan(state_vectors).any(axis=1)
    if mask is not None:
        valid &= mask.ravel()

    particle_idx, time_idx, state_vectors = (
        particle_idx[valid].astype(int),
        time_idx[valid].astype(int),
        state_vectors[valid],
    )

    return particle_idx, time_idx, state_vectors


def assemble_X_from_columns(
    particle_idx: np.ndarray,
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    *,
    fill_value: float = np.nan,
    relabel: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Rebuild an (T × N × d) trajectory tensor from tracking columns.

    Parameters
    ----------
    particle_idx, time_idx : 1-D integer arrays
    state_vectors          : (len, d) float array
    fill_value             : value for “missing” entries (default NaN)
    relabel                : if True, compress particle IDs to 0..N-1

    Returns
    -------
    X    : (T, N, d) ndarray
    mask : (T, N) bool mask indicating which entries are real
    """
    # Ensure NumPy
    particle_idx = np.asarray(particle_idx, dtype=int)
    time_idx     = np.asarray(time_idx,     dtype=int)
    state_vectors= np.asarray(state_vectors)

    # Ensure time indices are non-negative and start at 0:
    time_idx -= time_idx.min()
    
    if relabel:
        uniq = np.unique(particle_idx)
        remap = {old: new for new, old in enumerate(uniq)}
        particle_idx = np.vectorize(remap.__getitem__)(particle_idx)
        N = len(uniq)
    else:
        N = particle_idx.max() + 1

    T = time_idx.max() + 1
    d = state_vectors.shape[1]

    X    = np.full((T, N, d), fill_value, dtype=state_vectors.dtype)
    mask = np.zeros((T, N), dtype=bool)

    X[time_idx, particle_idx]     = state_vectors
    mask[time_idx, particle_idx]  = True
    return X, mask


def sanitize_metadata(obj):
    """Recursively convert JAX/NumPy arrays and scalars into vanilla Python types."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.generic, jnp.generic)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize_metadata(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_metadata(v) for v in obj]
    else:
        return obj

    
def save_trajectory_csv(
    filename: str,
    particle_idx: np.ndarray,
    time_idx: np.ndarray,
    state_vectors: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    float_fmt: str = "%.8f",
) -> None:
    """Write columns to a CSV compatible with
    ``load_trajectory_csv``."""
    yaml_str = yaml.dump(sanitize_metadata(metadata), sort_keys=False)
    yaml_header = "\n".join(f"# {line}" for line in yaml_str.strip().splitlines())
    yaml_header = "# ---\n" + yaml_header

    d = state_vectors.shape[-1]
    arr = np.column_stack([particle_idx, time_idx, np.asarray(state_vectors)])

    data_header = f"particle_id,time_step,{','.join(f'x{i}' for i in range(d))}"
    full_header = yaml_header + "\n" + data_header

    np.savetxt(
        filename,
        arr,
        header=full_header,
        delimiter=",",
        fmt=["%d", "%d"] + [float_fmt] * d,
        comments="",
    )


def load_trajectory_csv(filename, particle_column=0, time_column=1, state_columns=None, relabel = True):
    """Loads trajectory data and metadata from a CSV file.
    
    Args:
        filename (str): Path to the CSV file.
        particle_column (int): Index of the column containing particle IDs. If None, fill with zeros
                 (not a multiparticle file).
        time_column (int): Index of the column containing time indices.
        state_columns (list or None): List of indices for state columns to extract.
                                      If None, all columns except `particle_column` 
                                      and `time_column` are used.
        relabel: If True, map particle_indices to (0..Nmax) and time indices to (0..Tmax). Ensures no memory
                is wasted for particles / time points that are never there.

    Returns:
        metadata (dict): Metadata parsed from the header.
        column_headers (list): List of column headers.
        particle_indices (np.ndarray): jnp.array of particle indices.
        time_indices (np.ndarray): jnp.array of time indices.
        state_vectors (np.ndarray): jnp.array of state vectors (positions).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Extract YAML header
    yaml_lines = [line[2:] for line in lines if line.startswith("# ")]
    metadata = yaml.safe_load("\n".join(yaml_lines)) if yaml_lines else {}

    # Load the numerical data from the CSV file, skipping the header lines
    import pandas as pd
    df = pd.read_csv(filename, comment='#')
    data = df.to_numpy()  # Convert to NumPy array if needed

    # Extract particle indices and time indices
    time_indices = data[:, time_column].astype(int)
    if particle_column is None:
        particle_indices = jnp.zeros_like(time_indices,dtype=int)
    else:
        particle_indices = data[:, particle_column].astype(int)
    if relabel:
        # Basic relabeling - keep only particle indices that appear at
        # least once, and time starts at 0.
        if particle_column is not None:
            unique_ids, particle_indices = jnp.unique(particle_indices, return_inverse=True)
        time_indices = time_indices - time_indices.min()

    # Determine state columns
    if state_columns is None:
        # Use all columns except particle and time indices
        state_columns = [i for i in range(data.shape[1]) if i not in [particle_column, time_column]]

    # Extract the specified state vectors
    state_vectors = data[:, state_columns]
    return metadata, particle_indices, time_indices, state_vectors




### Data degradation utility (for synthetic data): ###

def degrade_data(
    metadata: Dict,
    particle_indices: np.ndarray,
    time_indices: np.ndarray,
    state_vectors: np.ndarray,
    *,
    downsample: int = 1,
    motion_blur: int = 0,
    data_loss_fraction: float = 0.0,
    noise: Union[None, float, np.ndarray] = None,
    ROI: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]] = None,
) -> Tuple[Dict, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """ Utility function to degrade synthetic data and make it mimic experiments.

    Arguments:
       - metadata, particle_indices, time_indices, state_vectors: as in load_trajectory_data output.
       - downsample: int (default 1). Only time points t with t%downsample==0 are kept.
       - motion_blur: int (default 0, should be < downsample). Each time point t is replaced by a time-blurred average over [t,t+motion_blur].
       - data_loss_fraction: fraction of points to randomly remove from the data set.
       - noise: Add Gaussian white "measurement noise" to the data.
            * None: skip this step.
            * float: iid noise on each coordinate with variance noise**2
            * array(dim): iid noise on each coordinate m with variance noise[m]**2
            * array(dim,dim): noise on each coordinate with covariance [m,n] = (noise@noise.T)[m,n]
       - ROI: Region of Interest selector - all particles that go outside are lost.
            * None: skip this step.
            * float: keep positions in [-ROI/2, ROI/2] for all dimensions.
            * array(2,dim): keep positions x such that (ROI[0] < x < ROI[1]).all()
            * callable array(dim) -> bool: custom function indicating whether a data point is within ROI or not.

    Returns:
       - Degraded metadata, particle_indices, time_indices, state_vectors as in load_trajectory_data.
         Metadata is updated with the degradation parameters.

    """
    # ------------------------ sanity checks -------------------------
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    if not (0.0 <= data_loss_fraction < 1.0):
        raise ValueError("data_loss_fraction must be in [0, 1).")
    if motion_blur < 0 or motion_blur >= downsample:
        raise ValueError("motion_blur must satisfy 0 <= motion_blur < downsample.")
    if len(particle_indices) != len(time_indices) or len(time_indices) != len(state_vectors):
        raise ValueError("particle_indices, time_indices and state_vectors must have the same length.")

    # Ensure we work with NumPy arrays (they'll be converted to JAX later if needed)
    particle_indices = np.asarray(particle_indices)
    time_indices = np.asarray(time_indices)
    state_vectors = np.asarray(state_vectors)

    rng = np.random.default_rng()  # Note: seeding could be added here for reproducibility
    dim = state_vectors.shape[1]

    # -----------------------------------------------------------------
    # 1) Motion blur + down-sampling  
    # -----------------------------------------------------------------
    # Rebuild full (T,N,d) tensor — we do *not* relabel because we want
    # original particle IDs preserved throughout degradation.
    X_full, mask_full = assemble_X_from_columns(
        particle_indices, time_indices, state_vectors,
        relabel=False
    )                                     # X_full  shape: (T,N,d)

    T, N, dim  = X_full.shape
    window     = motion_blur + 1          # number of frames averaged
    keep_times = np.arange(0, T - motion_blur, downsample)

    X_ds    = np.empty((keep_times.size, N, dim), dtype=X_full.dtype)
    mask_ds = np.empty((keep_times.size, N),     dtype=bool)

    for k, t0 in enumerate(keep_times):
        segment = X_full[t0 : t0 + window]          # (window,N,d)
        X_ds[k] = np.nanmean(segment, axis=0)
        if mask_full is None:
            # If no explicit mask, consider entry valid when not all-NaN
            mask_ds[k] = ~np.isnan(segment).all(axis=0)
        else:
            mask_ds[k] = mask_full[t0 : t0 + window].all(axis=0)

    # Flatten back to column representation
    particle_indices_ds, time_indices_ds, state_vectors_ds = flatten_X_to_columns(X_ds, mask_ds)

    # -----------------------------------------------------------------
    # 2) Measurement noise
    # -----------------------------------------------------------------
    if noise is not None:
        state_vectors_ds = _add_noise(state_vectors_ds, rng, noise)

    # -----------------------------------------------------------------
    # 3) Region‑of‑interest filtering
    # -----------------------------------------------------------------
    inside_roi = _validate_roi(ROI, dim)
    if ROI is not None:
        roi_mask = np.apply_along_axis(inside_roi, 1, state_vectors_ds).astype(bool)
        particle_indices_ds = particle_indices_ds[roi_mask]
        time_indices_ds = time_indices_ds[roi_mask]
        state_vectors_ds = state_vectors_ds[roi_mask]

    # -----------------------------------------------------------------
    # 4) Random data loss
    # -----------------------------------------------------------------
    if data_loss_fraction > 0.0 and len(particle_indices_ds) > 0:
        n_keep = int(round(len(particle_indices_ds) * (1.0 - data_loss_fraction)))
        keep_idx = rng.choice(len(particle_indices_ds), size=n_keep, replace=False)
        particle_indices_ds = particle_indices_ds[keep_idx]
        time_indices_ds = time_indices_ds[keep_idx]
        state_vectors_ds = state_vectors_ds[keep_idx]

    # -----------------------------------------------------------------
    # 5) Sort by (time, particle) for deterministic output
    # -----------------------------------------------------------------
    order = np.lexsort((particle_indices_ds, time_indices_ds))
    particle_indices_ds = particle_indices_ds[order]
    time_indices_ds = time_indices_ds[order]
    state_vectors_ds = state_vectors_ds[order]

    # Record degradation parameters so the provenance is clear
    metadata = dict(metadata) # Shallow copy
    metadata.update(
        {
            "original_dt": metadata['dt'],
            "dt": metadata['dt'] * downsample,
            "downsample": downsample,
            "motion_blur": motion_blur,
            "data_loss_fraction": data_loss_fraction,
            "noise_spec": None if noise is None else noise,
            "ROI_spec": ROI,
        }
    )

    return metadata, particle_indices_ds, time_indices_ds, state_vectors_ds


# Data degradation internal utils
def _validate_roi(roi: Union[None, float, np.ndarray, Callable[[np.ndarray], bool]], dim: int) -> Callable[[np.ndarray], bool]:
    """Return a callable that checks whether a point is inside the ROI."""
    if roi is None:
        return lambda x: True

    # Scalar ROI ⇒ centred box [−roi/2, +roi/2] in every dimension
    if np.isscalar(roi):
        half = float(roi) / 2.0
        return lambda x: np.all((-half <= x) & (x <= half))

    # array-like ROI
    roi_arr = np.asarray(roi, dtype=float)

    # (2, dim) array ⇒ rectangular box given by lower/upper corners
    if roi_arr.shape == (2, dim):
        lo, hi = roi_arr
        return lambda x: np.all((lo <= x) & (x <= hi))

    # Callable already – trust the user
    if callable(roi):
        return roi  # type: ignore[arg-type]

    raise ValueError("ROI must be None, a scalar, a (2,dim) array, or a callable.")


def _add_noise(x: np.ndarray, rng: np.random.Generator, noise: Union[float, np.ndarray]) -> np.ndarray:
    """Add Gaussian noise with various covariance specifications."""
    if np.isscalar(noise):
        return x + rng.normal(scale=float(noise), size=x.shape)

    noise_arr = np.asarray(noise, dtype=float)

    if noise_arr.ndim == 1:  # per–coordinate std‐devs
        return x + rng.normal(scale=noise_arr, size=x.shape)

    if noise_arr.ndim == 2:  # (dim, dim) – interpret as sqrt(covariance).
        dim = x.shape[1]
        if noise_arr.shape != (dim, dim):
            raise ValueError("Noise matrix must have shape (dim, dim).")
        white = rng.normal(size=x.shape)
        return x + white @ noise_arr

    raise ValueError("Noise must be a float, a 1‑D array, or a 2‑D array.")




############# Function naming and pretty printing ######################

def pretty_print_model(descriptors: list[str],
                       coeffs: jnp.ndarray,
                       *,
                       tol: float = 1e-12,
                       one_line: bool = True,
                       fmt: str = "{:+.4g}") -> str:
    """
    Return a compact human-readable polynomial:
        +1.2x0 -0.07x1^2 ...
    """
    pieces = [f"{fmt.format(float(c))}{d}"
              for d, c in zip(descriptors, coeffs) if abs(c) > tol]
    if not pieces:
        return "0"
    return " ".join(pieces) if one_line else "\n".join(pieces)


def make_variable_names(dim: int,
                        symbol: str = "x"
                        ) -> tuple[list[str], list[str]]:
    """
    Returns
        names      – symbols for the x–variables **already subscripted** if auto.
        subscripts – strings you can tack onto 'v' or 'F'.

          names      = ['x₀', 'x₁', …]
          subscripts = ['₀',  '₁',  …]
    """
    # --- automatic symbols with Unicode subscripts -----------------
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    names = [ symbol+str(i).translate(SUB) for i in range(dim)]
    subscripts = [f"{i}".translate(SUB)  for i in range(dim)]
    return names, subscripts



def simple_function_print(basis_names,
                          support,
                          coefficients,
                          *,
                          coeffs_stderr = None,
                          fmt: str = "{:+.4g}") -> str:
    """
    Return a single string for coefficients * function_names:
    """
    pieces = []
    for i,idx in enumerate(support):
        c = fmt.format(float(coefficients[i]))
        if coeffs_stderr is not None:
            c += f" (±{fmt.format(coeffs_stderr[i])[1:]}) "
        name = basis_names[idx]
        pieces.append(c + name)
    return " ".join(pieces) if pieces else "0"
