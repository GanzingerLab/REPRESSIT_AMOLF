# -*- coding: utf-8 -*-
"""
Batch LER computation using TRACKED cell IDs from linked_rois.pkl
================================================================

Overview
--------
This script computes LER (local enrichment ratio) for localization data using
tracked cell IDs obtained from `linked_rois.pkl`.

The main analysis region is split into:
1) An inner box around each localization
2) An outer ring around each localization

Important geometry
------------------
The ring is defined as:

    OUTER_BOX minus PICASSO_BOX

So the region between the inner box and the Picasso box is NOT used.

Example:
    PICK_BOX_SIDE  = 11
    INNER_BOX_SIDE = 7
    OUTER_BOX_SIDE = 15

This gives:
    - inner analysis region = 7x7
    - excluded middle region = 11x11 (Picasso box)
    - outer analysis ring = 15x15 minus 11x11

Extra handling included
-----------------------
1) Cells with NO assigned localizations in a run:
   - one synthetic row is added per missing tracked cell
   - with:
        LER = 1.0
        valid_LER = True
        synthetic_no_locs = True

2) Runs with NO locs file at all:
   - output CSV is still written
   - one synthetic row is added per tracked cell

This allows downstream cell-level aggregation to continue without needing to
reload PKL files.

Output
------
If locs CSV exists:
    *_locs_with_LER_trackedCellIDs.csv

If locs CSV does not exist:
    <RunName>_<CHANNEL_TAG>_locs_with_LER_trackedCellIDs.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff


# =============================
# USER SETTINGS
# =============================
ROOT = Path(r"\\sun\ganzinger\home-folder\raghuram\2026_02_analysis\output")
CHANNEL_TAG = "638nm"          # Example: "638nm" for ZAP70

# Box sizes (must be odd numbers)
PICK_BOX_SIDE = 11             # Picasso localization box size
INNER_BOX_SIDE = 7             # inner analysis box size
OUTER_BOX_SIDE = 15            # outer analysis box size

# Minimum number of pixels required after clipping to image / cell boundaries
MIN_IN_PX = 20                 # minimum pixels in inner box
MIN_RING_PX = 40               # minimum pixels in outer ring

# Runtime options
OVERWRITE = True
SEARCH_ONE_LEVEL_DEEP = False
DEBUG_TIFF = False

# LER value assigned to tracked cells that have no localizations
NO_LOCS_CELL_LER_VALUE = 1.0
# =============================

EPS = 1e-6  # small number to avoid divide-by-zero in LER calculation


def log(msg: str) -> None:
    """Print a message immediately."""
    print(msg, flush=True)


def is_run_dir(p: Path) -> bool:
    """Return True if the path is a run folder (name starts with 'run' / 'Run')."""
    return p.is_dir() and p.name.lower().startswith("run")


def find_run_dirs(root: Path) -> list[Path]:
    """Recursively find all run folders under the root directory."""
    return sorted([p for p in root.rglob("Run*") if is_run_dir(p)])


def detect_time_col(df: pd.DataFrame) -> str:
    """
    Detect the frame/time column in a localization dataframe.

    Accepted column names:
        frame, Frame, t, T
    """
    for c in ["frame", "Frame", "t", "T"]:
        if c in df.columns:
            return c
    raise ValueError(f"No time column found. Columns: {list(df.columns)}")


def maybe_fix_frame_indexing(df: pd.DataFrame, time_col: str, T: int) -> pd.DataFrame:
    """
    Convert frame numbering from 1..T to 0..T-1 if needed.

    If frames are already zero-based, nothing is changed.
    """
    if df.empty:
        return df

    tmin = int(df[time_col].min())
    tmax = int(df[time_col].max())

    if tmin == 1 and tmax == T:
        df = df.copy()
        df[time_col] = df[time_col] - 1

    return df


def pick_locs_csv(run_dir: Path, channel_tag: str) -> Path | None:
    """
    Find the localization CSV for a given run and channel.

    Looks for filenames matching:
        *_{channel_tag}_locs.csv
        *{channel_tag}_locs.csv

    Optionally searches one level deeper if SEARCH_ONE_LEVEL_DEEP is True.
    """
    pats = [f"*_{channel_tag}_locs.csv", f"*{channel_tag}_locs.csv"]

    for pat in pats:
        hits = sorted(run_dir.glob(pat))
        if hits:
            return hits[0]

    if SEARCH_ONE_LEVEL_DEEP:
        for sub in [p for p in run_dir.iterdir() if p.is_dir()]:
            for pat in pats:
                hits = sorted(sub.glob(pat))
                if hits:
                    return hits[0]

    return None


def candidate_tifs(run_dir: Path, channel_tag: str) -> list[Path]:
    """
    Collect possible TIFF movie files for a run and channel.

    Candidates are sorted by file size (largest first), which helps prefer
    the main movie stack over smaller files.
    """
    pats = [f"*{channel_tag}*.tif", f"*{channel_tag}*.tiff"]
    cand: list[Path] = []

    for pat in pats:
        cand += sorted(run_dir.glob(pat))

    if SEARCH_ONE_LEVEL_DEEP:
        for sub in [p for p in run_dir.iterdir() if p.is_dir()]:
            for pat in pats:
                cand += sorted(sub.glob(pat))

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for p in cand:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    # Prefer larger files first
    uniq.sort(key=lambda p: p.stat().st_size, reverse=True)
    return uniq


def pick_movie_tif(run_dir: Path, channel_tag: str) -> Path | None:
    """Pick the most likely TIFF movie file for a run."""
    cand = candidate_tifs(run_dir, channel_tag)
    return cand[0] if cand else None


def _score_series_shape(shape: tuple[int, ...]) -> int:
    """
    Score a TIFF series shape to help choose the best movie stack.

    A valid movie stack should have at least 3 dimensions and reasonably large
    spatial dimensions. Higher scores are better.
    """
    if not isinstance(shape, tuple) or len(shape) < 3:
        return -1
    if shape[-1] <= 32 or shape[-2] <= 32:
        return -1
    return int(np.prod(shape[:-2]))


def load_movie_stack_ultra(path: Path) -> np.ndarray:
    """
    Load a TIFF movie and return it as a numpy array with shape (T, H, W).

    Handles different TIFF layouts:
    - single series
    - multiple series
    - page-based TIFFs
    - 2D / 3D / 4D arrays
    """
    with tiff.TiffFile(path) as tf:
        if DEBUG_TIFF:
            series_shapes = [tuple(s.shape) for s in tf.series]
            log(f"TIFF DIAG: {path.name} | pages={len(tf.pages)} | series={series_shapes}")

        best_idx = None
        best_score = -1

        # Try to find the best series that looks like a movie stack
        for i, s in enumerate(tf.series):
            score = _score_series_shape(tuple(s.shape))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= 1:
            arr = tf.series[best_idx].asarray()
        else:
            # Fallback: stack pages manually if needed
            if len(tf.pages) > 1:
                arr = np.stack([p.asarray() for p in tf.pages], axis=0)
            else:
                arr = tf.asarray()

    arr = np.asarray(arr)

    # Force output into (T, H, W)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 4:
        tflat = int(np.prod(arr.shape[:-2]))
        arr = arr.reshape((tflat, arr.shape[-2], arr.shape[-1]))
    else:
        raise ValueError(f"Unexpected TIFF array shape {arr.shape} for {path}")

    return arr


def box_slices(cx: int, cy: int, hw: int, H: int, W: int) -> tuple[slice, slice]:
    """
    Return y/x slices for a square box centered at (cx, cy) with half-width hw,
    clipped to image boundaries.
    """
    x0 = max(cx - hw, 0)
    x1 = min(cx + hw + 1, W)
    y0 = max(cy - hw, 0)
    y1 = min(cy + hw + 1, H)
    return slice(y0, y1), slice(x0, x1)


def build_tracked_label_frames(linked_df: pd.DataFrame, T: int, H: int, W: int) -> list[np.ndarray]:
    """
    Build one label image per frame.

    Output:
        lbls[t][y, x] = tracked cell_id if pixel belongs to that tracked cell

    The contour coordinates from linked_rois.pkl are written directly into the
    label image for each frame.
    """
    lbls = [np.zeros((H, W), dtype=np.int32) for _ in range(T)]

    for t, sub in linked_df.groupby("frame"):
        t = int(t)
        if t < 0 or t >= T:
            continue

        lbl = lbls[t]

        for row in sub.itertuples(index=False):
            cid = int(getattr(row, "cell_id"))
            coords = getattr(row, "contour")

            if coords is None:
                continue

            coords = np.asarray(coords)
            if coords.size == 0:
                continue

            rr = coords[:, 0].astype(int, copy=False)
            cc = coords[:, 1].astype(int, copy=False)

            # Keep only coordinates inside the image bounds
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rr = rr[ok]
            cc = cc[ok]

            lbl[rr, cc] = cid

    return lbls


def compute_LER_tracked(stack: np.ndarray, locs_df: pd.DataFrame, linked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute LER for each localization using tracked cell identities.

    For each localization:
    - identify which tracked cell it belongs to
    - extract an inner region around the localization
    - extract an outer ring = outer box minus Picasso box
    - restrict both regions to the same tracked cell
    - compute:
          LER = median(inner) / median(ring)

    Also adds synthetic rows for tracked cells with no assigned localizations.
    """
    # Validate geometry settings
    if PICK_BOX_SIDE % 2 != 1:
        raise ValueError("PICK_BOX_SIDE must be odd (e.g., 11).")
    if INNER_BOX_SIDE % 2 != 1:
        raise ValueError("INNER_BOX_SIDE must be odd (e.g., 7).")
    if OUTER_BOX_SIDE % 2 != 1:
        raise ValueError("OUTER_BOX_SIDE must be odd (e.g., 15).")
    if INNER_BOX_SIDE > PICK_BOX_SIDE:
        raise ValueError("INNER_BOX_SIDE must be <= PICK_BOX_SIDE.")
    if OUTER_BOX_SIDE <= PICK_BOX_SIDE:
        raise ValueError("OUTER_BOX_SIDE must be > PICK_BOX_SIDE.")
    if stack.ndim != 3:
        raise ValueError(f"Movie is not (T,H,W). Got {stack.shape}")

    T, H, W = stack.shape
    df = locs_df.copy()

    # If no locs exist, create an empty dataframe with the required columns
    if df.empty:
        df = pd.DataFrame({
            "frame": pd.Series(dtype=int),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
        })

    time_col = detect_time_col(df)
    df = maybe_fix_frame_indexing(df, time_col, T)

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Locs missing x/y columns. Columns: {list(df.columns)}")

    # Create per-frame label maps from tracked contours
    lbls = build_tracked_label_frames(linked_df, T=T, H=H, W=W)

    hw_in = INNER_BOX_SIDE // 2
    hw_pick = PICK_BOX_SIDE // 2
    hw_out = OUTER_BOX_SIDE // 2

    n = len(df)

    # Output arrays
    I_in = np.full(n, np.nan, dtype=np.float32)
    I_ring = np.full(n, np.nan, dtype=np.float32)
    LER = np.full(n, np.nan, dtype=np.float32)
    n_in = np.zeros(n, dtype=np.int32)
    n_ring = np.zeros(n, dtype=np.int32)
    valid = np.zeros(n, dtype=bool)
    cell_id_tracked = np.zeros(n, dtype=np.int32)

    for i, row in enumerate(df.itertuples(index=False)):
        t = int(getattr(row, time_col))
        if t < 0 or t >= T:
            continue

        x = float(getattr(row, "x"))
        y = float(getattr(row, "y"))
        cx, cy = int(round(x)), int(round(y))

        # Skip out-of-bounds localizations
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            continue

        lbl = lbls[t]
        cid = int(lbl[cy, cx])

        # Skip if localization is not assigned to a tracked cell
        if cid == 0:
            continue

        cell_id_tracked[i] = cid
        frame = stack[t]

        # Outer box around localization
        y_out, x_out = box_slices(cx, cy, hw_out, H, W)
        img_out = frame[y_out, x_out]
        lbl_out = lbl[y_out, x_out]

        # Inner box
        y_in, x_in = box_slices(cx, cy, hw_in, H, W)

        # Picasso box (excluded from the ring)
        y_pick, x_pick = box_slices(cx, cy, hw_pick, H, W)

        # Coordinates relative to the outer box
        y0, x0 = y_out.start, x_out.start

        # Build inner mask inside the outer crop
        inner_mask = np.zeros_like(lbl_out, dtype=bool)
        inner_mask[
            slice(y_in.start - y0, y_in.stop - y0),
            slice(x_in.start - x0, x_in.stop - x0)
        ] = True

        # Build Picasso box mask inside the outer crop
        pick_mask = np.zeros_like(lbl_out, dtype=bool)
        pick_mask[
            slice(y_pick.start - y0, y_pick.stop - y0),
            slice(x_pick.start - x0, x_pick.stop - x0)
        ] = True

        # Ring = outer box minus Picasso box
        ring_mask = ~pick_mask

        # Only keep pixels belonging to the same tracked cell
        same_cell = (lbl_out == cid)
        inner_mask &= same_cell
        ring_mask &= same_cell

        nin = int(inner_mask.sum())
        nrg = int(ring_mask.sum())
        n_in[i] = nin
        n_ring[i] = nrg

        # Require enough pixels in both regions
        if nin < MIN_IN_PX or nrg < MIN_RING_PX:
            continue

        # Use median intensity for robustness
        Iin = float(np.median(img_out[inner_mask]))
        Irg = float(np.median(img_out[ring_mask]))

        I_in[i] = Iin
        I_ring[i] = Irg
        LER[i] = (Iin + EPS) / (Irg + EPS)
        valid[i] = True

    # Add computed results to dataframe
    df["cell_id_tracked"] = cell_id_tracked
    df["I_in"] = I_in
    df["I_ring"] = I_ring
    df["LER"] = LER
    df["n_in"] = n_in
    df["n_ring"] = n_ring
    df["valid_LER"] = valid
    df["synthetic_no_locs"] = False

    # Find tracked cells that never received any localization
    tracked_cells = set(pd.unique(linked_df["cell_id"].astype(int)))
    cells_with_any_assigned_locs = set(
        pd.unique(df.loc[df["cell_id_tracked"] > 0, "cell_id_tracked"].astype(int))
    )
    missing_cells = sorted(tracked_cells - cells_with_any_assigned_locs)

    # Add one synthetic row per missing cell
    if missing_cells:
        synth = pd.DataFrame({
            time_col: 0,
            "x": np.nan,
            "y": np.nan,
            "cell_id_tracked": missing_cells,
            "I_in": np.nan,
            "I_ring": np.nan,
            "LER": float(NO_LOCS_CELL_LER_VALUE),
            "n_in": 0,
            "n_ring": 0,
            "valid_LER": True,
            "synthetic_no_locs": True,
        })

        # Preserve any extra columns from the original locs dataframe
        for c in df.columns:
            if c not in synth.columns:
                synth[c] = np.nan

        synth = synth[df.columns]
        df = pd.concat([df, synth], ignore_index=True)

    # Quick sanity summary
    n_valid = int(np.sum(df["valid_LER"].to_numpy(bool)))
    n_with_cell = int(np.sum((df["cell_id_tracked"].to_numpy(int) > 0) & df["valid_LER"].to_numpy(bool)))
    n_synth = int(np.sum(df["synthetic_no_locs"].to_numpy(bool)))
    log(f"  Sanity: valid_LER={n_valid} | valid & cell_id_tracked>0={n_with_cell} | synthetic_no_locs_rows={n_synth}")

    return df


def _fallback_locs_df() -> pd.DataFrame:
    """
    Return an empty locs dataframe with the minimum required columns.

    Used when a run has no localization CSV at all.
    """
    return pd.DataFrame({
        "frame": pd.Series(dtype=int),
        "x": pd.Series(dtype=float),
        "y": pd.Series(dtype=float),
    })


def process_run(run_dir: Path) -> str:
    """
    Process a single run folder.

    Steps:
    - load tracked ROIs
    - load TIFF movie
    - load localization CSV if present
    - compute LER
    - save output CSV
    """
    linked_path = run_dir / "cell_detection" / "linked_rois.pkl"
    if not linked_path.exists():
        return f"SKIP no linked_rois.pkl: {run_dir}"

    movie_path = pick_movie_tif(run_dir, CHANNEL_TAG)
    if movie_path is None:
        return f"SKIP no tif/tiff for {CHANNEL_TAG}: {run_dir}"

    stack = load_movie_stack_ultra(movie_path)
    if stack.shape[0] <= 1:
        return f"SKIP stack has T<=1 (shape={stack.shape}): {movie_path.name} @ {run_dir}"

    linked_df = pd.read_pickle(linked_path)
    if linked_df.empty:
        return f"SKIP linked_rois.pkl empty: {run_dir}"

    need = {"frame", "cell_id", "contour"}
    if not need.issubset(set(linked_df.columns)):
        return f"SKIP linked_rois.pkl missing cols {need - set(linked_df.columns)}: {run_dir}"

    locs_path = pick_locs_csv(run_dir, CHANNEL_TAG)

    # Case 1: no locs file exists -> still produce synthetic output
    if locs_path is None:
        out_csv = run_dir / f"{run_dir.name}_{CHANNEL_TAG}_locs_with_LER_trackedCellIDs.csv"
        if out_csv.exists() and not OVERWRITE:
            return f"SKIP exists (no locs present in run): {out_csv.name} @ {run_dir}"

        df = _fallback_locs_df()
        scored = compute_LER_tracked(stack, df, linked_df)
        scored.to_csv(out_csv, index=False)

        n_cells_tracked = int(linked_df["cell_id"].nunique())
        n_cells_used = int(pd.Series(scored.loc[scored["valid_LER"], "cell_id_tracked"]).nunique())
        n_synth = int(scored["synthetic_no_locs"].sum())

        return (
            f"DONE {out_csv.name} (NO locs file) | tracked_cells_in_pkl={n_cells_tracked} "
            f"| cells_in_output={n_cells_used} | synthetic_rows={n_synth} | movie={movie_path.name} @ {run_dir}"
        )

    # Case 2: locs file exists -> compute LER normally
    out_csv = locs_path.with_name(locs_path.stem + "_with_LER_trackedCellIDs.csv")
    if out_csv.exists() and not OVERWRITE:
        return f"SKIP exists: {out_csv.name} @ {run_dir}"

    df = pd.read_csv(locs_path)
    scored = compute_LER_tracked(stack, df, linked_df)
    scored.to_csv(out_csv, index=False)

    valid_frac = float(scored["valid_LER"].mean()) if "valid_LER" in scored.columns and len(scored) else 0.0
    nonzero_frac = float((scored["cell_id_tracked"] > 0).mean()) if "cell_id_tracked" in scored.columns and len(scored) else 0.0

    n_cells_tracked = int(linked_df["cell_id"].nunique())
    n_cells_used = int(pd.Series(scored.loc[scored["valid_LER"], "cell_id_tracked"]).nunique())
    n_synth = int(scored["synthetic_no_locs"].sum())

    return (
        f"DONE {out_csv.name} | valid={valid_frac:.3f} | cell_id_tracked>0={nonzero_frac:.3f} "
        f"| tracked_cells_in_pkl={n_cells_tracked} | cells_seen_in_valid_LER={n_cells_used} "
        f"| synthetic_no_locs_rows={n_synth} | movie={movie_path.name} @ {run_dir}"
    )


def main() -> None:
    """Run the analysis over all detected run folders."""
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

    run_dirs = find_run_dirs(ROOT)
    log(f"ROOT: {ROOT}")
    log(f"Found {len(run_dirs)} Run folders")

    for rd in run_dirs:
        try:
            log(process_run(rd))
        except Exception as e:
            log(f"ERROR @ {rd}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()