#%%
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
from picasso.io import TiffMultiMap, load_movie
import os
from glob import glob
import matplotlib.pyplot as plt
from skimage.draw import polygon
from tqdm import tqdm
from numba import njit

# =============================
# USER SETTINGS
# =============================
ROOT = Path(r"D:\Data\20251215_T6_mutants\output")
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

#%%
def log(msg: str) -> None:
    """Print a message immediately."""
    print(msg, flush=True)

def build_tracked_label_frames(linked_df: pd.DataFrame, T: int, H: int, W: int) -> list[np.ndarray]:
    """
    Build one filled label image per frame.

    Output:
        lbls[t][y, x] = tracked cell_id if pixel is inside that tracked cell
        lbls[t][y, x] = 0 otherwise
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

            rr = coords[:, 0].astype(float)
            cc = coords[:, 1].astype(float)

            # Fill the contour polygon
            fill_rr, fill_cc = polygon(rr, cc, shape=(H, W))

            lbl[fill_rr, fill_cc] = cid

    return lbls

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

# def compute_LER_tracked(stack: np.ndarray, locs_df: pd.DataFrame, watershed: np.ndarray, linked_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute LER for each localization using tracked cell identities.

#     For each localization:
#     - identify which tracked cell it belongs to
#     - extract an inner region around the localization
#     - extract an outer ring = outer box minus Picasso box
#     - restrict both regions to the same tracked cell
#     - compute:
#           LER = median(inner) / median(ring)

#     Also adds synthetic rows for tracked cells with no assigned localizations.
#     """
#     # Validate geometry settings
#     if PICK_BOX_SIDE % 2 != 1:
#         raise ValueError("PICK_BOX_SIDE must be odd (e.g., 11).")
#     if INNER_BOX_SIDE % 2 != 1:
#         raise ValueError("INNER_BOX_SIDE must be odd (e.g., 7).")
#     if OUTER_BOX_SIDE % 2 != 1:
#         raise ValueError("OUTER_BOX_SIDE must be odd (e.g., 15).")
#     if INNER_BOX_SIDE > PICK_BOX_SIDE:
#         raise ValueError("INNER_BOX_SIDE must be <= PICK_BOX_SIDE.")
#     if OUTER_BOX_SIDE <= PICK_BOX_SIDE:
#         raise ValueError("OUTER_BOX_SIDE must be > PICK_BOX_SIDE.")

#     T, H, W = stack.shape
#     df = locs_df.copy()

#     # If no locs exist, create an empty dataframe with the required columns
#     if df.empty:
#         df = pd.DataFrame({
#             "frame": pd.Series(dtype=int),
#             "x": pd.Series(dtype=float),
#             "y": pd.Series(dtype=float),
#         })

#     time_col = 't'

#     # Create per-frame label maps from tracked contours
#     # lbls = build_tracked_label_frames(linked_df, T=T, H=H, W=W)
#     lbls = watershed.copy() 
    

#     hw_in = INNER_BOX_SIDE // 2
#     hw_pick = PICK_BOX_SIDE // 2
#     hw_out = OUTER_BOX_SIDE // 2

#     n = len(df)

#     # # Output arrays
#     I_in = np.full(n, np.nan, dtype=np.float32)
#     I_ring = np.full(n, np.nan, dtype=np.float32)
#     LER = np.full(n, np.nan, dtype=np.float32)
#     n_in = np.zeros(n, dtype=np.int32)
#     n_ring = np.zeros(n, dtype=np.int32)
#     valid = np.zeros(n, dtype=bool)
#     cell_id_tracked = np.zeros(n, dtype=np.int32)

#     for i, row in enumerate(df.itertuples(index=False)):
#         t = int(getattr(row, time_col))
#         if t < 0 or t >= T:
#             continue

#         x = float(getattr(row, "x"))
#         y = float(getattr(row, "y"))
#         cx, cy = int(round(x)), int(round(y))

#         # Skip out-of-bounds localizations
#         if cx < 0 or cx >= W or cy < 0 or cy >= H:
#             continue

#         lbl = lbls[t]
#         cid = int(lbl[cy, cx])


#         # Skip if localization is not assigned to a tracked cell
#         if cid == 0:
#             continue

#         cell_id_tracked[i] = cid
#         frame = stack[t]

#         # Outer box around localization
#         y_out, x_out = box_slices(cx, cy, hw_out, H, W)
#         img_out = frame[y_out, x_out]
#         lbl_out = lbl[y_out, x_out]

#         # Inner box
#         y_in, x_in = box_slices(cx, cy, hw_in, H, W)

#         # Picasso box (excluded from the ring)
#         y_pick, x_pick = box_slices(cx, cy, hw_pick, H, W)

#         # Coordinates relative to the outer box
#         y0, x0 = y_out.start, x_out.start

#         # Build inner mask inside the outer crop
#         inner_mask = np.zeros_like(lbl_out, dtype=bool)
#         inner_mask[
#             slice(y_in.start - y0, y_in.stop - y0),
#             slice(x_in.start - x0, x_in.stop - x0)
#         ] = True

#         # Build Picasso box mask inside the outer crop
#         pick_mask = np.zeros_like(lbl_out, dtype=bool)
#         pick_mask[
#             slice(y_pick.start - y0, y_pick.stop - y0),
#             slice(x_pick.start - x0, x_pick.stop - x0)
#         ] = True

#         # Ring = outer box minus Picasso box
#         ring_mask = ~pick_mask

#         # Only keep pixels belonging to the same tracked cell
#         same_cell = (lbl_out == cid)
#         inner_mask &= same_cell
#         ring_mask &= same_cell

#         nin = int(inner_mask.sum())
#         nrg = int(ring_mask.sum())
#         n_in[i] = nin
#         n_ring[i] = nrg

#         # Require enough pixels in both regions
#         if nin < MIN_IN_PX or nrg < MIN_RING_PX:
#             continue

#         # Use median intensity for robustness
#         Iin = float(np.median(img_out[inner_mask]))
#         Irg = float(np.median(img_out[ring_mask]))

#         I_in[i] = Iin
#         I_ring[i] = Irg
#         LER[i] = (Iin + EPS) / (Irg + EPS)
#         valid[i] = True

#     # Add computed results to dataframe
#     df["cell_id_tracked"] = cell_id_tracked
#     df["I_in"] = I_in
#     df["I_ring"] = I_ring
#     df["LER"] = LER
#     df["n_in"] = n_in
#     df["n_ring"] = n_ring
#     df["valid_LER"] = valid
#     df["synthetic_no_locs"] = False

#     # Find tracked cells that never received any localization
#     tracked_cells = set(pd.unique(linked_df["cell_id"].astype(int)))
#     cells_with_any_assigned_locs = set(
#         pd.unique(df.loc[df["cell_id_tracked"] > 0, "cell_id_tracked"].astype(int))
#     )
#     missing_cells = sorted(tracked_cells - cells_with_any_assigned_locs)

#     # Add one synthetic row per missing cell
#     if missing_cells:
#         synth = pd.DataFrame({
#             time_col: 0,
#             "x": np.nan,
#             "y": np.nan,
#             "cell_id_tracked": missing_cells,
#             "I_in": np.nan,
#             "I_ring": np.nan,
#             "LER": float(NO_LOCS_CELL_LER_VALUE),
#             "n_in": 0,
#             "n_ring": 0,
#             "valid_LER": True,
#             "synthetic_no_locs": True,
#         })

#         # Preserve any extra columns from the original locs dataframe
#         for c in df.columns:
#             if c not in synth.columns:
#                 synth[c] = np.nan

#         synth = synth[df.columns]
#         df = pd.concat([df, synth], ignore_index=True)

#     # Quick sanity summary
#     n_valid = int(np.sum(df["valid_LER"].to_numpy(bool)))
#     n_with_cell = int(np.sum((df["cell_id_tracked"].to_numpy(int) > 0) & df["valid_LER"].to_numpy(bool)))
#     n_synth = int(np.sum(df["synthetic_no_locs"].to_numpy(bool)))
#     log(f"  Sanity: valid_LER={n_valid} | valid & cell_id_tracked>0={n_with_cell} | synthetic_no_locs_rows={n_synth}")

#     return df

@njit
def median_from_buffer(buf, n):
    """
    Compute the median of the first n values in buf.
    """
    if n <= 0:
        return np.nan

    tmp = np.empty(n, dtype=np.float32)

    for i in range(n):
        tmp[i] = buf[i]

    tmp.sort()

    mid = n // 2

    if n % 2 == 1:
        return tmp[mid]
    else:
        return 0.5 * (tmp[mid - 1] + tmp[mid])

@njit
def median_from_buffer(buf, n):
    """
    Compute the median of the first n values in buf.
    """
    if n <= 0:
        return np.nan

    tmp = np.empty(n, dtype=np.float32)

    for i in range(n):
        tmp[i] = buf[i]

    tmp.sort()

    mid = n // 2

    if n % 2 == 1:
        return tmp[mid]
    else:
        return 0.5 * (tmp[mid - 1] + tmp[mid])

@njit
def compute_LER_core_numba(
    stack,
    lbls,
    work_idx,
    t_arr,
    cx_arr,
    cy_arr,
    watershed_label,
    I_in,
    I_ring,
    LER,
    n_in,
    n_ring,
    valid,
    H,
    W,
    hw_in,
    hw_pick,
    hw_out,
    min_in_px,
    min_ring_px,
    eps,
):
    """
    Fast LER core.

    Same logic as the Python loop:
    - use only pixels from the same cell
    - inner region = INNER_BOX
    - ring region = OUTER_BOX minus PICK_BOX
    - LER = median(inner) / median(ring)
    """

    max_inner_pixels = (2 * hw_in + 1) * (2 * hw_in + 1)
    max_outer_pixels = (2 * hw_out + 1) * (2 * hw_out + 1)

    inner_buf = np.empty(max_inner_pixels, dtype=np.float32)
    ring_buf = np.empty(max_outer_pixels, dtype=np.float32)

    for wi in range(len(work_idx)):
        i = work_idx[wi]

        t = t_arr[i]
        cx = cx_arr[i]
        cy = cy_arr[i]
        lab = watershed_label[i]

        nin = 0
        nrg = 0

        y0 = cy - hw_out
        y1 = cy + hw_out + 1
        x0 = cx - hw_out
        x1 = cx + hw_out + 1

        if y0 < 0:
            y0 = 0
        if x0 < 0:
            x0 = 0
        if y1 > H:
            y1 = H
        if x1 > W:
            x1 = W

        for yy in range(y0, y1):
            dy = yy - cy
            if dy < 0:
                dy = -dy

            for xx in range(x0, x1):
                dx = xx - cx
                if dx < 0:
                    dx = -dx

                # Only pixels from the same tracked cell
                if lbls[t, yy, xx] != lab:
                    continue

                val = stack[t, yy, xx]

                # Inner box
                if dx <= hw_in and dy <= hw_in:
                    inner_buf[nin] = val
                    nin += 1

                # Ring = outer box minus Picasso box
                if dx > hw_pick or dy > hw_pick:
                    ring_buf[nrg] = val
                    nrg += 1

        n_in[i] = nin
        n_ring[i] = nrg

        if nin < min_in_px or nrg < min_ring_px:
            continue

        Iin = median_from_buffer(inner_buf, nin)
        Irg = median_from_buffer(ring_buf, nrg)

        I_in[i] = Iin
        I_ring[i] = Irg
        LER[i] = (Iin + eps) / (Irg + eps)
        valid[i] = True

def compute_LER_tracked(
    stack: np.ndarray,
    locs_df: pd.DataFrame,
    watershed: np.ndarray,
    linked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute LER for each localization using watershed cell labels.

    For each localization:
    - get the tracked cell id from watershed[t, y, x]
    - extract an inner box around the localization
    - extract an outer ring = outer box minus Picasso box
    - restrict both regions to the same cell
    - compute:
          LER = median(inner) / median(ring)

    Also adds synthetic rows for tracked cells with no assigned localizations.
    """

    # Validate geometry settings
    if PICK_BOX_SIDE % 2 != 1:
        raise ValueError("PICK_BOX_SIDE must be odd, e.g. 11.")
    if INNER_BOX_SIDE % 2 != 1:
        raise ValueError("INNER_BOX_SIDE must be odd, e.g. 7.")
    if OUTER_BOX_SIDE % 2 != 1:
        raise ValueError("OUTER_BOX_SIDE must be odd, e.g. 15.")
    if INNER_BOX_SIDE > PICK_BOX_SIDE:
        raise ValueError("INNER_BOX_SIDE must be <= PICK_BOX_SIDE.")
    if OUTER_BOX_SIDE <= PICK_BOX_SIDE:
        raise ValueError("OUTER_BOX_SIDE must be > PICK_BOX_SIDE.")


    T, H, W = stack.shape

    if watershed.shape[:3] != stack.shape[:3]:
        raise ValueError(
            f"Stack and watershed shapes do not match. "
            f"stack={stack.shape}, watershed={watershed.shape}"
        )

    df = locs_df.copy()

    if df.empty:
        df = pd.DataFrame({
            "t": pd.Series(dtype=int),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
        })

    time_col = "t"

    required_cols = {time_col, "x", "y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Locs missing columns {missing}. Columns: {list(df.columns)}")

    # Do not copy watershed; just read from it
    lbls = watershed

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

    # --------------------------------------------------
    # Vectorized first pass: get t, x, y, cx, cy, cell_id
    # --------------------------------------------------
    t_arr = df[time_col].to_numpy(dtype=np.int32)
    x_arr = df["x"].to_numpy(dtype=float)
    y_arr = df["y"].to_numpy(dtype=float)

    cx_arr = np.rint(x_arr).astype(np.int32)
    cy_arr = np.rint(y_arr).astype(np.int32)

    in_bounds = (
        (t_arr >= 0) & (t_arr < T) &
        (cx_arr >= 0) & (cx_arr < W) &
        (cy_arr >= 0) & (cy_arr < H)
    )

    idx = np.where(in_bounds)[0]

    # Get raw watershed label for all valid localizations at once
    watershed_label = np.zeros(n, dtype=np.int32)

    watershed_label[idx] = lbls[
        t_arr[idx],
        cy_arr[idx],
        cx_arr[idx]
    ].astype(np.int32)

    # Add watershed label to df so we can join to linked_df
    df["watershed_label"] = watershed_label

    # linked_df maps:
    #   frame + label  ->  tracked cell_id
    #
    # locs df has:
    #   t + watershed_label
    lookup = (
        linked_df[["frame", "label", "cell_id"]]
        .drop_duplicates()
        .rename(columns={
            "frame": time_col,
            "label": "watershed_label",
            "cell_id": "cell_id_tracked",
        })
    )

    df = df.merge(
        lookup,
        on=[time_col, "watershed_label"],
        how="left"
    )

    df["cell_id_tracked"] = (
        df["cell_id_tracked"]
        .fillna(0)
        .astype(np.int32)
    )

    cell_id_tracked = df["cell_id_tracked"].to_numpy(dtype=np.int32)
    watershed_label = df["watershed_label"].to_numpy(dtype=np.int32)

    # Only localizations with a tracked cell_id need LER computation
    work_idx = np.where(cell_id_tracked > 0)[0]

    print(f"  Total locs: {n}")
    print(f"  In bounds: {len(idx)}")
    print(f"  Inside cell: {len(work_idx)}")

    # --------------------------------------------------
    # Main LER computation loop
    # --------------------------------------------------
    compute_LER_core_numba(
    stack=stack.astype(np.float32, copy=False),
    lbls=lbls.astype(np.int32, copy=False),
    work_idx=work_idx.astype(np.int64, copy=False),
    t_arr=t_arr.astype(np.int32, copy=False),
    cx_arr=cx_arr.astype(np.int32, copy=False),
    cy_arr=cy_arr.astype(np.int32, copy=False),
    watershed_label=watershed_label.astype(np.int32, copy=False),
    I_in=I_in,
    I_ring=I_ring,
    LER=LER,
    n_in=n_in,
    n_ring=n_ring,
    valid=valid,
    H=H,
    W=W,
    hw_in=hw_in,
    hw_pick=hw_pick,
    hw_out=hw_out,
    min_in_px=MIN_IN_PX,
    min_ring_px=MIN_RING_PX,
    eps=EPS,
)

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
        pd.unique(
            df.loc[df["cell_id_tracked"] > 0, "cell_id_tracked"].astype(int)
        )
    )

    missing_cells = sorted(tracked_cells - cells_with_any_assigned_locs)

    if missing_cells:
        synth_rows = []

        for cid in missing_cells:
            row = {c: np.nan for c in df.columns}

            row[time_col] = 0
            row["x"] = np.nan
            row["y"] = np.nan
            row["cell_id_tracked"] = cid
            row["I_in"] = np.nan
            row["I_ring"] = np.nan
            row["LER"] = float(NO_LOCS_CELL_LER_VALUE)
            row["n_in"] = 0
            row["n_ring"] = 0
            row["valid_LER"] = True
            row["synthetic_no_locs"] = True

            if "watershed_label" in row:
                row["watershed_label"] = 0

            synth_rows.append(row)

        synth = pd.DataFrame(synth_rows, columns=df.columns)

        for c in df.columns:
            try:
                synth[c] = synth[c].astype(df[c].dtype)
            except (ValueError, TypeError):
                pass

        df = pd.concat([df, synth], ignore_index=True)

    n_valid = int(np.sum(df["valid_LER"].to_numpy(bool)))
    n_with_cell = int(
        np.sum(
            (df["cell_id_tracked"].to_numpy(int) > 0) &
            df["valid_LER"].to_numpy(bool)
        )
    )
    n_synth = int(np.sum(df["synthetic_no_locs"].to_numpy(bool)))

    # log(
    #     f"  Sanity: valid_LER={n_valid} | "
    #     f"valid & cell_id_tracked>0={n_with_cell} | "
    #     f"synthetic_no_locs_rows={n_synth}"
    # )

    return df

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
    linked_path = os.path.join(run_dir, "cell_detection", "linked_rois.pkl")
    if not os.path.exists(linked_path):
        return f"SKIP no linked_rois.pkl: {run_dir}"
    movie_path = Path(glob(os.path.join(str(run_dir), f"*{CHANNEL_TAG}*.tif"))[0])
    # movie_path = pick_movie_tif(run_dir, CHANNEL_TAG)
    if movie_path is None:
        return f"SKIP no tif/tiff for {CHANNEL_TAG}: {run_dir}"
    stack = np.asarray(load_movie(movie_path)[0])
    if stack.shape[0] <= 1:
        return f"SKIP stack has T<=1 (shape={stack.shape}): {movie_path.name} @ {run_dir}"

    linked_df = pd.read_pickle(linked_path)
    if linked_df.empty:
        return f"SKIP linked_rois.pkl empty: {run_dir}"

    need = {"frame", "cell_id", "contour"}
    if not need.issubset(set(linked_df.columns)):
        return f"SKIP linked_rois.pkl missing cols {need - set(linked_df.columns)}: {run_dir}"

    locs_path = Path(glob(os.path.join(str(run_dir), f"*{CHANNEL_TAG}_locs.csv"))[0])
    # locs_path = pick_locs_csv(run_dir, CHANNEL_TAG)

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


    watershed = np.load(os.path.join(run_dir, "cell_detection", "watershed.npy"))
    scored = compute_LER_tracked(stack, df, watershed, linked_df)
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

    run_dirs = sorted(
        Path(p) for p in glob(str(ROOT / "**" / "Run*"), recursive=True)
        if Path(p).is_dir()
        )
    run_dirs = sorted(run_dirs)
    log(f"ROOT: {ROOT}")
    log(f"Found {len(run_dirs)} Run folders")

    for rd in tqdm(run_dirs, desc="Processing runs"):
        try:
            log(process_run(rd))
        except Exception as e:
            log(f"ERROR @ {rd}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()