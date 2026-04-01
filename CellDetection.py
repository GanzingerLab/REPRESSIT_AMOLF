# -*- coding: utf-8 -*-
"""
Consistent per-dataset cell detection + tracking pipeline

What this script does (high-level):
1) Load 561nm + 638nm stacks from a postSPIT Cell_Analyzer run folder
2) Segment cells in each channel (custom thresholds) → combine masks
3) Optionally split touching cells per-frame (watershed)
4) Extract per-frame connected components → (x,y,bbox, pixel coords)
5) Track components over time using trackpy
6) Filter:
   - remove tracks that ever touch the border (bbox-based)
   - keep only tracks that "ever land" (area + intensity gate)
   - require N frames after first landing frame
7) Save outputs (pkl, overlay tiff, rois.npy, params_used.json)
"""

import os
import json

import numpy as np
import pandas as pd
import trackpy as tp

from postSPIT import tirf_analysis as plc

from skimage import exposure
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_opening,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import find_boundaries, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from matplotlib import colormaps
from tifffile import imwrite
from pathlib import Path
from pathlib import Path

from pathlib import Path

def process_experiment_tree(
    root,
    params,
    run_prefix="Run",
    well_prefix="Well",
    depth=3,
):
    """
    Walk folder tree:
      cell_type -> ligand_condition -> experiment_date -> (optional) well -> run folders

    Examples supported:
      root/PD1wt/PD-L1/2026_02_01/Well 1/Run00001
      root/PD1wt/PD-L1/2026_02_01/Run00001

    Parameters
    ----------
    root : str or Path
        Base path that contains cell type folders.
    params : dict
        PARAMS dict.
    run_prefix : str
        Prefix for run folders (default "Run").
    well_prefix : str
        Prefix for well folders (default "Well").
    depth : int
        How many folder levels after root correspond to:
        cell_type (1) -> ligand (2) -> date (3). Default 3.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    def _is_valid_dir(p: Path) -> bool:
        return p.is_dir() and not p.name.startswith(".")

    # collect all "date-level" dirs = directories exactly `depth` below root
    # i.e. root/*/*/* when depth=3
    pattern = "*/" * depth
    date_level_dirs = sorted([p for p in root.glob(pattern) if _is_valid_dir(p)])

    if not date_level_dirs:
        print(f"[WARN] No experiment-date folders found under: {root}")
        print(f"       Expected something like: {root}/<cell_type>/<ligand>/<date>/ ...")
        return

    total_runs = 0
    total_failed = 0

    print(f"[INFO] Found {len(date_level_dirs)} experiment-date folders under: {root}")

    for date_dir in date_level_dirs:
        # date_dir = root/cell_type/ligand/date
        rel = date_dir.relative_to(root).parts
        cell_type = rel[0] if len(rel) > 0 else "UNKNOWN_CELLTYPE"
        ligand = rel[1] if len(rel) > 1 else "UNKNOWN_LIGAND"
        date_name = rel[2] if len(rel) > 2 else date_dir.name

        # Case A: date_dir contains well folders
        well_dirs = sorted([p for p in date_dir.glob(f"{well_prefix}*") if _is_valid_dir(p)])

        if well_dirs:
            for well_dir in well_dirs:
                run_dirs = sorted([p for p in well_dir.glob(f"{run_prefix}*") if _is_valid_dir(p)])
                if not run_dirs:
                    print(f"[WARN] No runs in: {well_dir}")
                    continue

                print(f"\n[INFO] {cell_type} / {ligand} / {date_name} / {well_dir.name}: {len(run_dirs)} runs")
                for run_dir in run_dirs:
                    total_runs += 1
                    print(f"[INFO] Processing: {cell_type} / {ligand} / {date_name} / {well_dir.name} / {run_dir.name}")
                    try:
                        process_run(str(run_dir), params)
                    except Exception as e:
                        total_failed += 1
                        print(f"[ERROR] Failed on {run_dir}: {e}")

        # Case B: date_dir directly contains run folders (no wells)
        else:
            run_dirs = sorted([p for p in date_dir.glob(f"{run_prefix}*") if _is_valid_dir(p)])
            if not run_dirs:
                print(f"[WARN] No wells AND no runs in: {date_dir}")
                continue

            print(f"\n[INFO] {cell_type} / {ligand} / {date_name}: {len(run_dirs)} runs (no wells)")
            for run_dir in run_dirs:
                total_runs += 1
                print(f"[INFO] Processing: {cell_type} / {ligand} / {date_name} / {run_dir.name}")
                try:
                    process_run(str(run_dir), params)
                except Exception as e:
                    total_failed += 1
                    print(f"[ERROR] Failed on {run_dir}: {e}")

    print(f"\n[OK] Finished. Total runs attempted: {total_runs} | failed: {total_failed} | succeeded: {total_runs - total_failed}")



# -------------------------
# PARAMS (freeze per dataset)
# -------------------------
PARAMS = dict(
    seg=dict(
        clip_limit=0.03,          # CLAHE clip limit
        phansalkar_k=0.02,
        phansalkar_radius=50,
        min_obj=800,             # remove tiny objects (per frame)
        min_hole=5000,           # fill holes inside cells (per frame)
        do_opening=True,          # morphological opening on final mask (per frame)
    ),
    split=dict(
        do_watershed=True,
        # Tune ONCE per dataset/day based on cell diameter in px.
        # If not splitting enough -> lower. If over-splitting -> raise.
        min_distance=4,
        min_size=900,
        # accept split only if both parts are plausible
        min_part_fraction=0.45,   # (currently not enforced in your logic; kept for future)
        max_parts_keep=2,         # (currently not enforced in your logic; kept for future)
    ),
    track=dict(
        search_range=50,          # max displacement per frame for linking
        memory=8,                 # how many frames a particle can vanish and come back
    ),
    filter=dict(
        # Border cleanup: remove whole tracks that ever touch border
        border_margin_px=1,

        # "Landing" definition: keep cells that EVER look like a real landed cell
        min_area_contact=1600,    # px; set once per dataset
        min_int_quantile=0.05,    # self-normalizing threshold based on max intensity per cell

        # Require at least N frames after first landing frame (inclusive)
        min_frames_after_land=8,
    ),
    io=dict(
        overlay_cmap="tab20",
    ),
)


# -------------------------
# Core helpers
# -------------------------
def find_cells_v2(analyzer, image_stack, seg_params):
    """
    Segment cells for an entire movie stack.

    Parameters
    ----------
    analyzer : plc.Cell_Analyzer
        Your postSPIT analyzer instance (provides custom threshold helpers).
    image_stack : np.ndarray, shape (T, H, W)
        Raw image stack for a single channel.
    seg_params : dict
        Segmentation parameters (CLAHE + threshold + cleanup).

    Returns
    -------
    phan_mask : np.ndarray, bool, shape (T, H, W)
        Mask from Phansalkar threshold.
    li_mask : np.ndarray, bool, shape (T, H, W)
        Mask from Li threshold.
    final_mask : np.ndarray, bool, shape (T, H, W)
        Combined + cleaned mask used downstream.
    """
    # CLAHE is 2D, so apply frame-by-frame
    eq = np.zeros_like(image_stack, dtype=float)
    for t in range(image_stack.shape[0]):
        eq[t] = exposure.equalize_adapthist(
            image_stack[t],
            clip_limit=seg_params["clip_limit"],
        )

    # Your custom threshold implementations (postSPIT)
    phan_mask = analyzer._phansalkar_threshold(
        eq,
        k=seg_params["phansalkar_k"],
        radius=seg_params["phansalkar_radius"],
    )
    li_mask = analyzer._li_threshold(eq, eq[0], mode="median")

    # Combine thresholds (logical AND) → then clean up
    mask = phan_mask & li_mask
    mask = analyzer._remove_small_objects_per_frame(
        mask,
        min_size=seg_params["min_obj"],
        connectivity=0,
    )
    mask = analyzer._remove_small_holes_per_frame(
        mask,
        min_size=seg_params["min_hole"],
        connectivity=0,
    )

    if seg_params.get("do_opening", True):
        mask = binary_opening(mask)

    return phan_mask, li_mask, mask


def split_touching_cells_per_frame(mask_stack, split_params):
    """
    Optional watershed split for touching cells, frame-by-frame.

    Notes
    -----
    - Uses EDT distance map and local maxima as markers.
    - Output is boolean union (labels_ws > 0).
      i.e., you are NOT keeping per-cell labels; you only improve separation for regionprops.

    Parameters
    ----------
    mask_stack : np.ndarray, bool, shape (T, H, W)
    split_params : dict

    Returns
    -------
    out : np.ndarray, bool, shape (T, H, W)
    """
    if not split_params.get("do_watershed", True):
        return mask_stack.astype(bool)

    min_distance = split_params["min_distance"]
    min_size = split_params.get("min_size", 2000)

    out = np.zeros_like(mask_stack, dtype=bool)

    for t in range(mask_stack.shape[0]):
        mask = mask_stack[t].astype(bool)

        # Quick cleanup helps watershed behave more consistently
        mask = remove_small_objects(mask, min_size=min_size)
        mask = remove_small_holes(mask, area_threshold=50000)

        if mask.sum() == 0:
            continue

        # Distance to nearest background pixel (peaks ≈ cell centers)
        dist = ndi.distance_transform_edt(mask)

        # Peak coordinates used as seed markers
        coords = peak_local_max(dist, min_distance=min_distance, labels=mask)

        # If only one peak, nothing to split
        if coords.shape[0] < 2:
            out[t] = mask
            continue

        # Build marker image from peak coords
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i

        # Ensure markers are properly labeled connected components
        markers = ndi.label(markers > 0)[0]

        # Watershed on negative distance (so peaks become basins)
        labels_ws = watershed(-dist, markers, mask=mask)

        # Keep union mask (boolean); regionprops() will label components again later
        out[t] = labels_ws > 0

    return out


def extract_regions_per_frame(mask_stack):
    """
    Extract connected components per frame into a DataFrame.

    IMPORTANT: 'contour' here is actually ALL PIXEL COORDS of the region (region.coords),
    not the boundary.

    Returns columns:
      frame, contour(coords), x, y, bbox_x0/x1/y0/y1
    """
    rows = []

    for frame_idx, frame_mask in enumerate(mask_stack):
        labeled = label(frame_mask)
        for region in regionprops(labeled):
            y0, x0, y1, x1 = region.bbox
            y, x = region.centroid

            rows.append(
                dict(
                    frame=frame_idx,
                    contour=region.coords,  # pixels belonging to the region
                    x=x,
                    y=y,
                    bbox_x0=x0,
                    bbox_x1=x1,
                    bbox_y0=y0,
                    bbox_y1=y1,
                )
            )

    return pd.DataFrame(rows)


def save_contours_overlay(images, tracked_df, out_path, cmap_name="tab20"):
    """
    Save an RGB TIFF stack with boundary overlays for each tracked cell_id.

    For each frame:
      - normalize grayscale image
      - render as RGB
      - draw boundary pixels (find_boundaries) in a unique color per cell_id
    """
    n_frames, h, w = images.shape

    if tracked_df.empty:
        print(f"[WARN] No tracked detections to overlay: {out_path}")
        return

    cell_ids = tracked_df["cell_id"].unique()

    cmap = colormaps[cmap_name].resampled(max(len(cell_ids), 1))
    color_map = {cid: np.array(cmap(i)[:3]) for i, cid in enumerate(cell_ids)}

    overlays = []
    for frame_idx in range(n_frames):
        img = images[frame_idx].astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_rgb = np.dstack([img, img, img])

        frame_rows = tracked_df[tracked_df["frame"] == frame_idx]
        for _, row in frame_rows.iterrows():
            cid = row["cell_id"]
            coords = row["contour"]
            color = color_map[cid]

            # Region pixels -> boundary -> color overlay
            region_mask = np.zeros((h, w), dtype=bool)
            region_mask[coords[:, 0], coords[:, 1]] = True
            perimeter = find_boundaries(region_mask, mode="outer")
            img_rgb[perimeter] = color

        overlays.append((img_rgb * 255).astype(np.uint8))

    imwrite(out_path, np.asarray(overlays), photometric="rgb")
    print(f"[OK] Saved overlay TIFF: {out_path}")


# -------------------------
# Filters (border + landed-ever)
# -------------------------
def apply_border_filter(df, H, W, margin):
    """
    Remove entire tracks (cell_id) that ever touch the border (within margin),
    based on bbox coords.
    """
    if df.empty:
        return df

    df = df.copy()
    df["touches_border"] = (
        (df["bbox_x0"] <= margin)
        | (df["bbox_y0"] <= margin)
        | (df["bbox_x1"] >= (W - margin))
        | (df["bbox_y1"] >= (H - margin))
    )

    edge_ids = df.loc[df["touches_border"], "cell_id"].unique()
    return df[~df["cell_id"].isin(edge_ids)].reset_index(drop=True)


def compute_area_and_intensity(df, img561, img638):
    """
    Add:
      - area: number of pixels in region (len(region.coords))
      - mean_int_561: mean intensity over region pixels in 561 channel for that frame
      - mean_int_638: mean intensity over region pixels in 638 channel for that frame
      - mean_int: combined intensity used for filtering (max of both by default)
    """
    if df.empty:
        return df

    df = df.copy()
    df["area"] = df["contour"].apply(lambda coords: int(coords.shape[0]))

    def _mean_intensity(img_stack, row):
        frame = int(row["frame"])
        coords = row["contour"]
        return float(img_stack[frame][coords[:, 0], coords[:, 1]].mean())

    df["mean_int_561"] = df.apply(lambda row: _mean_intensity(img561, row), axis=1)
    df["mean_int_638"] = df.apply(lambda row: _mean_intensity(img638, row), axis=1)

    # Combine channels for filtering:
    # max = "cell is bright in either channel" (recommended)
    df["mean_int"] = df[["mean_int_561", "mean_int_638"]].max(axis=1)

    # Alternative combos (pick one if you prefer):
    # df["mean_int"] = df["mean_int_561"] + df["mean_int_638"]     # sum
    # df["mean_int"] = 0.5*df["mean_int_561"] + 0.5*df["mean_int_638"]  # average

    return df



def apply_landing_filter(df, filter_params):
    """
    Keep cells that EVER pass:
      area >= min_area_contact AND mean_int >= intensity_threshold

    Intensity threshold is self-normalizing:
      - compute max mean_int per cell_id
      - threshold = quantile(q) of those max values

    Also require at least N frames after first landing frame (inclusive).
    """
    if df.empty:
        return df

    df = df.copy()
    min_area = filter_params["min_area_contact"]
    q = filter_params["min_int_quantile"]
    min_frames_after = filter_params["min_frames_after_land"]

    # Per-cell max stats (used only to set a run-specific intensity threshold)
    cell_stats = df.groupby("cell_id").agg(
    max_int_561=("mean_int_561", "max"),
    max_int_638=("mean_int_638", "max"),
    max_area=("area", "max"),
    )

    # threshold based on "best channel" per cell
    cell_stats["max_int_any"] = cell_stats[["max_int_561", "max_int_638"]].max(axis=1)
    int_thresh = float(cell_stats["max_int_any"].quantile(q))


    keep_ids = []
    for cid, d in df.groupby("cell_id"):
        landed_frames = d[(d["area"] >= min_area) & (d["mean_int"] >= int_thresh)]
        if landed_frames.empty:
            continue

        first_land = int(landed_frames["frame"].min())
        last_seen = int(d["frame"].max())

        # inclusive frame count from first landing to last seen
        frames_after = last_seen - first_land + 1

        if frames_after >= min_frames_after:
            keep_ids.append(cid)

    return df[df["cell_id"].isin(keep_ids)].reset_index(drop=True)


# -------------------------
# Main runner (single run)
# -------------------------
def process_run(path, params):
    """
    Process a single run folder:
      - load stacks
      - segment + combine
      - split touching
      - extract regions
      - track
      - filter
      - save outputs
    """
    analyzer = plc.Cell_Analyzer(path)

    # Load channels as (T,H,W)
    img561 = np.asarray(analyzer.images["561nm"])
    img638 = np.asarray(analyzer.images["638nm"])

    # Segment each channel using the same rules
    _, _, mask561 = find_cells_v2(analyzer, img561, params["seg"])
    _, _, mask638 = find_cells_v2(analyzer, img638, params["seg"])

    # Combine masks across channels and remove tiny objects again
    full_mask = analyzer._remove_small_objects_per_frame(
        (mask561 | mask638),
        min_size=params["seg"]["min_obj"],
        connectivity=0,
    )

    # Split touching cells (optional)
    full_mask = split_touching_cells_per_frame(full_mask, params["split"])

    # Extract per-frame connected components
    regions_df = extract_regions_per_frame(full_mask)
    if regions_df.empty:
        print(f"[WARN] No detections in: {path}")
        return

    # Track across frames (trackpy expects x,y,frame columns)
    linked_df = tp.link_df(
        regions_df,
        search_range=params["track"]["search_range"],
        memory=params["track"]["memory"],
    ).rename(columns={"particle": "cell_id"})

    # Output folder
    out_dir = os.path.join(path, "cell_detection")
    os.makedirs(out_dir, exist_ok=True)

    # Save params used (reproducibility)
    with open(os.path.join(out_dir, "params_used.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Filters
    H, W = full_mask.shape[1], full_mask.shape[2]
    linked_df = apply_border_filter(linked_df, H, W, params["filter"]["border_margin_px"])
    linked_df = compute_area_and_intensity(linked_df, img561, img638)
    linked_df = apply_landing_filter(linked_df, params["filter"])

    # Save outputs
    linked_df.to_pickle(os.path.join(out_dir, "linked_rois.pkl"))
    np.save(os.path.join(out_dir, "rois.npy"), full_mask)

    save_contours_overlay(
        img561,
        linked_df,
        os.path.join(out_dir, "overlay_cells.tiff"),
        cmap_name=params["io"]["overlay_cmap"],
    )

    print(f"[OK] Done: {path} | kept cell_ids: {linked_df['cell_id'].nunique()}")

# -------------------------
# Run it
# -------------------------
if __name__ == "__main__":
    root = r"\\sun\ganzinger\home-folder\raghuram\example\output"
    process_experiment_tree(root, PARAMS, run_prefix="Run", well_prefix="Well", depth=3)


