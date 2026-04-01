# -*- coding: utf-8 -*-
"""
SHP2 density analysis (ALL FRAMES) — save one CSV per run

Folder structure:
  root/<cell_type>/<condition>/<date>/(WellXX/)?RunXXXXX

For each run:
- loads linked_rois.pkl (linked_df)
- loads rois.npy just for (T,H,W)
- loads Picasso locs CSV (x,y,t)
- if locs CSV is missing, still processes the run using synthetic zero-spot data
- for each frame t:
    - assigns reproducible labels (stable sort)
    - builds markers_2d (H,W) by painting contour pixels
    - counts SHP2 spots per label using markers_2d[y,x]
    - computes density = n_shp2 / area
- saves one output CSV per run

Output:
- one CSV per run under OUT, preserving experiment subfolder structure
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# Config (fixed names)
# =========================
CELL_TYPES = ["PD1-wt", "PD1-T3", "PD1-absent"]
CONDITIONS = ["CD58", "pMHC", "PD-L1"]

RUN_PREFIX = "Run"
WELL_PREFIX = "Well"

CELL_DETECTION_DIR = "cell_detection"
LINKED_NAME = "linked_rois.pkl"
ROIS_NAME = "rois.npy"

SHP2_LOCS_FILENAME = "Pat01_561nm_locs.csv"


# =========================
# Helpers: directory walking
# =========================
def find_run_dirs(root: Path) -> list[Path]:
    """
    Scan tree:
      root/<cell_type>/<condition>/<date>/(WellXX/)?RunXXXXX
    """
    run_dirs: list[Path] = []
    date_dirs = sorted(root.glob("*/*/*"))

    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue

        well_dirs = sorted(date_dir.glob(f"{WELL_PREFIX}*"))
        if well_dirs:
            for w in well_dirs:
                run_dirs += sorted([p for p in w.glob(f"{RUN_PREFIX}*") if p.is_dir()])
        else:
            run_dirs += sorted([p for p in date_dir.glob(f"{RUN_PREFIX}*") if p.is_dir()])

    return run_dirs


def infer_metadata(root: Path, run_dir: Path) -> dict:
    rel = run_dir.relative_to(root).parts
    meta = dict(
        cell_type=rel[0],
        ligand_condition=rel[1],
        date=rel[2],
        well=rel[3] if len(rel) >= 5 and rel[3].startswith("Well") else "",
        run=run_dir.name,
    )
    return meta


# =========================
# Helpers: IO
# =========================
def load_picasso_locs(path: Path) -> pd.DataFrame:
    """
    Load picasso localize locs.
    Accepts:
      - columns (x,y,t)
      - or (x,y,frame) which we rename to t
    """
    for sep in [",", "\t", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if "t" not in df.columns and "frame" in df.columns:
                df = df.rename(columns={"frame": "t"})
            if {"x", "y", "t"}.issubset(df.columns):
                return df
        except Exception:
            continue

    raise ValueError(f"Could not parse locs file: {path}")


def empty_spots_df() -> pd.DataFrame:
    """
    Synthetic empty SHP2 localization table.
    Used when the run has no SHP2 locs file at all.
    """
    return pd.DataFrame({
        "x": pd.Series(dtype=float),
        "y": pd.Series(dtype=float),
        "t": pd.Series(dtype=int),
    })


# =========================
# Per-frame labeling + markers
# =========================
def build_markers_for_frame_and_add_label(linked_df: pd.DataFrame, frame: int, shape):
    """
    For one frame:
    - filter linked_df to that frame
    - stable sort -> reproducible labels within the frame
    - paint contour pixels into markers_2d (H,W) with those labels

    Returns
    -------
    df_f      : linked_df filtered to 'frame', with 'label' int column
    markers_2d: (H,W) int32 label image for this frame
    """
    _, H, W = shape

    df = linked_df.copy()
    df["frame"] = df["frame"].astype(int)

    df_f = df[df["frame"] == int(frame)].copy()
    if df_f.empty:
        return df_f, np.zeros((H, W), dtype=np.int32)

    # stable order -> reproducible labels (within this frame)
    df_f = df_f.sort_values(["bbox_y0", "bbox_x0", "y", "x"], kind="mergesort")
    df_f["label"] = np.arange(1, len(df_f) + 1, dtype=np.int32)

    markers_2d = np.zeros((H, W), dtype=np.int32)

    # paint contours into 2D markers
    for row in df_f.itertuples(index=False):
        coords = row.contour
        if coords is None or len(coords) == 0:
            continue

        yy = coords[:, 0]
        xx = coords[:, 1]

        m = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if np.any(m):
            markers_2d[yy[m], xx[m]] = int(row.label)

    return df_f, markers_2d


def add_spot_counts_and_density_for_frame(
    linked_f_df: pd.DataFrame,
    markers_2d: np.ndarray,
    spots_df: pd.DataFrame,
    frame: int,
    count_col="n_spots",
    density_col="spot_density",
    area_col="area",
):
    """
    For one frame, vectorized:
    - filter spots to this frame
    - label lookup via markers_2d[y,x]
    - count labels with np.bincount
    - attach counts to linked_f_df using 'label'
    - density = count / area
    """
    H, W = markers_2d.shape

    out = linked_f_df.copy()
    if out.empty:
        return out

    if "t" not in spots_df.columns:
        raise ValueError("spots_df must contain column 't'")

    s = spots_df[spots_df["t"].astype(int) == int(frame)].copy()
    if s.empty:
        out[count_col] = 0
        out[density_col] = 0.0
        return out

    # integer pixel positions
    x = np.rint(s["x"].to_numpy()).astype(np.int32)
    y = np.rint(s["y"].to_numpy()).astype(np.int32)

    # in-bounds
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[inb]
    y = y[inb]

    if x.size == 0:
        out[count_col] = 0
        out[density_col] = 0.0
        return out

    # label lookup
    labels = markers_2d[y, x].astype(np.int32)
    labels = labels[labels > 0]

    if labels.size == 0:
        out[count_col] = 0
    else:
        max_lab = int(out["label"].max())
        counts = np.bincount(labels, minlength=max_lab + 1)
        out[count_col] = out["label"].map(lambda lab: int(counts[int(lab)])).astype(np.int32)

    denom = out[area_col].replace(0, np.nan)
    out[density_col] = (out[count_col] / denom).fillna(0.0)

    return out


# =========================
# Save path helper
# =========================
def make_run_output_path(root: Path, out_dir: Path, run_dir: Path) -> Path:
    """
    Preserve folder structure under OUT and create one CSV per run.

    Example:
      run_dir = root/PD1-wt/PD-L1/2026-02-20/Well01/Run00012

    Output:
      out_dir/PD1-wt/PD-L1/2026-02-20/Well01/Run00012_shp2_density_all_frames.csv
    """
    rel_parent = run_dir.relative_to(root).parent
    out_subdir = out_dir / rel_parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    return out_subdir / f"{run_dir.name}_shp2_density_all_frames.csv"


# =========================
# Per-run analysis
# =========================
def analyze_single_run(root: Path, run_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    meta = infer_metadata(root, run_dir)

    print(f"  Cell type : {meta['cell_type']}")
    print(f"  Condition : {meta['ligand_condition']}")
    print(f"  Date      : {meta['date']}")
    if meta["well"]:
        print(f"  Well      : {meta['well']}")
    print(f"  Run       : {meta['run']}")

    if meta["cell_type"] not in CELL_TYPES or meta["ligand_condition"] not in CONDITIONS:
        print("  → SKIP (unknown cell type or condition)\n")
        return None

    cd = run_dir / CELL_DETECTION_DIR
    linked_p = cd / LINKED_NAME
    rois_p = cd / ROIS_NAME
    shp2_p = run_dir / SHP2_LOCS_FILENAME

    if not linked_p.exists() or not rois_p.exists():
        print("  → SKIP (missing cell_detection outputs)\n")
        return None

    linked_df = pd.read_pickle(linked_p)

    # mmap_mode helps if rois.npy is large; we only need shape
    rois = np.load(rois_p, mmap_mode="r")
    T, H, W = rois.shape

    # NEW: if locs file is missing, use synthetic empty spots dataframe
    if not shp2_p.exists():
        print("  → No SHP2 locs file found; using synthetic zero-SHP2 values for all tracked cells")
        shp2_df = empty_spots_df()
        shp2_locs_file_used = ""
        synthetic_no_locs_file = True
    else:
        print(f"  → Using SHP2 locs: {shp2_p.name}")
        shp2_df = load_picasso_locs(shp2_p)
        shp2_df["t"] = shp2_df["t"].astype(int)
        shp2_locs_file_used = shp2_p.name
        synthetic_no_locs_file = False

    print(f"Frames: {T}")

    # pre-group spots by frame
    spots_by_t = {t: g for t, g in shp2_df.groupby("t", sort=False)}

    run_out_frames = []

    for t in range(T):
        linked_f, markers_2d = build_markers_for_frame_and_add_label(linked_df, t, rois.shape)
        if linked_f.empty:
            continue

        spots_f = spots_by_t.get(t, None)
        if spots_f is None:
            spots_f = shp2_df.iloc[0:0].copy()

        out_f = add_spot_counts_and_density_for_frame(
            linked_f_df=linked_f,
            markers_2d=markers_2d,
            spots_df=spots_f,
            frame=t,
            count_col="n_shp2",
            density_col="shp2_density_px",
            area_col="area",
        )

        # keep 1 row per cell per frame
        out_f = out_f.drop_duplicates(subset=["cell_id", "frame"])
        run_out_frames.append(out_f)

        if (t % 50) == 0 or t == T - 1:
            print(f"    frame {t:>4}/{T-1} | cells: {out_f['cell_id'].nunique():>3}")

    if not run_out_frames:
        raise RuntimeError("No frames produced output (linked_df empty across frames?)")

    out_all = pd.concat(run_out_frames, ignore_index=True)

    out_all["cell_type"] = meta["cell_type"]
    out_all["ligand_condition"] = meta["ligand_condition"]
    out_all["date"] = meta["date"]
    out_all["well"] = meta["well"]
    out_all["run"] = meta["run"]
    out_all["run_path"] = str(run_dir)
    out_all["shp2_locs_file"] = shp2_locs_file_used
    out_all["synthetic_missing_shp2_locs_file"] = synthetic_no_locs_file
    out_all["n_frames_total"] = T
    out_all["image_height_px"] = H
    out_all["image_width_px"] = W

    out_csv = make_run_output_path(root, out_dir, run_dir)
    out_all.to_csv(out_csv, index=False)

    print(f"  → Total rows (cell,frame): {len(out_all)}")
    print(f"  → Unique cells in run    : {out_all['cell_id'].nunique()}")
    print(f"  → Saved: {out_csv}\n")

    return out_all


# =========================
# Main analysis with progress
# =========================
def analyze_root(root: Path, out_dir: Path):
    print(f"\n[INFO] Scanning experiment tree under:\n       {root}\n")

    run_dirs = find_run_dirs(root)
    total_runs = len(run_dirs)
    print(f"[INFO] Found {total_runs} run folders total\n")

    processed = 0
    failed = 0
    skipped = 0

    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[RUN {idx} / {total_runs}]")

        try:
            out = analyze_single_run(root, run_dir, out_dir)
            if out is None:
                skipped += 1
            else:
                processed += 1

        except Exception as e:
            failed += 1
            print(f"  → ERROR: {type(e).__name__}: {e}\n")
            continue

    print("\n[SUMMARY]")
    print(f"  Runs found total : {total_runs}")
    print(f"  Runs processed   : {processed}")
    print(f"  Runs skipped     : {skipped}")
    print(f"  Runs failed      : {failed}")
    print(f"  Output root      : {out_dir}")


if __name__ == "__main__":
    ROOT = Path(r"\\sun\ganzinger\home-folder\raghuram\2026_02_analysis\output")
    OUT = ROOT / "_shp2_density_per_run_2"

    analyze_root(ROOT, OUT)