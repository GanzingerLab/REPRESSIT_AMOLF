# -*- coding: utf-8 -*-
"""
Run-level SHP2 density analysis using the last N frames of each run
with optional track-length filtering
===============================================================

Folder structure:
  root/<cell_type>/<condition>/<date>/(WellXX/)?RunXXXXX

For each run:
- load linked_rois.pkl (linked_df)
- load rois.npy only to obtain (T, H, W)
- optionally load original Picasso locs CSV (x, y, t)
- optionally load localization-level trackpy CSV (locID, track.id, ...)
- optionally load trackpy stats HDF with per-track statistics
- if tracking files are present, keep only tracks with length >= MIN_TRACK_LENGTH
- analyze the last REQUIRED_N_FRAMES frames of each run
- compute mean cell area across frames for each cell
- count persistent SHP2 tracks per cell when possible
- if locs/tracks are missing or no tracks survive, assign SHP2 density = 0 to all cells
- save one CSV per run with one row per cell

Important
---------
This script counts persistent SHP2 tracks per cell when tracking files exist.
If tracking information is unavailable, or if no qualifying tracks are found,
all cells in the analyzed frame window are retained with zero SHP2 counts/density.

Input files used if available:
1) Pat01_561nm_locs.csv
2) Pat01_561nm_locs_nm_trackpy.csv
3) Pat01_561nm_locs_nm_trackpy_stats.hdf
4) linked_rois.pkl
5) rois.npy

Output example:
    Run00012_shp2_total_density_last10frames_tracklen_ge2.csv
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

# Optional original localization file
SHP2_LOCS_FILENAME = "Pat01_561nm_locs.csv"

# Optional localization-level tracking output
TRACKED_LOCS_FILENAME = "Pat01_561nm_locs_nm_trackpy.csv"

# Optional track summary output
TRACK_STATS_FILENAME = "Pat01_561nm_locs_nm_trackpy_stats.hdf"
TRACK_STATS_KEY = "df_stats"

# Keep only tracks present in at least this many frames
MIN_TRACK_LENGTH = 2

# Analyze the last N frames of each run
REQUIRED_N_FRAMES = 10


# =========================
# Directory discovery
# =========================
def find_run_dirs(root: Path) -> list[Path]:
    """
    Scan a directory tree with structure:
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
    """Infer metadata from the run directory path."""
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
# File loading
# =========================
def load_picasso_locs(path: Path) -> pd.DataFrame:
    """
    Load original Picasso localize output.

    Accepted column sets:
      - (x, y, t)
      - (x, y, frame), which is renamed to t
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


def load_tracked_locs_csv(path: Path) -> pd.DataFrame:
    """
    Load localization-level tracking CSV.

    Expected useful columns:
      - track.id
      - t
      - x
      - y
    """
    for sep in [",", "\t", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            required = {"track.id", "t", "x", "y"}
            if required.issubset(df.columns):
                return df
        except Exception:
            continue

    raise ValueError(
        f"Could not parse tracked locs file or missing required cols "
        f"{{'track.id','t','x','y'}}: {path}"
    )


def load_track_stats_hdf(path: Path, key: str = TRACK_STATS_KEY) -> pd.DataFrame:
    """
    Load track summary HDF with one row per track.

    Expected useful columns:
      - track.id
      - length
    """
    try:
        df = pd.read_hdf(path, key=key)
    except Exception as e:
        raise ValueError(f"Could not read track stats HDF {path}: {type(e).__name__}: {e}")

    required = {"track.id", "length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Track stats HDF missing required columns {missing}: {path}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


# =========================
# Marker helpers
# =========================
def build_cellid_marker_stack(linked_df: pd.DataFrame, shape):
    """
    Build a 3D marker stack whose values are cell_id + 1.

    Background remains 0.
    """
    T, H, W = shape

    df = linked_df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["cell_id"] = pd.to_numeric(df["cell_id"], errors="coerce")

    df = df.dropna(subset=["frame", "cell_id"]).copy()
    df["frame"] = df["frame"].astype(int)
    df["cell_id"] = df["cell_id"].astype(int)

    markers_3d = np.zeros((T, H, W), dtype=np.int32)

    for row in df.itertuples(index=False):
        t = int(row.frame)

        coords = row.contour
        if coords is None or len(coords) == 0:
            continue

        yy = coords[:, 0]
        xx = coords[:, 1]

        m = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if np.any(m):
            markers_3d[t, yy[m], xx[m]] = int(row.cell_id) + 1

    return markers_3d


def map_spot_to_cell_id(markers, spots_df):
    """
    Map tracked SHP2 spots to cell IDs using marker lookup.

    Assumes x/y in the tracked CSV are in nm and converts them to px
    by dividing by 91.
    """
    T, H, W = markers.shape

    s = spots_df.copy()
    if "t" not in s.columns:
        raise ValueError("spots_df must contain a 't' column (frame index).")

    s["t"] = pd.to_numeric(s["t"], errors="coerce")
    s["x"] = pd.to_numeric(s["x"], errors="coerce")
    s["y"] = pd.to_numeric(s["y"], errors="coerce")
    s = s.dropna(subset=["t", "x", "y"]).copy()

    if s.empty:
        s["cell_id"] = pd.Series(dtype=int)
        return s

    s["t_int"] = s["t"].astype(int)
    s["x_int"] = np.rint((s["x"] / 91.0).to_numpy()).astype(int)
    s["y_int"] = np.rint((s["y"] / 91.0).to_numpy()).astype(int)

    inb = (
        (s["t_int"] >= 0) & (s["t_int"] < T) &
        (s["x_int"] >= 0) & (s["x_int"] < W) &
        (s["y_int"] >= 0) & (s["y_int"] < H)
    )
    s = s.loc[inb].copy()

    if s.empty:
        s["cell_id"] = pd.Series(dtype=int)
        return s

    s["cell_id"] = markers[
        s["t_int"].to_numpy(),
        s["y_int"].to_numpy(),
        s["x_int"].to_numpy()
    ].astype(int)

    s = s[s["cell_id"] > 0].copy()
    s["cell_id"] = s["cell_id"] - 1
    return s


# =========================
# Output path helper
# =========================
def make_run_output_path(root: Path, out_dir: Path, run_dir: Path) -> Path:
    """
    Preserve the input folder structure under OUT and create one CSV per run.
    """
    rel_parent = run_dir.relative_to(root).parent
    out_subdir = out_dir / rel_parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    return out_subdir / f"{run_dir.name}_shp2_total_density_last{REQUIRED_N_FRAMES}frames_tracklen_ge{MIN_TRACK_LENGTH}.csv"


# =========================
# Per-run analysis
# =========================
def analyze_single_run(root: Path, run_dir: Path, out_dir: Path) -> pd.DataFrame | None:
    """Analyze one run and save a run-level SHP2 density table."""
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
    tracked_locs_p = run_dir / TRACKED_LOCS_FILENAME
    track_stats_p = run_dir / TRACK_STATS_FILENAME

    if not linked_p.exists() or not rois_p.exists():
        print("  → SKIP (missing cell_detection outputs)\n")
        return None

    rois = np.load(rois_p, mmap_mode="r")
    T, H, W = rois.shape

    if T < REQUIRED_N_FRAMES:
        print(f"  → SKIP (run has only {T} frames, fewer than required {REQUIRED_N_FRAMES})\n")
        return None

    start_frame = T - REQUIRED_N_FRAMES
    end_frame = T - 1
    print(f"  → Analyzing last {REQUIRED_N_FRAMES} frames: {start_frame} to {end_frame}")

    linked_df = pd.read_pickle(linked_p)
    linked_df["frame"] = pd.to_numeric(linked_df["frame"], errors="coerce")
    linked_df["cell_id"] = pd.to_numeric(linked_df["cell_id"], errors="coerce")
    linked_df = linked_df.dropna(subset=["frame", "cell_id"]).copy()
    linked_df["frame"] = linked_df["frame"].astype(int)
    linked_df["cell_id"] = linked_df["cell_id"].astype(int)

    linked_df_filtered = linked_df[linked_df["frame"] >= start_frame].copy()
    if linked_df_filtered.empty:
        print("  → SKIP (no tracked cells in analyzed frame window)\n")
        return None

    # Summarize all cells present in the analyzed frame window.
    area_summary = (
        linked_df_filtered
        .groupby("cell_id", as_index=True)
        .agg(area=("area", "mean"))
    )

    frames_present = (
        linked_df_filtered
        .groupby("cell_id")["frame"]
        .nunique()
        .rename("n_frames_cell_present")
    )

    base = area_summary.merge(frames_present, on="cell_id", how="left")
    base["n_frames_cell_present"] = base["n_frames_cell_present"].fillna(0).astype(int)

    # Default: assign zero SHP2 counts to all cells unless qualifying tracks are found.
    counts = pd.DataFrame(index=base.index, data={"n_shp2_total": 0})

    shp2_locs_file_used = shp2_p.name if shp2_p.exists() else ""
    tracked_locs_file_used = tracked_locs_p.name if tracked_locs_p.exists() else ""
    track_stats_file_used = track_stats_p.name if track_stats_p.exists() else ""

    synthetic_missing_locs_file = not shp2_p.exists()
    synthetic_missing_tracked_locs_file = not tracked_locs_p.exists()
    synthetic_missing_track_stats_file = not track_stats_p.exists()

    # Persistent-track counting is only attempted when both tracking files exist.
    if tracked_locs_p.exists() and track_stats_p.exists():
        print(f"  → Using tracked locs : {tracked_locs_p.name}")
        print(f"  → Using track stats  : {track_stats_p.name}")

        tracked_locs_df = load_tracked_locs_csv(tracked_locs_p)
        tracked_locs_df["t"] = pd.to_numeric(tracked_locs_df["t"], errors="coerce")
        tracked_locs_df = tracked_locs_df.dropna(subset=["t"]).copy()
        tracked_locs_df["t"] = tracked_locs_df["t"].astype(int)

        # Restrict to the analyzed frame window and renumber to 0..REQUIRED_N_FRAMES-1.
        tracked_locs_df = tracked_locs_df[tracked_locs_df["t"] >= start_frame].copy()
        tracked_locs_df["t"] = tracked_locs_df["t"] - start_frame

        if tracked_locs_df.empty:
            print("  → No tracked SHP2 spots in analyzed window; assigning zero density to all cells")
        else:
            markers_3d = build_cellid_marker_stack(linked_df_filtered, rois.shape)
            markers_3d_filtered = markers_3d[start_frame:].copy()

            tracked_locs_df_2 = map_spot_to_cell_id(markers_3d_filtered, tracked_locs_df)

            if tracked_locs_df_2.empty:
                print("  → No tracked SHP2 spots map to cells; assigning zero density to all cells")
            else:
                track_to_cell = (
                    tracked_locs_df_2[["track.id", "cell_id"]]
                    .dropna()
                    .drop_duplicates(subset="track.id")
                    .copy()
                )

                if track_to_cell.empty:
                    print("  → No track-to-cell mappings found; assigning zero density to all cells")
                else:
                    track_ids_in_window = set(
                        pd.to_numeric(track_to_cell["track.id"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .tolist()
                    )

                    track_stats_df = load_track_stats_hdf(track_stats_p)
                    track_stats_df["track.id"] = pd.to_numeric(track_stats_df["track.id"], errors="coerce")
                    track_stats_df["length"] = pd.to_numeric(track_stats_df["length"], errors="coerce")
                    track_stats_df = track_stats_df.dropna(subset=["track.id", "length"]).copy()
                    track_stats_df["track.id"] = track_stats_df["track.id"].astype(int)

                    track_stats_df_longer = track_stats_df[
                        (track_stats_df["length"] >= MIN_TRACK_LENGTH) &
                        (track_stats_df["track.id"].isin(track_ids_in_window))
                    ].copy()

                    if track_stats_df_longer.empty:
                        print("  → No tracks survive MIN_TRACK_LENGTH filter; assigning zero density to all cells")
                    else:
                        track_stats_df_longer = track_stats_df_longer.drop(columns="cell_id", errors="ignore")
                        track_stats_df_longer = track_stats_df_longer.merge(
                            track_to_cell,
                            on="track.id",
                            how="left"
                        )

                        counts = (
                            track_stats_df_longer
                            .dropna(subset=["cell_id"])
                            .groupby("cell_id", as_index=True)
                            .agg(n_shp2_total=("track.id", "count"))
                        )

                        if counts.empty:
                            print("  → No persistent tracks assigned to cells; assigning zero density to all cells")
                            counts = pd.DataFrame(index=base.index, data={"n_shp2_total": 0})
                        else:
                            print(f"  → Cells with persistent SHP2 tracks: {len(counts)}")

    else:
        if not tracked_locs_p.exists():
            print("  → Missing tracked locs CSV; assigning zero density to all cells")
        if not track_stats_p.exists():
            print("  → Missing track stats HDF; assigning zero density to all cells")

    # Merge counts back onto all cells present in the analyzed frame window.
    combined = base.merge(counts, on="cell_id", how="left")
    combined["n_shp2_total"] = combined["n_shp2_total"].fillna(0).astype(int)

    combined["shp2_density_total_per_mean_area_px"] = np.where(
        combined["area"] > 0,
        combined["n_shp2_total"] / combined["area"],
        0.0
    )

    combined["mean_n_shp2_per_frame"] = combined["n_shp2_total"] / REQUIRED_N_FRAMES
    combined["n_frames_cell_counted"] = REQUIRED_N_FRAMES
    combined["min_track_length"] = MIN_TRACK_LENGTH
    combined["required_n_frames"] = REQUIRED_N_FRAMES
    combined["n_frames_total"] = T
    combined["image_height_px"] = H
    combined["image_width_px"] = W
    combined["run_path"] = str(run_dir)

    combined["shp2_locs_file"] = shp2_locs_file_used
    combined["tracked_locs_file"] = tracked_locs_file_used
    combined["track_stats_file"] = track_stats_file_used

    combined["synthetic_missing_shp2_locs_file"] = synthetic_missing_locs_file
    combined["synthetic_missing_tracked_locs_file"] = synthetic_missing_tracked_locs_file
    combined["synthetic_missing_track_stats_file"] = synthetic_missing_track_stats_file

    combined["cell_type"] = meta["cell_type"]
    combined["ligand_condition"] = meta["ligand_condition"]
    combined["date"] = meta["date"]
    combined["well"] = meta["well"]
    combined["run"] = meta["run"]

    combined = combined.reset_index()

    combined = combined[
        [
            "cell_type",
            "ligand_condition",
            "date",
            "well",
            "run",
            "cell_id",
            "n_shp2_total",
            "area",
            "shp2_density_total_per_mean_area_px",
            "mean_n_shp2_per_frame",
            "n_frames_cell_present",
            "n_frames_cell_counted",
            "min_track_length",
            "required_n_frames",
            "n_frames_total",
            "image_height_px",
            "image_width_px",
            "run_path",
            "shp2_locs_file",
            "tracked_locs_file",
            "track_stats_file",
            "synthetic_missing_shp2_locs_file",
            "synthetic_missing_tracked_locs_file",
            "synthetic_missing_track_stats_file",
        ]
    ]

    out_csv = make_run_output_path(root, out_dir, run_dir)
    combined.to_csv(out_csv, index=False)

    print(f"  → Cells in output: {len(combined)}")
    print(f"  → Cells with zero SHP2 density: {(combined['n_shp2_total'] == 0).sum()}")
    print(f"  → Saved: {out_csv}\n")

    return combined


# =========================
# Main analysis
# =========================
def analyze_root(root: Path, out_dir: Path):
    """Run the analysis over all detected run folders and print progress."""
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
    OUT = ROOT / "_shp2_total_density_per_run_10frames"

    analyze_root(ROOT, OUT)