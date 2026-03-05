#%%
from postSPIT import plotting_classes as plc
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects
)
from glob import glob
from skimage import exposure
from skimage.measure import label, regionprops
import pandas as pd
import trackpy as tp
import os
from tqdm import tqdm 
import re
from skimage.draw import polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tifffile import imsave
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from matplotlib import colormaps
from tifffile import imwrite
import datetime
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
# %%
def find_cells_Gerard(image):
    image = exposure.equalize_adapthist(image, clip_limit=0.02)
    b = a._phansalkar_threshold(image, k = 0.05, radius = 80, p=5.0, q=12.0)
    c = a._li_threshold(image, image[0], mode = 'median')
    d = binary_opening(a._remove_small_holes_per_frame(a._remove_small_objects_per_frame(c&b, min_size=2000, connectivity=0), min_size = 50000, connectivity = 0))
    return b, c, d
def extract_contours_per_frame(mask_sequence):
    all_data = []

    for frame_index, frame_mask in enumerate(mask_sequence):
        labeled = label(frame_mask)
        regions = regionprops(labeled)

        for region in regions:
            y0, x0, y1, x1 = region.bbox
            y, x = region.centroid  # unpack centroid

            cell_info = {
                'frame': frame_index,
                'contour': region.coords,
                'x': x,
                'y': y,
                'bbox_x0': x0,
                'bbox_x1': x1,
                'bbox_y0': y0,
                'bbox_y1': y1
            }
            all_data.append(cell_info)

    return pd.DataFrame(all_data)
def contour_to_mask(contour, image_shape):
    y_coords, x_coords = contour[:, 0], contour[:, 1]
    rr, cc = polygon(y_coords, x_coords, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def save_contours_overlay(images, tracked_df, out_path, cmap_name='tab20'):
    n_frames, h, w = images.shape
    particle_ids = tracked_df['cell_id'].unique()
    
    
    cmap = colormaps[cmap_name]
    cmap = cmap.resampled(len(particle_ids))

    color_map = {pid: np.array(cmap(i)[:3]) for i, pid in enumerate(particle_ids)}

    overlays = []
    for frame_idx in range(n_frames):
        img = images[frame_idx].astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_rgb = np.dstack([img, img, img])

        regions = tracked_df[tracked_df['frame'] == frame_idx]
        for _, region in regions.iterrows():
            pid = region['cell_id']
            color = color_map[pid]
            coords = region['contour']

            from skimage.segmentation import find_boundaries
            mask = np.zeros((h, w), dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            perimeter = find_boundaries(mask, mode="outer")

            img_rgb[perimeter] = color

        overlays.append((img_rgb * 255).astype(np.uint8))

    imwrite(out_path, np.array(overlays), photometric='rgb')
    print(f"Saved overlay TIFF to {out_path}")
def segment_watershed(mask_frame, 
    min_distance=25):
    # Split fused nuclei using distance-based watershed on nuclei mask
    dist = distance_transform_edt(mask_frame)
    peaks = peak_local_max(
        dist,
        labels=mask_frame,
        min_distance=min_distance,   
        threshold_abs=0
    )
    seed = np.zeros_like(dist, dtype=np.int32)
    seed[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    
    markers = watershed(-dist, seed, mask=mask_frame)  # one label per nucleus
    return markers
def segment_watershed2(mask_frame, min_distance=25, h=25.0, smooth_sigma=1.0):
    dist = distance_transform_edt(mask_frame).astype(np.float32)

    # Optional: smooth the distance map to remove tiny ripples from bumps
    if smooth_sigma and smooth_sigma > 0:
        dist = gaussian_filter(dist, smooth_sigma)

    # Keep only “prominent” maxima
    maxima = h_maxima(dist, h=h)          # boolean image
    seed = label(maxima)                  # markers

    markers = watershed(-dist, seed, mask=mask_frame)
    return markers

def segment_watershed_stack(mask,
    min_distance=15):
    T = mask.shape[0]
    out = np.zeros_like(mask, dtype=np.int32)
    for i in range(T):
        out[i] = segment_watershed2(mask[i], min_distance)
    return out
def labels_to_trackpy_features(label_stack, contour_mode="outer"):
    """
    label_stack: (T,H,W) labeled objects (0 background)

    Returns a DataFrame with: frame, x, y, label (frame-local id), area, mass
    """
    rows = []
    T = label_stack.shape[0]

    for t in range(T):
        lab = label_stack[t]

        for rp in regionprops(lab):
            y, x = rp.centroid  # row, col
            y0, x0, y1, x1 = rp.bbox
            perim = find_boundaries(lab == rp.label, mode=contour_mode)
            coords = np.column_stack(np.nonzero(perim))  # (N,2)

            row = {
                "frame": t,
                "x": float(x),
                "y": float(y),
                "label": int(rp.label),     # label id within THIS frame
                "area": float(rp.area),
                "contour": coords,
                "bbox_x0": int(x0),
                "bbox_x1": int(x1),
                "bbox_y0": int(y0),
                "bbox_y1": int(y1),
            }

            rows.append(row)
    return pd.DataFrame(rows)


def add_spot_counts_and_density(
    linked_df,
    markers,
    spots_df,
    count_col="n_spots",
    density_col="spot_density",
    area_col="area",
):
    """
    Adds per-(frame,cell_id) spot counts and spot density.

    linked_df must have: frame, label, cell_id, and area_col (area in pixels)
    spots_df must have: t, x, y  (x=col, y=row)
    markers: (T,H,W) int labels (0 background)
    """

    # Map (frame,label) -> cell_id
    map_df = linked_df[["frame", "label", "cell_id"]].drop_duplicates()
    key_to_cell = {(int(r.frame), int(r.label)): int(r.cell_id) for r in map_df.itertuples(index=False)}

    T, H, W = markers.shape

    s = spots_df.copy()
    if "t" not in s.columns:
        raise ValueError("spots_df must contain a 't' column (frame index).")

    # integer coords
    s["t_int"] = s["t"].astype(int)
    s["x_int"] = np.rint(s["x"].to_numpy()).astype(int)
    s["y_int"] = np.rint(s["y"].to_numpy()).astype(int)

    # in-bounds
    inb = (
        (s["t_int"] >= 0) & (s["t_int"] < T) &
        (s["x_int"] >= 0) & (s["x_int"] < W) &
        (s["y_int"] >= 0) & (s["y_int"] < H)
    )
    s = s.loc[inb].copy()

    # label lookup
    s["label"] = markers[s["t_int"].to_numpy(), s["y_int"].to_numpy(), s["x_int"].to_numpy()].astype(int)
    s = s[s["label"] > 0].copy()

    # map to tracked cell_id
    s["frame"] = s["t_int"]
    s["cell_id"] = [key_to_cell.get((int(f), int(l)), -1) for f, l in zip(s["frame"], s["label"])]
    s = s[s["cell_id"] >= 0].copy()

    # counts per (frame, cell_id)
    counts = (
        s.groupby(["frame", "cell_id"])
         .size()
         .rename(count_col)
         .reset_index()
    )

    # merge counts into linked_df
    out = linked_df.merge(counts, on=["frame", "cell_id"], how="left")
    out[count_col] = out[count_col].fillna(0).astype(int)

    # density = count / area (area is pixels unless you convert)
    out[density_col] = out[count_col] / out[area_col].replace(0, np.nan)
    out[density_col] = out[density_col].fillna(0.0)

    return out


# %%
path = r'D:\Data\shraddha'
run_pattern = re.compile(r'^Run\d+$')
run_paths = []
for root, dirs, files in os.walk(path):
    matching_dirs = [d for d in dirs if run_pattern.match(d)]
    for d in matching_dirs:
        full_path = os.path.join(root, d)
        run_paths.append(full_path)
    dirs[:] = [d for d in dirs if not run_pattern.match(d)]

# %%
values  = []
file = r'cell_detection\intensities.pkl'

for fol in tqdm(run_paths):
    if os.path.exists(os.path.join(fol, file)):
        result = pd.read_pickle(os.path.join(fol, file))
        frames = result.frame.unique()
        num_frames = len(frames)
        
        for cell_id, cell_df in result.groupby('cell_id'):
            num_frames_cell = len(cell_df.frame.unique())
            if num_frames_cell/num_frames > 0.25:
                last_4_cols = cell_df.loc[:, ['norm_med_561nm', 'norm_sum_561nm', 'norm_mean_561nm', 'norm_max_561nm','norm_med_638nm', 'norm_sum_638nm',  'norm_mean_638nm',  'norm_max_638nm', 'n_spots_561', 'n_spots_638', 'spot_density_561', 'spot_density_638', 'area']]
                averages = {col: np.average(cell_df[col]) for col in last_4_cols}
                averages['cell_id'] = cell_id
                averages['folder'] = fol
                averages['surface'] = fol.split('\\')[5]
                averages['mut'] = fol.split('\\')[6]
                for col in last_4_cols:
                    median= np.average(cell_df[col])
                    averages[col] = median
                result_df = pd.DataFrame([averages])
                result_df = result_df[['folder', 'surface','mut', 'cell_id', 'norm_med_561nm', 'norm_sum_561nm', 'norm_mean_561nm', 'norm_max_561nm','norm_med_638nm', 'norm_sum_638nm',  'norm_mean_638nm',  'norm_max_638nm', 'n_spots_561', 'n_spots_638', 'spot_density_561', 'spot_density_638', 'area']]
                values.append(result_df)
final_av = pd.concat(values, ignore_index=True)
# %%
final_av.to_csv(r'D:\Data\20251215_T6_mutants\result_average.csv')
# %%
final_max= pd.read_csv(r'D:\Data\20251215_T6_mutants\result_max.csv')
final_med= pd.read_csv(r'D:\Data\20251215_T6_mutants\result_median.csv')
final_av= pd.read_csv(r'D:\Data\20251215_T6_mutants\result_average.csv')
# pmhc = pd.read_csv(r'D:\Data\20250901_analysis\plot_profile_pMHC.csv')
# pdl1 = pd.read_csv(r'D:\Data\20250901_analysis\plot_profile_pdl1.csv')
# %%
plt.figure(figsize=(12, 6))
plt.plot(pdl1['Distance_(_)'], pdl1['Gray_Value'], linestyle='-', marker='')
plt.xlabel('Distance (um)')
plt.ylabel('Intensity (A.U.)')
plt.xlim(0, 19)
plt.ylim(0)
plt.tight_layout()
plt.savefig(r'D:\Data\20250901_analysis\pdl1_plot_profile.png', dpi = 600)
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the fixed order of conditions
order = ['WT', 'T3', 'T6', 'T6.2']

# Filter the DataFrame to include only the specified conditions
final2 = final_av[(final_av['mut'].isin(order)) & (final_av['surface'] == 'PD-L1')]
final2['spot_density_561'] = final2['spot_density_561']/(0.09016**2)
final2['spot_density_638'] = final2['spot_density_638']/(0.09016**2)
final2['area'] = final2['area']/(0.09016**2)
# Group by condition and calculate median and standard error
grouped = final2.groupby('mut').agg(
    median_561=('spot_density_561', 'median'),
    median_638=('norm_med_638nm', 'median'),
    sem_561=('spot_density_561', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    sem_638=('norm_med_638nm', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
).reindex(order)

# Set white background with gridlines
sns.set_style("white")
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.grid(True, linestyle='--', linewidth=0.5)

# Use seaborn color palette
palette = sns.color_palette("tab10", len(order))

# Plot each condition with error bars
for i, condition in enumerate(order):
    plt.errorbar(
        grouped.loc[condition, 'median_638'],
        grouped.loc[condition, 'median_561'],
        xerr=grouped.loc[condition, 'sem_638'],
        yerr=grouped.loc[condition, 'sem_561'],
        fmt='o',
        label=condition,
        color=palette[i],
        capsize=5
    )

# Add labels and legend
plt.xlabel('Normalized median intensity ZAP70 (A.U.)')
plt.ylabel('density_spots_SHP2 (spots/um^2)')
# plt.xlim(0)
# plt.ylim(0)
plt.legend(title='Condition')
plt.tight_layout()
# plt.savefig(r'D:\Data\20250901_analysis\ZAPmaxVSSHPmax_medianwitherror_V2.pdf', dpi = 600)

# %%
# final.condition.unique()
import seaborn as sns
import matplotlib.pyplot as plt
# Define the fixed order of conditions
order = ['WT', 'T3', 'T6', 'T6.2']

# Filter the DataFrame to include only the specified conditions
final2 = final_med[(final_med['mut'].isin(order)) & (final_med['surface'] == 'PD-L1')]



 # optional fixed order


plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.grid(True, linestyle='--', linewidth=0.5)
sns.scatterplot(
    data=final2, x='norm_max_561nm', y='norm_med_638nm',
    hue='mut', hue_order=order, palette='tab10', s = 40
)
plt.legend(title='Condition')
plt.xlabel('normalized max intensity ZAP70')
plt.ylabel('Normalized median intensity SHP2')  # (you had a mismatch before)
plt.xlim(0)
plt.ylim(0)
plt.tight_layout()

# plt.savefig(r'D:\Data\20250901_analysis\ZAPmaxVSSHPmed2_V2.pdf', dpi = 600)
plt.show()
# %%

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

order = ['pMHC', 'PD-L1', 'CD58', 'PD-L1_2h5wt']  # enforce desired order
cats = pd.Categorical(final2['condition'], categories=order, ordered=True)
codes = cats.codes  # 0..N-1 in the given order

plt.figure(figsize=(8, 6))
sc = plt.scatter(
    final2['norm_max_561nm'], final2['norm_med_638nm'],
    c=codes, cmap='tab10'   # use a categorical-ish colormap
)

# Build legend with the exact same color mapping
handles = [
    Line2D([0], [0], marker='o', linestyle='', 
           color=sc.cmap(sc.norm(i)), label=lab)
    for i, lab in enumerate(cats.categories)
]
plt.legend(handles=handles, title='Condition')

plt.xlabel('norm_max_561nm')
plt.ylabel('norm_med_638nm')
plt.tight_layout()
plt.show()


# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming final2 is already defined and available in the environment
# Create a boxplot for the specified columns
columns_to_plot = ['norm_med_561nm', 'norm_sum_561nm', 'norm_med_638nm', 'norm_sum_638nm']

# Melt the DataFrame to long format for seaborn boxplot
melted_df = final2.melt(id_vars='condition', value_vars=columns_to_plot, var_name='Measurement', value_name='Value')

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=melted_df, x='Measurement', y='Value', hue='condition')
plt.title('Boxplot of Normalized Measurements by Condition')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Define the fixed order of conditions
# Define the fixed order of conditions
order = ['WT', 'T3', 'T6', 'T6.2']

# Filter the DataFrame to include only the specified conditions
final2 = final_av[(final_av['mut'].isin(order)) & (final_av['surface'] == 'PD-L1')]
final2['spot_density_561'] = final2['spot_density_561']/(0.09016**2)
final2['spot_density_638'] = final2['spot_density_638']/(0.09016**2)
final2['area'] = final2['area']/(0.09016**2)
columns_to_plot = [
    'norm_med_561nm', 'norm_mean_561nm', 'norm_max_561nm',
    'norm_med_638nm', 'norm_mean_638nm', 'norm_max_638nm', 
    'spot_density_561', 'spot_density_638', 'area'
]

for col in columns_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=final2, x='mut', y=col)#, showfliers=False)  # hide outlier dots
    sns.stripplot(data=final2, x='mut', y=col, 
                  jitter=True, color='black', alpha=0.5, size=2)  # add jittered dots
    
    plt.title(f'Boxplot of {col} by Condition')
    plt.ylabel(col)
    plt.xticks(rotation=-45)
    plt.tight_layout()
    plt.savefig(rf'D:\Data\20250901_analysis\boxplot_{col}.pdf', dpi = 600)
    plt.show()

    
#%% 
today = today = datetime.date.today()
for path in run_paths:
    print(path)
    intensity_path = os.path.join(path, 'cell_detection', 'rois.npy') #os.path.join(path, 'cell_detection', 'overlay_cells.tiff')
    result_path = os.path.join(path, 'cell_detection', 'intensities.pkl')
    # if os.path.exists(intensity_path):
    #     # get modification date
    #     mod_time = datetime.date.fromtimestamp(os.path.getmtime(result_path))

    #     # if file modified today → skip
    #     if mod_time == today:
    #         print(f"Skipping {path}, file already generated today.")
    #         continue
    # if os.path.exists(intensity_path):
    #     print(f"Skipping {path}, intensities already computed.")
    #     continue

    a = plc.Cell_Analyzer(path)
    
    
    phan1, li1, comb1 = find_cells_Gerard(np.array(a.images['638nm']))
    # plt.imshow(comb1[5])
    # plt.title('Zap')
    # plt.axis('off')
    # plt.show()
    phan2, li2, comb2 = find_cells_Gerard(np.array(a.images['561nm']))
    # plt.imshow(comb2[5])
    # plt.title('Shp')
    # plt.axis('off')
    # plt.show()
    if comb1.shape == comb2.shape:
        full_mask = a._remove_small_objects_per_frame(comb1|comb2, min_size=2000, connectivity=0)
    else: 
        if comb1.shape[0]> comb2.shape[0]:
            comb1 = comb1[:comb2.shape[0], :, :]
        elif comb2.shape[0]> comb1.shape[0]:
            comb2 = comb2[:comb1.shape[0], :, :]
        full_mask = a._remove_small_objects_per_frame(comb1|comb2, min_size=2000, connectivity=0)
    # plt.imshow(full_mask[5])
    # plt.title('comb')
    # plt.axis('off')
    # plt.show()
    markers = segment_watershed_stack(full_mask)

    # n = int(markers[5].max()) + 1
    # cmap = plt.cm.get_cmap("tab20", n)  # discrete colors
    # plt.figure(figsize=(5, 5))
    # plt.imshow(markers[5], cmap=cmap, vmin=0, vmax=n-1)
    # plt.axis("off")
    # plt.title("watershed")
    # plt.show()

    # full_mask = np.load(intensity_path)
    # cont = extract_contours_per_frame(full_mask)
    cont = labels_to_trackpy_features(markers)
    linked_df = tp.link_df(cont, search_range=100, memory=8)
    if not os.path.exists(os.path.join(path, 'cell_detection')):
        os.makedirs(os.path.join(path, 'cell_detection'))
    out = os.path.join(path, 'cell_detection', 'linked_rois.pkl')
    linked_df.rename(columns={'particle': 'cell_id'}, inplace = True)
    linked_df.to_pickle(out)
    if not os.path.exists(os.path.join(path, 'cell_detection')):
        os.makedirs(os.path.join(path, 'cell_detection'))
    out = os.path.join(path, 'cell_detection', 'overlay_cells.tiff')
    save_contours_overlay(np.array(a.images['561nm']), linked_df, out)

    out = os.path.join(path, 'cell_detection', 'rois.npy')
    np.save(out, full_mask)
    # Define the channels to analyze
    channels = ['561nm', '638nm']
    # Pre-allocate all columns at once
    for channel in channels:
        linked_df[f'norm_med_{channel}'] = 0.0
        linked_df[f'norm_sum_{channel}'] = 0.0
    frames = linked_df.frame.unique()
    num_frames = len(frames)
    spots_561 = pd.read_csv(glob(path + f'/**/**561nm_locs.csv', recursive=True)[0])
    # spots_638 = pd.read_csv(glob(path + f'/**/**638nm_locs.csv', recursive=True)[0])
    # linked_df = add_spot_counts_and_density(linked_df, markers, spots_561,count_col="n_spots_561",
                                            # density_col="spot_density_561", area_col="area")
    # linked_df = add_spot_counts_and_density(linked_df, markers, spots_638,count_col="n_spots_638",
                                            # density_col="spot_density_638", area_col="area")
    # Process each cell (compute masks once per cell)
    for cell_id, cell_df in tqdm(linked_df.groupby('cell_id')):
        print(f"Processing cell: {cell_id}")
        num_frames_cell = len(cell_df.frame.unique())
        if num_frames_cell/num_frames > 0.25:
            # Calculate crop boundaries once per cell
            x0 = cell_df['bbox_x0'].min()
            x1 = cell_df['bbox_x1'].max()
            y0 = cell_df['bbox_y0'].min()
            y1 = cell_df['bbox_y1'].max()
            x0_crop = int(x0 * 0.9)
            x1_crop = int(x1 * 1.1)
            y0_crop = int(y0 * 0.9)
            y1_crop = int(y1 * 1.1)
            
            # Pre-crop all channel images for this cell
            cropped_images = {}
            for channel in channels:
                cropped_images[channel] = a.images[channel][:, y0_crop:y1_crop, x0_crop:x1_crop]
            
            # Process each frame for this cell
            for idx, row in cell_df.iterrows():
                frame = row['frame']
                contour = row['contour']
                mask = full_mask[frame, y0_crop:y1_crop, x0_crop:x1_crop]
                # Compute mask once per frame
                corr = contour.copy()
                corr[:, 0] -= y0_crop
                corr[:, 1] -= x0_crop
                # start_time = time.time()
                # mask = contour_to_mask(corr, cropped_images[channels[0]][0].shape)
                # end_time = time.time()
                # print(f'mask took {end_time - start_time:.6f} sec')
                # Apply this mask to all channels
                for channel in channels:
                    cropped_image = cropped_images[channel]
                    background_median = np.median(cropped_image[frame][~mask])
                    background_mean = np.mean(cropped_image[frame][~mask])
                    background_max = np.max(cropped_image[frame][~mask])
                    
                    norm_median_col = f'norm_med_{channel}'
                    norm_sum_col = f'norm_sum_{channel}'
                    norm_mean_col = f'norm_mean_{channel}'
                    norm_max_col = f'norm_max_{channel}'
                    
                    linked_df.at[idx, norm_median_col] = np.median(cropped_image[frame][mask]) / background_median
                    linked_df.at[idx, norm_sum_col] = np.sum(cropped_image[frame][mask]) / background_median
                    linked_df.at[idx, norm_mean_col] = np.mean(cropped_image[frame][mask]) / background_mean
                    linked_df.at[idx, norm_max_col] = np.max(cropped_image[frame][mask]) / background_max
    
    out = os.path.join(path, 'cell_detection', 'intensities.pkl')
    linked_df.drop(['contour'], axis = 1)
    linked_df.to_pickle(out)
    
    
# %%
out = os.path.join(r'D:\Data\20250901_analysis\PD-L1\GCL0019\pMHC+PD-L1+CD58\Run00005', 'cell_detection', 'intenisties.hdf')
linked_df.to_hdf(out, key='df', mode='w', complevel=9, complib='blosc')
# %%
out = os.path.join(r'D:\Data\20250901_analysis\PD-L1\GCL0019\pMHC+PD-L1+CD58\Run00005', 'cell_detection', 'full_mask.npy')

np.save(out, full_mask)

#


# plt.imshow(full_mask[idx, y0_crop:y1_crop, x0_crop:x1_crop])
# %%





ch = '638nm'
path = r"D:\Data\20250901_analysis\PD-L1\GCL0024\Run00004"
a = plc.Cell_Analyzer(path)
image = np.array(a.images[ch])
phan1, li1, comb1 = find_cells_Gerard(np.array(a.images['638nm']))
phan2, li2, comb2 = find_cells_Gerard(np.array(a.images['561nm']))
# d = a._remove_small_objects_per_frame(c&b)

full_mask = a._remove_small_objects_per_frame(comb1|comb2, min_size=2000, connectivity=0) 

# %%





linked_df2 = pd.read_pickle(out)
# %%
from skimage.draw import polygon
import numpy as np



# Define the channels to analyze
channels = ['561nm', '638nm']

# Load the linked dataframe
out = os.path.join(path, 'cell_detection', 'linked_rois.pkl')
linked_df = pd.read_pickle(out)
linked_df.rename(columns={'particle': 'cell_id'}, inplace=True)

# Process each channel
for channel in channels:
    image = a.images[channel]
    print(f"Processing channel: {channel}")
    
    for cell_id, cell_df in linked_df.groupby('cell_id'):
        x0 = cell_df['bbox_x0'].min()
        x1 = cell_df['bbox_x1'].max()
        y0 = cell_df['bbox_y0'].min()
        y1 = cell_df['bbox_y1'].max()
        x0_crop = int(x0 * 0.9)
        x1_crop = int(x1 * 1.1)
        y0_crop = int(y0 * 0.9)
        y1_crop = int(y1 * 1.1)
        cropped_image = image[:, y0_crop:y1_crop, x0_crop:x1_crop]
        
        for idx, row in cell_df.iterrows():
            frame = row['frame']
            contour = row['contour']
            corr = contour.copy()
            corr[:, 0] -= y0_crop
            corr[:, 1] -= x0_crop
            mask = contour_to_mask(corr, cropped_image[0].shape)
            background_median = np.median(cropped_image[frame][~mask])
            
            # Create channel-specific column names
            norm_med_col = f'norm_med_{channel}'
            norm_sum_col = f'norm_sum_{channel}'
            
            linked_df.loc[idx, norm_med_col] = np.median(cropped_image[frame][mask]) / background_median
            linked_df.loc[idx, norm_sum_col] = np.sum(cropped_image[frame][mask]) / background_median
# %%

import numpy as np

def contour_to_mask(contour, image_shape):
    y_coords, x_coords = contour[:, 0], contour[:, 1]
    rr, cc = polygon(y_coords, x_coords, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr, cc] = True
    return mask



            
# %%
from skimage.draw import polygon
import numpy as np

def contour_to_mask(contour, image_shape):
    y_coords, x_coords = contour[:, 0], contour[:, 1]
    rr, cc = polygon(y_coords, x_coords, shape=image_shape)
    mask = np.zeros(image_shape, dtype=bool)
    mask[rr, cc] = True
    return mask

# Define the channels to analyze
channels = ['561nm', '638nm']

# Load the linked dataframe
out = os.path.join(path, 'cell_detection', 'linked_rois.pkl')
linked_df = pd.read_pickle(out)
linked_df.rename(columns={'particle': 'cell_id'}, inplace=True)

# Pre-allocate all columns at once
for channel in channels:
    linked_df[f'norm_med_{channel}'] = 0.0
    linked_df[f'norm_sum_{channel}'] = 0.0

# Process each cell (we still need this loop since crop regions differ per cell)
for cell_id, cell_df in linked_df.groupby('cell_id'):
    print(f"Processing cell: {cell_id}")
    
    # Calculate crop boundaries once per cell
    x0 = cell_df['bbox_x0'].min()
    x1 = cell_df['bbox_x1'].max()
    y0 = cell_df['bbox_y0'].min()
    y1 = cell_df['bbox_y1'].max()
    x0_crop = int(x0 * 0.9)
    x1_crop = int(x1 * 1.1)
    y0_crop = int(y0 * 0.9)
    y1_crop = int(y1 * 1.1)
    
    # Get all data for this cell
    indices = cell_df.index.values
    frames = cell_df['frame'].values
    contours = cell_df['contour'].values
    
    # Pre-crop all channel images for this cell
    cropped_images = {}
    for channel in channels:
        cropped_images[channel] = a.images[channel][:, y0_crop:y1_crop, x0_crop:x1_crop]
    
    ref_shape = cropped_images[channels[0]][0].shape
    
    # Create all masks at once
    masks = []
    for contour in contours:
        corr = contour.copy()
        corr[:, 0] -= y0_crop
        corr[:, 1] -= x0_crop
        mask = contour_to_mask(corr, ref_shape)
        masks.append(mask)
    
    masks = np.array(masks)
    
    # Process all channels for this cell
    for channel in channels:
        cropped_image = cropped_images[channel]
        
        # Vectorized computation for all frames at once
        norm_med_values = np.zeros(len(frames))
        norm_sum_values = np.zeros(len(frames))
        
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            frame_data = cropped_image[frame]
            background_median = np.median(frame_data[~mask])
            norm_med_values[i] = np.median(frame_data[mask]) / background_median
            norm_sum_values[i] = np.sum(frame_data[mask]) / background_median
        
        # Vectorized DataFrame assignment
        norm_med_col = f'norm_med_{channel}'
        norm_sum_col = f'norm_sum_{channel}'
        linked_df.loc[indices, norm_med_col] = norm_med_values
        linked_df.loc[indices, norm_sum_col] = norm_sum_values

# %%


import matplotlib.pyplot as plt
import numpy as np

# Get unique cell IDs
cell_ids = linked_df['cell_id'].unique()

# Create subplots - one for each cell
n_cells = len(cell_ids)
n_cols = min(4, n_cells)  # Max 4 columns
n_rows = int(np.ceil(n_cells / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
if n_cells == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes array for easy iteration
axes_flat = axes.flatten() if n_cells > 1 else axes

for i, cell_id in enumerate(cell_ids):
    if i >= len(axes_flat):
        break
        
    ax1 = axes_flat[i]
    
    # Get data for this cell
    cell_data = linked_df[linked_df['cell_id'] == cell_id].sort_values('frame')
    frames = cell_data['frame']
    norm_med_561 = cell_data['norm_med_561nm']
    norm_med_638 = cell_data['norm_med_638nm']
    
    # Plot 561nm on left axis (blue)
    color1 = 'tab:blue'
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('561nm norm_med', color=color1)
    line1 = ax1.plot(frames, norm_med_561, 'o-', color=color1, label='561nm', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create right axis for 638nm (red)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('638nm norm_med', color=color2)
    line2 = ax2.plot(frames, norm_med_638, 's-', color=color2, label='638nm', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and legend
    ax1.set_title(f'Cell {cell_id}')
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # Add grid
    ax1.grid(True, alpha=0.3)

# Hide any unused subplots
for i in range(n_cells, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout()
plt.show()

# Alternative: Single plot with all cells (if you prefer this format)
plt.figure(figsize=(12, 8))

# Create a colormap for different cells
colors = plt.cm.tab10(np.linspace(0, 1, len(cell_ids)))

ax1 = plt.gca()
ax2 = ax1.twinx()

for i, cell_id in enumerate(cell_ids):
    cell_data = linked_df[linked_df['cell_id'] == cell_id].sort_values('frame')
    frames = cell_data['frame']
    norm_med_561 = cell_data['norm_med_561nm']
    norm_med_638 = cell_data['norm_med_638nm']
    
    # Plot with different line styles and same color for each cell
    ax1.plot(frames, norm_med_561, 'o-', color=colors[i], alpha=0.7, 
             label=f'Cell {cell_id} (561nm)', markersize=4)
    ax2.plot(frames, norm_med_638, 's--', color=colors[i], alpha=0.7, 
             label=f'Cell {cell_id} (638nm)', markersize=4)

ax1.set_xlabel('Frame')
ax1.set_ylabel('561nm norm_med', color='tab:blue')
ax2.set_ylabel('638nm norm_med', color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

ax1.grid(True, alpha=0.3)
plt.title('Normalized Median Intensity Over Time - All Cells')
plt.tight_layout()
plt.show()



# %%


# %%


# Set the index
i = 15

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Display each image in the subplot with appropriate title and no axis
axs[0, 0].imshow(np.array(a.images['638nm'])[i])
axs[0, 0].set_title("Original Image (638nm)")
axs[0, 0].axis('off')

axs[0, 1].imshow(comb1[i], cmap='gray')
axs[0, 1].set_title("638 mask")
axs[0, 1].axis('off')

axs[1, 0].imshow(comb2[i], cmap='gray')
axs[1, 0].set_title("561 mask")
axs[1, 0].axis('off')

axs[1, 1].imshow(full_mask[i], cmap='gray')
axs[1, 1].set_title("Combined")
axs[1, 1].axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
