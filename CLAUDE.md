# Wildfire Spread Prediction — Project Guide

## Project Goal

Build an ML model that predicts wildfire burn perimeter expansion over 6-24 hour windows using satellite fire detections, weather reanalysis, fuel maps, and topography. Target region: California fires 2015-2023. Target architecture: ConvLSTM U-Net. Target benchmark: outperform FARSITE on held-out fires while running significantly faster.

---

## Stack

- **Python 3.11+**
- `torch`, `torchvision` — modeling
- `rasterio`, `pyproj`, `shapely` — geospatial raster/vector ops
- `xarray`, `cfgrib` — HRRR GRIB2 ingestion
- `zarr` or `h5py` — dataset storage
- `numpy`, `pandas` — array/tabular ops
- `matplotlib`, `cartopy` — visualization
- `requests`, `boto3` — data download (FIRMS uses REST; HRRR is on AWS S3)

---

## Project Structure

```
wildfire/
├── CLAUDE.md
├── data/
│   ├── raw/
│   │   ├── firms/         # FIRMS CSV/shapefiles
│   │   ├── hrrr/          # GRIB2 files
│   │   ├── landfire/      # Fuel/canopy rasters
│   │   └── dem/           # SRTM or 3DEP tiles
│   └── processed/
│       ├── samples/       # Zarr or HDF5 patch files
│       └── splits/        # Train/val/test fire ID lists
├── pipeline/
│   ├── download.py        # Data fetching scripts
│   ├── align.py           # CRS alignment, resampling
│   ├── labels.py          # Burn mask generation from FIRMS
│   ├── features.py        # Feature stack construction
│   └── dataset.py         # PyTorch Dataset class
├── model/
│   ├── unet.py            # Baseline U-Net
│   ├── convlstm.py        # ConvLSTM cell + encoder
│   └── convlstm_unet.py   # Full ConvLSTM U-Net
├── train.py
├── evaluate.py
└── configs/
    └── baseline.yaml      # Hyperparams, paths, resolution
```

---

## Phase 1: Data Acquisition

### 1.1 FIRMS Fire Detections (Labels)

- Register at https://firms.modaps.eosdis.nasa.gov/ and get an API key
- Download VIIRS S-NPP 375m active fire detections for California, 2015-2023
- Use the area-based download (bounding box: roughly -124.5, 32.5, -114.1, 42.0)
- Save raw CSVs to `data/raw/firms/`
- Each row is a fire pixel detection with lat, lon, acquisition datetime, confidence, FRP

### 1.2 HRRR Weather

- HRRR GRIB2 files are on AWS S3: `s3://noaa-hrrr-bdp-pds/`
- Variables needed: `UGRD` and `VGRD` (wind u/v at 10m), `RH` (relative humidity), `TMP` (2m temperature)
- For each fire event, download hourly files covering T-12h to T+24h
- Use `cfgrib` to extract variables; `xarray` to handle the time dimension
- Save per-event GRIB2 slices to `data/raw/hrrr/`
- Note: HRRR archive only goes back to ~2014; pre-2014 fires need ERA5 instead

### 1.3 LANDFIRE Fuels

- Download from https://landfire.gov/viewer/
- Layers needed: `FBFM40` (40 Scott-Burgan surface fuel models), `CC` (canopy cover), `CH` (canopy height)
- Download the CONUS mosaic at 30m; clip to California bounding box
- Save to `data/raw/landfire/`
- LANDFIRE updates periodically — use the version year closest to each fire event

### 1.4 DEM / Topography

- Download SRTM 30m via `earthaccess` or directly from USGS EarthExplorer
- Alternatively use `elevation` Python package for automated SRTM tile fetching
- Coverage: California bounding box, same as above
- Save GeoTIFF tiles to `data/raw/dem/`
- Topographic derivatives (slope, aspect) are computed in the pipeline, not stored raw

---

## Phase 2: Label Generation (`pipeline/labels.py`)

Goal: for each fire event and timestep T, produce a binary raster of burned vs. unburned at 100m resolution, aligned to EPSG:5070.

### Steps

1. Load FIRMS detections for a fire event; filter by confidence >= 'nominal'
2. Group detections by 6-hour windows to construct perimeter snapshots
3. Rasterize point detections onto a 100m EPSG:5070 grid using `rasterio.features.rasterize`
4. Apply a small dilation (1-2 pixels) to account for detection gaps at perimeter edges
5. For each timestep T, label = rasterized detections up to T; target = rasterized detections T to T+6h (or T+24h)
6. Save (label_mask, target_mask) pairs indexed by (fire_id, timestep)

### Edge Cases to Handle

- Fires with fewer than 3 timesteps: discard
- Detection gaps > 12h (cloud cover, satellite overpass gaps): flag and exclude that window
- Multi-polygon fires (fire splits or merges): treat as single union geometry

---

## Phase 3: Feature Stack Construction (`pipeline/features.py`)

All layers resampled to **100m resolution, EPSG:5070**, cropped to a **256x256 pixel patch** centered on the fire's active perimeter centroid at time T.

### Per-sample input tensor shape: `(T_steps=3, C=12, 256, 256)`

| Channel | Source | Notes |
|---|---|---|
| Current burn mask | FIRMS | Binary |
| Wind U (T, T-6h, T-12h) | HRRR | 3 timesteps |
| Wind V (T, T-6h, T-12h) | HRRR | 3 timesteps |
| Relative humidity | HRRR | At time T |
| Temperature | HRRR | At time T |
| Slope | DEM-derived | Static; `numpy` gradient on DEM |
| Aspect (sin) | DEM-derived | Encode as sin/cos to avoid 0/360 discontinuity |
| Aspect (cos) | DEM-derived | |
| Canopy height | LANDFIRE | Static |
| Fuel model | LANDFIRE | One-hot encode 40 classes; reduce via PCA or learned embedding to ~8 dims if needed |

### Derived Features to Compute

- Dead fuel moisture: estimate 10-hr fuel moisture from Fosberg equations using T and RH
- Topographic Position Index (TPI): local DEM mean subtracted from cell elevation, 300m radius

### Normalization

- Compute mean/std per channel from training set only
- Save normalization stats to `configs/norm_stats.json`
- Apply at dataset load time, not preprocessing time (keeps raw data reusable)

---

## Phase 4: Dataset Class (`pipeline/dataset.py`)

```python
class WildfireDataset(Dataset):
    def __init__(self, sample_dir, split_file, norm_stats, augment=False):
        # Load list of (fire_id, timestep) pairs from split_file
        # Load norm_stats for per-channel normalization
        # augment: random horizontal/vertical flip (preserve wind direction channels accordingly)

    def __getitem__(self, idx):
        # Load input tensor (T, C, H, W) and target mask (H, W)
        # Apply normalization
        # Apply augmentation if training
        # Return dict: {'input': tensor, 'target': mask, 'meta': {fire_id, timestep}}
```

### Augmentation Rules

- Horizontal/vertical flips are valid but require rotating wind U/V channels accordingly
- Do NOT apply random rotation — topographic features break under arbitrary rotation
- Do NOT apply color jitter or intensity shifts to the burn mask channel

---

## Phase 5: Model (`model/convlstm_unet.py`)

### Architecture

```
Input (T, C, H, W)
    └─► ConvLSTM Encoder (3 layers, hidden dims [64, 128, 256])
            └─► Takes temporal sequence; outputs final hidden state per layer
    └─► U-Net Decoder (skip connections from encoder spatial features)
            └─► Upsample blocks back to (1, H, W)
    └─► Sigmoid → burn probability map
```

### ConvLSTM Cell (`model/convlstm.py`)

- Standard ConvLSTM formulation (Shi et al. 2015)
- Kernel size 3x3, same padding
- Layer norm on hidden state (more stable than batch norm for sequences)

### U-Net Decoder

- Standard transposed conv upsample + skip concat
- Final layer: 1x1 conv → sigmoid
- Output: probability map, same spatial resolution as input

### Loss Function

Start with binary cross-entropy weighted by class imbalance ratio (unburned pixels vastly outnumber burned ones at each step). If class weighting is insufficient, switch to focal loss (gamma=2, alpha tuned on validation).

Do not use Dice loss alone — it is insensitive to small fires which are the hardest cases.

---

## Phase 6: Training (`train.py`)

### Config (`configs/baseline.yaml`)

```yaml
patch_size: 256
resolution_m: 100
t_steps: 3
t_horizon_h: 6
batch_size: 16
lr: 1e-4
epochs: 100
early_stopping_patience: 10
hidden_dims: [64, 128, 256]
loss: weighted_bce
device: cuda
```

### Training Loop Notes

- Use `torch.cuda.amp` (mixed precision) — input tensors are large
- Gradient clip at 1.0 — ConvLSTM is prone to exploding gradients
- LR schedule: cosine annealing with warm restart
- Log per-epoch CSI and IoU on validation set, not just loss
- Save best checkpoint by validation CSI, not loss

---

## Phase 7: Evaluation (`evaluate.py`)

### Splits

Construct splits at the **fire level**, not the sample level:

- `splits/train_fires.txt`: fire IDs from 2015-2019
- `splits/val_fires.txt`: fire IDs from 2020-2021
- `splits/test_fires.txt`: fire IDs from 2022-2023

Never mix timesteps from the same fire across splits.

### Metrics

| Metric | Formula | Notes |
|---|---|---|
| CSI (Critical Success Index) | TP / (TP + FP + FN) | Standard in fire/met literature |
| IoU | Same as CSI | Report both names for audience coverage |
| FAR (False Alarm Ratio) | FP / (TP + FP) | Operationally critical |
| POD (Probability of Detection) | TP / (TP + FN) | |
| Brier Score | Mean squared prob error | If outputting calibrated probabilities |

### Baselines to Implement

1. **Persistence**: predict T+6h burn mask = T burn mask. Deceptively strong.
2. **Circular spread**: expand perimeter radially at constant rate estimated from training set median.
3. **FARSITE** (optional but high value): run FARSITE on test fires and compare CSI directly.

---

## Phase 8: Known Failure Modes to Monitor

- **Wind shift events**: abrupt wind direction changes cause rapid perimeter shape changes the model will miss unless temporal weather context is long enough
- **Spotting (ember transport)**: long-range ignitions ahead of main perimeter will appear as false negatives; these are not model failures but labeling ambiguities
- **Cloud cover gaps**: FIRMS detections under smoke or cloud are unreliable; validate that your confidence filtering removes these
- **Fire suppression effects**: suppressed fires spread differently than free-burning fires; no good label for this exists in public data
- **Patch boundary artifacts**: fires that grow beyond the 256x256 patch window will have clipped targets; detect and exclude these during dataset construction

---

## Key References

- Shi et al. (2015) — ConvLSTM for precipitation nowcasting (architecture basis)
- Rothermel (1972) — Surface fire spread physics (for physics-informed loss baseline)
- FARSITE documentation (Finney 1998) — operational benchmark
- Huot et al. (2022) — "Next Day Wildfire Spread" (Google, similar task, useful benchmark numbers)
- Radke et al. (2019) — "FireCast" (LSTM on fire spread, direct prior work)

---

## MVP Checklist

- [ ] FIRMS download + fire event segmentation script
- [ ] HRRR download for 5 test fire events (manual before automating)
- [ ] Label rasterization pipeline producing clean burn masks
- [ ] Feature stack construction for those 5 fires
- [ ] Baseline U-Net training and overfitting test on single fire
- [ ] Persistence and circular spread baselines implemented
- [ ] Full dataset construction for California 2015-2022
- [ ] ConvLSTM U-Net training with fire-level temporal split
- [ ] Evaluation report: CSI/FAR/POD vs. baselines on 2022-2023 test fires
