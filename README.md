# FireSpread

Wildfire perimeter-spread prediction for California, 2015-2023. See [CLAUDE.md](CLAUDE.md)
for the full design; this file covers setup and current status.

## Setup

```bash
pip install -r requirements.txt
```

Everything installs from pip on Windows — no conda needed, and no `herbie-data` dependency since
`pipeline/hrrr.py` does its own index subsetting. `cfgrib`/`eccodes` ship working binary wheels
(verified on both Python 3.11 and 3.13). `cartopy` (Phase 7 figures) is still not installed.

**Pin the venv to 3.11 or 3.13, not the newest Python.** The geospatial and GRIB wheels lag new
releases by months, and this stack is entirely wheel-dependent.

### Current machine (rebuilt 2026-08-14)

| | previous laptop | current |
|---|---|---|
| GPU | RTX 4070 Laptop, 8 GB | **RTX 3090, 24 GB** |
| free disk | 83 GB | 209 GB |
| Python / torch | 3.13 / 2.7.0+cu118 | 3.11 / 2.11.0+cu128 |

The VRAM tables further down were measured on the 8 GB card and still describe the model's memory
appetite correctly, but the *conclusions* drawn from a 6.9 GB ceiling no longer bind here — see
"Known issues to revisit".

## FIRMS API key

The downloader needs a NASA FIRMS MAP_KEY:

1. Request one at <https://firms.modaps.eosdis.nasa.gov/api/area/> (email-based, arrives quickly).
2. Provide it either way:

```bash
setx FIRMS_MAP_KEY your_key_here
```

or create a `.env` at the project root containing `FIRMS_MAP_KEY=your_key_here`
(`.env` is gitignored).

## Phase 1.1 — run it

```bash
python -m pipeline.download firms --start 2020-08-01 --end 2020-09-30
```

Downloads in 5-day chunks into `data/raw/firms/`, one CSV per chunk, resumable — rerunning
skips chunks already on disk and retries only what failed. The full 2015-2023 pull is ~660
requests, well under the 5000-per-10-minutes quota. Start with a small range to confirm the
key works.

**The API maximum is 5 days per request, not 10** — above that it returns HTTP 400
`Invalid day range. Expects [1..5]`. Verified empirically; various docs suggest 10.

Then segment detections into fire events:

```bash
python -m pipeline.events
```

Writes `data/processed/fire_events.csv` (one row per fire: id, start/end, centroid, detection
count, timestep count, patch-fit flag) and `data/processed/detections_labeled.parquet`
(every kept detection tagged with its `fire_id`).

## Status

- [x] Repo scaffold, config, dependency baseline
- [x] FIRMS download script (chunked, resumable, retries)
- [x] Fire event segmentation (spatiotemporal single-link clustering)
- [x] Full 2015-2023 FIRMS pull — 658 chunks, 67 MB, 741,236 detections after filtering
- [x] Segmentation over the full archive — **622 usable fire events** at 24 h windows
- [x] Archive gap audit (`download.py audit`)
- [x] Phase 1.2 — HRRR for the 5 MVP fires (184/185 hours, 1.1 GB)
- [x] Phase 1.3 — LANDFIRE fuels via LFPS (5 fires, 70 MB)
- [x] Phase 1.4 — 3DEP elevation via COG windowed reads (5 fires, 1.16 GB)
- [x] Co-registration verified across all three sources
- [x] Phase 2 — label rasterisation (`pipeline/labels.py`), 24 h windows, acreage-calibrated
- [x] Phase 3a — static feature stack (terrain + fuel), co-registration verified
- [x] Phase 3b — tiled sample index, 661 samples over the MVP fires
- [x] HRRR re-pull for the MVP fires — 499/501 hours (+4 gap-repair), 2.9 GB fetched, **23.8 MB stored**
- [x] Phase 3c — weather channels + Fosberg fuel moisture, 661/661 samples validated
- [x] Phase 4 — `WildfireDataset` with on-the-fly tile cropping, splits, norm stats
- [x] Phase 5 — ConvLSTM U-Net + baseline U-Net, single-batch overfit test passes
- [x] Full dataset construction, California 2015-2023 — **7,070 samples over 450 fires**
- [x] Split boundary moved to 2015-2020 / 2021 / 2022-2023 — train is now 63.7%, was 21.7%
- [x] Phase 6 — `train.py`: AMP, grad clip, cosine warm restarts, best-by-CSI checkpointing
- [x] First full training run — **validation CSI 0.2641**, 3.2x the best naive baseline
- [ ] ConvLSTM-vs-U-Net ablation (baseline needs widening to ~5.3 M params first)
- [x] Phase 7 — `evaluate.py`: test **CSI 0.1849**, 1.82x the naive baseline, stratified by size
- [ ] Planned ablation: 12 h windows + night/day pass flag, scored against 24 h at a common horizon

### Events and split sizes (full archive)

| split | years | fires | detections |
|---|---|---|---|
| train | 2015-2019 | 320 | 213,754 |
| val | 2020-2021 | 166 | 386,690 |
| test | 2022-2023 | 136 | 30,645 |

**`keep` depends on `labels.window_hours`,** because `min_timesteps: 3` counts windows. The table
above is at the chosen 24 h. At the original 6 h spec it was 692 fires (363/187/142, 637,088
detections) — that earlier figure is what the first draft of this file recorded, and the 70-fire
drop is the Phase 2 window switch, not data loss. Verified by re-running segmentation at both
settings over the identical archive.

Per-year event counts track the real severity record: 2020 (104) and 2017 (95) high, 2019 (39)
low. **But note the detection totals**: the val years hold 387k detections against 31k in test.
The CLAUDE.md chronological split puts the two most extreme seasons in California history
(2020, 2021) in validation and two mild ones (2022, 2023) in test. Model selection will be tuned
on megafire behaviour and then scored on quiet fires, and headline test CSI will not say much
about extreme-event performance. Worth revisiting at Phase 7 — the cleanest fix is to keep the
chronological split but additionally report metrics stratified by fire size.

Ten of eleven spot-checked named fires segment correctly (Camp, Thomas, Tubbs, Carr, Mendocino,
Rough, Dixie, Caldor, Mosquito, Smith River). The eleventh, McKinney, is missing for the sensor
reason below.

### Validation — Aug-Sep 2020

The ten largest segmented events are all real, correctly dated, correctly located 2020 fires:

| fire_id | Fire | span (km) | est. acres | official | ratio |
|---|---|---|---|---|---|
| 2020_0162 | August Complex | 74 x 116 | 811,879 | 1,032,648 | 0.79 |
| 2020_0372 | Creek | 43 x 66 | 295,993 | 379,895 | 0.78 |
| 2020_0184 | North Complex | 68 x 42 | 219,649 | 318,935 | 0.69 |
| 2020_0229 | SQF Complex | 47 x 35 | 162,765 | 174,178 | 0.93 |
| 2020_0158 | SCU Lightning | 53 x 60 | 218,364 | 396,624 | 0.55 |
| 2020_0187 | LNU Lightning | 41 x 77 | 171,209 | 363,220 | 0.47 |
| 2020_0001 | Red Salmon Complex | 35 x 28 | 120,719 | 144,698 | 0.83 |
| 2020_0393 | Bobcat | 35 x 35 | 96,325 | 115,997 | 0.83 |
| 2020_0212 | Dolan | 31 x 32 | 92,745 | 124,924 | 0.74 |
| 2020_0419 | Slater | 34 x 34 | 92,155 | 157,220 | 0.59 |

Area is estimated from unique 375 m detection cells. Every fire lands **under** its official
acreage (0.47-0.93x), which is the expected signature: VIIRS sees actively burning fronts, not
cumulative burn scar, and misses fire under smoke, between overpasses, or below the detection
threshold. Nothing exceeding 1.0x is the useful signal here — over-merged clusters would
overshoot. LNU and SCU sit lowest, consistent with the August 2020 smoke inversion and their
large share of low-intensity grass and oak woodland burning.

## Decisions and deviations from CLAUDE.md

- **`pipeline/events.py` is a new file** not in the CLAUDE.md tree. Event segmentation is MVP
  checklist item 1 but has no listed home; it is separate from `download.py` because it is
  a transform, not a fetch.
- **Source is `VIIRS_SNPP_SP`**, not NRT. NRT only covers roughly the last two months; the
  standard-processing archive is what covers 2015-2023.
- **Detections filtered to `type == 0`** (vegetation fires), dropping volcanoes, static land
  sources such as gas flares, and offshore detections. Without this, industrial hot spots
  become permanent phantom "fires".
- **Clustering thresholds** (2 km, 96 h) are validated against the 2020 season (table above)
  and look correct: no cluster overshoots its official acreage, and the ten largest events map
  one-to-one onto named fires. Worth re-checking against Camp (2018) and Dixie (2021) once the
  full archive is down.
- **S-NPP only, deliberately.** `VIIRS_NOAA20_SP` is also available and returns roughly twice
  the detections, but only from 2018-04-01. Adding it would make label density jump partway
  through the training period and be systematically higher in the 2022-2023 test years than in
  the 2015-2019 training years — a split-dependent bias that would flatter test metrics.
  If you want the extra density, the clean options are to add NOAA-20 and restrict the whole
  study to 2019+, or to keep S-NPP as the label source throughout. Confirmed available range:
  `VIIRS_SNPP_SP` covers 2012-01-20 onward, so it spans 2015-2023 by itself.

## Data gap — 15-day S-NPP outage in peak 2022 season

```bash
python -m pipeline.download audit
```

Auditing the full archive turns up exactly one significant hole:

| span | days | note |
|---|---|---|
| 2016-01-04 .. 2016-01-06 | 3 | midwinter, plausibly genuine |
| 2019-02-13 .. 2019-02-15 | 3 | midwinter, plausibly genuine |
| **2022-07-27 .. 2022-08-10** | **15** | **peak fire season — sensor outage** |

The 2022 window is a real S-NPP VIIRS outage, not a download failure. Re-requesting returns
HTTP 200 with a header-only CSV every time, while over the identical window and bounding box
`VIIRS_NOAA20_SP` returns 1,609 detections and `MODIS_SP` returns 216.

It costs the **McKinney Fire** (60,138 acres, ignited 2022-07-29) entirely — it is the one fire
of eleven spot-checked that has no matching event — plus the run phase of Oak Fire. Both land in
the **test** split, so this bites evaluation rather than training.

Two guards were added after finding this: `download.py` now flags any header-only chunk inline
during the run, and `python -m pipeline.download audit` re-runs the whole gap scan on demand.
A zero-row response is always suspicious here — California has detections on essentially every
day of the year, so 69 empty days out of 3,287 is the entire base rate.

## Phase 1.2 — HRRR

```bash
python -m pipeline.hrrr --list-fires
python -m pipeline.hrrr --all-mvp
```

`pipeline/hrrr.py` fetches `UGRD`/`VGRD` at 10 m and `TMP`/`RH` at 2 m from
`s3://noaa-hrrr-bdp-pds/` anonymously. Full surface files are ~109 MB; each one has a `.idx`
sidecar giving per-message byte offsets, so we merge the wanted messages into contiguous runs and
pull only those with HTTP Range requests — **4.7 MB per hour instead of 109 MB, a 23x saving**.
Concatenated GRIB2 messages are a valid GRIB2 file, so the subset opens directly in cfgrib.

The five MVP fires span years, regions, sizes and all three splits:

| fire_id | fire | start | detections | split |
|---|---|---|---|---|
| 2017_2405 | (median-size Sierra fire) | 2017-08-05 | 386 | train |
| 2018_4037 | Camp | 2018-11-08 | 4,451 | train |
| 2020_3779 | Creek | 2020-09-05 | 37,855 | val |
| 2021_3526 | Caldor | 2021-08-15 | 19,456 | val |
| 2022_3298 | Mosquito | 2022-09-07 | 4,207 | test |

Window is T-12h to T+24h per CLAUDE.md Phase 1.2. **Full dataset construction will need the whole
active period instead** — these fires burn for weeks, and Caldor alone runs 1,224 hours.

### Verified against known meteorology

At the Camp Fire ignition cell (39.774N, -121.500W) on 2018-11-08 20:00 UTC the subset gives
wind **from 073 degrees at 22 mph** with **RH 10%** and T 18.3 C. That is the Jarbo Gap offshore
wind event that drove the fire into Paradise, including the warm downslope signature. Nearest
HRRR cell lands within 30 m of the target coordinate.

### HRRR archive has occasional missing hours

`hrrr.20170806/conus/hrrr.t00z.wrfsfcf00.grib2` is absent from the bucket entirely — both the
GRIB2 and its `.idx` — while 23z and 01z are present. One hour missing in 185 requested (0.5%).

Phase 3 must handle this, because the feature stack needs wind at T, T-6h and T-12h and any one
of those can be absent. Suggested rule: **linearly interpolate isolated single-hour gaps** from
the bracketing hours, which is physically reasonable for 10 m wind and 2 m T/RH, and drop samples
where two or more consecutive hours are missing. `download_event` already returns the missing
timestamps so the gap list can be persisted alongside the samples.

## Phase 1.3 — LANDFIRE fuels

```bash
python -m pipeline.landfire --all-mvp
```

**The documented endpoint is dead.** `GPServer/LandfireProductService/submitJob` is shadowed by
the current website and returns HTML for both GET and POST, though the service *metadata* still
serves JSON, which makes it look alive. The working API is:

| endpoint | purpose |
|---|---|
| `GET /api/products` | catalog, 136 CONUS products |
| `GET /api/job/submit` | `Email`, `Layer_List`, `Area_of_Interest`, `Output_Projection` |
| `GET /api/job/status?JobId=` | poll to `Succeeded`, then fetch `outputFile` (a zip) |

`Email` is mandatory; it lives in `.env` as `LFPS_EMAIL`. LFPS clips and reprojects server-side,
so we request per-fire AOIs already in EPSG:5070 instead of the 30 m CONUS mosaic. Output is one
multi-band GeoTIFF with bands in `Layer_List` order — nothing in the file records which band is
which, so `landfire.py` writes a `.bands.json` sidecar.

Five MVP fires: 70 MB total, 30 m, EPSG:5070, values in range (FBFM40 91-202, CC 0-95%,
CH 0-430 dm). Jobs completed in 10-40 s each.

### Fuel version strategy — leakage, not staleness, is the risk

Only **LF2016, LF2022, LF2023** (+2024/25) carry FBFM40/CC/CH. LF2014 ships disturbance only and
LF2020 topography only, so **there is no pre-2016 fuel map at all**.

CLAUDE.md Phase 1.3 says use the version year *closest* to each fire. That is unsafe: LANDFIRE
rewrites fuels to reflect burn scars, so a version published after a fire encodes that fire's own
footprint and the model can read the target off its input. The rule must be *latest version
strictly before ignition*.

Both strategies are implemented and switchable via `landfire.version_strategy`:

- `fixed` (**current**) — LF2016 for every fire. Homogeneous across train/val/test, no leakage
  for 2017+. Cost: fuels go stale, up to 7 years by the 2023 test fires.
- `latest_prior` — newest version predating each fire. Fresher, but the test split would straddle
  two fuel versions. Before switching, verify LF2022's true disturbance cutoff; if it includes
  2022, then 2022 fires leak under this strategy too.

Fires before `contaminated_before_year` (2017) are flagged in the manifest either way — LF2016
includes disturbance through 2016, so 2015-2016 fires (138 of 692) are mildly contaminated with
no clean alternative available.

## Phase 1.4 — DEM

```bash
python -m pipeline.dem --all-mvp
```

USGS 3DEP 1/3 arc-second (~10 m) from AWS Open Data, no credentials. Tiles are COGs, so
`/vsicurl/` windowed reads pull only each fire's footprint — 56 MB rather than a 468 MB tile.
Elevation is stored raw in native EPSG:4269; slope, aspect and TPI are derived later per
CLAUDE.md.

**This does not scale as-is.** 1.16 GB for five fires means roughly 160 GB across all 692. For
full dataset construction, build one statewide 100 m EPSG:5070 DEM (~0.5 GB) and clip from it;
the 10 m per-fire clips are worth keeping now only to validate that resampling.

## Co-registration verified

`python -m pipeline.viz layers --fire-id 2018_4037` puts all three sources on one EPSG:5070
extent. Sampling each source at every detection coordinate:

| fire | detections | on burnable fuel | inside DEM | elevation p5-p95 |
|---|---|---|---|---|
| 2017_2405 | 386 | 99.7% | 100% | 2039-2682 m |
| 2018_4037 Camp | 4,451 | 95.0% | 100% | 246-1277 m |
| 2020_3779 Creek | 37,855 | 95.0% | 100% | 1074-2582 m |
| 2021_3526 Caldor | 19,456 | 96.5% | 100% | 1079-2444 m |
| 2022_3298 Mosquito | 4,207 | 99.0% | 100% | 557-1518 m |

The 1-5% off burnable fuel is expected and informative rather than a defect: fires cross roads and
developed land, and a 375 m VIIRS pixel does not align to a 30 m fuel cell. Camp sits lowest
precisely because it burned through Paradise, which LANDFIRE maps as non-burnable developed land.

## Phase 2 — labels

```bash
python -m pipeline.labels --all-mvp
python -m pipeline.viz labels --fire-id 2018_4037
```

Stores **new burn per window**, not cumulative masks: the cumulative label at T is the running OR
up to T and the target is the OR over the horizon. One representation serves any `t_horizon_h`,
and burned pixels are not rewritten into every later timestep. Zarr + compression gets 190-470x
on these sparse binary masks — **0.2 MB for five fires against 93 MB raw**, so all 692 fires will
be single-digit MB rather than the 3.55 GB budgeted.

### 6 h windows do not work with one polar orbiter

S-NPP is sun-synchronous, so every one of the 741,236 detections lands in two ~4 h bands per day:
**08-11Z** (night pass, ~02:00 local) and **19-22Z** (day pass, ~13:00 local), about 11 h apart.
Nothing is ever observed in the other 16 hours.

| window | total | empty | % empty |
|---|---|---|---|
| 3 h | 56,769 | 42,929 | 75.6% |
| 6 h (spec) | 28,678 | 17,097 | **59.6%** |
| 12 h | 14,685 | 3,104 | 21.1% |
| **24 h (chosen)** | 7,709 | 684 | **8.9%** |

At 6 h, 60% of targets are empty *because no satellite was overhead* — training on them teaches
the model that fires stop spreading every other timestep. **Switched to 24 h windows and a 24 h
horizon**, which also matches Huot et al. 2022 "Next Day Wildfire Spread" (already in the
references), making CSI directly comparable. Result: 157 of 158 MVP windows usable (99.4%), the
single exception a genuine multi-day outage during Caldor.

### Dilation is two jobs, not one

CLAUDE.md prescribes a 1-2 px dilation "to account for detection gaps". That conflates restoring
the VIIRS footprint with bridging gaps between detections, and doing only the first leaves a
stippled perimeter — at `dil=2` the Camp mask broke into **274 components** with a 0.74 fill
ratio. Adding morphological **closing** (dilate then erode) joins them without pushing the outer
boundary out the way more dilation does. Calibrated against official acreage:

| config | Camp | Creek | Caldor | Mosquito | components |
|---|---|---|---|---|---|
| dil=2 | 0.57x | 0.96x | 0.97x | 0.85x | 64-274 |
| dil=4 | 0.87x | 1.13x | 1.13x | 1.15x | 1-5 (over-inflates) |
| **dil=2 close=4** | **0.77x** | **1.07x** | **1.07x** | **1.04x** | **1-14** |

Three of four fires land within 7% of official acreage. Camp stays low at 0.77x for a real
reason, not a rasterisation one: it burned most of its area between overpasses and much of it
through developed land where VIIRS detection is poor.

## Phase 3 — feature stack and sample index

```bash
python -m pipeline.features static --all-mvp      # terrain + fuel
python -m pipeline.features index  --all-mvp      # (fire_id, timestep, tile) samples
python -m pipeline.features hrrr-check --all-mvp  # what weather is still missing
```

Every layer is warped onto **the grid read back from that fire's label zarr**, not a
recomputed one, so features and targets cannot drift by a half pixel. LANDFIRE is already
EPSG:5070 at 30 m so it only downsamples — FBFM40 by **majority** (averaging fuel codes would
invent classes), CC and CH by area average. The DEM warps from EPSG:4269 by bilinear, and slope,
aspect (sin/cos) and TPI are derived on the 100 m grid.

Static stack is 7 continuous channels (float16) plus a categorical fuel index: **1.4-9.8 MB per
fire**.

### Co-registration check

The test that a projection bug would fail:

| fire | burn on non-burnable | slope, burned | unburned | CC, burned | unburned |
|---|---|---|---|---|---|
| 2017_2405 | 0.5% | 19.0&deg; | 14.3&deg; | 16.8% | 18.9% |
| Camp | 6.5% | 13.6&deg; | 9.3&deg; | 42.1% | 32.3% |
| Creek | 5.1% | 13.6&deg; | 14.1&deg; | 33.0% | 22.1% |
| Caldor | 3.4% | 12.7&deg; | 11.7&deg; | 45.9% | 30.4% |
| Mosquito | 1.2% | 19.3&deg; | 12.4&deg; | 54.4% | 42.0% |

Burned cells sit on steeper ground and denser canopy than unburned ones in 4 of 5 fires — fire
runs upslope and follows fuel. Camp's 6.5% non-burnable is Paradise, which also explains its
0.77x acreage.

### Sample index

661 samples over the 5 MVP fires. Tiles per timestep scale with fire size, from 1.00 for the
small 2017 Sierra fire to 8.70 for Creek — which is the option-A argument in one number.

Two bugs worth recording, both mine:

- Tile origins were snapped to a **global lattice** in absolute EPSG:5070 metres so tiles would
  align across fires. Lattice points do not reliably fall inside a small fire's raster, and
  `2017_2405` silently produced **zero samples**. Tiles never actually need to align between
  fires; origins are now perimeter-relative and clamped to the raster.
- `tile_clipped` compared each tile's target against the **whole fire's** target, which is false
  by construction whenever a timestep has several tiles — it flagged 85% of samples. It now tests
  whether target growth touches the tile edge, giving 57.6%.

### Weather — HRRR re-pull done

```bash
python -m pipeline.hrrr --all-mvp --dry-run   # hours and GB, fetch nothing
python -m pipeline.hrrr --all-mvp             # fetch, window, store, delete GRIBs
```

`download_event` used to be pinned to CLAUDE.md's T-12h..T+24h, which served under 5% of what
24 h windows with `t_steps: 3` require. It now derives its hours from the fire's own label zarr,
so weather can never span a different period than the targets.

**The needed set is much smaller than an hourly count suggests: 501 hours, not 3,917.** For each
usable window T the model sees `t_steps` windows ending at T, each needing wind at its own
T, T-6h, T-12h — so the union is

    {w - k*window_hours - lag : w in windows, k < t_steps, lag in (0, 6, 12)}

Consecutive 24 h windows overlap heavily in that set, which is where the 7.8x comes from. It is
also ~25% under a uniform `hrrr_step_hours: 6` grid, because the 18h-offset hour is never read by
any feature.

| fire | hours needed | present | fetched | stored |
|---|---|---|---|---|
| 2017_2405 | 57 | 55 | 254.6 MB | 1.7 MB |
| Camp | 48 | 48 | 175.8 MB | 2.1 MB |
| Creek | 195 | 195 | 946.5 MB | 10.6 MB |
| Caldor | 159 | 159 | 1,164.3 MB | 7.6 MB |
| Mosquito | 42 | 42 | 307.1 MB | 1.7 MB |
| **total** | **501** | **499 (99.6%)** | **2.8 GB** | **23.7 MB** |

Windowing to each fire's footprint plus a 60 km margin is a **120x** reduction, and the GRIBs are
deleted as they are consumed, so peak scratch is one file. The 4.52 GB HRRR line in the storage
budget below is now a large overestimate.

Stored per fire in `data/processed/hrrr/{fire_id}.zarr`: `data` (hours, 4, ny, nx) float32 with
channels `u10, v10, t2m, r2`, a `filled` flag per hour making the fetch resumable, and **`lon`/`lat`
arrays per cell**. Those coordinates are not optional — HRRR is Lambert Conformal, so there is no
correct way to resample onto the 100 m EPSG:5070 grid from the lon/lat bounding box alone.

Verified against the Camp Fire ignition cell (nearest stored cell 315 m from 39.774N, -121.504W):

| hour | wind | RH | T |
|---|---|---|---|
| 2018-11-08 12z | from 076&deg; at 21.6 mph | 23.2% | 10.1 &deg;C |
| 2018-11-08 18z | from 063&deg; at 22.4 mph | 18.5% | 12.9 &deg;C |
| 2018-11-09 00z | from 076&deg; at 15.9 mph | 8.3% | 17.7 &deg;C |

That is the Jarbo Gap offshore event: ENE downslope wind, RH collapsing through the day, and
temperature *rising* into the evening. It brackets the 073&deg;/22 mph/RH 10%/18.3 &deg;C recorded from the
earlier hourly pull, so the crop and the Lambert indexing are correct.

### Archive holes repair themselves

Two hours are absent from the bucket for 2017_2405: `20170806_00z` (the hole already documented
above) and `20170819_18z`. Both are isolated rather than consecutive.

On a 6-hourly stored set an isolated hole would have to be interpolated across **12 h**, which is
far too long an assumption for 10 m wind. So a missing hour now automatically pulls its **plus and
minus 1 h neighbours**, and the fetcher records them in `repair_hours`:

| gap | bracket | span |
|---|---|---|
| 20170806_00z | 20170805_23z / 20170806_01z | 2 h |
| 20170819_18z | 20170819_17z / 20170819_19z | 2 h |

Four extra requests, ~20 MB, and Phase 3c interpolates across 2 h instead of 12. Repair is
bounded by `repair_rounds` (default 2) so a genuine multi-hour outage walks outward once and then
stops rather than expanding indefinitely — and a real outage should still drop its windows, per
the rule in Phase 1.2.

The store is therefore a **superset** of the needed set. `hrrr_coverage` counts membership rather
than comparing lists, or repair hours would read as a coverage failure.

## Phase 3c — weather channels

```bash
python -m pipeline.features weather --all-mvp
```

**661/661 samples produce finite, physical weather.** Nothing is materialised — a per-fire
weather raster at 100 m would be ~2 GB for Creek alone against 23.8 MB for all five fires in
native HRRR resolution, so `weather_tile` regrids on demand for the tile being cropped.

| fire | samples | mean wind | RH | T | Fosberg 10-h |
|---|---|---|---|---|---|
| 2017_2405 | 10 | 5.2 m/s | 33.2% | 20.7 &deg;C | 8.3% |
| Camp | 25 | 2.2 m/s | **16.3%** | 16.1 &deg;C | **4.8%** |
| Creek | 522 | 3.9 m/s | 31.2% | 18.6 &deg;C | 8.1% |
| Caldor | 87 | 3.9 m/s | 27.3% | 21.3 &deg;C | 7.1% |
| Mosquito | 17 | 2.6 m/s | 32.4% | 23.7 &deg;C | 8.1% |

Camp is the driest fire with the lowest fuel moisture by a wide margin, which is the correct
physical ordering.

### Regridding is an exact affine, not a scattered interpolation

Projecting the stored per-cell lon/lat through the standard HRRR Lambert CRS reproduces a
**regular 3000 m grid to within 0.7 m** — float32 coordinate precision. So the 3 km to 100 m
resample is fire grid -> EPSG:5070 -> lon/lat -> HRRR Lambert -> fractional index, then bilinear.
`hrrr_affine` re-checks the regularity per fire and raises if it fails, which is what would catch
a wrong projection immediately.

Bilinear, not nearest: at a 30x resolution jump nearest-neighbour leaves 3 km plateaus that the
model would learn as terrain edges. Measured on a Camp tile, **0.00% of adjacent 100 m cells are
identical** (nearest-neighbour would be ~97%), median step 0.012 m/s.

Wind is interpolated as **u/v components**, never as speed and direction — direction wraps at
0/360, the same discontinuity aspect already encodes as sin/cos to avoid. Fosberg is computed
**after** regridding for the same class of reason: it is nonlinear in T and RH, so evaluating it
at 3 km and interpolating the result would smear exactly the dry extremes that drive spread.

### Verified: the Camp Fire's full meteorological arc in one tensor

Sample at T = 2018-11-11 00z, so step 0 is the window ending 11-09 — inside the offshore event:

| step | wind | RH | Fosberg |
|---|---|---|---|
| 0 (oldest, 11-09) | ENE 45-69&deg; at 12-17 mph | 7.7% | **2.4%** |
| 1 (11-10) | lag 12h ENE 14 mph -> lag 0h **SW 238&deg;** | 10.6% | 3.5% |
| 2 (most recent, 11-11) | SW/NW, light | 21.1% | 6.0% |

Step 0 is Jarbo Gap at full intensity with fuel moisture at 2.4%. Step 1 captures the **wind
reversal inside a single step's lags**. That is the Phase 8 wind-shift failure mode, and it is
visible only because of the 6 h lag structure — at 24 h sampling it would appear as 12 mph ENE
becoming 4 mph SW with nothing in between. This validates the temporal indexing and the regrid
together.

### Channel list — `C=12` in CLAUDE.md does not hold

Pinned in `model.channels`, which the feature builder, `norm_stats.json` and the Dataset's flip
augmentation all index into. CLAUDE.md declares `C=12`, but its own table lists 14 rows and omits
`cc`, `tpi`, `elevation` and Fosberg — all of which the pipeline builds or the "Derived Features"
section requires. The real count is **17 continuous + fuel**, embedded at 8 dims rather than
one-hot over 40 (29 classes observed, and the codes are not ordinal).

Each sequence step is a self-contained weather snapshot with its own 6 h history: step k is the
window ending at `T - k*window_hours`, and its lags are relative to *that step's* time, not the
sample's T. A sample therefore spans `window_hours*(t_steps-1) + 12 h` = **60 h** of weather.

### Elevation is centred per fire

`model.elevation_mode: fire_centred`, with `raw` and `none` as ablation arms. Variance
decomposition over all 661 tiles:

| channel | within-tile sd | between-tile sd | % variance between |
|---|---|---|---|
| **elevation** | **413.31** | **498.56** | **59.3%** |
| slope | 8.22 | 1.45 | 3.0% |
| aspect_sin / cos | 0.71 / 0.69 | 0.06 / 0.08 | 0.6% / 1.2% |
| tpi | 21.57 | 0.10 | 0.0% |
| cc | 18.13 | 8.27 | 17.2% |
| ch | 10.38 | 4.20 | 14.1% |

Elevation is the only static channel whose variance is mostly *between* tiles, which makes its
absolute level a per-fire identity handle — and with Creek alone contributing 522 of 661 MVP
samples, that is a memorisation pathway rather than terrain information. But its within-tile sd
is 413 m, so it is emphatically **not** a flat offset: dropping it outright would discard the
km-scale valley-to-ridge gradient that the 300 m TPI cannot see, and which is exactly the scale
upslope runs happen at. Centring per fire keeps that gradient and drops the identity component.
Per *fire* rather than per tile so overlapping tiles stay consistent for the same ground.

Caveat worth keeping: the 5-fire measurement overstates the identity risk relative to the full
622-fire archive, where elevation bands overlap far more. Hence the ablation arms.

## Phase 4 — dataset

```bash
python -m pipeline.dataset splits    # fire-level train/val/test lists
python -m pipeline.dataset norm      # per-channel stats, TRAIN fires only
python -m pipeline.dataset check     # shapes, splits, flip correctness, throughput
```

Batches as `(B, t_steps, C, 256, 256)` float32 with fuel as a separate int tensor for the
embedding, plus a `(B, 256, 256)` target. Nothing materialised — each `__getitem__` crops the
burn mask and static stack and regrids weather for that tile.

### Two normalisation bugs worth remembering

Both were caught by looking at the printed stats rather than by anything failing.

**Flips did not commute with normalisation.** `__getitem__` normalises and then flips, and the
flip negates the *normalised* value. For a channel with mean `mu`, negating gives
`(mu - x)/sigma` where the true flipped value is `(-x - mu)/sigma` — off by `2*mu/sigma`. The
augmentation was quietly teaching shifted wind physics.

**Per-lag means invented a temporal signal.** `u10_lag0` came out at +1.15 and `u10_lag12` at
-2.30. Those are one variable at three times, so normalising each by its own mean would make a
genuinely constant wind field appear to change across lags — corrupting exactly the channels
that exist to carry temporal change.

Both are fixed by `DIRECTION_GROUPS`: signed direction components (`u10_*`, `v10_*`,
`aspect_sin/cos`) get **zero mean and a std shared across the group**. Zero mean is not a
convenience — it is the symmetry the flip augmentation asserts is true, so making it exact is
what lets negation and normalisation commute. `check` asserts it as a regression test.

### Flip augmentation

`aspect_sin` is the east component of the downslope bearing and `aspect_cos` the north
component, so they mirror exactly as `u` and `v` do:

| flip | mirrors | negates |
|---|---|---|
| east-west | columns | `u10_*`, `aspect_sin` |
| north-south | rows | `v10_*`, `aspect_cos` |

`_flip` takes explicit `ew`/`ns` flags so this is testable — the first version of the test seeded
the RNG, drew 0.549 and 0.715, fired neither flip, and passed vacuously.

### Throughput

136 ms/sample single-threaded (7.4/s), after two fixes worth 1.7x: the hour store was being
fully decompressed on every sample (27 MB per call for Creek) and is now read lazily per chunk
with an LRU cache, and the bilinear gather is `map_coordinates` with cached coordinates rather
than recomputed fancy-index weights per lag. On 8 workers that is ~0.3 s per batch of 16, which
is roughly balanced against the GPU step.

### Splits

320 / 166 / 136 fires, grouped by `fire_id` and asserted disjoint. **`norm_stats.json` is
provisional**: the MVP index only covers 2 train fires and 35 samples, so the values will move
once the full archive is built. Regenerate before any run whose metrics matter.

## Phase 5 — model

`ConvLSTMUNet` is 5.35 M params: a ConvLSTM per scale at [64, 128, 256] with max-pooling
between, each layer's final hidden state doubling as that scale's skip connection, then a
transposed-conv decoder and a 1x1 head. `UNet` is the 1.90 M baseline with the time axis folded
into channels.

**Both return logits, not probabilities.** CLAUDE.md specifies a sigmoid on the final layer, but
`pos_weight` reaches ~9,400 at the median sample, and `sigmoid` then `BCELoss` takes the log of a
saturated probability and loses the gradient in float16. `BCEWithLogitsLoss` fuses them via
log-sum-exp and stays stable; `predict()` applies the sigmoid for inference and metrics. The
architecture is unchanged — only where the exponential is evaluated.

Layer norm on the hidden state is implemented as `GroupNorm(1, C)`, which normalises over
(C, H, W) per sample. Identical to LayerNorm but without pinning the spatial size into the
module, so the same encoder runs at 256 or 512 px unmodified.

### The fuel index was per-fire — a silent bug

`fuel_dense_index` derived its vocabulary from the codes each fire happened to contain. So
**index 5 meant fuel code 101 (short grass) in Camp and code 99 (barren) in Creek**, with 15 of
27 shared codes disagreeing. A single embedding table would have been learning contradictory
semantics per fire, and nothing would have failed — it would simply have trained on noise.

Now fixed to the published SB40 vocabulary (45 codes + nodata), identical for every fire and
stable across splits and any later LANDFIRE re-clip. Found only because the model needed a
`fuel_classes` argument and the per-fire counts (20/29/30/30/28) did not agree.

### Measured VRAM — RTX 3090, real 17+8 channel input, AMP, sustained

| config | batch | peak | step |
|---|---|---|---|
| 256 px | 12 | 16.45 GB | 663 ms |
| **256 px** | **16** | **21.91 GB** | **828 ms** |
| 256 px | 24 | spills to system RAM | — |
| 512 px in / 256 supervised | 4 | 21.87 GB | — |
| 512 px in / 256 supervised | 8 | OOM | — |

`grad_accum_steps` is **dropped** — it existed only to reach an effective batch of 16 on 8 GB.

**But batch 16 is the wrong choice, and only end-to-end measurement showed it.** The table above
is the model in isolation. With DataLoader workers and pinned buffers also resident, 21.91 of
22.8 GB is 96% occupancy and the allocator thrashes:

| batch | workers | s/step | samples/s | min/epoch | peak |
|---|---|---|---|---|---|
| 16 | 4 | 1.89 | 8.5 | 11.1 | 21.9 GB |
| **12** | **6** | **0.58** | **20.7** | **3.6** | 16.5 GB |
| 8 | 4 | 0.43 | 18.7 | 4.0 | 11.0 GB |

**Batch 12 is 2.4x faster than batch 16 despite being smaller**, and raising workers made 16
*worse* (2.38 s at 8, 2.58 s at 12) — contention, not starvation. Fitting is not the same as
being fast. 100 epochs goes from ~21 h to ~8 h.

**The overlap-tile idea does not pay after all.** The earlier analysis guessed a 16 GB card would
allow batch 6 at 512 px; measured on 24 GB it is batch 4, OOM at 8. That is 4x the per-sample cost
for 4x fewer samples per step, on top of the 1.69x more pixel work per epoch already documented.
Left off via `model.supervise_centre: null`, with the code path in place should it ever be wanted.

### Single-batch overfit

The fastest end-to-end check that data, model and loss are connected. 4 real samples, 60 steps,
positive fraction 0.28% (`pos_weight` 353):

| step | loss | CSI | TP | FP | FN |
|---|---|---|---|---|---|
| 0 | 1.4257 | 0.023 | 312 | 12,942 | 428 |
| 30 | 0.1498 | 0.271 | 716 | 1,903 | 24 |
| 59 | 0.0488 | 0.342 | 740 | 1,426 | 0 |

False negatives reach zero first and false positives then fall — the trajectory a heavily
weighted BCE should produce, recall before precision.

## Full dataset — California 2015-2023

```bash
python -m pipeline.labels   --all
python -m pipeline.landfire --all --quiet --workers 6
python -m pipeline.dem      --statewide
python -m pipeline.hrrr     --all                    # deduped by calendar hour
python -m pipeline.features static  --all --quiet
python -m pipeline.features index   --all
python -m pipeline.features weather --all --prune    # validate, then drop what fails
python -m pipeline.dataset  splits
python -m pipeline.dataset  norm
```

**7,070 samples over 450 fires**, every one verified to produce a finite tensor.

| stage | result | on disk |
|---|---|---|
| labels | 622 fires | 22 MB (1,614 MB raw) |
| LANDFIRE | 622 fires, 0 failures | 3.0 GB |
| statewide DEM | 12,162 x 13,245 @ 100 m | 226 MB |
| static features | 610 fires (12 refused) | 1.0 GB |
| HRRR windows | 25,979/27,249 fire-hours (95.3%) | **0.91 GB** |
| sample index | 7,070 samples, 450 fires | 0.5 MB |

The HRRR figure is the one worth noting: **5,850 unique calendar hours served 25,527 fire-hours
(4.36x dedupe)**, ~15 GB fetched compressed to 0.91 GB stored, and peak scratch stayed one GRIB.

### Attrition, honestly accounted

622 kept fires become 450 trainable ones:

| stage | fires | why |
|---|---|---|
| kept by `events` | 622 | |
| static built | 610 | 12 refused: DEM coverage below `min_dem_coverage` |
| produced samples | 479 | `min_timesteps: 3` is one short — see below |
| survived weather | **450** | 716 samples lost to unbridgeable HRRR gaps |

**`min_timesteps: 3` is one short of usable.** A sample needs `t_steps` windows of history
*and* a target window, so 4 usable windows is the real floor. Of 105 fires with exactly 3,
five yielded any sample. `sample_index.parquet`, not `keep`, is the authority on what is trainable.

### Split boundary moved off the CLAUDE.md years

CLAUDE.md Phase 7 specifies 2015-2019 / 2020-2021 / 2022-2023. Measured on the built dataset that
is unusable: it puts **69.7% of samples in validation against 21.7% in training**, because 2020 and
2021 are the two most extreme seasons on record and tiling multiplies each megafire into many
tiles. Train was hit twice over — it also lost the most to weather gaps (2,253 -> 1,537, -32%),
since 2015-2016 HRRR is the least complete. Training on 22% while selecting models on 70% is not
defensible.

Moving 2020 into training (`sampling.split_years`) keeps strict chronology, so the
temporal-generalisation claim survives, and keeps `fire_id` grouping intact:

| split | years | samples | share | fires | median positive | implied `pos_weight` | clipped |
|---|---|---|---|---|---|---|---|
| train | 2015-2020 | **4,507** | 63.7% | 290 | 0.247% | 404 | 34.5% |
| val | 2021 | 1,956 | 27.7% | 47 | 0.339% | 294 | 44.2% |
| test | 2022-2023 | 607 | 8.6% | 113 | 0.078% | 1,284 | 4.0% |

Train grew 2.9x, from 1,537 to 4,507.

**Two costs to keep in view.** Validation is now a single season and only 47 fires, dominated by
Dixie and Caldor, so val CSI is a narrower signal than 1,956 samples suggests — early stopping on
it will be noisier than the count implies. And the test split is both small (607) and much
sparser: its median positive fraction is 0.078% against 0.247% in training, a 5x harsher implied
`pos_weight`, because 2022-2023 were mild seasons of mostly small fires. Test CSI will therefore
run lower than validation CSI for reasons that have nothing to do with generalisation, so the two
are not directly comparable. Report test metrics stratified by fire size.

**Norm stats must be regenerated with any split change** — moving 2020 into training shifted the
distribution measurably: RH 31.6% -> 28.5%, temperature 20.7 -> 23.3 &deg;C, Fosberg 8.08% -> 7.36%.
That is the 2020 heat and drought entering the training set.

### Two HRRR data-quality findings

**Pre-2017 HRRR carries no 2 m RH at all.** Its 2 m fields are DPT, SPFH and TMP; RH arrives with
the HRRRv2 upgrade, probed to **2016-08-23**. That would have silently cost 858 hours in 2015 and
316 in 2016 — disproportionately the training split. RH is now derived from dewpoint via
Magnus-Tetens, validated against an hour carrying both fields:

| T range | mean error |
|---|---|
| 20-30 &deg;C | **0.65 pts** |
| 10-20 &deg;C | 2.4 pts |
| below 10 &deg;C | 11-23 pts |

Accurate where fire spreads, poor where it does not. **3,230 fire-hours across 104 fires** use it,
recorded per fire in `rh_derived` so the provenance is auditable rather than invisible.

**153 hours are genuinely absent** from the bucket; 1,140 repair hours were fetched around them so
those gaps interpolate across 2 h rather than 12.

### Three crashes and a near-miss, all in the ingest path

Worth recording because two were robustness and one was correctness:

- `NoSuchKey` on the byte-range fetch. The `.idx` guard did not cover the object fetch, and the
  bucket holds hours whose index exists but whose GRIB does not. Killed a run 2,400 hours in.
- `int('')` in `parse_idx`. Some `.idx` files carry blank or truncated lines; one aborted the
  repair stage. The parser now skips unparseable lines.
- **The window slice was recomputed per run.** HRRR files from different years encode coordinates
  at slightly different precision, so a fire's window could land a column differently depending on
  which hour was fetched first. It surfaced as a broadcast error — but with coincidentally matching
  shapes it would have written weather **offset by a cell against the stored lon/lat**, silently
  corrupting every downstream regrid. The window is now pinned in the store on first use, with
  recovery for stores written before the pin.

## First training run

```bash
python train.py --model convlstm_unet
```

Early-stopped at epoch 26, best at **epoch 16**, ~2 h on the 3090 at 4.4 min/epoch.

| | value |
|---|---|
| validation CSI / IoU | **0.2641** @ threshold 0.95 |
| FAR | 0.635 |
| POD | 0.489 |

CSI ran 0.041 -> 0.264 over 16 epochs. The LR warm restart is visible at epoch 10 (train loss
jumps 0.374 -> 0.457, val CSI dips then recovers), and train/val diverge from about epoch 8
(train 0.26 against val 1.51 by the end) — early stopping caught the overfit.

### The threshold sweep had to be widened, and it changed the ranking

`pos_weight` of ~103 deliberately pushes over-prediction, so the model's probabilities are
inflated and the CSI-optimal threshold is high. A first run capped the sweep at 0.7 and selected
0.7 in *every* epoch — pinned at the boundary. Re-run with the sweep extended to 0.98:

| epoch | capped at 0.7 | swept to 0.98 | understated |
|---|---|---|---|
| 1 | 0.1374 @0.7 | 0.2163 @0.9 | 57% |
| 2 | 0.1796 @0.7 | 0.2390 @0.9 | 33% |
| 3 | 0.1597 @0.7 | 0.2411 @0.95 | 51% |
| 4 | 0.1456 @0.7 | 0.2506 @0.95 | 72% |

The damage was not just an understated number: under the capped sweep epochs 3-4 looked like
*regressions* (0.180 -> 0.160 -> 0.146) and counted toward early stopping, when correctly measured
they were monotonic improvements (0.239 -> 0.241 -> 0.251). `train.py` now warns whenever the best
threshold is the largest in the sweep.

### Baselines — persistence is NOT strong here

CLAUDE.md Phase 7 calls persistence "deceptively strong". That is true of a **cumulative** burn
target, where most correct pixels were already burning. Ours is **new burn only**
(`burn[t+1] & ~cumulative[t]`), so persistence predicts exactly the cells the target excludes.
Measured over the full validation split:

| baseline | CSI | FAR | POD |
|---|---|---|---|
| persistence (= current burn) | **0.0000** | 1.000 | 0.000 |
| dilate 1 ring | 0.0591 | 0.894 | 0.117 |
| dilate 3 ring | 0.0811 | 0.899 | 0.296 |
| **dilate 5 ring** (best naive) | **0.0833** | 0.906 | 0.419 |
| dilate 8 ring | 0.0802 | 0.914 | 0.542 |
| all-positive | 0.0128 | 0.987 | 1.000 |
| **ConvLSTM U-Net** | **0.2641** | **0.635** | 0.489 |

**3.2x the best naive baseline**, and the manner of the win matters more than the ratio: at
comparable detection rate (POD 0.489 vs 0.419) the model's false-alarm ratio is 0.635 against
0.906. A dilation ring sprays predictions in every direction and is wrong ~90% of the time; the
model is wrong 64% of the time. It has learned *where* fire goes, not just that it spreads.

Replace persistence with the dilation ring as the naive baseline in Phase 7 — persistence is
uninformative against this target.

### Two caveats on the 0.2641

- **Not comparable to published CSI on cumulative masks.** Huot et al. and similar predict the
  fire mask at T+1 day, including all the already-burning cells, which are trivially correct.
  Ours excludes them, so it measures a strictly harder quantity and will read lower than
  literature numbers that are not measuring the same thing.
- **Validation is one season**, 47 fires dominated by Dixie and Caldor. The test split is smaller
  and ~3x sparser (median positive 0.078% vs 0.339%), so test CSI should be expected to land below
  this for reasons unrelated to generalisation.

## Test-set evaluation

```bash
python evaluate.py --split test
```

The operating threshold comes from the checkpoint, where it was chosen on validation.
Re-choosing it on test would be selecting a hyperparameter on the test set; the test-optimal
value is reported as an oracle number, not the headline.

| | CSI/IoU | FAR | POD |
|---|---|---|---|
| **ConvLSTM U-Net @ val threshold 0.95** | **0.1849** | 0.783 | 0.558 |
| (oracle: best test threshold 0.98) | 0.1953 | 0.689 | 0.345 |
| persistence (= current burn) | 0.0000 | 1.000 | 0.000 |
| dilate 1 ring | 0.0872 | 0.866 | 0.201 |
| **dilate 3 ring** (best naive on test) | **0.1019** | 0.886 | 0.484 |
| dilate 5 ring | 0.0910 | 0.904 | 0.651 |

Brier score 0.03179. **Model beats the best naive baseline 1.82x** — real, but narrower than the
3.2x on validation.

**The validation-selected threshold transfers.** The test oracle gains only 0.0104 CSI over it
(0.1953 vs 0.1849), so the operating point is not overfit to validation and the honest protocol
costs almost nothing.

### Pooled CSI flatters; report per fire

| aggregation | CSI |
|---|---|
| pooled over all tiles | 0.1849 |
| **per fire, median** | **0.0990** |
| per fire, mean | 0.1120 |

Pooling weights each *pixel* equally, so a handful of large fires dominate. Per-fire aggregation
weights each *fire* equally and is the honest summary — it is also what the 50% tile overlap
requires, since adjacent tiles are not independent samples. Report both, and never quote the
pooled figure alone.

### Skill scales with fire size

| quartile (burned cells) | fires | median CSI | mean CSI |
|---|---|---|---|
| Q1 smallest | 29 | 0.0556 | 0.0772 |
| Q2 | 28 | 0.0971 | 0.1012 |
| Q3 | 28 | 0.0919 | 0.1102 |
| **Q4 largest** | 28 | **0.1887** | 0.1605 |

**3.4x between the smallest and largest quartiles.** This is the stratification the split analysis
predicted was necessary: 2022-2023 were mild seasons of mostly small fires, so the test split is
weighted toward exactly the cases the model handles worst, and a single headline CSI hides it.
Small fires are genuinely harder — fewer active pixels, a larger share of the perimeter influenced
by suppression, and less signal per tile.

### Val 0.2641 -> test 0.1849

A 30% drop, from two causes that should not be conflated:

- **Composition.** Test is ~3x sparser (median positive 0.078% vs 0.339%) and skewed small, and
  the size table above shows how much that costs.
- **Generalisation.** Whatever remains after composition. Separating the two properly needs the
  model scored on a size-matched subset — worth doing before quoting a generalisation gap.

## Storage budget — do NOT materialise samples

Measured on this machine: **83 GB free of 475 GB.** That is the binding constraint on the whole
project, and CLAUDE.md Phase 3 as written does not fit inside it.

Phase 3 says to save patch samples to zarr/HDF5. With tiling that is 26,876 samples of shape
`(3, 12, 256, 256)`:

| approach | size | verdict |
|---|---|---|
| materialise tiles, float32 | 253.6 GB | 3x over budget |
| materialise tiles, float16 | 126.8 GB | 1.5x over budget |
| **per-fire rasters, crop tiles at `__getitem__`** | **~11 GB** | fits |

**Decision: store per-fire rasters and crop tiles on the fly in the Dataset class.** Nothing
downstream needs materialised patches, and 50%-overlapping tiles duplicate data when written out
but cost nothing when cropped on demand — so this makes the tiling scheme cheaper, not dearer.

Composition of the ~11 GB:

| component | format | size |
|---|---|---|
| static layers (slope, aspect sin/cos, CH, CC, fuel) | float16 | 2.99 GB (1.39 GB if statewide) |
| burn masks, 11,581 timesteps | uint8 | 3.55 GB |
| HRRR local windows | float32 | 4.52 GB |

### Two further traps, each fatal on its own

- **Never retain raw HRRR GRIBs.** ~204 GB across all fires. Stream, extract the local window,
  delete the GRIB — stored footprint becomes 4.52 GB.
- **Never store per-fire 10 m DEM.** ~160 GB across 692 fires. Use one statewide 100 m
  EPSG:5070 DEM (~0.5 GB).

Full dataset then lands at roughly **15-20 GB including raw keeps**.

### Bandwidth is a separate budget

Fires overlap heavily, so 176,413 fire-hours dedupe to 43,366 unique calendar hours (4.1x). Still
~204 GB of transfer at hourly resolution.

- **6-hourly: ~34 GB.** Matches the spec — the feature table only uses wind at T, T-6h, T-12h.
- **hourly: ~204 GB.** Enables 6 h-aggregated wind stats (mean/max/variability) instead of
  instantaneous snapshots, which is better physics and speaks to the Phase 8 wind-shift mode.

Starting 6-hourly. HRRR-Zarr (`s3://hrrrzarr/`) would allow spatially chunked reads and cut this
sharply, but only covers 2016-10 onward — it would mean mixing weather sources mid-study, the same
homogeneity problem already avoided for sensors and fuels.

## Decided — patch scheme: tile the active perimeter (option A)

Samples are indexed by `(fire_id, timestep, tile_index)`, 256x256 tiles on a 128 px stride
covering the active perimeter at time T. Configured under `sampling:` in `baseline.yaml`.
Two consequences to hold onto downstream:

- **Splits must group by `fire_id`.** Tiles of one fire landing in both train and test would
  leak badly and inflate CSI. This is now `sampling.group_splits_by`.
- **Adjacent tiles overlap 50%,** so per-sample confidence intervals computed as if samples were
  independent will be too narrow. Report metrics aggregated per fire, not per tile.

The analysis behind the choice follows.

### Patch scheme comparison

Across the full archive there are **11,581 (fire, 6 h window) pairs over 692 fires**. Measured
against a 256x256 patch at 100 m (25.6 km across):

| scheme | windows usable | usable for top-5% largest fires | samples | disk (fp32) |
|---|---|---|---|---|
| **A** 256 tiled, 128 px stride | 100% | 100% | 26,876 | 0.25 TB |
| **B** 256 centred (as written) | 90.9% | 64.0% | 10,529 | 0.10 TB |
| **C** 512 centred | 98.0% | 91.7% | 11,350 | 0.43 TB |

The headline 90.9% for option B hides the actual problem, which is that the loss is not spread
evenly — it falls almost entirely on the biggest fires:

| fire-size quintile | windows | usable under B |
|---|---|---|
| Q1 smallest | 2,317 | 100.0% |
| Q2 | 2,344 | 100.0% |
| Q3 | 2,289 | 99.9% |
| Q4 | 2,361 | 95.5% |
| **Q5 largest** | 2,270 | **58.5%** |

No fire loses *all* its windows under B — every fire contributes its early growth phase, and
only the run phase is clipped. So B is not "megafires are excluded" but the subtler and arguably
worse "megafires are included only while they are small", which biases the model toward
under-predicting the fast run phase that operationally matters most.

Under A, tiling is cheap for typical fires and only expands where the fire is genuinely big:
median 1 tile per window, p90 5, max 27. Half of all tile-samples come from the top-5% largest
fires — that is the 2.6x sample gain, and it lands exactly where B is starved.

### Measured VRAM (this machine, RTX 4070 Laptop, 8.00 GB, ~6.9 GB free)

Peak allocated for a spec-faithful ConvLSTM U-Net (3 layers, hidden [64,128,256]), forward +
backward + optimizer step, AMP on, T=3, C=12:

| batch | 256x256 | 512x512 |
|---|---|---|
| 1 | 0.63 GB | 2.43 GB |
| 2 | 1.23 GB | 4.82 GB |
| 4 | 2.43 GB | 9.62 GB (spills) |
| 8 | 4.82 GB | OOM |
| 16 | 9.62 GB (spills) | OOM |

"Spills" means it exceeded free VRAM and Windows fell back to system RAM — it completes rather
than crashing, but it is PCIe-bound and far too slow to train on. Practical maxima here are
**batch 8 at 256** and **batch 2 at 512**.

This is what makes option C expensive in practice: it needs a 4x smaller batch than A/B, its
samples are 4x larger on disk, and it still leaves 8.3% of large-fire windows clipped. It buys
less than tiling and costs more.

### Would more VRAM have changed the answer?

Asked and checked, because it is the obvious objection. On a 16 GB card, 512 px goes from batch 2
to batch 6, which removes C's practical blocker. Three costs survive that are properties of the
data rather than the hardware, so the answer stayed A:

- **Coverage.** 512 px still covers only 91.7% of large-fire windows. Closing that needs ~1024 px
  (99.6%), which is ~9.6 GB/sample — batch 1 even on 16 GB.
- **Epoch cost.** A pushes 1.76 G px/epoch, C pushes 2.98 G px — C does **1.69x the pixel work**
  despite 2.4x fewer samples, because tiles materialise only where fire is (median 1 per window)
  while C pays full 512^2 on every window including the small fires that dominate.
- **Class imbalance.** Median positive fraction drops from 0.011% (A) to 0.003% (C), pushing
  implied `pos_weight` from ~9,400 to ~29,100. CLAUDE.md already names imbalance as the loss
  design driver; C makes the median sample 3.5x emptier.

C's real advantage is context: 51.2 km of upwind terrain and fuel visible to the model versus
25.6 km. If this ever moves to a 16 GB card, the better use of the headroom is **feeding a 512 px
input while supervising only the central 256 px** (the U-Net overlap-tile strategy) — C's context
with A's coverage and positive fraction. Not worth the complexity on 8 GB.

This needs a decision because it changes the sample index scheme that Phases 3, 4, and 7 all
build on — in particular, splits must then group by `fire_id` so tiles of one fire never straddle
train/test.

## Known issues to revisit

- `batch_size: 16` in the config comes from CLAUDE.md, but measured 9.62 GB against the old 8 GB
  card's 6.9 GB free — it spilled to system RAM rather than training at speed. The config's
  **8 + `grad_accum_steps: 2`** was that workaround. **On the 24 GB 3090 this no longer binds:**
  bs16 at 256 fits natively, so the accumulation can be dropped (keep AMP either way).
- **The 512 px context question is reopened by the 3090.** The analysis below concluded option A
  partly because 512 px meant batch 2 on 8 GB. It also concluded that *above 16 GB* the right use
  of headroom is feeding 512 px while supervising only the central 256 px — the U-Net overlap-tile
  strategy, giving C's upwind context with A's coverage and positive fraction. That is now
  affordable. It does not change the sample index (still A-tiled), only what the Dataset crops
  around each tile, so it can be deferred to Phase 5 without reworking Phases 3-4.
- The August Complex (2020) and similar complexes are genuinely multiple ignitions that merged.
  CLAUDE.md says to treat merges as a single union geometry, which the clustering does by
  construction — worth confirming that this is still what you want for the largest events.
