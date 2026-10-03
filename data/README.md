# Data

All files in `data/raw` are the real datasets used in the article, exactly as obtained
from their providers (Farm-Flow: the subset of records available to the authors).
Nothing in `data/raw` is synthetic. `data/processed` is created by the experiment scripts
(trained estimator ensembles, cross-fitted estimators, trained policies in `policies/`, and
`e3_slow_v1_heuristics.json`, the heuristic evaluations reused by the reported E3 run).

| Folder | Content | Source / citation | Licence / terms |
|---|---|---|---|
| `smap/` | AppEEARS area statistics of SMAP SPL3SMP_E v006 (AM/PM soil moisture, AM vegetation water content and surface temperature, quality-flag statistics) over a 722-cell Uttarakhand region, 1 Oct 2022 - 31 Oct 2024 | O'Neill et al., NSIDC DAAC, doi:10.5067/M20OXIZHY3RJ | NASA open data |
| `imd_power_s1/uttarakhand_imd_gridded.csv` | IMD gridded daily rainfall and max/min temperature, 5 districts, 2022-2024 | Pai et al., Mausam 65(1), 2014 | IMD data policy - check before redistribution |
| `imd_power_s1/*_power.csv` | NASA POWER daily meteorology (Almora, Dehradun, Nainital, Pithoragarh), 1 Oct 2022 - 31 Oct 2024. Almora and Nainital are identical (same grid cell); Haridwar is not covered | NASA POWER project (Sparks, JOSS 2018) | NASA open data |
| `imd_power_s1/uttarakhand_sentinel1_vvvh.csv` | Sentinel-1 VV/VH backscatter point extractions, 5 districts; 72 of 810 rows are empty and removed | Copernicus Sentinel-1 (Torres et al., RSE 2012) | Copernicus open licence |
| `era5/` | ERA5-Land, Dehradun box (99 cells, 0.1 deg), 6-hourly, Feb-Dec 2024 without Sep (validation only) | Munoz-Sabater et al., ESSD 2021 | Copernicus licence |
| `mendeley/` | Smart Agriculture and Plant Health Monitoring using IoT (100 readings, 27 Sep 2024) | Kathole et al., Mendeley Data V1, doi:10.17632/65jxyrxv7b.1 | CC BY 4.0 |
| `farmflow/farmflow_available_raw_flows.csv` | 545 unique raw flow records (normal + 8 attack classes) of the Farm-Flow release that were available to the authors | Ferreira et al., Zenodo doi:10.5281/zenodo.10964648; Comput. Electr. Eng. 121, 2025 | CC BY 4.0 |
| `farmflow/Farm-Flow_Test_Binary.csv` | Standardised Farm-Flow test split (kept for completeness; not used in the experiments) | as above | CC BY 4.0 |
| `landsat_mtl/` | Metadata (MTL) of five Landsat 8/9 Collection-2 Level-2 scenes (27-29 Oct 2022); used for raster payload sizes only | USGS, doi:10.5066/P9OGBGM6 | USGS public domain |

## Obtaining the full datasets
* SMAP: NASA AppEEARS (https://appeears.earthdatacloud.nasa.gov), product SPL3SMP_E.006, area request.
* ERA5-Land: Copernicus Climate Data Store, dataset "ERA5-Land hourly data from 1950 to present".
* NASA POWER: https://power.larc.nasa.gov (daily, point, agroclimatology community).
* IMD gridded: India Meteorological Department (e.g., via the `imdlib` Python package).
* Sentinel-1: Google Earth Engine collection `COPERNICUS/S1_GRD` (point means per acquisition).
* Farm-Flow: https://zenodo.org/records/10964648 (about 0.5 GB archive).

## Synthetic data
The synthetic network dataset is not stored; it is regenerated deterministically by
`src/terai/synth.py` (Gaussian copula per Farm-Flow class fitted to the 545 real records,
telemetry inter-arrivals resampled from the Mendeley log). Its fidelity statistics are
written by `experiments/e5_data_synth.py` to `results/raw/e5_data_synth.json`.
