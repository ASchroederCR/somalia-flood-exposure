# Somalia El Niño Flood Exposure

**Live app:** https://aschroedercr.github.io/somalia-flood-exposure/

An interactive map that ranks Somalia's 91 districts by flood exposure under two scenarios:

- **Live forecast.** Google FloodHub significant river-flood events, flash-flood alerts, and forecasts for ~1,000 river gauges. Each gauge popup shows a forecast discharge chart against its warning, danger and extreme thresholds.
- **2023 El Niño analog.** UNOSAT's satellite-mapped flood extents from the 2023 Deyr floods (1 Nov – 3 Dec 2023), the last El Niño / positive-IOD season.

For each district it counts the people, buildings, roads and IDP sites in the flood zone, and combines them into an exposure index whose weights you can adjust. The app explains how the index is calculated.

## Data sources

| Layer | Source |
|---|---|
| Flood forecast, gauges, thresholds | [Google FloodHub API](https://developers.google.com/flood-forecasting) |
| 2023 flood extent | UNOSAT FL20231105SOM, via HDX |
| Boundaries | OCHA COD-AB Somalia v03 (HDX) |
| Population | WorldPop 2026 constrained, 100 m (R2025A) |
| Buildings | Overture Maps (buildings theme) |
| Roads | OpenStreetMap via Geofabrik |
| Displacement | IOM DTM district IDP figures and Emergency Trend Tracking (HDX) |
| IDP sites | CCCM Cluster IDP Site Master List, Q4 2024 (HDX) |

The forecast counts use FloodHub's event outline, which is much larger than the flooded area FloodHub itself reports, so treat them as upper bounds. The index ranks relative exposure. It does not predict damage or the number of people affected.

## Rebuilding the data

Requires Python 3.11+ with `geopandas`, `pyogrio`, `pyarrow`, `pillow`, `scipy`, `requests` and `openpyxl`.

1. Put your FloodHub API key in a file named `FloodHub API Key.txt` in the repo root. It is git-ignored; never commit it.
2. Run the pipeline:

```bash
python scripts/run_all.py --full
```

`--full` downloads everything (about 1.2 GB, including Overture and OSM). Without it, only the FloodHub forecast is re-pulled and cached downloads are reused.

| Script | Step |
|---|---|
| `01_floodhub.py` | FloodHub events, gauges, statuses, forecast series, thresholds |
| `02_download_hdx.py` | OCHA boundaries, IOM DTM, CCCM sites |
| `03_download_pop_osm.py` | WorldPop raster, OSM extract |
| `04_osm_roads.py` | Road network from the OSM extract |
| `05_overture_buildings.py` | Overture building footprints for Somalia |
| `05b_unosat_2023.py` | Cleaned UNOSAT 2023 flood extent |
| `06_exposure.py` | District exposure metrics and web layers in `app/data/` |

To preview locally, run `python -m http.server 8765 --directory app` and open http://localhost:8765.

Pushing changes under `app/` to `main` redeploys GitHub Pages.
