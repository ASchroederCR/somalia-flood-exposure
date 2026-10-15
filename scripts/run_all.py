"""Rebuild everything. Static downloads are skipped once cached; FloodHub is always re-pulled.
    python scripts/run_all.py            # refresh FloodHub forecast + exposure + app data
    python scripts/run_all.py --full     # also re-run every download step
"""
import subprocess, sys, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
steps = ["01_floodhub.py"]
if "--full" in sys.argv or not (ROOT / "data/raw/overture/overture_buildings_somalia_bbox.parquet").exists():
    steps += ["02_download_hdx.py", "03_download_pop_osm.py", "04_osm_roads.py", "05_overture_buildings.py", "05b_unosat_2023.py"]
steps += ["06_exposure.py"]
for s in steps:
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True, cwd=ROOT)
print("Done. Serve the app with:  python -m http.server 8765 --directory app")
