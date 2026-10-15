"""Download WorldPop 2026 constrained population (100 m) and the OSM Somalia extract (Geofabrik)."""
import pathlib, requests, shutil
ROOT = pathlib.Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw"
FILES = {
 "worldpop": "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2026/SOM/v1/100m/constrained/som_pop_2026_CN_100m_R2025A_v1.tif",
 "osm": "https://download.geofabrik.de/africa/somalia-latest.osm.pbf",
}
for name, url in FILES.items():
    d = RAW / name; d.mkdir(parents=True, exist_ok=True)
    f = d / url.rsplit("/", 1)[-1]
    if not f.exists():
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(f.with_suffix(".part"), "wb") as fh: shutil.copyfileobj(r.raw, fh, 1 << 20)
        f.with_suffix(".part").rename(f)
    print(name, f.name, f.stat().st_size)
