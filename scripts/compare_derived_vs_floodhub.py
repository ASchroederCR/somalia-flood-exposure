"""Validate the derived flood zone against a real FloodHub significant-event polygon.
Rebuilds the derived zone from the gauge statuses of a past day (default: the 2026-10-16 commit) and compares it
with that day's event polygon: area, overlap, IoU and WorldPop population. Writes data/processed/compare_<date>.json.

    python scripts/compare_derived_vs_floodhub.py [COMMIT] [POLYGON_ID]
"""
import io, json, pathlib, subprocess, sys, urllib.request
import numpy as np, geopandas as gpd, shapely
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
Image.MAX_IMAGE_PIXELS = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import importlib
dz = importlib.import_module("05c_derived_zone")

COMMIT = sys.argv[1] if len(sys.argv) > 1 else "fc700d9"
PID = sys.argv[2] if len(sys.argv) > 2 else "EiQKIDBjNDgwNjgzMTRjYzRiN2FiNmZmNzk2MTY4ZWNjMGNiEAE"
KEY = (ROOT / "FloodHub API Key.txt").read_text().strip().split()[-1]

# past gauge statuses and the FloodHub event polygon of that day
gj = subprocess.run(["git", "show", f"{COMMIT}:app/data/gauges.geojson"], cwd=ROOT, capture_output=True, check=True).stdout
gauges = gpd.read_file(io.BytesIO(gj))[["gauge_id", "severity"]]
past = json.loads(subprocess.run(["git", "show", f"{COMMIT}:app/data/summary.json"], cwd=ROOT, capture_output=True, check=True).stdout)["floodhub"]
day = past["pulled_utc"][:10]
ev = next((e for e in past.get("significant", []) if e.get("poly_id") == PID), {})
with urllib.request.urlopen(f"https://floodforecasting.googleapis.com/v1/serializedPolygons/{PID}?key={KEY}", timeout=120) as r:
    kml = json.loads(r.read())["kml"]
k = gpd.read_file(io.BytesIO(kml.encode()))
fh = k.geometry.make_valid().union_all()

fc = json.loads(subprocess.run(["git", "show", f"{COMMIT}:app/data/gauge_forecasts.json"], cwd=ROOT, capture_output=True, check=True).stdout)
danger = dz.danger_levels(fc)
z, meta = dz.build(gauges, danger)
dv = shapely.union_all(z.geometry.values) if len(z) else shapely.Polygon()
som = gpd.read_file(ROOT / "data/raw/ocha_codab/som_admin2.geojson").geometry.make_valid().union_all()
fh_s, dv_s = fh.intersection(som), dv.intersection(som)

eq = lambda g: float(gpd.GeoSeries([g], crs=4326).to_crs(6933).area.iloc[0] / 1e6)
inter, union = fh_s.intersection(dv_s), fh_s.union(dv_s)

# WorldPop people inside each footprint
im = Image.open(ROOT / "data/raw/worldpop/som_pop_2026_CN_100m_R2025A_v1.tif")
tp, ps = im.tag_v2[33922], im.tag_v2[33550]
X0, Y0, RES = tp[3], tp[4], ps[0]
pop = np.array(im, dtype=np.float64)
pop[~np.isfinite(pop) | (pop < 0)] = 0


def people(g):
    """WorldPop sum over every cell a footprint touches (rasterio's all_touched): Pillow burns the cells whose
    centres are inside, then the one-cell ring around them is tested exactly with shapely. The derived zone is
    made of 90 m strips, so centre sampling alone on the 100 m grid misses ~8% of its people."""
    if g.is_empty:
        return 0
    x0, y0, x1, y1 = g.bounds
    c0, r0 = max(int((x0 - X0) / RES) - 1, 0), max(int((Y0 - y1) / RES) - 1, 0)
    c1, r1 = min(int((x1 - X0) / RES) + 2, pop.shape[1]), min(int((Y0 - y0) / RES) + 2, pop.shape[0])
    if c0 >= c1 or r0 >= r1:
        return 0
    img = Image.new("1", (c1 - c0, r1 - r0), 0)
    dr = ImageDraw.Draw(img)
    px = lambda cs: [((x - X0) / RES - c0 - 0.5, (Y0 - y) / RES - r0 - 0.5) for x, y in cs]   # pixel centres
    for p in shapely.get_parts(g):
        if p.geom_type != "Polygon":
            continue
        dr.polygon(px(p.exterior.coords), fill=1, outline=1)
        for h in p.interiors:
            dr.polygon(px(h.coords), fill=0)
    m = np.array(img, dtype=bool)
    rr, cc = np.nonzero(binary_dilation(m, np.ones((3, 3), bool)) & ~m)
    shapely.prepare(g)
    hit = shapely.intersects(g, shapely.box(X0 + (c0 + cc) * RES, Y0 - (r0 + rr + 1) * RES, X0 + (c0 + cc + 1) * RES, Y0 - (r0 + rr) * RES))
    m[rr[hit], cc[hit]] = True
    return round(float(pop[r0:r1, c0:c1][m].sum()))


res = dict(day=day, commit=COMMIT, derived=meta, fh_area_km2=ev.get("area_km2"), fh_affected_pop=ev.get("fh_affected_pop"),
           km2_floodhub=round(eq(fh_s)), km2_derived=round(eq(dv_s)), km2_overlap=round(eq(inter)),
           iou=round(eq(inter) / max(eq(union), 1e-9), 3),
           share_derived_inside_floodhub=round(eq(inter) / max(eq(dv_s), 1e-9), 3),
           share_floodhub_inside_derived=round(eq(inter) / max(eq(fh_s), 1e-9), 3),
           pop_floodhub=people(fh_s), pop_derived=people(dv_s), pop_overlap=people(inter))
# how much of the FloodHub footprint the derived zone could ever reach (its sub-basins, before the JRC clip)
bas = dz.load_basins()
zb = dz.zone_basins(gauges, bas, dz.skipped(gauges, bas, danger))
cand = shapely.union_all(bas.loc[list(zb)].geometry.values).intersection(som)
res["share_floodhub_inside_candidate_basins"] = round(eq(fh_s.intersection(cand)) / max(eq(fh_s), 1e-9), 3)
out = ROOT / f"data/processed/compare_{day}.json"
out.write_text(json.dumps(res, indent=1))
z.to_file(ROOT / f"data/processed/derived_zone_{day}.geojson", driver="GeoJSON")
gpd.GeoDataFrame(geometry=[fh_s], crs=4326).to_file(ROOT / f"data/processed/floodhub_event_{day}.geojson", driver="GeoJSON")
print(json.dumps(res, indent=1))
