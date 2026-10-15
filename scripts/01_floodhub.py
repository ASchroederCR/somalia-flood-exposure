"""Pull live Google FloodHub forecasts for Somalia (adapted from the QGIS sample, no QGIS needed).
Outputs data/processed/floodhub_*.geojson + floodhub_meta.json"""
import json, urllib.request, urllib.error, pathlib, io, time
import geopandas as gpd, pandas as pd
from shapely.geometry import Point
ROOT = pathlib.Path(__file__).resolve().parents[1]
KEY = (ROOT / "FloodHub API Key.txt").read_text().strip().split()[-1]
BASE = "https://floodforecasting.googleapis.com/v1"
OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)
CC = "SO"

def _req(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 3: time.sleep(2 ** attempt); continue
            raise RuntimeError(f"{e.code} {e.read().decode()[:300]}") from None

def post_all(endpoint, payload, key):
    """POST with pagination; returns concatenated list under `key`."""
    items, token = [], None
    while True:
        p = dict(payload, **({"pageToken": token} if token else {}))
        r = _req(f"{BASE}/{endpoint}?key={KEY}", p)
        items += r.get(key, [])
        token = r.get("nextPageToken")
        if not token: return items

_poly_cache = {}
def polygon(pid):
    if not pid: return None
    if pid in _poly_cache: return _poly_cache[pid]
    d = _req(f"{BASE}/serializedPolygons/{pid.split('/')[-1]}?key={KEY}")
    kml = d.get("kml")
    geom = None
    if kml:
        g = gpd.read_file(io.BytesIO(kml.encode()))
        g = g[g.geometry.notna()]
        if len(g): geom = g.geometry.make_valid().union_all()
    _poly_cache[pid] = geom
    return geom

# 1. Flash floods
ff = post_all("flashFloods:search", {"countryCodes": [CC]}, "flashFloodEvents")
rows = []
for ev in ff:
    for t in ("eventPolygonId", "likelyAffectedPolygonId", "highlyLikelyAffectedPolygonId"):
        g = polygon(ev.get(t))
        if g is not None:
            rows.append(dict(source="flashFloods", poly_type=t, issue_time=ev.get("forecastIssueTime", ""),
                             valid_hrs=ev.get("forecastPeriodHours", 0), poly_id=ev.get(t), geometry=g))
flash = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(columns=["source","poly_type","geometry"], geometry="geometry", crs=4326)
flash.to_file(OUT / "floodhub_flash.geojson", driver="GeoJSON")

# 2. Significant (riverine/basin) events touching Somalia
sig = [e for e in post_all("significantEvents:search", {}, "significantEvents") if CC in e.get("affectedCountryCodes", [])]
rows = []
for ev in sig:
    g = polygon(ev.get("eventPolygonId"))
    if g is not None:
        iv = ev.get("eventInterval", {})
        rows.append(dict(source="significantEvents", start=iv.get("startTime"), end=iv.get("endTime"),
                         fh_affected_pop=ev.get("affectedPopulation"), area_km2=ev.get("areaKm2"),
                         n_gauges=len(ev.get("gaugeIds", [])), countries=",".join(ev.get("affectedCountryCodes", [])),
                         poly_id=ev.get("eventPolygonId"), geometry=g))
sigdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(columns=["source","geometry"], geometry="geometry", crs=4326)
sigdf.to_file(OUT / "floodhub_significant.geojson", driver="GeoJSON")

# 3. Gauges + latest status (+ inundation maps where a gauge is flooding)
gauges = {g["gaugeId"]: g for g in post_all("gauges:searchGaugesByArea", {"regionCode": CC, "includeNonQualityVerified": True}, "gauges")}
stat = {s["gaugeId"]: s for s in post_all("floodStatus:searchLatestFloodStatusByArea", {"regionCode": CC, "includeNonQualityVerified": True}, "floodStatuses")}
grow, inund = [], []
for gid, g in gauges.items():
    s = stat.get(gid, {})
    loc = g.get("location") or s.get("gaugeLocation") or {}
    grow.append(dict(gauge_id=gid, site=g.get("siteName", ""), river=g.get("river", ""), source=g.get("source"),
                     verified=g.get("qualityVerified"), severity=s.get("severity", "UNKNOWN"),
                     trend=s.get("forecastTrend", ""), issued=s.get("issuedTime", ""),
                     fc_start=s.get("forecastTimeRange", {}).get("start", ""),
                     geometry=Point(loc.get("longitude"), loc.get("latitude"))))
    for ms in s.get("inundationMapSet", {}).get("inundationMaps", []) or []:
        geo = polygon(ms.get("serializedPolygonId"))
        if geo is not None:
            inund.append(dict(gauge_id=gid, level=ms.get("level"), severity=s.get("severity"), geometry=geo))
gdf = gpd.GeoDataFrame(grow, geometry="geometry", crs=4326)
gdf.to_file(OUT / "floodhub_gauges.geojson", driver="GeoJSON")
ind = gpd.GeoDataFrame(inund, geometry="geometry", crs=4326) if inund else gpd.GeoDataFrame(columns=["gauge_id","level","geometry"], geometry="geometry", crs=4326)
ind.to_file(OUT / "floodhub_inundation.geojson", driver="GeoJSON")

# 4. Gauge hydrographs: latest forecast series + model thresholds (for the app's popup charts)
def batches(seq, n=50):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

ids = sorted(gauges)
since = (pd.Timestamp.now("UTC") - pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
models, series = {}, {}
for b in batches(ids):
    r = _req(f"{BASE}/gaugeModels:batchGet?" + "&".join(f"names=gaugeModels/{i}" for i in b) + f"&key={KEY}")
    for m in r.get("gaugeModels", []):
        models[m["gaugeId"]] = m
    r = _req(f"{BASE}/gauges:queryGaugeForecasts?" + "&".join(f"gaugeIds={i}" for i in b) + f"&issuedTimeStart={since}&key={KEY}")
    for gid, fc in (r.get("forecasts") or {}).items():
        latest = max(fc.get("forecasts", []), key=lambda f: f["issuedTime"], default=None)
        if latest and latest.get("forecastRanges"):
            series[gid] = latest
fc_out = {}
for gid in ids:
    s, m = series.get(gid), models.get(gid, {})
    if not s:
        continue
    th = m.get("thresholds", {})
    rng = sorted(s["forecastRanges"], key=lambda x: x["forecastStartTime"])
    fc_out[gid] = dict(issued=s["issuedTime"][:16] + "Z", start=rng[0]["forecastStartTime"][:10],
                       unit=m.get("gaugeValueUnit", ""), th=[round(th[k], 2) if k in th else None for k in ("warningLevel", "dangerLevel", "extremeDangerLevel")],
                       v=[round(x["value"], 2) for x in rng])
(OUT / "floodhub_gauge_forecasts.json").write_text(json.dumps(fc_out, separators=(",", ":")))
print(f"gauge forecast series: {len(fc_out)} of {len(ids)} gauges")

meta = dict(pulled_utc=pd.Timestamp.now("UTC").isoformat(), n_flash_events=len(ff), n_flash_polys=len(flash),
            n_significant=len(sigdf), n_gauges=len(gdf), n_inundation_maps=len(ind),
            severity_counts=gdf.severity.value_counts().to_dict(),
            significant=[{k: v for k, v in r.items() if k != "geometry"} for r in rows])
(OUT / "floodhub_meta.json").write_text(json.dumps(meta, indent=1, default=str))
print(json.dumps(meta, indent=1, default=str))
