"""Derived flood zone, for days when FloodHub publishes no significant event polygon.

FloodHub gauge IDs are HydroBASINS level-12 sub-basin IDs (hybas_<HYBAS_ID>). For every gauge forecast
SEVERE or EXTREME we take its sub-basin plus the sub-basins downstream of it (following NEXT_DOWN until
the next gauged sub-basin, which has its own forecast), then keep only the cells that the JRC / GloFAS
global river flood hazard map floods at a return period matching the severity:
    SEVERE  (FloodHub danger level, ~5-year flow)          -> JRC RP10 (lowest JRC return period)
    EXTREME (FloodHub extreme danger level, ~20-year flow) -> JRC RP20
JRC "spurious depth" areas are removed. Gauges on small channels that drain inland (endorheic sub-basin and a
danger-level flow under MIN_ENDO_DANGER m3/s) are skipped: there the 90 m JRC map spreads shallow water across
flat ground, e.g. a ~26 km2 sheet over Gaalkacyo town from a 9 m3/s channel. That one sub-basin was 63% of the
derived zone's population when checked against the 2026-10-16 FloodHub event.
Output: data/processed/derived_zone.geojson (+ derived_zone_meta.json).

    python scripts/05c_derived_zone.py                      # from today's gauge statuses
    python scripts/05c_derived_zone.py GAUGES.geojson OUT   # any gauge file with gauge_id + severity
                                                            # (danger levels come from floodhub_gauge_forecasts.json)
"""
import functools, json, pathlib, sys, zlib
import numpy as np, geopandas as gpd, pandas as pd, shapely
from PIL import Image
# No rasterio: Windows application control on the pipeline machine blocks its DLLs. The JRC tiles are read with
# the small GeoTiff reader below, the sub-basin mask uses shapely.contains_xy and the wet cells are vectorized
# by unioning per-row runs of cells.
Image.MAX_IMAGE_PIXELS = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
JRC = RAW / "jrc_flood_hazard"
TILES = ["ID160_N20_E40", "ID161_N10_E40", "ID162_N0_E40", "ID170_N20_E50", "ID171_N10_E50"]
RP = {2: 10, 3: 20}           # gauge severity -> JRC return period
MAX_DOWN = 40                 # safety cap on the downstream walk
MIN_ENDO_DANGER = 20          # m3/s; endorheic gauges with a smaller danger-level flow are skipped
BBOX = (38.0, -5.0, 54.0, 14.0)


def log(*a):
    print(*a, flush=True)


def fetch():
    """Cache the static inputs on first run (~190 MB)."""
    import requests
    jrc = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
    want = {RAW / "hydrobasins/hybas_af_lev12_v1c.zip": "https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_af_lev12_v1c.zip"}
    for t in TILES:
        for rp in RP.values():
            want[JRC / f"{t}_RP{rp}_depth.tif"] = f"{jrc}/RP{rp}/{t}_RP{rp}_depth.tif"
        want[JRC / f"{t}_spurious_depth_areas.tif"] = f"{jrc}/Spurious_Depths/{t}_spurious_depth_areas.tif"
    for f, url in want.items():
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, timeout=600); r.raise_for_status(); f.write_bytes(r.content)
            log("downloaded", f.name)


def load_basins():
    f = RAW / "hydrobasins/hybas_af_lev12_v1c.zip"
    b = gpd.read_file(f"zip://{f}!hybas_af_lev12_v1c.shp", bbox=BBOX, columns=["HYBAS_ID", "NEXT_DOWN", "SUB_AREA", "ENDO"])
    return b.set_index("HYBAS_ID")


def danger_levels(fc):
    """{gauge_id: danger-level flow} from a gauge_forecasts.json dict (th = warning, danger, extreme)."""
    return {k: v["th"][1] for k, v in fc.items() if v.get("th") and v["th"][1] is not None}


def skipped(gauges, basins, danger):
    """Flagged gauges on small inland-draining channels, where the JRC map is unreliable."""
    endo = basins.ENDO.to_dict()
    return {g for g, s in zip(gauges.gauge_id, gauges.severity) if s in ("SEVERE", "EXTREME")
            and endo.get(int(g[6:]), 0) >= 1 and danger.get(g, np.inf) < MIN_ENDO_DANGER}


def zone_basins(gauges, basins, skip=()):
    """{HYBAS_ID: severity} for flagged gauge sub-basins and their downstream reaches (max severity wins)."""
    gid = gauges.gauge_id.str.replace("hybas_", "", regex=False).astype("int64")
    sev = gauges.severity.map({"SEVERE": 2, "EXTREME": 3}).fillna(0).astype(int).values
    gauged = set(gid)
    nxt = basins.NEXT_DOWN.to_dict()
    out = {}
    for g, h, s in zip(gauges.gauge_id, gid, sev):
        if s < 2 or h not in nxt or g in skip:
            continue
        cur, steps = h, 0
        while cur in nxt and steps <= MAX_DOWN:
            out[cur] = max(out.get(cur, 0), s)
            cur, steps = nxt[cur], steps + 1
            if cur == 0 or cur in gauged:
                break
    return out


class GeoTiff:
    """Window reader for the JRC tiles: tiled, deflate, no predictor, GDAL-sparse (empty tiles have byte count 0,
    which Pillow's libtiff refuses to decode). Pillow parses the tags; each tile is inflated with zlib."""

    def __init__(self, path):
        im = Image.open(path)
        t = im.tag_v2
        if t.get(259) != 8 or t.get(317, 1) != 1 or t.get(277, 1) != 1 or 322 not in t:
            raise ValueError(f"{path.name}: expected a single-band tiled deflate GeoTIFF without predictor")
        endian = "<" if open(path, "rb").read(2) == b"II" else ">"
        self.dtype = np.dtype(endian + {(3, 32): "f4", (1, 8): "u1", (1, 16): "u2", (2, 16): "i2"}[(t.get(339, (1,))[0], t[258][0])])
        self.W, self.H = im.size
        self.tw, self.th = t[322], t[323]
        self.ntx = -(-self.W // self.tw)
        self.off, self.cnt = t[324], t[325]
        tp, ps = t[33922], t[33550]
        self.x0, self.y0, self.res = tp[3], tp[4], ps[0]
        self.nodata = self.dtype.type(float(t[42113])) if 42113 in t else None
        self.f = open(path, "rb")
        self.tile = functools.lru_cache(maxsize=256)(self._tile)
        im.close()

    def _tile(self, i):
        if not self.cnt[i]:                       # sparse: never written, reads as nodata
            return np.full((self.th, self.tw), self.nodata or 0, self.dtype)
        self.f.seek(self.off[i])
        return np.frombuffer(zlib.decompress(self.f.read(self.cnt[i])), self.dtype).reshape(self.th, self.tw)

    def window(self, x0, y0, x1, y1):
        """(row, col, height, width) of the cells whose centres fall in the bounds, clipped to the raster."""
        c0, c1 = max(int(np.floor((x0 - self.x0) / self.res + 0.5)), 0), min(int(np.ceil((x1 - self.x0) / self.res - 0.5)), self.W)
        r0, r1 = max(int(np.floor((self.y0 - y1) / self.res + 0.5)), 0), min(int(np.ceil((self.y0 - y0) / self.res - 0.5)), self.H)
        return r0, c0, r1 - r0, c1 - c0

    def read(self, r0, c0, h, w):
        out = np.empty((h, w), self.dtype)
        for tr in range(r0 // self.th, (r0 + h - 1) // self.th + 1):
            for tc in range(c0 // self.tw, (c0 + w - 1) // self.tw + 1):
                a = self.tile(tr * self.ntx + tc)
                ya, yb = max(r0, tr * self.th), min(r0 + h, (tr + 1) * self.th)
                xa, xb = max(c0, tc * self.tw), min(c0 + w, (tc + 1) * self.tw)
                out[ya - r0:yb - r0, xa - c0:xb - c0] = a[ya - tr * self.th:yb - tr * self.th, xa - tc * self.tw:xb - tc * self.tw]
        return out

    def close(self):
        self.f.close()


def cells_to_polygons(mask, x0, y0, res):
    """Union of the cells set in a boolean grid (row 0 at top, cell edges at x0 + c*res, y0 - r*res)."""
    boxes = []
    for r in np.flatnonzero(mask.any(axis=1)):
        d = np.diff(np.concatenate(([0], mask[r].view(np.int8), [0])))
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        boxes.append(shapely.box(x0 + s * res, y0 - (r + 1) * res, x0 + e * res, y0 - r * res))
    return shapely.union_all(np.concatenate(boxes)) if boxes else None


def flooded(geom, rp, srcs):
    """Polygons of JRC-flooded cells (depth > 0, not spurious) whose centres lie inside one sub-basin."""
    parts = []
    for t, (dep, spu) in srcs[rp].items():
        r0, c0, h, w = dep.window(*geom.bounds)
        if h < 1 or w < 1:
            continue
        d = dep.read(r0, c0, h, w)
        s = spu.read(r0, c0, h, w)
        wet = np.isfinite(d) & (d > 0) & ((d != dep.nodata) if dep.nodata is not None else True) & (s != 1)  # spurious = 1, else nodata
        if not wet.any():
            continue
        x0, y0 = dep.x0 + c0 * dep.res, dep.y0 - r0 * dep.res
        rr, cc = np.nonzero(wet)
        wet[rr, cc] = shapely.contains_xy(geom, x0 + (cc + 0.5) * dep.res, y0 - (rr + 0.5) * dep.res)
        g = cells_to_polygons(wet, x0, y0, dep.res)
        if g is not None:
            parts.append(g)
    return shapely.union_all(parts) if parts else None


def build(gauges, danger, basins=None):
    basins = load_basins() if basins is None else basins
    skip = skipped(gauges, basins, danger)
    zb = zone_basins(gauges, basins, skip)
    srcs = {rp: {t: (GeoTiff(JRC / f"{t}_RP{rp}_depth.tif"), GeoTiff(JRC / f"{t}_spurious_depth_areas.tif")) for t in TILES}
            for rp in RP.values()} if zb else {}
    rows = []
    for h, s in zb.items():
        g = flooded(basins.geometry[h], RP[s], srcs)
        if g is not None and not g.is_empty:
            rows.append(dict(hybas_id=int(h), severity="EXTREME" if s == 3 else "SEVERE", rp=RP[s], geometry=g))
    for d in srcs.values():
        for a, b in d.values():
            a.close(); b.close()
    z = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else \
        gpd.GeoDataFrame(columns=["hybas_id", "severity", "rp", "geometry"], geometry="geometry", crs=4326)
    n_flag = int(gauges.severity.isin(["SEVERE", "EXTREME"]).sum())
    meta = dict(n_flagged_gauges=n_flag, n_skipped_endorheic=len(skip), min_endo_danger_m3s=MIN_ENDO_DANGER,
                n_zone_basins=len(zb), n_wet_basins=len(z),
                basin_km2=round(float(basins.loc[list(zb), "SUB_AREA"].sum())) if zb else 0,
                rp_by_severity={"SEVERE": RP[2], "EXTREME": RP[3]})
    return z, meta


if __name__ == "__main__":
    gpath = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else PROC / "floodhub_gauges.geojson"
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else PROC / "derived_zone.geojson"
    fetch()
    danger = danger_levels(json.loads((PROC / "floodhub_gauge_forecasts.json").read_text()))
    z, meta = build(gpd.read_file(gpath)[["gauge_id", "severity"]], danger)
    sig = gpd.read_file(PROC / "floodhub_significant.geojson")
    meta["fallback_active"] = len(sig) == 0     # FloodHub gave no significant event polygon
    meta["source_gauges"] = gpath.name
    z.to_file(out, driver="GeoJSON")
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps(meta))
