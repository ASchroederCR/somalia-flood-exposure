"""Compute district-level El Nino flood exposure for Somalia and write the web-app data layers.

Hazards (rasterised onto the WorldPop 100 m grid)
  fh     : Google FloodHub live forecast (significant riverine event + flash-flood + inundation polygons)
  2023   : UNOSAT observed flood extent, 2023 Deyr / El Nino floods (historical analog)
  gauges : FloodHub gauge forecast severity (points)
Exposure
  WorldPop 2026 population, Overture buildings, OSM roads, IOM DTM IDPs, CCCM IDP sites
"""
import json, pathlib
import numpy as np, pandas as pd, geopandas as gpd, shapely, pyarrow.parquet as pq
from shapely import STRtree
from pyproj import Geod
from scipy import ndimage
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, PROC, APP = ROOT / "data/raw", ROOT / "data/processed", ROOT / "app/data"
APP.mkdir(parents=True, exist_ok=True)
geod = Geod(ellps="WGS84")
SEV = {"NO_FLOODING": 0, "ABOVE_NORMAL": 1, "SEVERE": 2, "EXTREME": 3}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- grid: WorldPop 2026 constrained, 100 m
im = Image.open(RAW / "worldpop/som_pop_2026_CN_100m_R2025A_v1.tif")
tp, ps = im.tag_v2[33922], im.tag_v2[33550]
X0, Y0, RES = tp[3], tp[4], ps[0]
pop = np.array(im, dtype=np.float32)
pop[~np.isfinite(pop) | (pop < 0)] = 0
H, W = pop.shape


def rc(x, y):
    """Grid row/col for lon/lat arrays, plus in-grid mask."""
    r, c = ((Y0 - np.asarray(y)) / RES).astype(np.int64), ((np.asarray(x) - X0) / RES).astype(np.int64)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    return np.clip(r, 0, H - 1), np.clip(c, 0, W - 1), ok


def rasterize(geoms, all_touched=True):
    """Burn polygons into a boolean grid (holes respected). Tiny polygons still mark their pixel."""
    img = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(img)
    to_px = lambda a: [((x - X0) / RES, (Y0 - y) / RES) for x, y in a]
    for g in shapely.get_parts(geoms):
        if g.geom_type != "Polygon" or g.is_empty:
            continue
        ext = to_px(g.exterior.coords)
        if len(ext) >= 3:
            dr.polygon(ext, fill=1, outline=1 if all_touched else None)
        for hole in g.interiors:
            h = to_px(hole.coords)
            if len(h) >= 3:
                dr.polygon(h, fill=0)
    return np.array(img, dtype=bool)


# ---------------------------------------------------------------- boundaries
adm2 = gpd.read_file(RAW / "ocha_codab/som_admin2.geojson")[["adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode", "area_sqkm", "geometry"]]
adm1 = gpd.read_file(RAW / "ocha_codab/som_admin1.geojson")[["adm1_name", "adm1_pcode", "geometry"]]
adm2["geometry"] = adm2.geometry.make_valid()
D = adm2.drop(columns="geometry").set_index("adm2_pcode")
N = len(adm2)
NIDX = range(N)


def district_of(x, y):
    """adm2 row index per point (-1 outside Somalia). Points sit in the tree so each district polygon is prepared once."""
    di, pi = STRtree(shapely.points(x, y)).query(adm2.geometry.values, predicate="intersects")
    out = np.full(len(x), -1, dtype=np.int64)
    out[pi] = di
    return out


def by_district(df, col=None):
    s = df.groupby("d").size() if col is None else df.groupby("d")[col].sum()
    return s.reindex(NIDX, fill_value=0).values


# district id grid (for raster zonal sums)
r_, c_ = np.nonzero(pop > 0)
pv = pop[r_, c_].astype(np.float64)
p_d = district_of(X0 + (c_ + 0.5) * RES, Y0 - (r_ + 0.5) * RES)
log("pop pixels", len(pv), "| total", round(pv.sum()), "| in districts", round(pv[p_d >= 0].sum()))

# ---------------------------------------------------------------- hazards
def load(f):
    g = gpd.read_file(PROC / f)
    return g[g.geometry.notna() & ~g.geometry.is_empty] if len(g) else g


sig, flash, inund = load("floodhub_significant.geojson"), load("floodhub_flash.geojson"), load("floodhub_inundation.geojson")
parts = [g[["geometry"]].assign(kind=k) for g, k in ((sig, "significant"), (flash, "flash"), (inund, "inundation")) if len(g)]
# quiet days have no FloodHub polygons at all: keep an empty layer so FloodHub exposure is zero
fh = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"geometry": [], "kind": []}),
                      geometry="geometry", crs=4326)
fh["geometry"] = fh.geometry.make_valid()
h23 = gpd.read_file(PROC / "unosat_2023_flood_extent.gpkg", layer="raw")
# derived zone (scripts/05c_derived_zone.py): severe/extreme gauge sub-basins clipped to the JRC flood hazard map
dvz = load("derived_zone.geojson") if (PROC / "derived_zone.geojson").exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
dv_meta = json.loads((PROC / "derived_zone_meta.json").read_text()) if (PROC / "derived_zone_meta.json").exists() else {}
log("hazard polygons: FloodHub", len(fh), "| UNOSAT 2023", len(h23), "| derived", len(dvz))
M_fh = rasterize(fh.geometry.values)
M_23 = rasterize(h23.geometry.values)
M_dv = rasterize(dvz.geometry.values, all_touched=False)  # already a 90 m cell map, so don't grow it
log("hazard pixels (100 m): FloodHub", int(M_fh.sum()), "| 2023", int(M_23.sum()), "| derived", int(M_dv.sum()))


def zsum(mask=None):
    m = p_d >= 0
    if mask is not None:
        m &= mask[r_, c_]
    return np.bincount(p_d[m], weights=pv[m], minlength=N)


D["pop"], D["pop_fh"], D["pop_2023"], D["pop_dv"] = zsum(), zsum(M_fh), zsum(M_23), zsum(M_dv)

# hazard area per district (km2), from hazard pixel centres
def hazard_area(mask):
    hr, hc = np.nonzero(mask)
    if not len(hr):
        return np.zeros(N)
    lat = Y0 - (hr + 0.5) * RES
    d = district_of(X0 + (hc + 0.5) * RES, lat)
    px_km2 = (RES * 111.32) ** 2 * np.cos(np.radians(lat))
    return np.bincount(d[d >= 0], weights=px_km2[d >= 0], minlength=N)


D["haz_km2_fh"], D["haz_km2_2023"], D["haz_km2_dv"] = hazard_area(M_fh), hazard_area(M_23), hazard_area(M_dv)

# ---------------------------------------------------------------- buildings (Overture)
bt = pq.read_table(RAW / "overture/overture_buildings_somalia_bbox.parquet", columns=["lon", "lat"])
bx, by = bt.column("lon").to_numpy(), bt.column("lat").to_numpy()
b_d = district_of(bx, by)
keep = b_d >= 0
bx, by, b_d = bx[keep], by[keep], b_d[keep]
br, bc, bok = rc(bx, by)
b_fh, b_23, b_dv = M_fh[br, bc] & bok, M_23[br, bc] & bok, M_dv[br, bc] & bok
D["bldg"] = np.bincount(b_d, minlength=N)
D["bldg_fh"], D["bldg_2023"] = np.bincount(b_d[b_fh], minlength=N), np.bincount(b_d[b_23], minlength=N)
D["bldg_dv"] = np.bincount(b_d[b_dv], minlength=N)
log("buildings in Somalia", len(bx), "| in FloodHub zone", int(b_fh.sum()), "| in 2023 extent", int(b_23.sum()))

# ---------------------------------------------------------------- roads (OSM): sample every ~50 m against hazard grids
roads = gpd.read_file(PROC / "osm_roads.gpkg")
roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)
roads["len_km"] = [geod.geometry_length(g) / 1000 for g in roads.geometry.values]
rp = roads.geometry.representative_point()
roads["d"] = district_of(rp.x.values, rp.y.values)
dens = shapely.segmentize(roads.geometry.values, 0.00045)
coords, ridx = shapely.get_coordinates(dens, return_index=True)
cr, ccol, cok = rc(coords[:, 0], coords[:, 1])
npts = np.bincount(ridx, minlength=len(roads))
for key, M in (("km_fh", M_fh), ("km_2023", M_23), ("km_dv", M_dv)):
    hit = np.bincount(ridx, weights=(M[cr, ccol] & cok).astype(float), minlength=len(roads))
    roads[key] = roads.len_km * hit / np.maximum(npts, 1)
MAJOR = ["motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"]
roads["major"] = roads.highway.isin(MAJOR)
rr = roads[roads.d >= 0]
for col, src in (("road_km", "len_km"), ("road_km_fh", "km_fh"), ("road_km_2023", "km_2023"), ("road_km_dv", "km_dv")):
    D[col] = by_district(rr, src)
    D["m" + col] = by_district(rr[rr.major], src)
log("road km", round(rr.len_km.sum()), "| in FloodHub zone", round(rr.km_fh.sum()), "| in 2023 extent", round(rr.km_2023.sum()))

# ---------------------------------------------------------------- IOM DTM displacement (district stock)
dtm = pd.read_csv(RAW / "iom_dtm_api/som-iom-dtm-from-api-admin-0-to-2.csv")
dtm = dtm[dtm.adminLevel == 2]
latest = dtm.reportingDate.max()
dtm = dtm[dtm.reportingDate == latest].copy()
dtm["pc"] = dtm.admin2Pcode.str.replace("^SOM", "SO", regex=True)


def allocate(series):
    """Map DTM district totals onto OCHA districts. DTM reports Banadir as a single unit while the
    OCHA COD splits it into Mogadishu districts, so unmatched codes are split by WorldPop population."""
    out = series.reindex(D.index).fillna(0).astype(float)
    for pc, val in series.items():
        if pc in D.index:
            continue
        kids = D.index[D.index.str.startswith(pc[:4])]
        if len(kids):
            w = D.loc[kids, "pop"]
            out.loc[kids] += val * w / w.sum()
        else:
            log("  unmatched DTM pcode", pc, val)
    return out


D["idp_dtm"] = allocate(dtm.groupby("pc").numPresentIdpInd.sum())
D["idp_dtm_natural"] = allocate(dtm[dtm.displacementReason.str.contains("Natural", na=False)].groupby("pc").numPresentIdpInd.sum())
log("DTM", latest[:10], "IDPs", int(D.idp_dtm.sum()))

# IOM DTM Emergency Trend Tracking (new arrivals since June 2026)
ett = pd.read_excel(RAW / "iom_dtm_ett/iom_dtm_ett_som_tracker_sincejune2026_w16.xlsx", sheet_name=0, skiprows=[1])
ett["n"] = pd.to_numeric(ett["Total new arrivals since last week"], errors="coerce").fillna(0)
hz = ett["Main Cause of Displacement (type of Natural hazard)"].fillna("")
D["ett_arrivals"] = ett.groupby("OCHA District Pcode").n.sum().reindex(D.index).fillna(0)
D["ett_flood_arrivals"] = ett[hz.str.contains("flood", case=False)].groupby("OCHA District Pcode").n.sum().reindex(D.index).fillna(0)
ett_range = (str(ett["Date of Assessment"].min().date()), str(ett["Date of Assessment"].max().date()))
ett_causes = ett.groupby(ett["Main Cause of Displacement"].fillna("Unknown")).n.sum().round().to_dict()

# ---------------------------------------------------------------- CCCM IDP sites (coordinates)
cc = pd.read_excel(RAW / "cccm_sites/idp-site-master-list-4th-quarter-2024.xlsx", sheet_name="CCCM IDP Site List (Verified)", header=1)
cc = cc.loc[:, ~cc.columns.astype(str).str.startswith("Unnamed")]
indcol = [c for c in cc.columns if str(c).startswith("Individual")][-1]
hhcol = [c for c in cc.columns if str(c).startswith("HH")][-1]
cc = cc.rename(columns={"CCCM IDP Site Code": "site_code", "IDP Site": "site", indcol: "ind", hhcol: "hh"})
cc["Latitude"], cc["Longitude"] = pd.to_numeric(cc.Latitude, errors="coerce"), pd.to_numeric(cc.Longitude, errors="coerce")
cc = cc.dropna(subset=["Latitude", "Longitude"])
cc = cc[cc.Latitude.between(-2, 12.5) & cc.Longitude.between(40.5, 52)].reset_index(drop=True)
cc["ind"] = pd.to_numeric(cc.ind, errors="coerce").fillna(0)
cc["hh"] = pd.to_numeric(cc.hh, errors="coerce").fillna(0)
sx, sy = cc.Longitude.values, cc.Latitude.values
cc["d"] = district_of(sx, sy)
sr, sc, sok = rc(sx, sy)
# site coordinates are approximate, so "near" = within ~1 km (10 pixels) of a hazard pixel
ring = ndimage.generate_binary_structure(2, 1)
near_fh = ndimage.binary_dilation(M_fh, ring, iterations=10)
near_23 = ndimage.binary_dilation(M_23, ring, iterations=10)
near_dv = ndimage.binary_dilation(M_dv, ring, iterations=10)
cc["in_fh"], cc["near_fh"] = M_fh[sr, sc] & sok, near_fh[sr, sc] & sok
cc["in_2023"], cc["near_2023"] = M_23[sr, sc] & sok, near_23[sr, sc] & sok
cc["near_dv"] = near_dv[sr, sc] & sok
cs = cc[cc.d >= 0]
D["sites"], D["site_ind"] = by_district(cs), by_district(cs, "ind")
D["sites_fh"], D["site_ind_fh"] = by_district(cs[cs.near_fh]), by_district(cs[cs.near_fh], "ind")
D["sites_2023"], D["site_ind_2023"] = by_district(cs[cs.near_2023]), by_district(cs[cs.near_2023], "ind")
D["sites_dv"], D["site_ind_dv"] = by_district(cs[cs.near_dv]), by_district(cs[cs.near_dv], "ind")
log("CCCM sites", len(cc), "| <=1 km of FloodHub zone:", int(cc.near_fh.sum()), "sites,", int(cc[cc.near_fh].ind.sum()), "people",
    "| <=1 km of 2023 extent:", int(cc.near_2023.sum()), "sites")

# ---------------------------------------------------------------- health facilities (Somalia - Health Facilities, HDX / SoDMA portal)
hx = pd.read_excel(RAW / "health_facilities_hdx/health-facilities.xlsx", skiprows=[1])  # row 1 holds HXL tags
hx.columns = [c.strip() for c in hx.columns]
hx = hx.rename(columns={"Health Facility name": "name", "Type:": "kind", "Latiitude": "lat", "Longititude": "lon"})
for c in ("Region", "District", "name", "kind"):
    hx[c] = hx[c].astype("string").str.strip()
hx["kind"] = hx.kind.fillna("Unknown")


def dms(v):
    """Some coordinates were typed as packed degrees-minutes-seconds (20327.3 = 2°03'27.3")."""
    v = pd.to_numeric(v, errors="coerce")
    deg, rest = np.floor(v / 10000), v % 10000
    return deg + np.floor(rest / 100) / 60 + (rest % 100) / 3600


def in_som(la, lo):
    return la.between(-2, 12.5) & lo.between(40.5, 52)


la, lo = pd.to_numeric(hx.lat, errors="coerce"), pd.to_numeric(hx.lon, errors="coerce")
hx["loc"] = np.where(in_som(la, lo), "as given", None)
la2, lo2 = dms(hx.lat), dms(hx.lon)
fix = hx["loc"].isna() & la.notna() & in_som(la2, lo2)
la, lo = la.where(~fix, la2), lo.where(~fix, lo2)
hx.loc[fix, "loc"] = "decoded DMS"
hx["lat"], hx["lon"] = la, lo
hx["d"] = -1
okm = hx["loc"].notna()
hx.loc[okm, "d"] = district_of(lo[okm].values, la[okm].values)
# keep decoded points only when they land in the stated region
reg = adm2.adm1_name.str.lower().values
bad = (hx["loc"] == "decoded DMS") & ((hx.d < 0) | (pd.Series([reg[i] if i >= 0 else "" for i in hx.d]) != hx.Region.str.lower()))
hx.loc[bad, ["loc", "d"]] = [None, -1]
n_hf_listed, n_hf_decoded = len(hx), int((hx["loc"] == "decoded DMS").sum())
hs = hx[hx["loc"].notna() & (hx.d >= 0)].reset_index(drop=True)
hp = gpd.points_from_xy(hs.lon, hs.lat)
hr_, hc_, hok = rc(hs.lon.values, hs.lat.values)
# coordinates are field-reported and often rounded, so "exposed" = within ~1 km, as for IDP sites
hs["near_fh"], hs["near_2023"], hs["near_dv"] = near_fh[hr_, hc_] & hok, near_23[hr_, hc_] & hok, near_dv[hr_, hc_] & hok
D["hf"], D["hf_fh"], D["hf_2023"] = by_district(hs), by_district(hs[hs.near_fh]), by_district(hs[hs.near_2023])
D["hf_dv"] = by_district(hs[hs.near_dv])
log("health facilities listed", n_hf_listed, "| located", len(hs), f"({n_hf_decoded} from decoded DMS)",
    "| <=1 km of FloodHub zone", int(hs.near_fh.sum()), "| <=1 km of 2023 extent", int(hs.near_2023.sum()))

# ---------------------------------------------------------------- gauges
g = gpd.read_file(PROC / "floodhub_gauges.geojson")
g["sev"] = g.severity.map(SEV).fillna(-1).astype(int)
g["d"] = district_of(g.geometry.x.values, g.geometry.y.values)
gg = g[g.d >= 0]
D["g_total"] = by_district(gg)
D["g_above"], D["g_severe"], D["g_extreme"] = (by_district(gg[gg.sev == s]) for s in (1, 2, 3))
D["g_maxsev"] = gg.groupby("d").sev.max().reindex(NIDX, fill_value=-1).values
D["g_rising"] = by_district(gg[gg.trend == "RISE"])

# ---------------------------------------------------------------- derived
D["pct_pop_fh"] = np.where(D["pop"] > 0, D.pop_fh / D["pop"].clip(lower=1) * 100, 0)
D["pct_pop_2023"] = np.where(D["pop"] > 0, D.pop_2023 / D["pop"].clip(lower=1) * 100, 0)
D["pct_pop_dv"] = np.where(D["pop"] > 0, D.pop_dv / D["pop"].clip(lower=1) * 100, 0)
D["gauge_score"] = np.where(D.g_total > 0, (D.g_above + 2 * D.g_severe + 3 * D.g_extreme) / (3 * D.g_total.clip(lower=1)) * 100, 0)
D = D.reset_index()
num = D.select_dtypes("number").columns
D[num] = D[num].round(2)
D.to_csv(PROC / "district_exposure.csv", index=False)

meta = json.loads((PROC / "floodhub_meta.json").read_text())
summary = dict(
    generated=pd.Timestamp.now("UTC").isoformat(timespec="minutes"), floodhub=meta,
    pop_somalia=round(D["pop"].sum()), pop_fh=round(D.pop_fh.sum()), pop_2023=round(D.pop_2023.sum()),
    haz_km2_fh=round(D.haz_km2_fh.sum()), haz_km2_2023=round(D.haz_km2_2023.sum()),
    bldg_total=int(len(bx)), bldg_fh=int(b_fh.sum()), bldg_2023=int(b_23.sum()),
    road_km=round(rr.len_km.sum()), road_km_fh=round(rr.km_fh.sum()), road_km_2023=round(rr.km_2023.sum()),
    mroad_km_fh=round(rr[rr.major].km_fh.sum()), mroad_km_2023=round(rr[rr.major].km_2023.sum()),
    idp_dtm=round(D.idp_dtm.sum()), dtm_date=latest[:10], ett_range=ett_range, ett_arrivals=round(D.ett_arrivals.sum()),
    ett_causes=ett_causes,
    sites=int(len(cc)), site_ind=round(cc.ind.sum()), sites_fh=int(cc.near_fh.sum()), site_ind_fh=round(cc[cc.near_fh].ind.sum()),
    sites_2023=int(cc.near_2023.sum()), site_ind_2023=round(cc[cc.near_2023].ind.sum()),
    hf_listed=n_hf_listed, hf_total=int(len(hs)), hf_fh=int(hs.near_fh.sum()), hf_2023=int(hs.near_2023.sum()),
    hf_by_kind=hs.kind.value_counts().to_dict(),
    pop_dv=round(D.pop_dv.sum()), haz_km2_dv=round(D.haz_km2_dv.sum()), bldg_dv=int(b_dv.sum()),
    road_km_dv=round(rr.km_dv.sum()), mroad_km_dv=round(rr[rr.major].km_dv.sum()),
    sites_dv=int(cc.near_dv.sum()), site_ind_dv=round(cc[cc.near_dv].ind.sum()), hf_dv=int(hs.near_dv.sum()),
    derived=dv_meta,
    derived_check=json.loads(max(PROC.glob("compare_*.json")).read_text()) if list(PROC.glob("compare_*.json")) else None,
    overture_release=(RAW / "overture/release.txt").read_text().strip(),
    gauges_by_sev=g.severity.value_counts().to_dict(),
    unosat_layers=int(h23.layer.nunique()),
)
log(json.dumps({k: v for k, v in summary.items() if k != "floodhub"}, indent=1, default=str))


# ---------------------------------------------------------------- web vector layers
def write(gdf, name, tol=None, prec=5):
    gdf = gdf.copy()
    if tol:
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)
    gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 10 ** -prec)
    gdf = gdf[~gdf.geometry.is_empty]
    f = APP / f"{name}.geojson"
    f.write_text(gdf.to_json(drop_id=True, na="null"))
    log(f"  {f.name} {f.stat().st_size / 1e6:.2f} MB")


write(adm2[["adm2_pcode", "geometry"]].merge(D, on="adm2_pcode"), "districts", tol=0.003, prec=4)
write(adm1, "regions", tol=0.005, prec=4)
write(fh, "floodhub_zones", tol=0.002, prec=4)
write(dvz.dissolve("severity", as_index=False)[["severity", "rp", "geometry"]] if len(dvz) else dvz, "derived_zones", tol=0.0006, prec=4)
gout = g[["gauge_id", "severity", "trend", "verified", "fc_start", "issued", "geometry"]].copy()
gout[["fc_start", "issued"]] = gout[["fc_start", "issued"]].astype(str)
gout["district"] = [adm2.adm2_name.iat[i] if i >= 0 else None for i in g.d]
write(gout, "gauges")
write(gpd.GeoDataFrame(cc[["site_code", "site", "Region", "District", "ind", "hh", "in_fh", "near_fh", "in_2023", "near_2023", "near_dv"]],
                       geometry=gpd.points_from_xy(sx, sy), crs=4326), "idp_sites")
hout = hs[["name", "kind", "Region", "District", "loc", "near_fh", "near_2023", "near_dv"]].copy()
write(gpd.GeoDataFrame(hout, geometry=hp, crs=4326), "health_sites")
mr = roads[roads.major | (((roads.km_fh > 0) | (roads.km_dv > 0)) & roads.highway.isin(["tertiary", "tertiary_link"]))]
mr = mr[["name", "highway", "surface", "len_km", "km_fh", "km_2023", "km_dv", "geometry"]].copy()
mr[["len_km", "km_fh", "km_2023", "km_dv"]] = mr[["len_km", "km_fh", "km_2023", "km_dv"]].round(2)
write(mr, "roads_major", tol=0.0005, prec=4)
(APP / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
(APP / "gauge_forecasts.json").write_text((PROC / "floodhub_gauge_forecasts.json").read_text())


# ---------------------------------------------------------------- raster overlays (rows resampled to Web Mercator)
def merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def to_png(grid, lat_top, lat_bot, name, stops, vmax, gamma=0.5):
    h = grid.shape[0]
    lats = np.degrees(2 * np.arctan(np.exp(np.linspace(merc_y(lat_top), merc_y(lat_bot), h))) - np.pi / 2)
    gm = grid[np.clip(((lat_top - lats) / (lat_top - lat_bot) * h).astype(int), 0, h - 1)]
    t = np.clip(gm / vmax, 0, 1) ** gamma
    stops = np.array(stops, float)
    pos = np.linspace(0, 1, len(stops))
    rgba = np.stack([np.interp(t, pos, stops[:, k]) for k in range(4)], -1)
    rgba[gm <= 0] = 0
    f = APP / f"{name}.png"
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(f, optimize=True)
    log(f"  {f.name} {f.stat().st_size / 1e6:.2f} MB")


def block(a, f, how="sum"):
    hh, ww = a.shape[0] // f, a.shape[1] // f
    v = a[:hh * f, :ww * f].reshape(hh, f, ww, f)
    return v.sum((1, 3)) if how == "sum" else v.max((1, 3))


rasters = {}
popc = block(pop, 10)  # 1 km
to_png(popc, Y0, Y0 - popc.shape[0] * 10 * RES, "pop_1km",
       [(20, 60, 25, 0), (35, 100, 45, 150), (60, 150, 70, 200), (110, 200, 110, 230), (200, 245, 190, 250)], vmax=5000)
rasters["pop_1km"] = [[Y0 - popc.shape[0] * 10 * RES, X0], [Y0, X0 + popc.shape[1] * 10 * RES]]
bg = np.zeros((H, W), np.float32)
np.add.at(bg, (br[bok], bc[bok]), 1)
bg = block(bg, 5)  # 500 m
to_png(bg, Y0, Y0 - bg.shape[0] * 5 * RES, "bldg_500m",
       [(70, 60, 0, 0), (120, 105, 10, 150), (180, 160, 20, 200), (227, 201, 28, 230), (255, 240, 150, 250)], vmax=400)
rasters["bldg_500m"] = [[Y0 - bg.shape[0] * 5 * RES, X0], [Y0, X0 + bg.shape[1] * 5 * RES]]
m23 = block(M_23.astype(np.float32), 2, "max")  # 200 m
to_png(m23, Y0, Y0 - m23.shape[0] * 2 * RES, "flood_2023_200m", [(42, 80, 214, 0), (42, 80, 214, 215)], vmax=1, gamma=1)
rasters["flood_2023_200m"] = [[Y0 - m23.shape[0] * 2 * RES, X0], [Y0, X0 + m23.shape[1] * 2 * RES]]
(APP / "rasters.json").write_text(json.dumps(rasters))
log("done")
