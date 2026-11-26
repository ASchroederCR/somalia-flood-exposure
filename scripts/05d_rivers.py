"""River network for the map, from HydroRIVERS v1.0 (Africa).

HydroRIVERS is derived from the same HydroSHEDS data as the HydroBASINS sub-basins that FloodHub gauge IDs refer to,
so gauges sit on these lines. Keeps reaches draining at least MIN_UPLAND km2, simplifies them for the web and
writes app/data/rivers.geojson with:
    dis  long-term average discharge, m3/s (HydroRIVERS DIS_AV_CMS, modelled)
    ord  river order by discharge (ORD_FLOW: 1 = largest, e.g. >100,000 m3/s ... 9 = smallest)
    up   upstream area, km2
Static: run_all.py only runs it when the output is missing or with --full. Downloads ~108 MB on first run.
"""
import pathlib, sys
import geopandas as gpd, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/raw/hydrorivers/HydroRIVERS_v10_af_shp.zip"
URL = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip"
OUT = ROOT / "app/data/rivers.geojson"
BBOX = (40.0, -2.5, 52.0, 12.5)      # Somalia plus the Ethiopian and Kenyan reaches that flow into it
MIN_UPLAND = 2000                    # km2
TOL = 0.004                          # degrees (~400 m) simplification


def main():
    if not ZIP.exists():
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL, timeout=900); r.raise_for_status(); ZIP.write_bytes(r.content)
        print("downloaded", ZIP.name, flush=True)
    r = gpd.read_file(f"zip://{ZIP}!HydroRIVERS_v10_af_shp/HydroRIVERS_v10_af.shp", bbox=BBOX,
                      columns=["UPLAND_SKM", "DIS_AV_CMS", "ORD_FLOW"])
    r = r[r.UPLAND_SKM >= MIN_UPLAND].copy()
    r = r.clip(BBOX)
    r["geometry"] = r.geometry.simplify(TOL, preserve_topology=False)
    r = r[~r.geometry.is_empty]
    out = gpd.GeoDataFrame({"dis": r.DIS_AV_CMS.round(1), "ord": r.ORD_FLOW.astype(int),
                            "up": r.UPLAND_SKM.round(0).astype(int)}, geometry=r.geometry, crs=4326)
    out = out.sort_values("dis")      # largest rivers drawn last, on top
    out.to_file(OUT, driver="GeoJSON", COORDINATE_PRECISION=4)
    print(f"{len(out)} reaches -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(out.ord.value_counts().sort_index().to_string())


if __name__ == "__main__":
    sys.exit(main())
