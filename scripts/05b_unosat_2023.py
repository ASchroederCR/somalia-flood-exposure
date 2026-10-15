"""Merge UNOSAT 2023 Deyr (El Nino) flood-extent layers into one cleaned polygon layer.
Source: UNOSAT FL20231105SOM geodatabase (cumulative to 3 Dec 2023), via HDX."""
import pathlib, time
import geopandas as gpd, pandas as pd, pyogrio, shapely

ROOT = pathlib.Path(__file__).resolve().parents[1]
GDB = ROOT / "data/raw/unosat_2023/FL20231105SOM_3765.gdb/FL20231105SOM.gdb"
OUT = ROOT / "data/processed/unosat_2023_flood_extent.gpkg"
# Default ring organisation is extremely slow on the multi-million-vertex layers; make_valid repairs afterwards.
pyogrio.set_gdal_config_options({"OGR_ORGANIZE_POLYGONS": "ONLY_CCW"})

parts = []
for name, _ in pyogrio.list_layers(GDB):
    if "FloodExtent" not in name:
        continue
    t = time.time()
    g = pyogrio.read_dataframe(GDB, layer=name, columns=[]).to_crs(4326)
    geoms = shapely.get_parts(shapely.force_2d(g.geometry.values))  # repair piecewise: whole-layer make_valid stalls
    geoms = shapely.get_parts(shapely.make_valid(shapely.simplify(geoms, 0.00005)))
    geoms = geoms[shapely.get_type_id(geoms) == 3]
    parts.append(gpd.GeoDataFrame(geometry=geoms, crs=4326).assign(layer=name, sensor=name.split("_")[0], date=name.split("_")[1]))
    print(f"{name}: {len(geoms)} polygons ({time.time() - t:.0f}s)", flush=True)
out = pd.concat(parts, ignore_index=True)
out.to_file(OUT, layer="raw", driver="GPKG")
print(len(out), "polygons ->", OUT.name)

