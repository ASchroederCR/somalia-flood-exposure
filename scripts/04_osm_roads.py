"""Extract OSM roads (highway=*) for Somalia from the Geofabrik PBF using GDAL's OSM driver."""
import pathlib, pyogrio, os
ROOT = pathlib.Path(__file__).resolve().parents[1]
PBF = ROOT / "data/raw/osm/somalia-latest.osm.pbf"; OUT = ROOT / "data/processed"
os.environ["OSM_USE_CUSTOM_INDEXING"] = "NO"
KEEP = ("motorway","trunk","primary","secondary","tertiary","motorway_link","trunk_link","primary_link","secondary_link","tertiary_link","unclassified","residential","track","service","road")
sql = ("SELECT osm_id, name, highway, other_tags FROM lines WHERE highway IN (" + ",".join(f"'{k}'" for k in KEEP) + ")")
roads = pyogrio.read_dataframe(PBF, sql=sql, sql_dialect="OGRSQL")
roads = roads.set_crs(4326, allow_override=True)
roads["surface"] = roads.other_tags.fillna("").str.extract(r'"surface"=>"([^"]+)"')[0]
roads = roads.drop(columns="other_tags")
roads.to_file(OUT / "osm_roads.gpkg", driver="GPKG")
print(len(roads)); print(roads.highway.value_counts())
