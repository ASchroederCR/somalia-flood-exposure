"""Pull Overture Maps building footprints for Somalia (latest release) as centroid + area points.
Uses parquet row-group bbox statistics to read only the relevant row groups from S3."""
import pathlib, sys, concurrent.futures as cf
import pyarrow as pa, pyarrow.fs as fs, pyarrow.parquet as pq, pyarrow.compute as pc
ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "data/raw/overture"; OUT.mkdir(parents=True, exist_ok=True)
XMIN, YMIN, XMAX, YMAX = 40.95, -1.75, 51.45, 12.05   # Somalia
s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
rel = sorted(i.path for i in s3.get_file_info(fs.FileSelector("overturemaps-us-west-2/release")))[-1]
files = [f.path for f in s3.get_file_info(fs.FileSelector(rel + "/theme=buildings/type=building")) if f.path.endswith(".parquet")]
print("release", rel, len(files), "files", flush=True)

def rg_stats(md, rg, name):
    r = md.row_group(rg)
    for c in range(r.num_columns):
        col = r.column(c)
        if col.path_in_schema == name and col.statistics is not None and col.statistics.has_min_max:
            return col.statistics.min, col.statistics.max
    return None

def scan(path):
    pf = pq.ParquetFile(path, filesystem=s3)
    md = pf.metadata
    keep = []
    for rg in range(md.num_row_groups):
        xs, ys = rg_stats(md, rg, "bbox.xmin"), rg_stats(md, rg, "bbox.ymin")
        xe, ye = rg_stats(md, rg, "bbox.xmax"), rg_stats(md, rg, "bbox.ymax")
        if None in (xs, ys, xe, ye): keep.append(rg); continue
        if xs[0] <= XMAX and xe[1] >= XMIN and ys[0] <= YMAX and ye[1] >= YMIN: keep.append(rg)
    if not keep: return None
    t = pf.read_row_groups(keep, columns=["id", "bbox", "height", "num_floors"])
    b = t.column("bbox").combine_chunks()
    x0, x1, y0, y1 = (pc.struct_field(b, k) for k in ("xmin", "xmax", "ymin", "ymax"))
    m = pc.and_(pc.and_(pc.greater_equal(x0, XMIN), pc.less_equal(x1, XMAX)), pc.and_(pc.greater_equal(y0, YMIN), pc.less_equal(y1, YMAX)))
    t = t.filter(m)
    if t.num_rows == 0: return None
    b = t.column("bbox").combine_chunks(); x0, x1, y0, y1 = (pc.struct_field(b, k) for k in ("xmin", "xmax", "ymin", "ymax"))
    return pa.table({"id": t.column("id"),
                     "lon": pc.divide(pc.add(x0, x1), 2), "lat": pc.divide(pc.add(y0, y1), 2),
                     "w_deg": pc.subtract(x1, x0), "h_deg": pc.subtract(y1, y0),
                     "height": t.column("height"), "num_floors": t.column("num_floors")})

parts, done = [], 0
with cf.ThreadPoolExecutor(16) as ex:
    for res in ex.map(scan, files):
        done += 1
        if res is not None: parts.append(res); print(f"{done}/{len(files)} +{res.num_rows}", flush=True)
        elif done % 50 == 0: print(f"{done}/{len(files)}", flush=True)
tab = pa.concat_tables(parts)
pq.write_table(tab, OUT / "overture_buildings_somalia_bbox.parquet")
(OUT / "release.txt").write_text(rel.rsplit("/", 1)[-1])
print("buildings in bbox:", tab.num_rows)
