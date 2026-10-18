"""Download OCHA COD-AB boundaries and IOM DTM displacement data from HDX."""
import pathlib, requests, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw"
FILES = {
 "ocha_codab": "https://data.humdata.org/dataset/ec140a63-5330-4376-a3df-c7ebf73cfc3c/resource/79f7f826-6028-4650-b8f7-2d5e53032955/download/som_admin_boundaries.geojson.zip",
 "iom_dtm_api": "https://data.humdata.org/dataset/b21eb34d-6ca1-47f5-a4ca-cc83a351369c/resource/97729555-bbaf-4d7c-ac44-e698ebc7b6b1/download/som-iom-dtm-from-api-admin-0-to-2.csv",
 "iom_dtm_baseline": "https://data.humdata.org/dataset/b421e1e5-a8cc-48c1-8e85-dc2d01706584/resource/343967e0-533d-4af2-aefe-85ff949061a9/download/somalia-baseline-assessment-dataset_-round-3-february-september-2024_hdx.xlsx",
 "iom_dtm_ett": "https://data.humdata.org/dataset/7f9b9d5c-f629-441a-80d2-109ec30efca4/resource/9733c8a7-2b49-4edd-8eb8-3949515ebfca/download/iom_dtm_ett_som_tracker_sincejune2026_w16.xlsx",
 "cccm_sites": "https://data.humdata.org/dataset/5c334c3d-3b40-4819-ae8d-450f7e5fc104/resource/1519fb91-f1a8-4723-bdaa-0d6f5f768e1d/download/idp-site-master-list-4th-quarter-2024.xlsx",
 # Somalia - Health Facilities (HDX, mirrored on the SoDMA data portal)
 "health_facilities_hdx": "https://data.humdata.org/dataset/9fd710d4-ab87-43c9-b6da-39847af345b1/resource/3bc1cf5a-a46d-47a5-8f7f-93789f5db5b9/download/health-facilities.xlsx",
}
for name, url in FILES.items():
    d = RAW / name; d.mkdir(parents=True, exist_ok=True)
    f = d / url.rsplit("/", 1)[-1]
    if not f.exists():
        r = requests.get(url, timeout=300); r.raise_for_status(); f.write_bytes(r.content)
    if f.suffix == ".zip":
        zipfile.ZipFile(f).extractall(d)
    print(name, f.name, f.stat().st_size)
