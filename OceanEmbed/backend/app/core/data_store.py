import json
from fastapi import HTTPException


class DataStore:
    """
    Loads data/demo/backend_bundle.json (the real output of
    ml/inference/build_demo_bundle.py) once and serves it from memory.
    In production mode, this class's methods are the seam where you'd swap
    in live NetCDF/Zarr reads + Redis caching (section 36/37) without
    touching the API layer above it.
    """

    def __init__(self, bundle_path: str):
        with open(bundle_path, "r") as f:
            self.bundle = json.load(f)
        self.dataset_label = self.bundle.get("dataset_label", "Demo / Prototype Dataset")

    def region_keys(self):
        return list(self.bundle["regions"].keys())

    def get_region(self, region: str):
        if region not in self.bundle["regions"]:
            raise HTTPException(status_code=404, detail=f"Unknown region '{region}'. Valid: {self.region_keys()}")
        return self.bundle["regions"][region]

    def get_dates(self, region: str):
        return self.get_region(region)["dates"]

    def get_depths(self):
        return self.bundle["depths_m"]

    def nearest_grid_index(self, region: str, lat: float, lon: float):
        r = self.get_region(region)
        lats, lons = r["lats"], r["lons"]
        i = min(range(len(lats)), key=lambda k: abs(lats[k] - lat))
        j = min(range(len(lons)), key=lambda k: abs(lons[k] - lon))
        return i, j
