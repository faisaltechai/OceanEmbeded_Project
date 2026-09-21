"""
Global region catalogue -- expands the original Bay of Bengal / Arabian Sea
scope to the full ocean list from the brief, while keeping the existing
`region` query-param contract every panel already calls with.

Only Bay of Bengal and Arabian Sea have a precomputed demo bundle (they're
the actual SIH26066 problem-statement scope). Every other region here is
real bbox/metadata plus wired-up live clients (satellite fetch works for
any lat/lon globally); the demo-fallback grid just won't have a synthetic
map for them until someone runs `build_demo_bundle.py` with a wider config
-- the gateway will label those responses status="error" with a clear
message rather than inventing a fake grid, per RULE #2.
"""
from __future__ import annotations

REGIONS: dict[str, dict] = {
    "bay_of_bengal": {"name": "Bay of Bengal", "bbox": [5, 78, 23, 95], "has_demo_bundle": True},
    "arabian_sea": {"name": "Arabian Sea", "bbox": [0, 45, 25, 78], "has_demo_bundle": True},
    "indian_ocean": {"name": "Indian Ocean", "bbox": [-60, 20, 30, 120], "has_demo_bundle": False},
    # Pacific spans the antimeridian AND has a very different eastern coastline
    # by latitude (California ~-125 in the north, Chile ~-70 in the south) --
    # a single rectangle either clips the North Pacific or swallows the North
    # American interior (caught by tests/test_regions.py: a naive [-70] east
    # bound incorrectly matched Kansas). Modeled as two latitude bands instead.
    "pacific_ocean": {
        "name": "Pacific Ocean",
        "bbox": [-60, 120, 60, -70],  # kept for display/back-compat only
        "sub_bboxes": [
            [0, 120, 60, -125],    # North Pacific: east edge ~California/Baja
            [-60, 120, 0, -70],    # South Pacific: east edge ~Chile
        ],
        "has_demo_bundle": False,
    },
    "atlantic_ocean": {"name": "Atlantic Ocean", "bbox": [-60, -70, 60, 20], "has_demo_bundle": False},
    "southern_ocean": {"name": "Southern Ocean", "bbox": [-90, -180, -60, 180], "has_demo_bundle": False},
    "mediterranean_sea": {"name": "Mediterranean Sea", "bbox": [30, -6, 46, 37], "has_demo_bundle": False},
    "red_sea": {"name": "Red Sea", "bbox": [12, 32, 30, 44], "has_demo_bundle": False},
    "gulf_of_mexico": {"name": "Gulf of Mexico", "bbox": [18, -98, 31, -80], "has_demo_bundle": False},
    "south_china_sea": {"name": "South China Sea", "bbox": [-3, 99, 25, 122], "has_demo_bundle": False},
    "north_sea": {"name": "North Sea", "bbox": [51, -4, 62, 9], "has_demo_bundle": False},
    "coral_sea": {"name": "Coral Sea", "bbox": [-30, 142, -8, 172], "has_demo_bundle": False},
}


def _box_contains(bbox: list[float], lat: float, lon: float) -> bool:
    south, west, north, east = bbox
    if west <= east:
        lon_ok = west <= lon <= east
    else:  # bbox crosses the antimeridian
        lon_ok = lon >= west or lon <= east
    return south <= lat <= north and lon_ok


def bbox_contains(region_key: str, lat: float, lon: float) -> bool:
    r = REGIONS.get(region_key)
    if not r:
        return False
    boxes = r.get("sub_bboxes") or [r["bbox"]]
    return any(_box_contains(b, lat, lon) for b in boxes)


def region_for_point(lat: float, lon: float) -> str | None:
    """First matching region for a clicked point -- supports 'click anywhere
    on Earth' by falling back to the containing ocean basin.
    """
    for key in REGIONS:
        if bbox_contains(key, lat, lon):
            return key
    return None
