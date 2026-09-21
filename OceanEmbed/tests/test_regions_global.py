"""
Tests for the global region catalogue (backend/app/core/regions_global.py).

Run with: PYTHONPATH=<repo>/backend python3 -m pytest tests/test_regions_global.py
(or plain `python3 tests/test_regions_global.py` -- no pytest required, see
the __main__ block below, since this sandbox couldn't install pytest either).

History: an earlier version modeled the Pacific as one rectangle with its
east edge at the Chile coastline (~-70 deg), which is correct for the South
Pacific but put the entire North American interior "inside" the Pacific for
any longitude west of -70 -- e.g. Kansas (38.5, -98) incorrectly resolved to
pacific_ocean. Fixed by splitting the Pacific into latitude-banded
sub-bboxes with a realistic eastern edge per hemisphere. These tests pin
that fix.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.regions_global import REGIONS, bbox_contains, region_for_point  # noqa: E402


def test_bay_of_bengal_point():
    assert region_for_point(12.5, 86.3) == "bay_of_bengal"


def test_pacific_antimeridian_crossing():
    assert bbox_contains("pacific_ocean", 0, 179) is True
    assert bbox_contains("pacific_ocean", 0, -179) is True
    assert bbox_contains("pacific_ocean", 0, 0) is False


def test_pacific_does_not_swallow_north_america():
    # Regression test for the Kansas bug described above.
    assert region_for_point(38.5, -98.0) is None


def test_pacific_hemisphere_bands():
    assert bbox_contains("pacific_ocean", 20, -155) is True   # Hawaii (north band)
    assert bbox_contains("pacific_ocean", 35, 140) is True    # off Japan (north band)
    assert bbox_contains("pacific_ocean", -30, -72) is True   # off Chile (south band)


def test_all_regions_have_required_fields():
    for key, meta in REGIONS.items():
        assert "name" in meta and "bbox" in meta and "has_demo_bundle" in meta, key
        assert len(meta["bbox"]) == 4, key


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    if failed:
        sys.exit(1)
    print(f"\n{len(tests)} tests passed.")
