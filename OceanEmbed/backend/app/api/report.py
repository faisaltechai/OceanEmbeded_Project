from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.overview import get_overview
from app.api.ocean import get_profile
from app.reports.generator import build_report

router = APIRouter()


class ReportRequest(BaseModel):
    region: str
    lat: float | None = None
    lon: float | None = None


@router.post("/report")
async def generate_report(request: Request, body: ReportRequest):
    store = request.app.state.store
    r = store.get_region(body.region)

    overview = await get_overview(request, region=body.region)
    profile = None
    if body.lat is not None and body.lon is not None:
        profile = get_profile(request, region=body.region, lat=body.lat, lon=body.lon)

    path = build_report(
        region=body.region,
        region_name=r["region_name"],
        lat=body.lat,
        lon=body.lon,
        overview=overview,
        profile=profile,
        events=r.get("events", []),
    )
    return {"status": "ok", "file_path": path, "download_url": f"/api/report/download?path={path}"}


@router.get("/report/download")
def download_report(path: str):
    import os
    if not os.path.isfile(path):
        raise HTTPException(404, "Report file not found.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
