from fastapi import APIRouter, Request, Query

router = APIRouter()


@router.get("")
def list_events(request: Request, region: str, severity: str | None = None):
    store = request.app.state.store
    r = store.get_region(region)
    events = r["events"]
    if severity:
        events = [e for e in events if e["severity"].lower() == severity.lower()]
    return {"region": region, "count": len(events), "events": events}


@router.get("/extreme")
def list_extreme_events(request: Request, region: str):
    return list_events(request, region=region, severity="High")
