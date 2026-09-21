from fastapi import APIRouter, HTTPException
import httpx
from app.services.satellite.erddap import noaa_coastwatch_sst_client

router=APIRouter()

@router.get("/surface")
async def get_surface(region:str):
    bounds={"bay_of_bengal":((5,24),(80,100)),"arabian_sea":((5,25),(55,75))}
    if region not in bounds:
        raise HTTPException(404,"Region not found")
    lat_range,lon_range=bounds[region]
    async with httpx.AsyncClient(timeout=30) as client:
        live=await noaa_coastwatch_sst_client().fetch_grid(client,lat_range,lon_range)
    return live.model_dump()

@router.get("/point")
async def get_point(lat:float,lon:float):
    async with httpx.AsyncClient(timeout=30) as client:
        live=await noaa_coastwatch_sst_client().fetch_point(client,lat,lon)
    return live.model_dump()
