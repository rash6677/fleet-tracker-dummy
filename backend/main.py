# backend/main.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
import math

from .ml_model import EtaModelManager   # <-- uses backend/ml_model.py

app = FastAPI(title="Fleet Tracking Backend", version="1.0")

# Allow calls from Streamlit or other local apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

eta_manager = EtaModelManager()


class TelemetryIn(BaseModel):
    vehicle_id: str
    trip_id: str
    ts: datetime
    lat: float
    lon: float
    speed: float
    fuel_rate: float
    engine_rpm: float
    load: float

    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float


class RouteFeatures(BaseModel):
    distance_km: float
    avg_speed: float
    model_name: Optional[str] = None


# store latest telemetry per vehicle
TELEMETRY_STORE: Dict[str, TelemetryIn] = {}


def to_dict(model: BaseModel) -> dict:
    """Compatible with Pydantic v1 and v2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


@app.get("/")
def root():
    return {
        "status": "API running",
        "available_models": list(eta_manager.models.keys()),
    }


@app.post("/ingest_telematics")
def ingest_telematics(t: TelemetryIn):
    TELEMETRY_STORE[t.vehicle_id] = t
    return {"status": "ok"}


@app.get("/current_positions")
def current_positions():
    return [to_dict(t) for t in TELEMETRY_STORE.values()]


@app.get("/current_positions_with_eta")
def current_positions_with_eta(
    model: Literal["rf", "svr", "gbr", "xgb"] = Query("xgb"),
):
    now = datetime.now(timezone.utc)
    result: List[dict] = []

    for t in TELEMETRY_STORE.values():
        ts = t.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_sec = (now - ts).total_seconds()

        total_route_km = haversine(
            t.origin_lat, t.origin_lon, t.dest_lat, t.dest_lon
        )
        distance_remaining_km = haversine(
            t.lat, t.lon, t.dest_lat, t.dest_lon
        )

        progress_percent = 0.0
        if total_route_km > 0:
            progress_percent = max(
                0.0,
                min(100.0, (1 - distance_remaining_km / total_route_km) * 100),
            )

        avg_speed = max(t.speed, 5.0)
        eta_min = eta_manager.predict_eta(
            {"distance_km": distance_remaining_km, "avg_speed": avg_speed},
            model,
        )

        row = to_dict(t)
        row.update(
            {
                "total_route_km": total_route_km,
                "distance_remaining_km": distance_remaining_km,
                "progress_percent": progress_percent,
                "eta_minutes": eta_min,
                "last_update_seconds_ago": age_sec,
                "model_used": model,
            }
        )
        result.append(row)

    return result


@app.post("/predict_eta")
def predict_eta(
    payload: RouteFeatures,
    model: Literal["rf", "svr", "gbr", "xgb"] = Query("xgb"),
):
    model_key = payload.model_name or model
    eta = eta_manager.predict_eta(
        {"distance_km": payload.distance_km, "avg_speed": payload.avg_speed},
        model_key,
    )
    return {"eta_minutes": eta, "model_used": model_key}
