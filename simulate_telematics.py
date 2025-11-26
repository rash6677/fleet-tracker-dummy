

import time
from datetime import datetime, timezone
import random
import requests

API_BASE = "http://127.0.0.1:8000"

CITIES = {
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nashik": (19.9975, 73.7898),
    "Nagpur": (21.1458, 79.0882),
    "Surat": (21.1702, 72.8311),
    "Ahmedabad": (23.0225, 72.5714),
    "Indore": (22.7196, 75.8577),
    "Jaipur": (26.9124, 75.7873),
}

NUM_TRUCKS = 10
NUM_STEPS = 120
SLEEP_SEC = 2


def pick_route():
    names = list(CITIES.keys())
    o_name, d_name = random.sample(names, 2)
    o_lat, o_lon = CITIES[o_name]
    d_lat, d_lon = CITIES[d_name]
    return o_lat, o_lon, d_lat, d_lon


def send_point(vehicle_id, trip_id, lat, lon, speed, o_lat, o_lon, d_lat, d_lon):
    payload = {
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "lat": lat,
        "lon": lon,
        "speed": speed,
        "fuel_rate": random.uniform(6, 20),
        "engine_rpm": random.uniform(1200, 2500),
        "load": random.uniform(0.6, 1.0),
        "origin_lat": o_lat,
        "origin_lon": o_lon,
        "dest_lat": d_lat,
        "dest_lon": d_lon,
    }

    try:
        r = requests.post(f"{API_BASE}/ingest_telematics", json=payload, timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(" Failed to send telemetry:", e)


if __name__ == "__main__":
    print(" Starting multi-truck telematics simulation...")

    trucks = []
    for i in range(1, NUM_TRUCKS + 1):
        vid = f"TRUCK_{i:03d}"
        tid = f"TRIP_{i:03d}"
        o_lat, o_lon, d_lat, d_lon = pick_route()
        trucks.append(
            {
                "vehicle_id": vid,
                "trip_id": tid,
                "o_lat": o_lat,
                "o_lon": o_lon,
                "d_lat": d_lat,
                "d_lon": d_lon,
                "step": 0,
            }
        )

    try:
        while True:
            for t in trucks:
                if t["step"] >= NUM_STEPS:
                    t["step"] = 0
                    o_lat, o_lon, d_lat, d_lon = pick_route()
                    t.update(
                        {
                            "o_lat": o_lat,
                            "o_lon": o_lon,
                            "d_lat": d_lat,
                            "d_lon": d_lon,
                            "trip_id": f"TRIP_{random.randint(1000, 9999)}",
                        }
                    )

                alpha = t["step"] / NUM_STEPS
                lat = t["o_lat"] + alpha * (t["d_lat"] - t["o_lat"])
                lon = t["o_lon"] + alpha * (t["d_lon"] - t["o_lon"])
                speed = random.uniform(35, 75)

                send_point(
                    t["vehicle_id"],
                    t["trip_id"],
                    lat,
                    lon,
                    speed,
                    t["o_lat"],
                    t["o_lon"],
                    t["d_lat"],
                    t["d_lon"],
                )

                t["step"] += 1

            print(" Telemetry sent for all trucks.")
            time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("\n Simulation stopped.")
