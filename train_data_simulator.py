# train_data_simulator.py

import os
import numpy as np
import pandas as pd

DATA_PATH = "data/raw"
os.makedirs(DATA_PATH, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute distance (km) between two lat/lon points.
    """
    R = 6371.0  # Earth radius (km)
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def simulate_trips(num_trips: int = 500):
    # Some Indian city coordinates (approx)
    cities = {
        "Mumbai": (19.0760, 72.8777),
        "Pune": (18.5204, 73.8567),
        "Nashik": (19.9975, 73.7898),
        "Nagpur": (21.1458, 79.0882),
        "Surat": (21.1702, 72.8311),
        "Ahmedabad": (23.0225, 72.5714),
    }

    city_names = list(cities.keys())
    rng = np.random.default_rng(42)

    rows = []
    for i in range(num_trips):
        origin_name, dest_name = rng.choice(city_names, size=2, replace=False)
        o_lat, o_lon = cities[origin_name]
        d_lat, d_lon = cities[dest_name]

        distance_km = haversine(o_lat, o_lon, d_lat, d_lon)

        # Assume base speed 40–70 km/h + noise
        base_speed = rng.uniform(40, 70)
        duration_hours = distance_km / base_speed

        # Add traffic / delay noise (up to +50%)
        delay_factor = rng.uniform(1.0, 1.5)
        actual_duration_hours = duration_hours * delay_factor

        # Convert to minutes
        duration_min = actual_duration_hours * 60

        # Fake timestamps: not used in training deeply, but nice to have
        # We just use a numeric index here; you can extend later
        rows.append(
            {
                "trip_id": f"TRIP_{i:04d}",
                "vehicle_id": f"TRUCK_{rng.integers(1, 21):03d}",
                "origin_city": origin_name,
                "origin_lat": o_lat,
                "origin_lon": o_lon,
                "dest_city": dest_name,
                "dest_lat": d_lat,
                "dest_lon": d_lon,
                "distance_km": distance_km,
                "duration_min": duration_min,
                "avg_speed": distance_km / (actual_duration_hours + 1e-6),
            }
        )

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = simulate_trips(num_trips=800)
    out_path = os.path.join(DATA_PATH, "trips.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved simulated trips to {out_path}")
    print(df.head())
