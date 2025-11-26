# frontend/app.py

import requests
import streamlit as st
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Fleet Tracking Dashboard", layout="wide")
st.title("🚚 Fleet Tracking & ML ETA Dashboard")


# ---- Sidebar: manual ETA ----
st.sidebar.header("Manual ETA Prediction")

distance_km = st.sidebar.number_input("Distance (km)", min_value=1.0, value=150.0)
avg_speed = st.sidebar.number_input("Average speed (km/h)", min_value=1.0, value=55.0)
model_choice = st.sidebar.selectbox("Model", ["xgb", "rf", "gbr", "svr"], index=0)

if st.sidebar.button("Predict ETA"):
    payload = {
        "distance_km": distance_km,
        "avg_speed": avg_speed,
        "model_name": model_choice,
    }
    try:
        r = requests.post(f"{API_BASE}/predict_eta", json=payload, timeout=5)
        r.raise_for_status()
        res = r.json()
        st.sidebar.success(
            f"ETA: {res['eta_minutes']:.1f} min (model: {res['model_used']})"
        )
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


# ---- Main: fleet tracking ----
st.subheader("📡 Live Fleet Positions (with ETA & Progress)")

fleet_model = st.selectbox(
    "Model for Fleet ETA", ["xgb", "rf", "gbr", "svr"], index=0
)

if st.button("Refresh Fleet Data"):
    st.session_state["refresh"] = True


def fetch_fleet():
    try:
        r = requests.get(
            f"{API_BASE}/current_positions_with_eta",
            params={"model": fleet_model},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error fetching fleet: {e}")
        return []


vehicles = fetch_fleet()

if not vehicles:
    st.info("No telemetry yet. Run simulate_telematics.py.")
else:
    df = pd.DataFrame(vehicles)

    preferred_cols = [
        "vehicle_id",
        "trip_id",
        "origin_lat",
        "origin_lon",
        "dest_lat",
        "dest_lon",
        "lat",
        "lon",
        "speed",
        "total_route_km",
        "distance_remaining_km",
        "progress_percent",
        "eta_minutes",
        "last_update_seconds_ago",
        "fuel_rate",
        "engine_rpm",
        "load",
        "ts",
        "model_used",
    ]

    cols = [c for c in preferred_cols if c in df.columns] + [
        c for c in df.columns if c not in preferred_cols
    ]
    df = df[cols]

    st.dataframe(df, use_container_width=True)

    if {"lat", "lon"}.issubset(df.columns):
        st.map(df[["lat", "lon"]])
