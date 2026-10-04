"""
Data loader — generates and caches all synthetic/sample datasets.
Provides a clear interface so that real MOIL data can be substituted later.
"""
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta

# ── MOIL mine locations (publicly known approximate coordinates) ──────────────
MOIL_MINES = {
    "Balaghat":       {"lat": 21.83, "lon": 80.18, "state": "MP",  "capacity_tpd": 2800},
    "Dongri Buzurg":  {"lat": 21.69, "lon": 80.06, "state": "MP",  "capacity_tpd": 2200},
    "Kandri":         {"lat": 21.22, "lon": 79.17, "state": "MH",  "capacity_tpd": 1800},
    "Munsar":         {"lat": 21.19, "lon": 79.20, "state": "MH",  "capacity_tpd": 1600},
    "Gumgaon":        {"lat": 21.32, "lon": 79.32, "state": "MH",  "capacity_tpd": 1400},
    "Chikla":         {"lat": 21.45, "lon": 79.48, "state": "MH",  "capacity_tpd": 1200},
    "Sitapatore":     {"lat": 21.50, "lon": 79.55, "state": "MH",  "capacity_tpd":  900},
    "Ukwa":           {"lat": 21.62, "lon": 80.02, "state": "MP",  "capacity_tpd":  750},
    "Tirodi":         {"lat": 21.68, "lon": 79.72, "state": "MH",  "capacity_tpd":  600},
    "New Chikla":     {"lat": 21.47, "lon": 79.51, "state": "MH",  "capacity_tpd":  500},
}

LITHOLOGY_CLASSES = [
    "Gondite / Kodurite",
    "Khondalite",
    "Charnockite",
    "Banded Iron Formation",
    "Metamorphic Schist",
    "Granitic Gneiss",
    "Quartzite",
    "Calc-silicate",
]

FORMATION_CLASSES = [
    "Sausar Group",
    "Eastern Ghat Mobile Belt",
    "Vindhyan Supergroup",
    "Peninsular Gneissic Complex",
    "Deccan Trap",
    "Bastar Craton",
]


# ─────────────────────────────────────────────────────────────────────────────
#  SATELLITE FEATURES  (Sentinel-2 derived, prototype / synthetic proxy)
# ─────────────────────────────────────────────────────────────────────────────
def generate_satellite_features(n_points: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic Sentinel-2 surface spectral features for a prototype.
    In production, replace with actual band reflectance values from Copernicus
    Data Space (https://dataspace.copernicus.eu/).
    """
    rng = np.random.default_rng(seed)

    # Seed ~25 % of points near known MOIL mine centres
    mine_list = list(MOIL_MINES.values())
    n_near = int(n_points * 0.25)
    n_rand = n_points - n_near

    # Near-mine points
    mine_idx = rng.integers(0, len(mine_list), n_near)
    lat_near = np.array([mine_list[i]["lat"] for i in mine_idx]) + rng.normal(0, 0.15, n_near)
    lon_near = np.array([mine_list[i]["lon"] for i in mine_idx]) + rng.normal(0, 0.15, n_near)

    # Random background (central India bounding box)
    lat_rand = rng.uniform(20.5, 22.5, n_rand)
    lon_rand = rng.uniform(78.5, 81.0, n_rand)

    lat = np.concatenate([lat_near, lat_rand])
    lon = np.concatenate([lon_near, lon_rand])

    # Sentinel-2 bands (reflectance 0–1 scale)
    b2  = rng.uniform(0.05, 0.15, n_points)   # Blue
    b3  = rng.uniform(0.06, 0.18, n_points)   # Green
    b4  = rng.uniform(0.05, 0.20, n_points)   # Red
    b8  = rng.uniform(0.15, 0.45, n_points)   # NIR
    b11 = rng.uniform(0.10, 0.35, n_points)   # SWIR-1
    b12 = rng.uniform(0.08, 0.30, n_points)   # SWIR-2

    # Derived indices
    eps = 1e-6
    ndvi  = (b8  - b4)  / (b8 + b4 + eps)
    ndwi  = (b3  - b8)  / (b3 + b8 + eps)
    bsi   = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)  # Bare Soil Index
    clay  = b11 / (b12 + eps)    # Clay mineral ratio
    iron  = b4  / (b2  + eps)    # Iron oxide ratio
    swir_ratio = b11 / (b8 + eps)

    # Elevation & slope (proxy via DEM — synthetic here)
    elevation = rng.uniform(350, 750, n_points)
    slope     = rng.uniform(0, 25, n_points)

    return pd.DataFrame({
        "latitude": lat,
        "longitude": lon,
        "b2_blue": b2, "b3_green": b3, "b4_red": b4,
        "b8_nir": b8, "b11_swir1": b11, "b12_swir2": b12,
        "ndvi": ndvi, "ndwi": ndwi, "bsi": bsi,
        "clay_ratio": clay, "iron_ratio": iron, "swir_ratio": swir_ratio,
        "elevation_m": elevation, "slope_deg": slope,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  GEOLOGICAL FEATURES  (GSI-style data, synthetic proxy for prototype)
# ─────────────────────────────────────────────────────────────────────────────
def generate_geological_features(sat_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic geological attributes attached to the satellite sample points.
    In production, replace with GSI digital geological map data:
    https://www.gsi.gov.in/
    """
    rng = np.random.default_rng(seed)
    n = len(sat_df)

    lithology_codes = {name: i for i, name in enumerate(LITHOLOGY_CLASSES)}
    formation_codes = {name: i for i, name in enumerate(FORMATION_CLASSES)}

    lithology  = rng.choice(LITHOLOGY_CLASSES, n)
    formation  = rng.choice(FORMATION_CLASSES, n)

    # Distance to nearest known MOIL mine (simplified Euclidean in degrees)
    mine_lats = np.array([v["lat"] for v in MOIL_MINES.values()])
    mine_lons = np.array([v["lon"] for v in MOIL_MINES.values()])
    dist_km = []
    for la, lo in zip(sat_df["latitude"], sat_df["longitude"]):
        d = np.sqrt((mine_lats - la)**2 + (mine_lons - lo)**2) * 111
        dist_km.append(d.min())
    dist_km = np.array(dist_km)

    # Fault / lineament proximity (synthetic, 0=far, 1=close)
    fault_proximity = np.exp(-dist_km / 50) + rng.normal(0, 0.05, n)
    fault_proximity = np.clip(fault_proximity, 0, 1)

    # USP-1: Structural Density Index (Zhao 2025 -- most influential Mn predictor)
    # Simulates fault/lineament density from DEM-derived lineament mapping
    structural_density = (
        fault_proximity * 0.6
        + np.exp(-dist_km / 30) * 0.3
        + rng.uniform(0, 0.15, n)
    )
    structural_density = np.clip(structural_density, 0, 1)

    # Known occurrence label
    near_mine = dist_km < 20
    gondite    = lithology == "Gondite / Kodurite"
    known_occurrence = (near_mine | gondite).astype(int)
    flip = rng.random(n) < 0.08
    known_occurrence = np.where(flip, 1 - known_occurrence, known_occurrence)

    geo_df = sat_df.copy()
    geo_df["lithology"]           = lithology
    geo_df["lithology_code"]      = [lithology_codes[l] for l in lithology]
    geo_df["formation"]           = formation
    geo_df["formation_code"]      = [formation_codes[f] for f in formation]
    geo_df["dist_to_mine_km"]     = dist_km
    geo_df["fault_proximity"]     = fault_proximity
    geo_df["structural_density"]  = structural_density
    geo_df["known_occurrence"]    = known_occurrence

    return geo_df


# ─────────────────────────────────────────────────────────────────────────────
#  WEATHER / ENVIRONMENTAL DATA  (ERA5 / Copernicus, synthetic proxy)
# ─────────────────────────────────────────────────────────────────────────────
def generate_weather_data(
    start: str = "2022-01-01",
    end:   str = "2024-12-31",
    seed:  int = 42
) -> pd.DataFrame:
    """
    Daily weather dataset for the MOIL operating region.
    In production, replace with ERA5 data from:
    https://cds.climate.copernicus.eu/
    """
    rng  = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n    = len(dates)

    # Seasonal rainfall pattern (monsoon June–Sept)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    monsoon = np.exp(-((doy - 240)**2) / (2 * 50**2))  # peak ~day 240

    rainfall_mm   = np.clip(rng.exponential(5, n) * (1 + 15 * monsoon), 0, 120)
    temp_max_c    = 25 + 10 * np.sin(2 * np.pi * (doy - 90) / 365) + rng.normal(0, 2, n)
    temp_min_c    = temp_max_c - 8 + rng.normal(0, 1.5, n)
    humidity_pct  = 40 + 40 * monsoon + rng.normal(0, 5, n)
    soil_moisture = 0.15 + 0.3 * monsoon + rng.normal(0, 0.03, n)

    return pd.DataFrame({
        "date":          dates,
        "rainfall_mm":   rainfall_mm,
        "temp_max_c":    temp_max_c,
        "temp_min_c":    temp_min_c,
        "humidity_pct":  np.clip(humidity_pct, 20, 100),
        "soil_moisture": np.clip(soil_moisture, 0.05, 0.55),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  OPERATIONAL / PRODUCTION DATA  (Synthetic prototype — NOT official MOIL data)
# ─────────────────────────────────────────────────────────────────────────────
def generate_operational_data(
    start: str = "2022-01-01",
    end:   str = "2024-12-31",
    seed:  int = 42
) -> pd.DataFrame:
    """
    Prototype operational dataset — synthetic due to non-public operational records.

    IMPORTANT: This dataset is entirely synthetic and does NOT represent actual
    MOIL Ltd. operational records. It is designed to demonstrate AI/ML forecasting
    capability and can be replaced with real MOIL data.

    Variables based on realistic open-pit / underground manganese mining operations.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_dates = len(dates)
    mines   = list(MOIL_MINES.keys())

    records = []
    for mine_name, mine_info in MOIL_MINES.items():
        capacity = mine_info["capacity_tpd"]
        prev_actual = capacity * 0.85

        for i, date in enumerate(dates):
            doy = date.timetuple().tm_yday

            # Seasonal effects
            monsoon_factor = np.exp(-((doy - 240)**2) / (2 * 50**2))
            seasonal_eff   = 1 - 0.25 * monsoon_factor

            # Planned production
            planned = capacity * (0.90 + rng.uniform(-0.05, 0.05))

            # Operational factors
            equip_avail_pct  = rng.uniform(65, 98)
            downtime_hrs     = rng.exponential(2.5) * (1 + 2 * monsoon_factor)
            blasting_delay   = rng.exponential(1.5) * (1 + monsoon_factor)
            working_hrs      = np.clip(24 - downtime_hrs - blasting_delay, 8, 22)
            rainfall_day     = rng.exponential(4) * (1 + 12 * monsoon_factor)
            ore_grade_pct    = rng.normal(38, 4)  # Mn% typical for MOIL
            maintenance_evt  = int(rng.random() < 0.05)

            # Actual production (physics-informed formula)
            eff_factor = (
                (equip_avail_pct / 100) *
                (working_hrs / 18) *
                seasonal_eff *
                (1 - 0.015 * (rainfall_day > 30)) *
                (1 - 0.12 * maintenance_evt)
            )
            actual = planned * eff_factor + rng.normal(0, planned * 0.03)
            actual = max(actual, 0)

            shortfall     = max(planned - actual, 0)
            shortfall_pct = shortfall / planned * 100

            records.append({
                "date":               date,
                "mine":               mine_name,
                "state":              mine_info["state"],
                "planned_tpd":        round(planned, 1),
                "actual_tpd":         round(actual, 1),
                "equipment_avail_pct":round(equip_avail_pct, 1),
                "equipment_downtime_hrs": round(downtime_hrs, 2),
                "blasting_delay_hrs": round(blasting_delay, 2),
                "working_hrs":        round(working_hrs, 2),
                "rainfall_mm":        round(rainfall_day, 1),
                "ore_grade_pct":      round(ore_grade_pct, 1),
                "maintenance_event":  maintenance_evt,
                "prev_actual_tpd":    round(prev_actual, 1),
                "shortfall_tpd":      round(shortfall, 1),
                "shortfall_pct":      round(shortfall_pct, 2),
            })
            prev_actual = actual

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS  (call these from pages)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st

@st.cache_data(show_spinner="Loading satellite features...")
def load_satellite_data():
    sat_df = generate_satellite_features(n_points=500)
    geo_df = generate_geological_features(sat_df)
    return geo_df

@st.cache_data(show_spinner="Loading weather data...")
def load_weather_data():
    return generate_weather_data()

@st.cache_data(show_spinner="Loading operational data...")
def load_operational_data():
    return generate_operational_data()

@st.cache_data(show_spinner="Loading all datasets...")
def load_all_data():
    sat  = load_satellite_data()
    wx   = load_weather_data()
    ops  = load_operational_data()
    return sat, wx, ops


# ─────────────────────────────────────────────────────────────────────────────
#  USP-5: TEMPORAL SATELLITE DATA  (two snapshots for change monitoring)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading temporal satellite snapshots...")
def load_temporal_satellite_data():
    """
    USP-5: Returns two satellite snapshots (T1=dry season, T2=post-monsoon)
    with computed spectral change deltas for temporal monitoring.
    """
    # T1: Dry season (lower vegetation, higher bare soil exposure)
    sat_t1 = generate_satellite_features(n_points=500, seed=42)
    geo_t1 = generate_geological_features(sat_t1, seed=42)
    geo_t1["snapshot"] = "T1: Dry Season (Jan 2024)"

    # T2: Post-monsoon (higher vegetation, moisture, different spectral signature)
    rng2 = np.random.default_rng(99)
    sat_t2 = generate_satellite_features(n_points=500, seed=99)
    # Simulate seasonal shifts: higher NDVI, lower BSI after monsoon
    sat_t2_mod = sat_t2.copy()
    sat_t2_mod["ndvi"]  = np.clip(sat_t2_mod["ndvi"]  + rng2.uniform(0.05, 0.15, 500), -1, 1)
    sat_t2_mod["bsi"]   = np.clip(sat_t2_mod["bsi"]   - rng2.uniform(0.02, 0.10, 500), -1, 1)
    sat_t2_mod["ndwi"]  = np.clip(sat_t2_mod["ndwi"]  + rng2.uniform(0.03, 0.12, 500), -1, 1)
    sat_t2_mod["iron_ratio"] = np.clip(sat_t2_mod["iron_ratio"] + rng2.uniform(-0.05, 0.05, 500), 0.5, 3)
    geo_t2 = generate_geological_features(sat_t2_mod, seed=99)
    geo_t2["latitude"]  = geo_t1["latitude"].values   # same spatial grid
    geo_t2["longitude"] = geo_t1["longitude"].values
    geo_t2["snapshot"] = "T2: Post-Monsoon (Oct 2024)"

    # Compute delta (change) between snapshots
    delta = geo_t1[["latitude", "longitude", "lithology", "prospectivity_score"
                     if "prospectivity_score" in geo_t1.columns else "dist_to_mine_km"]].copy()
    for col in ["ndvi", "bsi", "ndwi", "iron_ratio", "clay_ratio"]:
        if col in geo_t1.columns and col in geo_t2.columns:
            delta[f"delta_{col}"] = geo_t2[col].values - geo_t1[col].values

    return geo_t1, geo_t2, delta
