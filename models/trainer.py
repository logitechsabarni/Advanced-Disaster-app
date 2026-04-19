"""
ML Model Trainer - Trains all disaster prediction models
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)


def generate_earthquake_data(n=5000):
    np.random.seed(42)
    lat = np.random.uniform(-60, 70, n)
    lon = np.random.uniform(-180, 180, n)
    depth = np.random.exponential(30, n)
    seismic_activity = np.random.uniform(0, 10, n)
    fault_distance = np.random.exponential(50, n)
    historical_freq = np.random.poisson(3, n)
    tectonic_stress = np.random.uniform(0, 100, n)
    
    magnitude = (
        2.0 +
        0.05 * seismic_activity +
        0.01 * tectonic_stress -
        0.005 * fault_distance +
        np.random.normal(0, 0.5, n)
    ).clip(1.0, 9.5)
    
    risk = np.where(magnitude < 3, "Low", np.where(magnitude < 5, "Moderate", np.where(magnitude < 7, "High", "Critical")))
    
    X = np.column_stack([lat, lon, depth, seismic_activity, fault_distance, historical_freq, tectonic_stress])
    return X, magnitude, risk


def generate_flood_data(n=5000):
    np.random.seed(123)
    rainfall = np.random.exponential(50, n)
    river_level = np.random.uniform(0, 15, n)
    soil_moisture = np.random.uniform(0, 100, n)
    elevation = np.random.exponential(100, n)
    drainage = np.random.uniform(0, 10, n)
    population_density = np.random.exponential(500, n)
    
    flood_prob = (
        0.3 * (rainfall / 200) +
        0.25 * (river_level / 15) +
        0.2 * (soil_moisture / 100) +
        0.15 * (1 - elevation / 500).clip(0, 1) +
        0.1 * (1 - drainage / 10)
    ).clip(0, 1)
    flood_prob += np.random.normal(0, 0.05, n)
    flood_prob = flood_prob.clip(0, 1)
    
    X = np.column_stack([rainfall, river_level, soil_moisture, elevation, drainage, population_density])
    return X, flood_prob


def generate_cyclone_data(n=5000):
    np.random.seed(456)
    sst = np.random.uniform(24, 32, n)
    pressure = np.random.uniform(900, 1010, n)
    wind_shear = np.random.uniform(0, 30, n)
    humidity = np.random.uniform(50, 100, n)
    lat = np.random.uniform(5, 40, n)
    
    wind_speed = (
        50 +
        5 * (sst - 26) +
        0.3 * (1010 - pressure) -
        1.5 * wind_shear +
        0.3 * humidity +
        np.random.normal(0, 10, n)
    ).clip(20, 200)
    
    category = np.digitize(wind_speed, [74, 96, 111, 130, 157]) + 1
    category = category.clip(1, 5)
    
    X = np.column_stack([sst, pressure, wind_shear, humidity, lat])
    return X, wind_speed, category


def generate_wildfire_data(n=5000):
    np.random.seed(789)
    temperature = np.random.uniform(10, 45, n)
    humidity = np.random.uniform(5, 95, n)
    wind_speed = np.random.uniform(0, 50, n)
    drought_index = np.random.uniform(0, 10, n)
    vegetation_density = np.random.uniform(0, 100, n)
    
    fire_prob = (
        0.25 * ((temperature - 10) / 35) +
        0.3 * (1 - humidity / 100) +
        0.2 * (wind_speed / 50) +
        0.15 * (drought_index / 10) +
        0.1 * (vegetation_density / 100)
    ).clip(0, 1)
    fire_prob += np.random.normal(0, 0.05, n)
    fire_prob = fire_prob.clip(0, 1)
    
    X = np.column_stack([temperature, humidity, wind_speed, drought_index, vegetation_density])
    return X, fire_prob


def generate_tsunami_data(n=5000):
    np.random.seed(321)
    magnitude = np.random.uniform(5.0, 9.5, n)
    depth = np.random.uniform(0, 100, n)
    coastal_distance = np.random.exponential(200, n)
    bathymetry = np.random.uniform(100, 8000, n)
    
    wave_height = (
        0.5 * np.exp(0.5 * (magnitude - 6)) -
        0.002 * depth +
        0.001 * bathymetry / 1000 +
        np.random.normal(0, 0.5, n)
    ).clip(0, 30)
    
    eta = (coastal_distance / 200 * 60 + np.random.normal(0, 10, n)).clip(5, 600)
    
    X = np.column_stack([magnitude, depth, coastal_distance, bathymetry])
    return X, wave_height, eta


def generate_drought_data(n=5000):
    np.random.seed(654)
    spi = np.random.normal(0, 1.5, n)
    temp_anomaly = np.random.normal(0, 2, n)
    precip_deficit = np.random.uniform(0, 100, n)
    evapotranspiration = np.random.uniform(0, 10, n)
    ndvi = np.random.uniform(-0.2, 0.9, n)
    
    severity_score = (
        -0.5 * spi +
        0.2 * temp_anomaly +
        0.3 * (precip_deficit / 100) +
        0.2 * (evapotranspiration / 10) +
        np.random.normal(0, 0.2, n)
    )
    
    severity = np.where(severity_score < 0, "No Drought",
               np.where(severity_score < 0.5, "Mild",
               np.where(severity_score < 1, "Moderate",
               np.where(severity_score < 1.5, "Severe", "Extreme"))))
    
    duration = (severity_score * 4 + np.random.poisson(2, n)).clip(0, 52).astype(int)
    
    X = np.column_stack([spi, temp_anomaly, precip_deficit, evapotranspiration, ndvi])
    return X, severity_score, duration


def train_all_models():
    """Train and save all disaster prediction models"""
    results = {}
    
    # ── Earthquake Model ──
    X, mag, risk = generate_earthquake_data()
    X_train, X_test, y_train, y_test = train_test_split(X, mag, test_size=0.2)
    eq_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42))
    ])
    eq_model.fit(X_train, y_train)
    score = eq_model.score(X_test, y_test)
    joblib.dump(eq_model, os.path.join(MODEL_DIR, "earthquake_model.pkl"))
    results["earthquake"] = score
    
    # ── Flood Model ──
    X, prob = generate_flood_data()
    X_train, X_test, y_train, y_test = train_test_split(X, prob, test_size=0.2)
    fl_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42))
    ])
    fl_model.fit(X_train, y_train)
    score = fl_model.score(X_test, y_test)
    joblib.dump(fl_model, os.path.join(MODEL_DIR, "flood_model.pkl"))
    results["flood"] = score
    
    # ── Cyclone Model ──
    X, wind, cat = generate_cyclone_data()
    X_train, X_test, y_train, y_test = train_test_split(X, wind, test_size=0.2)
    cy_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42))
    ])
    cy_model.fit(X_train, y_train)
    score = cy_model.score(X_test, y_test)
    joblib.dump(cy_model, os.path.join(MODEL_DIR, "cyclone_model.pkl"))
    results["cyclone"] = score
    
    # ── Wildfire Model ──
    X, prob = generate_wildfire_data()
    X_train, X_test, y_train, y_test = train_test_split(X, prob, test_size=0.2)
    wf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42))
    ])
    wf_model.fit(X_train, y_train)
    score = wf_model.score(X_test, y_test)
    joblib.dump(wf_model, os.path.join(MODEL_DIR, "wildfire_model.pkl"))
    results["wildfire"] = score
    
    # ── Tsunami Model ──
    X, wave, eta = generate_tsunami_data()
    X_train, X_test, y_train, y_test = train_test_split(X, wave, test_size=0.2)
    ts_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42))
    ])
    ts_model.fit(X_train, y_train)
    score = ts_model.score(X_test, y_test)
    joblib.dump(ts_model, os.path.join(MODEL_DIR, "tsunami_model.pkl"))
    results["tsunami"] = score
    
    # ── Drought Model ──
    X, severity_score, duration = generate_drought_data()
    X_train, X_test, y_train, y_test = train_test_split(X, severity_score, test_size=0.2)
    dr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42))
    ])
    dr_model.fit(X_train, y_train)
    score = dr_model.score(X_test, y_test)
    joblib.dump(dr_model, os.path.join(MODEL_DIR, "drought_model.pkl"))
    results["drought"] = score
    
    return results


def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def models_exist():
    names = ["earthquake", "flood", "cyclone", "wildfire", "tsunami", "drought"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f"{n}_model.pkl")) for n in names)
