"""
Model C (Random Forest) - Ideal Yield Prediction
==================================================
Uses Random Forest Regressor to predict the ideal maize yield (tons/hectare)
based on soil macronutrients (N, P, K, pH) and weather data (Rainfall, Temperature).

This model predicts the yield assuming the crop is 100% healthy.
The actual/final yield is adjusted by Model D using the DSI from Model A.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
def load_dataset(csv_path):
    """
    Loads the crop yield dataset from a CSV file.
    
    Expected CSV columns:
        N, P, K, Soil_pH, Soil_Moisture, Temperature, Humidity, Rainfall, Sunlight_Hours, Yield
    """
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Map common column name variations
    column_map = {
        'nitrogen': 'n', 'phosphorus': 'p', 'potassium': 'k',
        'soil_ph': 'soil_ph', 'ph_level': 'soil_ph', 'ph': 'soil_ph',
        'rain': 'rainfall', 'precipitation': 'rainfall',
        'temp': 'temperature', 'avg_temp': 'temperature',
        'moisture': 'soil_moisture', 'humidity': 'humidity', 'sunlight': 'sunlight_hours',
        'crop_yield': 'yield', 'yield_tons': 'yield',
        'yield_ton_per_hectare': 'yield', 'crop_yield_ton_per_hectare': 'yield',
    }
    df.rename(columns=column_map, inplace=True)
    
    required_cols = ['n', 'p', 'k', 'soil_ph', 'soil_moisture', 'temperature', 'humidity', 'rainfall', 'sunlight_hours', 'yield']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n[ERROR] Missing columns: {missing}")
        print(f"[INFO]  Available columns: {list(df.columns)}")
        raise ValueError(f"Dataset is missing required columns: {missing}")
    
    before = len(df)
    df = df[required_cols].dropna()
    after = len(df)
    if before != after:
        print(f"[INFO] Dropped {before - after} rows with missing values.")
    
    print(f"[OK] Dataset loaded: {after} samples")
    print(f"     Yield range: {df['yield'].min():.2f} - {df['yield'].max():.2f} tons/hectare")
    
    feature_cols = ['n', 'p', 'k', 'soil_ph', 'soil_moisture', 'temperature', 'humidity', 'rainfall', 'sunlight_hours']
    X = df[feature_cols].values
    y = df['yield'].values
    
    return X, y, feature_cols

# ==========================================
# 2. Model Training (Random Forest)
# ==========================================
def train_model(csv_path, test_size=0.2, save_dir="."):
    """
    Trains a Random Forest model for yield prediction.
    
    Args:
        csv_path:   Path to CSV with soil, weather, and yield data.
        test_size:  Fraction for testing (default 20%).
        save_dir:   Directory to save model files.
    
    Saves:
        model_cRF_weights.pkl  - Trained Random Forest model
        model_cRF_scaler.pkl   - Feature scaler
        model_cRF_config.json  - Metadata and metrics
    """
    print("=" * 50)
    print("  Model C (RF) - Yield Prediction Training")
    print("=" * 50)
    
    # Load data
    X, y, feature_names = load_dataset(csv_path)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest model
    print(f"\nTraining with: RANDOM FOREST")
    print("-" * 40)
    
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    print(f"\n{'='*40}")
    print(f"  TRAINING RESULTS")
    print(f"{'='*40}")
    print(f"  Train R² Score : {train_r2:.4f}")
    print(f"  Test  R² Score : {test_r2:.4f}")
    print(f"  Test  MAE      : {test_mae:.4f} tons/ha")
    print(f"  Test  RMSE     : {test_rmse:.4f} tons/ha")
    print(f"  CV R² (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*40}")
    
    # Feature importance
    importances = model.feature_importances_
    print(f"\n  Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {name:>12s}: {imp:.4f} {bar}")
    
    # Save
    model_path = os.path.join(save_dir, "model_cRF_weights.pkl")
    scaler_path = os.path.join(save_dir, "model_cRF_scaler.pkl")
    config_path = os.path.join(save_dir, "model_cRF_config.json")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    config = {
        "model_type": "random_forest",
        "feature_names": feature_names,
        "metrics": {
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "test_mae": round(test_mae, 4),
            "test_rmse": round(test_rmse, 4),
            "cv_r2_mean": round(cv_scores.mean(), 4),
        },
        "data_stats": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n  Saved: {model_path}")
    print(f"  Saved: {scaler_path}")
    print(f"  Saved: {config_path}")
    
    return model, scaler

# ==========================================
# 3. Inference Function
# ==========================================
def predict_yield(n, p, k, soil_ph, soil_moisture, temperature, humidity, rainfall, sunlight_hours,
                  model_path="model_cRF_weights.pkl",
                  scaler_path="model_cRF_scaler.pkl"):
    """
    Predicts the ideal yield given soil and weather inputs.
    
    Returns:
        dict with 'ideal_yield' in tons/hectare
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found. Please train first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    features = np.array([[n, p, k, soil_ph, soil_moisture, temperature, humidity, rainfall, sunlight_hours]])
    features_scaled = scaler.transform(features)
    
    ideal_yield = model.predict(features_scaled)[0]
    ideal_yield = max(0.0, ideal_yield)
    
    return {
        "ideal_yield": round(ideal_yield, 4),
        "unit": "tons/hectare",
        "inputs": {
            "N": n, "P": p, "K": k,
            "Soil_pH": soil_ph, "Soil_Moisture": soil_moisture,
            "Temperature": temperature, "Humidity": humidity, 
            "Rainfall": rainfall, "Sunlight": sunlight_hours
        }
    }

# ==========================================
# 4. Model D - Health-Adjusted Final Yield
# ==========================================
def calculate_final_yield(ideal_yield, dsi, crop_sensitivity_K=0.6):
    """
    Model D: Final Yield = Ideal Yield × (1 - K × DSI)
    
    Args:
        ideal_yield:        From Model C (100% healthy prediction)
        dsi:                Disease Severity Index from Model A (0.0 to 1.0)
        crop_sensitivity_K: Crop Sensitivity Constant (default 0.6 for maize)
    """
    dsi = max(0.0, min(1.0, dsi))
    
    final_yield = ideal_yield * (1 - crop_sensitivity_K * dsi)
    final_yield = max(0.0, final_yield)
    
    yield_loss = ideal_yield - final_yield
    yield_loss_pct = (yield_loss / ideal_yield * 100) if ideal_yield > 0 else 0.0
    
    return {
        "ideal_yield": round(ideal_yield, 2),
        "final_yield": round(final_yield, 2),
        "yield_loss": round(yield_loss, 2),
        "yield_loss_percentage": round(yield_loss_pct, 2),
        "dsi_used": round(dsi, 4),
        "crop_sensitivity_K": crop_sensitivity_K
    }


# ==========================================
# Usage
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Model C (RF) & D - Yield Prediction Pipeline")
    print("=" * 50)
    
    # --- Training Workflow ---
    csv_file = "yield-predict-augmented.csv"
    if os.path.exists(csv_file):
        train_model(csv_path=csv_file)
    else:
        print(f"Dataset '{csv_file}' not found. Please run augment_data.py first.")
    
    # --- Inference Workflow ---
    if os.path.exists("model_cRF_weights.pkl"):
        result = predict_yield(
            n=80, p=45, k=60, soil_ph=6.5, soil_moisture=25.0, 
            temperature=26, humidity=60, rainfall=900, sunlight_hours=8
        )
        print(f"\nIdeal Yield Prediction (Model C - RF):")
        print(f"  Ideal Yield: {result['ideal_yield']} {result['unit']}")
        print(f"  Inputs: {result['inputs']}")
        
        # Apply Model D with DSI from Model A
                # Dynamically get DSI from Model A
        try:
            from model_a import predict_image
            test_image_path = 'test_4-B.jpg'
            if os.path.exists(test_image_path):
                model_a_result = predict_image(test_image_path)
                dsi_from_model_a = model_a_result["dsi"]
                print(f"  [Integration] Extracted DSI from '{test_image_path}': {dsi_from_model_a:.4f}")
            else:
                dsi_from_model_a = 0.67
                print("  [Integration] Image not found. Using fallback DSI: 0.67")
        except ImportError:
            dsi_from_model_a = 0.67
            print("  [Integration] Could not import model_a. Using fallback DSI: 0.67")

        final = calculate_final_yield(
            ideal_yield=result['ideal_yield'],
            dsi=dsi_from_model_a,
            crop_sensitivity_K=0.6
        )
        print(f"\nFinal Yield After Disease Adjustment (Model D):")
        print(f"  Ideal Yield       : {final['ideal_yield']} tons/ha")
        print(f"  Final Yield       : {final['final_yield']} tons/ha")
        print(f"  Yield Loss        : {final['yield_loss']} tons/ha ({final['yield_loss_percentage']}%)")
        print(f"  DSI Used          : {final['dsi_used']}")
        print(f"  Sensitivity (K)   : {final['crop_sensitivity_K']}")
    else:
        print("\n[INFO] No trained model found. Train first using your CSV dataset.")
