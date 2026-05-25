import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_regression_model(csv_path="yield-predict.csv", model_path="model_c_weights.pkl", scaler_path="model_c_scaler.pkl", model_name="Model C"):
    """
    Evaluates a regression model for yield prediction.
    """
    if not os.path.exists(model_path):
        print(f"[SKIP] Model file {model_path} not found.")
        return
    if not os.path.exists(scaler_path):
        print(f"[SKIP] Scaler file {scaler_path} not found.")
        return
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset {csv_path} not found for evaluation.")
        return

    print(f"\n{'='*50}")
    print(f"  Evaluating {model_name}")
    print(f"{'='*50}")

    # 1. Load Data
    df = pd.read_csv(csv_path)
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
    
    feature_cols = ['n', 'p', 'k', 'soil_ph', 'soil_moisture', 'temperature', 'humidity', 'rainfall', 'sunlight_hours']
    X = df[feature_cols].values
    y = df['yield'].values

    # 2. Train/Test Split (Same as training for fair evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Load Model and Scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 4. Predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # 5. Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"  R² Score         : {r2:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f} tons/ha")
    print(f"  Mean Squared Error : {mse:.4f}")
    print(f"  Root MSE         : {rmse:.4f} tons/ha")
    print(f"{'='*50}")

    # with open(F"evaluate/model_C_report.txt_{model_name.lower().replace(' ', '_')}", "w") as f:
    #     f.write(f"R2 Score: {r2:.4f}\n"+ f"Mean Absolute Error: {mae:.4f} tons/ha\n"
    #     + f"Mean Squared Error: {mse:.4f}\n"+ f"Root MSE: {rmse:.4f} tons/ha\n")
    
    # 6. Visualization
    plt.figure(figsize=(15, 6))

    # Actual vs Predicted Plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Yield (tons/ha)')
    plt.ylabel('Predicted Yield (tons/ha)')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.grid(True)

    # Residuals Plot
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, color='blue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.title(f'{model_name}: Residuals Distribution')
    plt.grid(True)

    save_path = f"evaluate/{model_name.lower().replace(' ', '_')}_results.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Plots saved to: {save_path}")

def find_file(filename):
    """Checks current and parent directory for a file."""
    if os.path.exists(filename):
        return filename
    parent_path = os.path.join("..", filename)
    if os.path.exists(parent_path):
        return parent_path
    return filename

if __name__ == "__main__":
    # Evaluate Model C (XGBoost)
    evaluate_regression_model(
        csv_path=find_file("yield-predict.csv"),
        model_path=find_file("model_c_weights.pkl"),
        scaler_path=find_file("model_c_scaler.pkl"),
        model_name="Model C XGBoost"
    )

    # Evaluate Model C-RF (Random Forest)
    evaluate_regression_model(
        csv_path=find_file("yield-predict.csv"),
        model_path=find_file("model_cRF_weights.pkl"),
        scaler_path=find_file("model_cRF_scaler.pkl"),
        model_name="Model C Random Forest"
    )
