import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATA INGESTION & PARTITIONING STEP
# ==========================================
def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """
    Loads raw simulation data from a specified file path and establishes 
    train/test data matrices.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[+] Loaded data from: '{file_path}' successfully ({len(df)} records).")
    except FileNotFoundError:
        absolute_target = os.path.abspath(file_path)
        raise FileNotFoundError(
            f"[!] Could not locate file at: {file_path}\n"
            f"    Checked absolute path: {absolute_target}\n"
            f"    Please check your directory string or run the Phase 1 engine first!"
        )

    X = df[['initial_velocity', 'launch_angle']]
    y = df['landing_distance']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ==========================================
# 2. MODEL INITIALIZATION STEP
# ==========================================
def initialize_pipeline(hidden_layers=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42):
    """
    Instantiates the structural pipeline containing data scales and hyperparameters.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
    )
    return pipeline


# ==========================================
# 3. TRAINING STEP
# ==========================================
def train_pipeline(pipeline, X_train, y_train):
    """
    Executes feature standard transformations and trains the neural network weights.
    """
    print(f"[~] Fitting pipeline weights to training partition...")
    pipeline.fit(X_train, y_train)
    print("[+] Model pipeline training complete.")
    return pipeline


# ==========================================
# 4. EXECUTION / INFERENCE STEP
# ==========================================
def run_predictions(pipeline, X_test):
    """
    Generates model range predictions for new input configurations.
    """
    return pipeline.predict(X_test)


# ==========================================
# 5. METRICS & PERFORMANCE EVALUATION STEP
# ==========================================
def generate_performance_report(y_true, y_pred, verbose=True):
    """
    Calculates operational accuracy scores and returns metrics for tracking.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    if verbose:
        print("\n================= MODEL PERFORMANCE REPORT ================")
        print(f" Root Mean Square Error (RMSE) : {rmse:.4f} meters")
        print(f" R² Prediction Accuracy Score  : {r2 * 100:.2f}%")
        print("===========================================================")
        
        print("\n[i] Sample Trajectory Analysis Matrix:")
        comparison_df = pd.DataFrame({
            'True Physics Range (m)': y_true.values[:5],
            'AI Predicted Range (m)': y_pred[:5],
            'Absolute Discrepancy (m)': np.abs(y_true.values[:5] - y_pred[:5])
        })
        print(comparison_df.to_string(index=False))
        
    return {"rmse": rmse, "r2": r2}


# ==========================================
# CENTRAL PIPELINE EXECUTION ENGINE
# ==========================================

if __name__ == "__main__":
    print("Executing Isolated Modular Architecture...")
    
    # Define your configuration path variables here
    # (Can be a relative name if in the same folder, or a full system path)
    file_path = "projectile_physics_data.csv"
    
    print(f"[i] Resolving target environment path...")
    print(f"    Target path: {os.path.abspath(file_path)}")
    
    # Run Step 1: Load and Split using the file path
    X_tr, X_te, y_tr, y_te = load_and_split_data(file_path)
    
    # Run Step 2: Initialize parameters 
    physics_model = initialize_pipeline(hidden_layers=(64, 32), max_iter=1000)
    
    # Run Step 3: Train
    trained_model = train_pipeline(physics_model, X_tr, y_tr)
    
    # Run Step 4: Predict
    predictions = run_predictions(trained_model, X_te)
    
    # Run Step 5: Evaluate
    metrics = generate_performance_report(y_te, predictions, verbose=True)

