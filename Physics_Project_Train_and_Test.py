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
"""
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

"""
import time
import matplotlib.pyplot as plt

def generate_scree_plot(X_train, X_test, y_train, y_test):
    """
    Sweeps through varying neuron network sizes, extracts evaluation metrics,
    and constructs a highly detailed dual-axis execution screen plot.
    """
    # Define the range of network complexities to test
    neuron_configurations = [16, 32, 64, 128, 256]
    
    rmse_results = []
    timing_results = []
    
    print("\n[~] Beginning Architecture Parameter Sweep...")
    
    for size in neuron_configurations:
        print(f"    Evaluating architecture: ({size}, 32)... ", end="", flush=True)
        
        # 1. Initialize custom configurations using your modular function
        pipeline = initialize_pipeline(hidden_layers=(size, 32), max_iter=1000)
        
        # 2. Benchmark training performance and execution speeds
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        # 3. Predict and evaluate
        preds = pipeline.predict(X_test)
        report = generate_performance_report(y_test, preds, verbose=False)
        
        # 4. Save metrics across arrays
        rmse_results.append(report["rmse"])
        timing_results.append(elapsed_time)
        print(f"Done! RMSE: {report['rmse']:.4f}m | Time: {elapsed_time:.3f}s")
        
    print("[+] Parameter sweep complete. Rendering plot...")

    # ===========================================================
    # OBJECT-ORIENTED MATPLOTLIB RENDERING (Single Screen Axis Framework)
    # ===========================================================
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Initialize the figure and core axis object
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
    
    # --- Plot Axis 1: Root Mean Square Error (Left Axis) ---
    color_rmse = '#1f77b4'  # Professional Steel Blue
    ax1.set_xlabel('Hidden Layer 1 Capacity (Number of Neurons)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Validation Error (RMSE in Meters)', color=color_rmse, fontsize=12, fontweight='bold', labelpad=10)
    
    # Draw line and scatter markers
    line1 = ax1.plot(neuron_configurations, rmse_results, color=color_rmse, linestyle='-', marker='o', 
                     linewidth=2.5, markersize=8, label='Validation RMSE (Lower is Better)')
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.set_xticks(neuron_configurations)
    
    # Add localized value labels for absolute precision data tracking
    for x, y in zip(neuron_configurations, rmse_results):
        ax1.annotate(f"{y:.3f}m", xy=(x, y), xytext=(0, 8), textcoords='offset points', 
                     ha='center', fontsize=9, fontweight='semibold', color=color_rmse)

    # --- Plot Axis 2: Computational Overhead (Twin Right Axis) ---
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    color_time = '#d62728'  # Deep Crimson Red
    ax2.set_ylabel('Model Training Duration (Seconds)', color=color_time, fontsize=12, fontweight='bold', labelpad=10)
    
    # Draw line and scatter markers
    line2 = ax2.plot(neuron_configurations, timing_results, color=color_time, linestyle='--', marker='s', 
                     linewidth=2.0, markersize=8, label='Training Runtime (Seconds)')
    ax2.tick_params(axis='y', labelcolor=color_time)
    
    # Add runtime labels over the data markers
    for x, y in zip(neuron_configurations, timing_results):
        ax2.annotate(f"{y:.2f}s", xy=(x, y), xytext=(0, -15), textcoords='offset points', 
                     ha='center', fontsize=9, fontweight='semibold', color=color_time)

    # --- Unified Screen Details & Annotations ---
    plt.title('Neural Network Hyperparameter Optimization Scree Plot\nNetwork Topology Error Decay vs. Computational Training Cost', 
              fontsize=14, fontweight='bold', pad=15)
    
    # Merge legends from both independent axis maps into a single unified key window
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True, fontsize=11)
    
    # Clean padding management
    fig.tight_layout()
    
    # Save a high-resolution print copy straight to disk for report documentation sheets
    plt.savefig('nn_architecture_elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# CENTRAL PIPELINE EXECUTION CALL
# ==========================================
if __name__ == "__main__":
    file_path = "projectile_physics_data.csv"
    
    # Ingest using your pre-built modular code pipeline
    X_tr, X_te, y_tr, y_te = load_and_split_data(file_path)
    
    # Spin up the optimization analyzer
    generate_scree_plot(X_tr, X_te, y_tr, y_te)