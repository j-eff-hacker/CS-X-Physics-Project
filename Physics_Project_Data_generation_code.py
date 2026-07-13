import numpy as np
import pandas as pd
import math
import os

def simulate_trajectory(v0, angle_deg):
    """
    Simulates a single projectile launch factoring in quadratic air resistance
    using Euler's method for numerical integration.
    
    Inputs:
        v0 (float): Initial launch velocity in m/s
        angle_deg (float): Launch angle in degrees
        
    Returns:
        float: Total horizontal distance traveled (Range R) when y hits 0.
    """
    # --- Controlled Constants (Fixed Environment) ---
    g = 9.81              # Acceleration due to gravity (m/s^2)
    mass = 0.145          # Mass of a standard baseball (kg)
    r = 0.037             # Radius of the ball (m)
    A = math.pi * (r**2)  # Cross-sectional area (m^2)
    Cd = 0.5              # Drag coefficient (dimensionless sphere profile)
    rho = 1.225           # Standard air density at sea level (kg/m^3)
    dt = 0.001            # Microsecond time step (seconds)
    
    # --- State Initialization ---
    x, y = 0.0, 0.0
    angle_rad = math.radians(angle_deg)
    
    # Resolve initial velocity into vector components
    vx = v0 * math.cos(angle_rad)
    vy = v0 * math.sin(angle_rad)
    
    # --- The Physics Numerical Integration Loop ---
    while y >= 0:
        # 1. Calculate current total velocity magnitude
        v = math.sqrt(vx**2 + vy**2)
        if v == 0: 
            break  # Prevent division by zero if the object drops dead dead
            
        # 2. Calculate the total magnitude of quadratic aerodynamic drag force
        F_drag = 0.5 * rho * (v**2) * Cd * A
        
        # 3. Use trig components to find how much drag opposes each axis
        F_drag_x = F_drag * (vx / v)
        F_drag_y = F_drag * (vy / v)
        
        # 4. Net forces mapped to Newton's Second Law (a = F/m)
        ax = (-F_drag_x) / mass
        ay = (-mass * g - F_drag_y) / mass  # Gravity and drag both pull down during descent
        
        # 5. Euler integration steps (Updating state variables for the next millisecond)
        vx += ax * dt
        vy += ay * dt  # Fixed: Changed from vy += vy + (ay * dt)
        
        x += vx * dt          # Update horizontal displacement
        y += vy * dt          # Update vertical displacement         # Update vertical displacement
        
    return x


def generate_physics_dataset(num_samples=5000):
    """
    Automates the physics engine to run thousands of times with randomized
    inputs to compile a raw dataset for our machine learning pipeline.
    """
    raw_data = []
    print(f"Executing {num_samples} discrete physics loops...")
    
    # Setting a random seed ensures reproducibility across runs
    np.random.seed(42) 
    
    for i in range(num_samples):
        # Generate random inputs within our defined experimental limits
        initial_velocity = np.random.uniform(5.0, 50.0)   # 5 to 50 m/s
        launch_angle = np.random.uniform(10.0, 80.0)     # 10 to 80 degrees
        
        # Pass the parameters into our non-linear physics engine
        final_range = simulate_trajectory(initial_velocity, launch_angle)
        
        # Append rows directly to our matrix dictionary
        raw_data.append({
            'initial_velocity': initial_velocity,
            'launch_angle': launch_angle,
            'landing_distance': final_range
        })
        
    # Compile the matrix directly into a clean structured Pandas DataFrame
    df = pd.DataFrame(raw_data)
    print("Dataset generation complete!")
    return df

# --- Execution ---
if __name__ == "__main__":
    # Create the structured spreadsheet matrix containing 5,000 unique runs
    dataset = generate_physics_dataset(num_samples=5000)
    
    # Inspect the head of the pipeline matrix
    print("\nFirst 5 rows of our generated dataset features and labels:")
    print(dataset.head())

    # Define the file name
    filename = "projectile_physics_data.csv"
    
    # 1. Save it to your disk properly
    dataset.to_csv(filename, index=False) # index=False prevents an extra unnamed row column
    
    # 2. Get and print the absolute path so you know exactly where it is on your machine
    absolute_path = os.path.abspath(filename)
    print(f"\n✅ Dataset successfully saved to file!")
    print(f"Absolute File Path: {absolute_path}")
