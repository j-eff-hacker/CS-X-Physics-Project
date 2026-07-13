from Physics_Project_Train_and_Test import *

import math

def traditional_predict_landing(v0, theta_deg, dt=0.005):
    """
    Predicts the horizontal landing range using a traditional iterative loop.
    No models, no weights—just basic physics rules, loops, and conditions.
    """
    g = 9.81         # acceleration due to gravity (m/s^2)
    c = 0.005        # quadratic drag coefficient constant
    
    # Convert angle to radians for trigonometry
    theta_rad = math.radians(theta_deg)
    
    # Initialize state variables
    x, y = 0.0, 0.0
    vx = v0 * math.cos(theta_rad)
    vy = v0 * math.sin(theta_rad)
    
    # Condition: Keep loop running while projectile is in the air
    while y >= 0.0:
        v = math.sqrt(vx**2 + vy**2)  # Current absolute velocity
        
        # Calculate net deceleration components
        ax = -c * v * vx
        ay = -g - (c * v * vy)
        
        # Update positions using basic numerical integration steps
        x += vx * dt
        y += vy * dt
        
        # Update velocity vectors for the next iteration
        vx += ax * dt
        vy += ay * dt
        
        # Safety terminal condition to prevent infinite loops on bad data
        if x > 5000:
            break
            
    return round(x, 2)

# Quick sanity check printout
print(f"Traditional Prediction (45m/s at 45°): {traditional_predict_landing(45, 45)} meters")
