# Data-Driven Kinematics & Computational Physics

An A2-Level computational physics and machine learning project exploring whether an Artificial Neural Network (ANN) can inherently decipher non-linear aerodynamic kinematics without explicit mathematical programming.

---

## 📌 Project Overview & Thesis

> **Thesis Question:** *Can a simple Artificial Neural Network accurately predict the non-linear landing coordinates of a projectile subjected to quadratic aerodynamic drag without utilizing explicit kinematic equations?*

Standard high school kinematics models projectile trajectories as symmetrical, uniform parabolas using basic equations like $s = ut + \frac{1}{2}at^2$. However, in real-world fluid dynamics, turbulent aerodynamic drag scales with the square of the velocity ($F_d \propto v^2$), making acceleration variable and non-linear. 

This project bridges **Computational Physics** and **Data Science**:
1. **The Simulator:** Built a custom numerical integration engine utilizing **Euler's Method** ($\Delta t = 0.001\text{s}$) to resolve differential equations and generate thousands of chaotic, realistic trajectory data rows.
2. **The AI Brain:** Isolated the underlying physics equations from an Artificial Neural Network (`MLPRegressor`) and trained it purely on tabular inputs and outputs to see if it could "learn" the curve of nature through pattern recognition.

---

## 🔬 The Physics Engine

In a fluid medium like standard air, a moving projectile experiences a dynamic drag force opposing its velocity vector:

$$F_d = \frac{1}{2} \rho v^2 C_d A$$

Because the magnitude of $F_d$ changes dynamically every millisecond alongside velocity, the acceleration is non-linear and lacks a simple algebraic solution. To automate the data generation pipeline, the diagonal force vector is resolved into coordinate acceleration components at every microscopic time step ($\Delta t$):

$$\text{Net } a_x = \frac{-F_{dx}}{m} = \frac{-F_d \cdot \left(\frac{v_x}{v}\right)}{m}$$

$$\text{Net } a_y = -g - \frac{F_{dy}}{m} = -g - \frac{F_d \cdot \left(\frac{v_y}{v}\right)}{m}$$

The simulation steps sequentially through time, iteratively updating velocities and spatial displacements ($x, y$) until the vertical boundary constraint ($y \le 0$) is violated.

---

## 📊 Experimental Variables

### Independent Variables (AI Model Features)
* **Launch Velocity ($v_0$):** Randomized uniformly from $5.0 \text{ m/s}$ to $50.0 \text{ m/s}$.
* **Launch Angle ($\theta$):** Randomized uniformly from $10.0^\circ$ to $80.0^\circ$.

### Dependent Variable (AI Model Target)
* **Horizontal Range ($R$):** The ultimate $x$-coordinate intersection value mapping exactly where the projectile lands on the ground ($y = 0$).

### Controlled Constants (Static Environment)
| Variable | Description | Value |
| :--- | :--- | :--- |
| $g$ | Acceleration due to gravity | $9.81 \text{ m/s}^2$ |
| $m$ | Projectile Mass (Standard Baseball) | $0.145 \text{ kg}$ |
| $r$ | Ball Radius | $0.037 \text{ m}$ |
| $A$ | Cross-Sectional Area ($\pi r^2$) | $\approx 0.0043 \text{ m}^2$ |
| $C_d$ | Drag Coefficient (Smooth Sphere) | $0.5$ |
| $\rho$ | Air Density (Sea Level Standard) | $1.225 \text{ kg/m}^3$ |

---

## 🛠️ Tech Stack & Environment

* **Language:** Python 3
* **Libraries Utilized:**
  * `NumPy`: High-performance vector mathematics and trigonometry operations.
  * `Pandas`: Data structure serialization, analysis, and matrix export to tabular CSV format.
  * `Scikit-Learn`: Implements preprocessing mechanisms (`StandardScaler`), data splitting partitions, and the Multi-Layer Perceptron architecture (`MLPRegressor`).
  * `Matplotlib`: Renders true trajectory pathways overlaid against neural network predictions for final validation graphs.

---

## 🏗️ Core Architecture & Pipeline

The pipeline is split into structural milestones:

### 1. Preprocessing & Feature Scaling
Because launch angles ($10^\circ \text{ to } 80^\circ$) operate on a larger numerical scale than launch velocities ($5 \text{ to } 50 \text{ m/s}$), the network would naturally over-index on angles during gradient descent weight adjustment. We mitigate this using Z-score standardization via `StandardScaler`:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$

*The scaler is fitted strictly on the training partition to eliminate data leakage.*

### 2. Neural Network Configuration
* **Architecture:** Multi-Layer Perceptron (MLP) with two hidden layers configured at `(64, 32)` nodes respectively.
* **Activation Function:** Rectified Linear Unit (`relu`) to map non-linear continuous regression paths.
* **Optimization Solver:** `adam` for dynamic learning rate adjustments.

### 3. Error Metrics Evaluation
Model validation uses **Root Mean Square Error (RMSE)** to measure performance, directly mirroring the mathematical RMS concepts used to calculate effective value magnitudes across physics tracks:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}} - y_{\text{pred}})^2}$$

---

## 🚀 How To Run

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/projectile-ai-physics.git](https://github.com/yourusername/projectile-ai-physics.git)
   cd projectile-ai-physics
