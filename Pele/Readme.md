# Reconstructing Historical F1 Standings: AI-Hybrid Approach

This project utilizes a **Transformer-LSTM Hybrid Model** enhanced by **XGBoost-based Soft Scaling** to reconstruct historical Formula 1 standings. By simulating "Sprint Races" across F1 history (1950–2024), this model analyzes how modern scoring formats and AI predictions would have altered historical championship outcomes.

---

## Project Overview

### The Challenge

Historical F1 data is inherently noisy and lacks the temporal context required for modern deep learning models. Raw race data fails to capture critical factors such as "driver momentum," "team form," or "track-specific proficiency."

### The Solution

We implemented a **two-stage pipeline** to maximize predictive accuracy:

1.  **Feature Engineering:** Transformed raw data into "V2 Engineered Data" focused on relative pace and form.
2.  **Hybrid Modeling:** A **Long Short-Term Memory (LSTM)** network with an **Attention Mechanism**, initialized using feature importance weights derived from **XGBoost**.

---

## Model Architecture

The architecture is designed to capture the sequential nature of F1 seasons while effectively filtering out historical noise.

### 1. Hybrid Architecture

- **Sequential Memory (LSTM):** A 2-layer LSTM (64 hidden units) captures the temporal trajectory of driver and team performance over the season.
- **Selective Focus (Attention):** A mechanism that dynamically weights historical data, prioritizing recent races (last 3 rounds) and performance on similar circuit types over outdated data.

### 2. XGBoost Soft Scaling (Knowledge Transfer)

To overcome the "black box" initialization problem of deep learning, we utilized XGBoost to calculate feature importance scores beforehand. These scores were mapped to the model's initial weights to accelerate convergence.

- **Team Form:** 0.4506 (Priority 1)
- **Driver Form:** 0.1893
- **Qual Position:** 0.0947
- **Circuit Avg Pos:** 0.0613

---

## Performance & Results

### Feature Engineering Impact

Comparing **V1 (Raw Data)** vs. **V2 (Engineered Data)** using the identical architecture. The engineered features significantly reduced error rates.

| Metric   | V1 (Raw) | V2 (Engineered) | Improvement |
| :------- | :------- | :-------------- | :---------- |
| **MAE**  | 3.46     | **3.00**        | **13.3% ↓** |
| **MSE**  | 17.57    | 13.96           | 20.6% ↓     |
| **RMSE** | 4.19     | 3.74            | 10.7% ↓     |

### Hyperparameter Tuning Impact

Comparing the **Base Hybrid** model vs. the **Tuned Hybrid (XGBoost Weighted)** model.

| Metric   | Base Hybrid | Tuned Hybrid | Improvement |
| :------- | :---------- | :----------- | :---------- |
| **MAE**  | 3.0038      | **2.7526**   | **8.4% ↓**  |
| **MSE**  | 13.9592     | 11.8846      | 14.9% ↓     |
| **RMSE** | 3.7362      | 3.4474       | 7.7% ↓      |

---

## Simulation Outcomes: The "AI-Sprint" Era

Applying this model to reconstruct history resulted in significant changes to World Drivers' Championship (WDC) titles. The AI simulation suggests a more meritocratic distribution of titles based on pure performance data.

### Revised Championship Titles

- **Lewis Hamilton:** 8 Titles (Historical: 7)
- **Michael Schumacher:** 7 Titles
- **Alain Prost:** 7 Titles (Historical: 4)
- **Max Verstappen:** 4 Titles
- **Juan Fangio:** 4 Titles

### Notable Historical Flips

| Year     | Original Champion | AI Predicted Champion |
| :------- | :---------------- | :-------------------- |
| **2010** | Sebastian Vettel  | **Lewis Hamilton**    |
| **2007** | Kimi Räikkönen    | **Fernando Alonso**   |
| **1983** | Nelson Piquet     | **Alain Prost**       |
| **1982** | Keke Rosberg      | **Alain Prost**       |
| **1968** | Graham Hill       | **Denny Hulme**       |

---

## 💻 System Environment

This project was developed and tested on the following system configuration:

| Component         | Specification      |
| :---------------- | :----------------- |
| **OS**            | Ubuntu 20.04 LTS   |
| **Kernel**        | 5.15.0-139-generic |
| **CPU**           | Intel i9-9900KF    |
| **RAM**           | 64GB               |
| **GPU**           | NVIDIA RTX 2080TI  |
| **NVIDIA_Driver** | 570.172.08         |
| **CUDA**          | Version 12.2       |
| **Python**        | 3.11               |

---

## 🛠️ Installation

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
