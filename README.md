# Spatio-Temporal Solar Power Forecasting with Gated Identity T-GCNs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A benchmark framework for multi-node spatio-temporal solar power forecasting using real-world data from the **Desert Knowledge Australia Solar Centre (DKASC)** in Alice Springs. 

This repository implements and evaluates a **Proposed Gated Identity Temporal Graph Convolutional Network (Gated T-GCN)** alongside 5 standard baselines. The core innovation addresses **localized weather shocks** (e.g., cloud cover over a single physical array) where standard graph spatial aggregation causes spatial oversmoothing and error propagation.

---

## Key Highlights

- **Dynamic Identity Preservation Gate:** Uses a feature-wise gating vector $g \in [0, 1]^d$ to dynamically control the balance between spatial neighborhood blending and local node identity preservation during localized volatility shocks.
- **Leak-Free Pipeline:** Pre-tensorized sliding-window data loader calculating mean and standard deviation strictly on training splits to eliminate temporal data leakage.
- **Physical Scale Evaluation:** Evaluates models on actual raw physical scale (kW) using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE with daytime masking), and $R^2$ Score.
- **Multi-Baseline Benchmarking:** Includes non-parametric (Historical Average), spatial (GCN), recurrent (GRU), spatio-temporal (Vanilla T-GCN), and attention-driven (A3T-GCN) baseline architectures.

---

## Mathematical Formulation

Standard spatio-temporal graph networks aggregate neighboring node representations via graph convolution:

$$H_{spatial} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X W_s \right)$$

During localized weather anomalies, neighborhood aggregation can introduce noise from unaffected adjacent nodes. Our **Gated Identity T-GCN** computes a feature-wise identity gate $g$:

$$g = \sigma(W_g X + b_g)$$

$$H_{final} = g \odot X_{shortcut} + (1 - g) \odot H_{spatial}$$

The gated hidden state $H_{final}$ is then passed through a GRU cell for sequential temporal updates.

---

#### 1. Volatility Case Study
![Volatility Case Study](figure_1_1_spatial_anomaly_topology.png)
*Figure 1: Time-series evaluation comparing actual solar power generation versus predicted outputs during localized cloud transient shocks.

#### 1. Volatility Case Study
![Volatility Case Study](figure_1_2_gated_routing_comparison.png)
*Figure 1: Time-series evaluation comparing actual solar power generation versus predicted outputs during localized cloud transient shocks.

## Repository Architecture

```text
.
├── data/
│   ├── __init__.py
│   └── build_dataset.py       # Raw DKASC CSV parser & synthetic simulation generator
├── models/
│   ├── __init__.py
│   └── tgcn_models.py         # PyTorch Gated T-GCN & baseline model definitions
├── utils/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch dataset & sliding-window pre-tensorizer
│   ├── graph_utils.py         # Correlation matrix & normalized adjacency solver
│   ├── metrics.py             # MAE, RMSE, MAPE, R2 evaluation engine
│   └── plotting.py            # Benchmark visualization generator
├── train.py                   # Master pipeline, training loop & evaluation execution
└── README.md                  # Project documentation
````
## Dataset

The framework benchmarks 6 physical solar PV arrays from the **Desert Knowledge Australia Solar Centre (DKASC)** facility in Alice Springs, Australia:

| Node Index | System ID | Array Capacity | Installation & Orientation |
| :---: | :---: | :---: | :--- |
| **Node 1** | `16A` | 2.0 kW | BP Solar (Fixed North) |
| **Node 2** | `16B` | 2.0 kW | BP Solar (Horizontal / Flat) |
| **Node 3** | `16C` | 2.0 kW | BP Solar (Fixed East) |
| **Node 4** | `16D` | 2.0 kW | BP Solar (Fixed West) |
| **Node 5** | `3` | 5.0 kW | BP Solar (Roof Mounted) |
| **Node 6** | `11` | 5.0 kW | BP Solar (Ground Mounted) |

The preprocessing pipeline extracts, cleans, and synchronizes the following feature channels at a 5-minute sampling resolution across all nodes:
* **Active Power (kW)** (*Target variable*)
* **Global Horizontal Radiation (GHI)**
* **Diffuse Horizontal Radiation (DHI)**
* **Ambient Weather Temperature (°C)**

---

## Quick Start

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/gated-tgcn-solar.git](https://github.com/your-username/gated-tgcn-solar.git)
cd gated-tgcn-solar
pip install torch pandas numpy scikit-learn matplotlib seaborn psutil
````
### 2. Dataset Setup

Place your 6 raw DKASC CSV files into the root directory of the repository:
* `16A_BP_Solar_North.csv`
* `16B_BP_Solar_Flat.csv`
* `16C_BP_Solar_East.csv`
* `16D_BP_Solar_West.csv`
* `3_BP_Solar_Roof.csv`
* `11_BP_Solar_Ground.csv`

> **Synthetic Fallback:** If raw files are absent, executing `train.py` automatically constructs a multi-node synthetic dataset matching real PV capacity and directional orientation profiles to allow immediate sandbox execution.

---

### 3. Execution

Run the complete data processing, training, metric calculation, and visualization pipeline:

```bash
python train.py

```
## Evaluated Models & Baselines

The framework benchmarks the **Proposed Gated Identity T-GCN** against five comparative spatial, temporal, and spatio-temporal baseline architectures:

| Model Architecture | Category | Spatio-Temporal Mechanism | Description |
| :--- | :--- | :--- | :--- |
| **Proposed Gated T-GCN** | Spatio-Temporal | GCN + Feature Identity Gate ($g$) + GRU | Dynamically decouples neighborhood spatial aggregation during local volatility shocks using a learned feature-wise identity shortcut. |
| **Vanilla T-GCN** | Spatio-Temporal | GCN + GRU Cell | Standard Temporal Graph Convolutional Network combining fixed normalized graph convolution with a Recurrent GRU Cell. |
| **A3T-GCN** | Spatio-Temporal | GCN + GRU + Temporal Attention | Soft Attention-based Temporal Graph Convolutional Network designed to re-weight historical time slots dynamically. |
| **Standard GCN** | Spatial Only | Graph Convolution (Static) | Multi-layer Graph Convolutional Network operating across spatial dimensions without sequential memory. |
| **Standard GRU** | Temporal Only | Recurrent Gated Unit | Isolated Gated Recurrent Unit trained independently per PV node without spatial connectivity information. |
| **Historical Average** | Non-Parametric | Statistical Rolling Mean | Non-learned baseline forecasting future timesteps based on the mean of the historical observation window. |

## Output Metrics & Artifacts

Upon execution completion, `train.py` evaluates model predictions on actual raw physical scales (kW) using daytime-filtered metrics. The output is printed directly to the console:

```text
=====================================================================================
COMPREHENSIVE STATISTICAL VALIDATION TABLE
=====================================================================================
Proposed Gated T-GCN   | MAE:  0.082 kW | RMSE:  0.185 kW | MAPE:  4.12% | R2: 0.9845
Vanilla T-GCN          | MAE:  0.114 kW | RMSE:  0.241 kW | MAPE:  6.85% | R2: 0.9712
A3T-GCN                | MAE:  0.098 kW | RMSE:  0.210 kW | MAPE:  5.40% | R2: 0.9790
Standard GCN           | MAE:  0.165 kW | RMSE:  0.312 kW | MAPE: 10.21% | R2: 0.9520
Standard GRU           | MAE:  0.102 kW | RMSE:  0.218 kW | MAPE:  5.92% | R2: 0.9775
Historical Average     | MAE:  0.245 kW | RMSE:  0.450 kW | MAPE: 18.30% | R2: 0.8910
=====================================================================================
```
### Generated Figures

The master script automatically exports four publication-ready figures to the root directory to visualize model performance, spatial decoupling behavior, and training dynamics:

* `reviewer_plot_a_volatility.png`: Time-series evaluation comparing actual generation versus predicted outputs during localized cloud transient shocks.
* `reviewer_plot_b_gate_analysis.png`: Dynamic inspection of feature-wise identity gate activations ($g$) demonstrating automated spatial decoupling during high-volatility events.
* `reviewer_plot_c_residuals.png`: Distribution boxplots of absolute prediction residuals ($|y - \hat{y}|$) across all evaluated benchmark architectures.
* `reviewer_plot_d_convergence.png`: Training loss convergence trajectories over training epochs of proposed Gated T-GCN model.

The master script automatically exports four publication-ready figures to visualize model performance, spatial decoupling behavior, and training dynamics.

#### 1. Volatility Case Study
![Volatility Case Study](reviewer_plot_a_volatility.png)
*Figure 1: Time-series evaluation comparing actual solar power generation versus predicted outputs during localized cloud transient shocks.

---

#### 2. Identity Gate Activation Analysis
![Gate Activation Analysis](reviewer_plot_b_gate_analysis.png)
*Figure 2: Dynamic inspection of feature-wise identity gate activations ($g$) demonstrating automated spatial decoupling during high-volatility events.

---

#### 3. Residual Distribution Analysis
![Residual Distribution](reviewer_plot_c_residuals.png)
*Figure 3: Distribution boxplots of global absolute prediction residuals ($|y - \hat{y}|$) across all evaluated benchmark architectures.

---

#### 4. Training Convergence Trajectories
![Training Convergence](reviewer_plot_d_convergence.png)
*Figure 4: Training loss convergence trajectories across training epochs for all neural network baselines.
