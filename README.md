# HEM-DeepRL-v2 — Home Energy Management with Deep Reinforcement Learning

A Deep Reinforcement Learning framework for optimizing residential energy management. The agent learns to minimize electricity costs by intelligently scheduling battery charging from solar panels (PV) and the smart grid (SG), while ensuring the electric vehicle (EV) is always ready for daily use.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Data Format](#data-format)
- [Algorithms](#algorithms)
- [Parameter Settings](#parameter-settings)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Overview

This project models a smart home equipped with:

- A **home battery** (100 kWh capacity)
- An **electric vehicle battery** (100 kWh capacity, 11 kWh daily consumption)
- A **photovoltaic (PV) solar panel** system
- A connection to the **smart grid (SG)**

A Deep RL agent is trained using **Proximal Policy Optimization (PPO)** to decide, at each hour, whether to charge the home or EV battery from either PV or the grid — or take no action. The goal is to **minimize the total annual electricity cost** while meeting household demand and keeping the EV charged.

---

## Project Structure

```
HEM-DeepRL-v2/
├── main.py                  # Entry point: training, testing, and plotting
├── algos/
│   ├── mcPPO.py             # Monte Carlo PPO (Actor-Critic)
│   └── DQN.py               # Double DQN with experience replay
├── pycode/
│   ├── smart_home.py        # Gym-style smart home environment
│   ├── constant.py          # Action definitions and system specifications
│   ├── processed_data.py    # Data access layer for PV, prices, and consumption
│   ├── plotter.py           # Visualization utilities
│   └── train_dqn.py         # DQN training script
├── data/
│   ├── H4.csv               # Household power consumption (minute-level)
│   ├── PV.csv               # Photovoltaic production data
│   └── Prices.csv           # Smart grid electricity prices
├── trained_models/
│   └── neural_network_trained.h5   # Pre-trained PPO model
├── generated/               # Reward logs from training runs
└── results/                 # Output plots
```

---

## Environment

The smart home environment follows the **OpenAI Gym** interface.

### State Space (5 dimensions, normalized to [0, 1])

| Feature             | Description                        |
|---------------------|------------------------------------|
| `month`             | Current month of the year          |
| `day`               | Current day of the month           |
| `time`              | Current hour of the day            |
| `home_battery`      | Home battery state of charge       |
| `ev_battery`        | EV battery state of charge         |

### Action Space (5 discrete actions)

| Action | Description                          |
|--------|--------------------------------------|
| 0      | No action                            |
| 1      | Charge home battery from PV          |
| 2      | Charge home battery from smart grid  |
| 3      | Charge EV battery from PV            |
| 4      | Charge EV battery from smart grid    |

### Reward Design

- **Primary**: Negative electricity cost (in euros) for the current time step
- **Penalty (-10)**: EV not sufficiently charged by 6:00 AM
- **Penalty (-10)**: Attempting to charge the EV during work hours (6:00–18:00)

### Episode

Each episode spans **720 time steps** (24 hours x 30 days), starting from a random date and battery state.

---

## Data Format

The model reads three CSV files from the `data/` directory. You can replace them with your own data as long as you follow the format described below. All files use **semicolon (`;`)** as the delimiter.

### `PV.csv` — Photovoltaic Production

Hourly solar panel output in kWh. Must contain **8,760 rows** (365 days x 24 hours). Decimal separator: **dot (`.`)**.

```
Time;P_PV_
01.01.16 00:00;0
01.01.16 01:00;0
01.01.16 06:00;0.523
...
```

### `Prices.csv` — Smart Grid Electricity Prices

Hourly electricity price in EUR/MWh. Must contain **8,760 rows**. Decimal separator: **comma (`,`)**. The code converts prices from EUR/MWh to EUR/kWh internally.

```
Time;Price;;
01.01.16 00:00;22,39;;
01.01.16 01:00;20,59;;
...
```

### `H4.csv` — Household Power Consumption

Minute-level power consumption in kWh. Must contain **525,600 rows** (365 days x 24 hours x 60 minutes). The code aggregates every 60 rows into hourly values. Decimal separator: **comma (`,`)**.

```
Time;Power
01.01.17 00:00;0,00129330485928748
01.01.17 00:01;0,00123732723485149
...
```

### Using Custom Data

To use your own data:
1. Prepare your CSV files following the column names and formats above
2. Ensure the data covers a full year (365 days)
3. Place the files in the `data/` directory with the same filenames (`PV.csv`, `Prices.csv`, `H4.csv`)
4. Adjust the system parameters in `pycode/constant.py` if your battery specs differ

---

## Algorithms

### Proximal Policy Optimization (PPO)
- **Actor-Critic** architecture with two 64-unit hidden layers
- Supports both **discrete** and **continuous** action spaces
- Clipped surrogate objective (clip value = 0.2)
- Discount factor (gamma) = 0.99
- Network updates every 5 episodes

### Double DQN
- Two 64-unit hidden layers with **soft target updates** (tau = 0.005)
- Experience replay buffer (size = 2000)
- Epsilon-greedy exploration with decay (0.995)
- Discount factor (gamma) = 0.95

---

## Parameter Settings

Below is a detailed reference of all configurable parameters in the project.

### System Specifications (`pycode/constant.py`)

| Parameter              | Value   | Description                                      |
|------------------------|---------|--------------------------------------------------|
| `HOME_CAPACITY`        | 100 kWh | Maximum capacity of the home battery             |
| `EV_CAPACITY`          | 100 kWh | Maximum capacity of the EV battery               |
| `EV_DAILY_CONSUME`     | 11 kWh  | Energy the EV consumes per day                   |
| `MAX_CHARGE_FOR_HOUR`  | 11 kWh  | Maximum energy that can be charged in one hour   |
| `EV_CHARGE_WINDOW`     | [18, 6] | Hours when the EV is available for charging      |
| `MAX_STEP_HOURS`       | 720     | Episode length in hours (24h x 30 days)          |

### PPO Hyperparameters (`algos/mcPPO.py`)

| Parameter              | Value   | Description                                      |
|------------------------|---------|--------------------------------------------------|
| `gamma`                | 0.99    | Discount factor for cumulative rewards           |
| `batch_size`           | 128     | Mini-batch size for network updates              |
| `epoch`                | 10      | Number of optimization epochs per update         |
| `clip_val`             | 0.2     | PPO clipping range for the surrogate objective   |
| `sigma`                | 1.0     | Initial exploration noise (continuous only)       |
| `exploration_decay`    | 1.0     | Sigma decay rate (no decay by default)           |
| Hidden layers          | 2 x 64  | Actor and critic network architecture            |
| Optimizer              | Adam    | Optimizer with default learning rate             |
| Update frequency       | Every 5 episodes | Buffer is cleared after each update     |

### DQN Hyperparameters (`algos/DQN.py`)

| Parameter              | Value   | Description                                      |
|------------------------|---------|--------------------------------------------------|
| `gamma`                | 0.95    | Discount factor for future rewards               |
| `batch_size`           | 32      | Mini-batch size sampled from replay buffer       |
| `memory_size`          | 2000    | Maximum replay buffer capacity                   |
| `exploration_rate`     | 1.0     | Initial epsilon for epsilon-greedy exploration   |
| `exploration_decay`    | 0.995   | Epsilon decay rate per episode                   |
| `tau`                  | 0.005   | Soft update rate for the target network          |
| Hidden layers          | 2 x 64  | Network architecture                             |
| Optimizer              | Adam    | Optimizer with default learning rate             |

### Reward Function Parameters (`pycode/smart_home.py`)

| Parameter              | Value   | Description                                      |
|------------------------|---------|--------------------------------------------------|
| Cost penalty           | `-cost` | Negative of electricity cost at each step        |
| EV not ready penalty   | -10     | Applied at 6:00 AM if EV charge < daily need     |
| EV charge during work  | -10     | Applied if EV is charged between 6:00–18:00      |

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wa7med/HEM-DeepRL-v2.git
   cd HEM-DeepRL-v2
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow numpy pandas matplotlib gym
   ```

---

## Usage

All commands should be run from the project root directory.

### Train a New Model

```bash
python main.py -train_model <number_of_episodes>
```

> 300 episodes is typically sufficient for convergence.

**Example:**
```bash
python main.py -train_model 300
```

The trained model will be saved to `trained_models/neural_network_trained.h5`.

### Test a Trained Model

```bash
python main.py -test_model
```

A pre-trained model is included in the `trained_models/` directory.

### Plot Results

```bash
python main.py -plot_graph <graph_type>
```

Available graph types: `money_spent`, `battery_charge`, `reward_function`, `demands`, `generation`

---

## Results

### Money Spent (Trained vs. Random)

Comparison of monthly electricity costs between the trained PPO agent (**171.79 EUR/year**) and a random policy (**1141.75 EUR/year**) — an **85% cost reduction**.

```bash
python main.py -plot_graph money_spent
```

![Money Spent](results/money_spent.png)

### Battery Charging Strategy

The agent learns to charge from the grid during **off-peak hours** (midnight, late evening) and leverage **PV during daytime** — aligning charging with low prices.

```bash
python main.py -plot_graph battery_charge
```

![Battery Charge](results/battery_charge.png)

### Reward Function Convergence

The PPO agent converges within approximately **150 episodes**, with the reward stabilizing near zero (minimal cost).

```bash
python main.py -plot_graph reward_function
```

![Reward Function](results/reward_function.png)

### House Demands vs. Battery Charge

Hourly household energy demand overlaid with battery charging activity.

```bash
python main.py -plot_graph demands
```

![Demands](results/demands.png)

### Energy Generation Sources

Breakdown of energy supply by source: battery discharge, PV generation, and grid imports.

```bash
python main.py -plot_graph generation
```

![Generation](results/generation.png)

---

## License

This project is developed as part of a Master's thesis research.
