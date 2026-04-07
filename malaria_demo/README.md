# Malaria modeling Starsim demo

> [!WARNING]
> **This is a demonstration project only. It is not intended to be used as a functional malaria model.**

Technology demonstrator for modeling malaria transmission dynamics in R, Python, and Starsim. Uses a Ross-Macdonald-type metapopulation ODE model incorporating human mobility between patches (unions/upazilas).

Based on the following paper:

> Mahmud, A.S., Chang, MC., Citron, D.T. et al. Identifying malaria elimination strategies in the presence of human movement in Bangladesh. Commun Med 5, 461 (2025). https://doi.org/10.1038/s43856-025-01145-6

Converted from R → Python → Starsim using [Starsim-AI](https://ai.starsim.org).

## Model evolution

The project tracks the progressive simplification and translation of a malaria model through four stages:

### 1. Original Mahmud et al. model
The original R codebase from the published analysis. Runs a multi-patch Ross-Macdonald ODE model on real Bangladesh data (118 unions), with perturbation experiments parallelized across 80 HPC cores. Uses a custom Rcpp package for fast matrix multiplication. Not included in the repo.

### 2. Simplified R demo (`demo_sim.R`)
A self-contained R script that distills the core model into a 3-patch toy example with **seasonal mosquito density forcing**. No external data or Rcpp dependencies required. Computes baseline mosquito density analytically from equilibrium prevalence, then runs a 5-year seasonal simulation using `deSolve`. Produces plots of prevalence, incidence, mosquito density, and EIR.

### 3. Python translation (`demo_sim.py`)
Direct translation of `demo_sim_endemic.R` to Python using scipy (`solve_ivp` with LSODA), numpy, and matplotlib. Same model, same 3-patch synthetic data, same seasonal forcing. Serves as a stepping stone toward integration with Python-based modeling frameworks.

### 4. Starsim version (`demo_sim_ss.py`)
Reimplements the model as a [Starsim](https://starsim.org) `Module` with Euler integration. Can be embedded in an `ss.Sim` for integration with other Starsim disease modules. Uses `sciris` for plotting and parameter management. Importable as a library (`from demo_sim_ss import Malaria_SS`) or can be run standalone.

## Model parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `a` | Biting rate | 0.3 |
| `b` | Mosquito-to-human transmission probability | 0.1-0.54 |
| `c` | Human-to-mosquito transmission probability | 0.214-0.423 |
| `r` | Recovery rate | 1/150 |
| `mu` | Mosquito death rate | 1/10 |
| `tau` | Extrinsic incubation period | 10 |

## Running

**R scripts** (require `deSolve`, `ggplot2`):
```bash
Rscript demo_sim.R          # seasonal 3-patch demo
```

**Python script**:
```bash
python demo_sim.py                  # scipy ODE version
```

**Starsim version**:
```bash
python demo_sim_ss.py               # Starsim module version
```

Or use as a library:
```python
import starsim as ss
from demo_sim import Malaria_SS

mal = Malaria_SS()
sim = ss.Sim(modules=mal, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
sim.run()
mal.plot()
```
