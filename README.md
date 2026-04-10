# Starsim-NGM

This repo contains four Starsim implementations of malaria models, from simple compartmental through to complex agent-based ones.

## Quick start
1) Create a virtual environment (optional)
2) Install: `pip install -e .`
3) Run: `python run_malaria.py`

## Model comparison

### Simple compartmental: `malaria_demo`

This is a direct translation of the Mahmud et al. (2025) compartmental malaria model into Starsim.

### Simple agent-based: `malaria_sslaser`

This is a direct translation of the LASER-NGM malaria model into Starsim.

### Hybrid compartmental/agent-based: `malaria_hybrid`

This is a Ross-Macdonald malaria model implemented in Starsim with four interchangeable module combinations: ODE or agent-based (ABM) humans crossed with ODE or ABM mosquitoes. All four share the same `sim.c` compartment bus so any human module works with any mosquito module. Mosquitoes use SEI dynamics with seasonal forcing.

### Complex agent-based: `malariasim`

This is a more complete agent-based malaria model, drawing on EMOD-Malaria among other sources.
