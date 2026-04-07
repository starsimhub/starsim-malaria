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

This is a loose translation of the [Gervas et al. 2025](https://link.springer.com/article/10.1186/s12936-025-05633-x) compartmental malaria model into Starsim. It includes both compartmental and agent-based options for both humans and mosquitoes, plus simple interventions (ITNs and vaccines).

### Complex agent-based: `malariasim`

This is a more complete agent-based malaria model, drawing on EMOD-Malaria among other sources.
