"""
Malaria modeling framework for Starsim.

Modules
-------
MalariaHumanBase
    Abstract base class providing shared biting infrastructure and observable
    states (rel_biting_rate, infectiousness, hrp2_pg, fever_F, symptom_state).
    Not intended for direct use; inherit to create a new human model variant.

MalariaHumanSIS
    SIS disease module: susceptible → infected → susceptible.
    Transmission is driven by EIR from a companion mosquito module.

MalariaHumanSEIS
    SEIS disease module: susceptible → exposed → infected → susceptible.
    Adds an explicit incubation period (~12 days) before agents become
    infectious. Drop-in replacement for MalariaHumanSIS.

MosquitoesEirOnly
    Simple mosquito module with no agents. EIR is a fixed parameter (scalar
    or monthly array with linear interpolation). Use when mosquito population
    dynamics are not needed.

MosquitoesAdultFemaleAgentsOnly
    SEI agent-based mosquito module. Population is dynamic and seasonal,
    driven by a monthly emergence rate. Use when explicit mosquito lifecycle
    dynamics are needed.

Sim
    Subclass of ``ss.Sim`` that assembles a ready-to-run malaria simulation
    from the modules above.

Interface contract
------------------
The human and mosquito modules are fully decoupled — neither imports the
other. They communicate each timestep through a shared ``NodeState`` object,
which both modules discover at runtime by searching for the ``_is_node_state``
duck-type marker.

    Mosquito → Human  :  ``eir_indoor`` & ``eir_outdoor``
    Human   → Mosquito:  ``p_inf_indoor`` & ``p_inf_outdoor``
    
Any user-defined module that reads/writes the NodeState fields will work as
a drop-in replacement for either side.

Quick start
-----------
EIR-only mode (no mosquito agents)::

    import malariasim as ms
    sim = ms.Sim(mosquito='eir_only')
    sim.run()
    sim.plot()

Agent-based mosquito mode::

    sim = ms.Sim(mosquito='agents')
    sim.run()
    sim.plot()

Custom parameters::

    sim = ms.Sim(
        mosquito='agents',
        n_agents=20_000,
        start=2010,
        stop=2025,
        human_pars=dict(b=0.4, c=0.3, init_prev=ss.bernoulli(p=0.05)),
        mosquito_pars=dict(
            emergence=[0.1, 0.1, 0.2, 0.5, 1.5, 3.0, 3.0, 1.5, 0.5, 0.2, 0.1, 0.1],
            life_expectancy=14.0,
        ),
    )
    sim.run()
"""

import starsim as ss

from .human_base        import MalariaHumanBase, SYMPTOM_NONE, SYMPTOM_CLINICAL, SYMPTOM_SEVERE, SYMPTOM_DISEASE_DEATH
from .human_sis         import MalariaHumanSIS
from .human_seis        import MalariaHumanSEIS
from .human_ode         import MalariaHumanODE
from .mosquito_eir_only import MosquitoesEirOnly
from .mosquito_agents   import MosquitoesAdultFemaleAgentsOnly
from .mosquito_ode      import MosquitoesODE
from .node_grid         import NodeGrid
from .node_state        import NodeState
from .sim               import Sim
from .demographics      import compute_mortality_schedule
from .defaults          import (
    AGE_BIN_EDGES, AGE_BIN_FRACTIONS,
    URBAN_EIR, PERI_URBAN_EIR, RURAL_EIR,
    URBAN_EMERGENCE, PERI_URBAN_EMERGENCE, RURAL_EMERGENCE,
    URBAN_F_INDOOR, PERI_URBAN_F_INDOOR, RURAL_F_INDOOR,
    make_steady_state_demographics, make_age_data, make_mortality_dataframe,
)
from .performance       import PerformanceMonitor
from .results           import AgeBinPopulation, MalariaMultiSim

__all__ = [
    'MalariaHumanBase',
    'MalariaHumanSIS',
    'MalariaHumanSEIS',
    'MalariaHumanODE',
    'MosquitoesEirOnly',
    'MosquitoesAdultFemaleAgentsOnly',
    'MosquitoesODE',
    'NodeGrid',
    'NodeState',
    'Sim',
    'compute_mortality_schedule',
    'AGE_BIN_EDGES', 'AGE_BIN_FRACTIONS',
    'URBAN_EIR', 'PERI_URBAN_EIR', 'RURAL_EIR',
    'URBAN_EMERGENCE', 'PERI_URBAN_EMERGENCE', 'RURAL_EMERGENCE',
    'URBAN_F_INDOOR', 'PERI_URBAN_F_INDOOR', 'RURAL_F_INDOOR',
    'make_steady_state_demographics', 'make_age_data', 'make_mortality_dataframe',
    'PerformanceMonitor',
    'AgeBinPopulation',
    'MalariaMultiSim',
    'SYMPTOM_NONE',
    'SYMPTOM_CLINICAL',
    'SYMPTOM_SEVERE',
    'SYMPTOM_DISEASE_DEATH',
]
