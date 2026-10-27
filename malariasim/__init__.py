"""
Malaria modeling framework for Starsim

The model is one declared disease, ``ms.Malaria``: people (SIS or SEIS) infected by the bites
of mosquitoes, which are counts ('ode'), individuals written as code ('agents', ``ms.MosquitoAgents``),
or an EIR given directly ('eir_only'). Being declared, the same model runs as agents or as an ODE
(``solver='abm'`` or ``solver='ode'``). Nodes are a patch stratification (``ss.geo.geography``).

``ms.Sim`` assembles a ready-to-run simulation from these.

Quick start
-----------
EIR given, no mosquitoes::

    import malariasim as ms
    sim = ms.Sim(mosquito='eir_only')
    sim.run()
    sim.plot()

Mosquitoes as counts, run as an ODE::

    sim = ms.Sim(mosquito='ode', solver='ode', init_prev=0.1)

Three nodes, with a vaccine for children under 5 in two of them::

    import starsim as ss
    nodes = ss.geo.geography(names=['Urban', 'Peri-urban', 'Rural'], pop=[6000, 3000, 1000])
    vaccine = ss.Vaccine(efficacy=0.8, waning=ss.box(dur=ss.days(180), decay=ss.days(730)))
    sim = ms.Sim(nodes=nodes, mosquito='ode', init_prev=0.1, eir=[4, 6, 8],
                 interventions=ss.vaccinate(vaccine, at='2005-01-01', coverage=0.8,
                                            where=(ss.ppl.age < 5) & (ss.ppl.patch != 'Urban')))
"""

from .defaults          import *
from .demographics      import compute_mortality_schedule
from .malaria           import *
from .mosquito_agents   import *
from .interventions     import *
from .utils             import *
from .sim               import *
from .results           import *
from .performance       import PerformanceMonitor
