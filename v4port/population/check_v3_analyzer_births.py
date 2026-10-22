""" Check: does malariasim's ResultsAnalyzer record births and deaths per step with its own demographics (ss.Births/ss.Deaths, and FastBirths/FastDeaths)? """
import numpy as np, sciris as sc
import starsim as ss; print(ss.__version__, ss.__file__)
import v3_baseline as b
for fast in [False, True]:
    sim = b.make_sim(0, fast=fast); sim.pars.stop = 2002; sim = b.make_sim(0, fast=fast)
    sim.pars.stop = 2002
    sim.run()
    az = sim.analyzers[0]
    nb = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Births)); nd = sum(m.results.new.sum() for m in sim.demographics.values() if isinstance(m, ss.Deaths))
    print(f'{"Fast" if fast else "plain"}: analyzer births {az.births_hist.sum()}, deaths {az.deaths_hist.sum()}; module results births {nb:.0f}, deaths {nd:.0f}; attrs: {[ (type(m).__name__, hasattr(m, "n_births_this_step"), hasattr(m, "n_deaths")) for m in sim.demographics.values()]}')
