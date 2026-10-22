""" malariasim's own bugs, found while porting its interventions (run on v3) """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
import malariasim as ms
from scen import *
import run_v3, run_v3_rm

# 1. Building a second ms.Sim clears the first one's campaign registrations (the updaters are module-level singletons)
def cum_y2(sim):
    return sim.analyzers.counts.new[365:].sum()
a = run_v3.make('vax', 1); a.run(); alone = cum_y2(a)
a = run_v3.make('vax', 1); b = run_v3.make('base', 1); a.run(); after_b = cum_y2(a)
base = run_v3.make('base', 1); base.run(); nothing = cum_y2(base)
print(f'1. vaccine scenario, new infections in year 2: built and run alone {alone:.0f}; built, then another sim built, then run {after_b:.0f}; no vaccine {nothing:.0f}')

# 2. Bednets lower indoor bites twice for the people who hold one: the node's indoor EIR is cut by the mean block for everyone,
#    and each holder's acquisition is cut again by their own block. Count the infectious bites each side sees on one step.
sim = run_v3_rm.make('itn', 1)
sim.init()
for ti in range(T_INTV + 2):
    sim.run_one_step()
d = sim.diseases[0]
ns = next(m for m in sim.connectors.values() if getattr(m, '_is_node_state', False))
mos = next(m for m in sim.connectors.values() if isinstance(m, ms.MosquitoesODE))
alive = sim.people.auids
received = (ns.eir_indoor[0] * np.asarray(d._prob_acquire_indoor[alive])).sum()   # infectious indoor bites the humans get this step
made     = RM['a'] * F_INDOOR * (1 - ns.itn_biting_reduction[0]) * mos._Im[0] * 1.0  # infectious indoor bites the mosquitoes land (a f (1 - br) Im dt)
held = np.asarray(d._prob_acquire_indoor[alive]) < 1
print(f'2. one step after bednets: mean block br = {ns.itn_biting_reduction[0]:.3f}; infectious indoor bites landed by mosquitoes {made:.3f}, received by humans {received:.3f} (ratio {received/made:.3f}, about 1 - br = {1 - ns.itn_biting_reduction[0]:.3f})')
print(f'   a holder\'s indoor multiplier is (1 - br)(1 - block) = {ns.eir_indoor[0]*np.asarray(d._prob_acquire_indoor[alive])[held].mean()/(RM["a"]*F_INDOOR*mos._Im[0]/len(alive)):.3f}, an unprotected person\'s (1 - br) = {1 - ns.itn_biting_reduction[0]:.3f}')
