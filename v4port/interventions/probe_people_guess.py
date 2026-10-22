""" Probe: which states v4 takes to be people in the declared Ross-Macdonald model, and why """
import starsim as ss; print(ss.__version__, ss.__file__)
from malaria_rm_v4 import make_disease
mal = make_disease()
sim = ss.Sim(mal, start='2000-01-01', stop=ss.days(30), dt=ss.days(1), n_agents=4000, verbose=0)
sim.init()
print(sim.describe())
