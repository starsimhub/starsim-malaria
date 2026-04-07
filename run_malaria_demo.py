"""
Example simulation from Mahmud et al. 2025
"""
import starsim as ss
from malaria_demo import Malaria_SS

mal = Malaria_SS()
sim = ss.Sim(modules=mal, copy_inputs=False, start=0, stop=365*5, dt=1, n_agents=1)
sim.run()
mal.plot()