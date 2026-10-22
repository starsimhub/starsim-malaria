""" Minimal: ss.diff(sim1, sim2) on two plain sims holding a coded disease (ssl.CodedSIS) """
import starsim as ss, starsim.library as ssl; print(ss.__version__, ss.__file__)
mk = lambda beta: ss.Sim(diseases=ssl.CodedSIS(beta=beta), networks='random', n_agents=200, stop=2001, verbose=0)
try:
    ss.diff(mk(0.05), mk(0.1))
except Exception as e:
    print(f'{type(e).__name__}: {e}')
print('declared:'); ss.diff(ss.Sim(diseases=ss.SIS(beta=0.05), networks='random', n_agents=200, stop=2001), ss.Sim(diseases=ss.SIS(beta=0.1), networks='random', n_agents=200, stop=2001))
