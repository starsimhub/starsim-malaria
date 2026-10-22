""" Order of calls and the first steps in v3 and v4: when does ResultsAnalyzer.step run, relative to the modules' step()? """
import sys, os
import starsim as ss; print(ss.__version__, ss.__file__)
if ss.__version__.startswith('4'):
    import malshim # noqa
import numpy as np
import malariasim as ms

calls = []
def wrap(cls, name):
    orig = getattr(cls, name)
    def f(self, *a, **k):
        calls.append((type(self).__name__, name, self.ti if hasattr(self, 't') and self.t is not None else None))
        return orig(self, *a, **k)
    setattr(cls, name, f)

for cls in [ms.ResultsAnalyzer, ms.MosquitoesAdultFemaleAgentsOnly, ms.MalariaHumanSIS, ms.NodeState]:
    for name in ['step', 'update_results', 'finish_step', 'start_step']:
        if name in vars(cls) or True:
            try: wrap(cls, name)
            except Exception as e: pass

sim = ms.Sim(human='sis', mosquito='agents', start=2000, stop=2000.02, nodes=ms.NodeGrid(populations=[500]), verbose=0, rand_seed=0, analyzers=ms.ResultsAnalyzer())
try:
    sim.init()
    print('after init:', calls); calls.clear()
    sim.run()
except Exception as e:
    print('ERROR', type(e).__name__, e)
for c in calls[:30]: print(c)
print('n timepoints', len(sim.t.tvec) if hasattr(sim.t,'tvec') else sim.t.npts)
