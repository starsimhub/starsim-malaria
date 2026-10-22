"""Profile the declared hybrid as agents on v4 (why it is slower than v3's HumanABM x MosquitoODE)"""
import starsim as ss; print(ss.__version__, ss.__file__)
import cProfile, pstats, io, sciris as sc
from hybrid_v4 import run
run('abm', seed=0) # Warm up
pr = cProfile.Profile(); pr.enable(); sim, t = run('abm', seed=1); pr.disable()
print('run time', round(t, 2))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(35); print(s.getvalue()[:9000])
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(20); print(s.getvalue()[:5000])
