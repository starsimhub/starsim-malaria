""" Is the small v3-v4 gap in agents (B, SIS: node 1 0.783 vs 0.777) malariasim's linear per-step probability (p = b*EIR_dt*rel, clipped)?
Rerun v3 scenario B with p = 1 - exp(-b*EIR_dt*rel), as v4 converts a rate, patched in at run time (no file is changed) """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, malariasim as ms
from malariasim.human_base import MalariaHumanBase
import run_v3
orig = MalariaHumanBase._p_inf_bite_callable
def expprob(self, sim, uids):
    p = orig(self, sim, uids) # clipped linear; recover the hazard when unclipped
    return 1 - np.exp(-p)
P = []
for patched in [False, True]:
    MalariaHumanBase._p_inf_bite_callable = expprob if patched else orig
    prevs = []
    for seed in range(5):
        r = run_v3.run('B', 'sis', seed)
        I = np.array(r['I']); N = np.array(r['N'])
        prevs.append(I[-365:].sum(0)/N[-365:].sum(0))
    prevs = np.array(prevs)
    print(f'v3 B SIS, {"1 - exp(-x)" if patched else "linear x (unchanged)"}: last-year prevalence by node', np.round(prevs.mean(0), 3), '±', np.round(prevs.std(0, ddof=1), 3))
print('v4 B SIS agents (compare_out.md): 0.498, 0.777, 0.898')
