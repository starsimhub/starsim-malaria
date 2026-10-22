"""
The shim beyond the smallest one (malshim.py): what else unchanged malariasim needs to pass its own suite
on v4. Each patch is a v4 time-convention change that coded modules must follow (pytest -p malshim_full).
"""
import numpy as np
import malshim # noqa: the three renames
import malariasim as ms

# 1. v4 calls update_results() and every analyzer at the initial time point, before any module's step().
#    ResultsAnalyzer reads mosquito arrays that MosquitoesAdultFemaleAgentsOnly allocates in its first step().
_az_step = ms.ResultsAnalyzer.step
def az_step(self):
    if self.ti == 0: # The initial point, before any step: v3 never called analyzers here
        return
    return _az_step(self)
ms.ResultsAnalyzer.step = az_step

# 2. v4 (like Starsim 3.6+, 7bae8b1e) counts new_infections in Infection.set_prognoses(), which malariasim overrides
#    without calling super(); so new_infections is 0 at every step. Count them where v4 expects, at the step's end.
for cls in [ms.MalariaHumanBase, ms.MalariaHumanSEIS]:
    def make(orig):
        def set_prognoses(self, uids, sources=None):
            out = orig(self, uids, sources)
            if self.sim.initialized:
                self.results.new_infections[self.ti + 1] += len(uids)
            return out
        return set_prognoses
    cls.set_prognoses = make(cls.set_prognoses)

# 3. MalariaHumanODE writes its results in step() at self.ti (the step's start in v4): the last point is never
#    written (0, so prevalence is 0/0), and index 0 holds the first step's value. Move them to ti + 1, and fill index 0.
_ode_step = ms.MalariaHumanODE.step
KEYS = ['n_susceptible', 'n_infected', 'new_infections']
def ode_step(self):
    ti = self.ti
    if ti == 0:
        self.results.n_susceptible[0] = self._Sh.raw.sum()
        self.results.n_infected[0] = self._Ih.raw.sum()
    old = {k:self.results[k][ti] for k in KEYS}
    _ode_step(self)
    for k in KEYS:
        r = self.results[k]
        r[ti+1], r[ti] = r[ti], old[k]
ms.MalariaHumanODE.step = ode_step
