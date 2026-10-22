"""
Smallest shim to run unchanged malariasim on Starsim v4 (a pytest plugin: `pytest -p malshim`). Updated 2026-10-21 for v4 ee5d1fb0.

- `ss.SIS` is now declared, and a subclass defining set_prognoses() is refused at class creation; Starsim 3's coded SIS is `ssl.AgentSIS`.
- `ss.Disease` is now the declared base; Starsim 3's bare coded base has no v4 name, so it is rebuilt here on `DiseaseBase`.
"""
import starsim as ss
import starsim.library as ssl


class V3Disease(ss.diseases.DiseaseBase, ss.Module):
    """ Starsim 3's ss.Disease: a coded module with no declared graph, no states and no results of its own """
    def __init__(self, pars=None, **kwargs):
        ss.Module.__init__(self)
        self.engine = None
        self.update_pars(pars, **kwargs)

    def step_state(self):         pass
    def step(self):               pass
    def set_prognoses(self, uids, sources=None): pass
    @property
    def transitions(self):        return None


_orig = (ss.SIS, ss.Disease)
ss.SIS, ss.Disease = ssl.AgentSIS, V3Disease
try:
    import malariasim # noqa: F401  (the classes are built now, on the v3 meanings)
finally:
    ss.SIS, ss.Disease = _orig

ss.Rate.__float__ = lambda self: float(self.value)
