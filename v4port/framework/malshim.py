"""
Smallest shim to run unchanged malariasim on Starsim v4 (a pytest plugin: `pytest -p malshim`).

v4 changed the meaning of two names malariasim subclasses:
- `ss.SIS` is now a declared disease (no `ti_infected`, `ti_recovered`, `set_prognoses()`);
  Starsim 3's coded SIS is `ssl.CodedSIS`.
- `ss.Disease` is now the declared base and refuses a class with no states or transitions;
  Starsim 3's bare coded base (no states, no results) has no v4 name, so it is rebuilt here.
So, while malariasim is imported (and only then), the two names point to their v3 meaning.
"""
import starsim as ss
import starsim.library as ssl


class V3Disease(ss.Disease):
    """ Starsim 3's ss.Disease: a coded module with no declared graph, no states and no results of its own """
    def __init__(self, pars=None, **kwargs):
        ss.Module.__init__(self)
        self.infection_log = None
        self.engine = None
        self.update_pars(pars, **kwargs)

    def init_pre(self, sim):      return ss.Module.init_pre(self, sim)
    def init_post(self):          return ss.Module.init_post(self)
    def start_step(self):         return ss.Module.start_step(self)
    def step_state(self):         pass
    def step_die(self, uids):     pass
    def step(self):               pass
    def update_results(self):     return ss.Module.update_results(self)
    def finalize_results(self):   return ss.Module.finalize_results(self)
    def shrink(self):             return ss.Module.shrink(self)
    def set_prognoses(self, uids, sources=None): pass
    @property
    def transitions(self):        return None


_orig = (ss.SIS, ss.Disease)
ss.SIS, ss.Disease = ssl.CodedSIS, V3Disease
try:
    import malariasim # noqa: F401  (the classes are built now, on the v3 meanings)
finally:
    ss.SIS, ss.Disease = _orig

# Starsim 3.6 (ea12d268) and v4 refuse float() of a rate; malariasim's ODE does float(p.death_rate_per_year)
# (human_ode.py:205). The one-line fix in malariasim is `.value`; from outside, restore the v3.5 behavior.
ss.Rate.__float__ = lambda self: float(self.value)
