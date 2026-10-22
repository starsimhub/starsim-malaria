""" malariasim shim for either v4 (1a5b7a77 or ee5d1fb0) """
import starsim as ss
import starsim.library as ssl
new = hasattr(ss.diseases, 'DiseaseBase')
if new:
    class V3Disease(ss.diseases.DiseaseBase, ss.Module):
        def __init__(self, pars=None, **kwargs):
            ss.Module.__init__(self); self.engine = None; self.update_pars(pars, **kwargs)
        def step_state(self): pass
        def step(self): pass
        def set_prognoses(self, uids, sources=None): pass
        @property
        def transitions(self): return None
    SIS = ssl.AgentSIS
else:
    class V3Disease(ss.Disease):
        def __init__(self, pars=None, **kwargs):
            ss.Module.__init__(self); self.infection_log = None; self.engine = None; self.update_pars(pars, **kwargs)
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
    SIS = ssl.CodedSIS
_orig = (ss.SIS, ss.Disease)
ss.SIS, ss.Disease = SIS, V3Disease
try:
    import malariasim # noqa
finally:
    ss.SIS, ss.Disease = _orig
ss.Rate.__float__ = lambda self: float(self.value)
