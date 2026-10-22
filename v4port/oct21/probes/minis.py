""" Small checks of Oct 17 malaria findings on the current v4 """
import numpy as np, sciris as sc, starsim as ss, starsim.library as ssl
print(ss.__version__, ss.__file__)
def tryit(label, fn):
    try: print(f'{label}: {fn()}')
    except Exception as e: print(f'{label}: {type(e).__name__}: {str(e)[:300]}')

# MA22: a bare coded disease (no states), as malariasim's MalariaHumanODE
def ma22():
    class Bare(ss.Infection):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.define_states(ss.FloatArr('susceptible', default=1.0))
    s = ss.Sim(diseases=Bare(), n_agents=100, stop=2001, verbose=0).run(); return 'ran'
tryit('MA22 ss.Infection subclass defining its own "susceptible"', ma22)
tryit('MA22 public bare base?', lambda: [n for n in ['DiseaseBase', 'CodedDisease', 'AgentDisease'] if hasattr(ss, n) or hasattr(ssl, n)])
def ma22b():
    class Bare(ss.Disease): pass
    Bare()
tryit('MA22 message for ss.Disease with no states', ma22b)

# MA13: .n and .count() under both solvers
def ma13(solver):
    S, I = ss.states('S I'); d = ss.Disease(infection=S >> I @ ss.peryear(1)*I, recovery=I >> S @ ss.years(1), init=dict(I=0.1), name='d')
    kw = dict(n_agents=500) if solver == 'abm' else dict(total_pop=500)
    s = ss.Sim(d, stop=2001, solver=solver, verbose=0, **kw).run()
    return (float(s.diseases.d.I.n), float(s.diseases.d.I.count()))
for sv in ['abm', 'ode']: tryit(f'MA13 .n/.count() {sv}', lambda: ma13(sv))

# MA31: .by() on a MultiSim plots
def ma31():
    S, I = ss.states('S I'); d = ss.Disease(infection=S >> I @ ss.peryear(1)*I, recovery=I >> S @ ss.years(1), init=dict(I=0.1), name='d')
    d.stratify(patch=['a', 'b'])
    m = ss.parallel(ss.Sim(d, stop=2002, n_agents=500, verbose=0), n_runs=2)
    r = m.results.d.n_I.by('patch'); r.plot(); return type(r).__name__
tryit('MA31 msim by().plot()', ma31)

# MA32: people_results=False with ss.Deaths and ss.Births
tryit('MA32 people_results=False + Births + Deaths', lambda: ss.Sim(diseases=ssl.AgentSIS(), demographics=[ss.Births(), ss.Deaths()], people_results=False, n_agents=200, stop=2002, verbose=0).run() and 'ran')

# FP3: resample('year') on a 3-year dated sim
def fp3():
    S, I = ss.states('S I'); d = ss.Disease(infection=S >> I @ ss.peryear(1)*I, recovery=I >> S @ ss.years(1), init=dict(I=0.1), name='d')
    s = ss.Sim(d, start='2000-01-01', stop='2003-01-01', dt=ss.days(7), total_pop=100, verbose=0).run()
    r = s.results.d.infection.resample('year'); return len(r), list(r.index)[-1] if hasattr(r, 'index') else None
tryit('FP3 resample(year) bins', fp3)

# ST11: public way for code to end an infection in a declared disease
tryit('ST11 Disease.move_to', lambda: hasattr(ss.Disease, 'move_to'))

# FP1 last point: a coded module writing results at self.ti leaves the last point 0 -- is there a warning?
def fp1():
    class Writer(ss.Module):
        def init_results(self):
            super().init_results(); self.define_results(ss.Result('x'))
        def step(self):
            self.results.x[self.ti] = 1.0
    with sc.capture() as txt:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            s = ss.Sim(modules=Writer(name='w'), n_agents=10, stop=2001, dt=ss.years(0.25), verbose=0).run()
    return f'x = {s.results.w.x.values}; warnings: {[str(x.message)[:120] for x in w]}; printed: {txt.strip()[:200]!r}'
tryit('FP1 last point', fp1)
