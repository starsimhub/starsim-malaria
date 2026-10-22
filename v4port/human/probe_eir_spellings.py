""" Which spellings of "b times a seasonal EIR given as 12 monthly anchors (per person per year), linearly interpolated, repeating yearly" work? """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, traceback
monthly = [2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2] # annual EIR at the start of each month
b = 0.4
days = [k*365/12 for k in range(12)]

def ts_bare(): return ss.timeseries(dict(zip(days, monthly)), mode='linear', period=ss.days(365))
def ts_typed(): return ss.timeseries(dict(zip(days, [ss.peryear(v) for v in monthly])), mode='linear', period=ss.days(365))
def ts_years(): return ss.timeseries(dict(zip([k/12 for k in range(12)], [ss.peryear(v) for v in monthly])), mode='linear', period=ss.years(1))
spellings = {
    'b * ts(typed values)':           lambda: b * ts_typed(),
    'ts(typed values) * b':           lambda: ts_typed() * b,
    'b * ss.peryear(ts(bare))':       lambda: b * ss.peryear(ts_bare()),
    'b * ss.peryear(1) * ts(bare)':   lambda: b * ss.peryear(1) * ts_bare(),
    'b * ts(bare) * ss.peryear(1)':   lambda: b * ts_bare() * ss.peryear(1),
    'ss.par(ts typed) * b':           lambda: ss.par(ts_typed(), name='eir') * b,
    'b * ts(typed, times in years)':  lambda: b * ts_years(),
}
for label, f in spellings.items():
    try:
        S, I = ss.states('S I')
        m = ss.Disease(infection = S >> I @ f(), recovery = I >> S @ ss.days(200), init=dict(I=0.0), name='malaria')
        sim = ss.Sim(m, start=2000, stop=2002, dt=ss.days(1), n_agents=5000, verbose=0).run()
        r = sim.results.malaria
        inf = r.infection[:]; Sn = r.n_S[:]
        haz = inf[1:]/np.maximum(Sn[:-1], 1)*365 # per-year hazard per susceptible, from the flow
        print(f'OK   {label:32s} mean annual hazard / b by month in year 2:', np.round([haz[365+int(k*365/12):365+int((k+1)*365/12)].mean()/b for k in range(12)], 1))
    except Exception as e:
        print(f'FAIL {label:32s} {type(e).__name__}: {str(e)[:250]}')
