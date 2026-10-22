""" ss.timeseries with bare-number times: the docstring says "a number, which is a time in the sim's unit (in a sim with dates, that long after the start)".
In a sim whose unit is days (describe() says so), times 0..334 given in days are read as years, so a seasonal EIR is flat at its first value. """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
monthly = [2, 2, 5, 15, 30, 50, 50, 30, 15, 5, 2, 2]
for label, times, period in [('times in days, period ss.days(365)', [k*365/12 for k in range(12)], ss.days(365)), ('times in years, period ss.years(1)', [k/12 for k in range(12)], ss.years(1))]:
    for start in [2000, 0]:
        ts = ss.timeseries(dict(zip(times, [ss.peryear(v) for v in monthly])), mode='linear', period=period)
        S, I = ss.states('S I')
        d = ss.Disease(infection=S >> I @ ts, recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='m')
        sim = ss.Sim(d, start=start, stop=start + 365, dt=ss.days(1), n_agents=20_000, verbose=0, unit='day') if start == 0 else ss.Sim(d, start=start, stop=start + 1, dt=ss.days(1), n_agents=20_000, verbose=0)
        sim.run()
        r = sim.results.m
        haz = r.infection[1:]/np.maximum(r.n_S[:-1], 1)*365
        print(f'start={start:<5} unit={str(getattr(sim, 'unit', '?')):4s} {label:36s} hazard per year by month:', np.round([haz[int(k*365/12):int((k+1)*365/12)].mean() for k in range(12)], 1))
print('expected (monthly means of the linear interpolation)      ', np.round([np.mean(np.interp(np.linspace(k/12, (k+1)/12, 30, endpoint=False), np.arange(13)/12, monthly + monthly[:1])) for k in range(12)], 1))
