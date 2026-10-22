""" Why is the declared ODE slower than malariasim's hand-written Euler for 3 nodes? Time the ODE for EIR forms: a constant, a time series, ss.by(patch=) of constants, ss.by(patch=) of time series (best of 3 after a warm-up) """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np, sciris as sc
from human_v4 import eir_series
forms = {
    'constant, 1 node':                      (lambda: ss.peryear(10), 1),
    'time series, 1 node':                   (lambda: eir_series([10]*12), 1),
    'by(patch=) of 3 constants':             (lambda: ss.by(patch=[ss.peryear(5), ss.peryear(20), ss.peryear(50)]), 3),
    'by(patch=) of 3 time series':           (lambda: ss.by(patch=[eir_series([5]*12), eir_series([20]*12), eir_series([50]*12)]), 3),
    '3 time series, one transition per patch (where=)': (None, 3),
}
for label, (f, n) in forms.items():
    times = []
    for rep in range(4):
        S, I = ss.states('S I')
        if f is not None:
            d = ss.Disease(infection=S >> I @ 0.4 * f(), recovery=I >> S @ ss.days(200), init=dict(I=0.0), name='m')
        else:
            continue
        if n > 1: d.stratify(patch=['a', 'b', 'c'], split=[0.3, 0.3, 0.4])
        sim = ss.Sim(d, start=2000, stop=2005, dt=ss.days(1), solver='ode', total_pop=10_000, verbose=0)
        T = sc.timer(); sim.run(); times.append(T.toc(output=True))
    if times: print(f'{label:32s} best of 3 {min(times[1:]):.3f} s (warm-up {times[0]:.3f} s)')
