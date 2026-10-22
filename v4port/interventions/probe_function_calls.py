""" Probe: how often v4 calls a function-valued parameter, and a product's waning function, per step (agents) """
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
import malaria_rm_v4 as rm, malaria_v4 as mv
calls = dict(n=0)
orig = rm.node_means
def counted(*args, **kw):
    calls['n'] += 1
    return orig(*args, **kw)
rm.node_means = counted
sim = rm.make('itn', 'abm', 1); sim.run()
print(f'node_means calls per step (it is used by 3 rate terms): {calls["n"]/sim.t.npts:.2f}')
w = dict(n=0)
orig_box = mv.box_exp
def box_counted(*a):
    f = orig_box(*a)
    def g(t):
        w['n'] += 1
        return f(t)
    return g
mv.box_exp = box_counted
sim = mv.make('vax', 'abm', 1); sim.run()
print(f'vaccine waning function calls per step (efficacy on 2 transitions): {w["n"]/sim.t.npts:.2f}')
