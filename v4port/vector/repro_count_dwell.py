"""
A fixed extrinsic incubation period on mosquitoes (states that are not people), `Ev >> Iv @ ss.days(12)`:
as an ODE it is Erlang stages; in an agent-based run the count is advanced with hazard 1/mean, an exponential.
With mortality 0.1/day, the share surviving a fixed 12 days is exp(-1.2) = 0.30, an exponential one 0.45.
Expected Iv: fixed 5,020; exponential 7,576 (inflow 5,000/day, biting 0.05/day).
"""
import starsim as ss; print(ss.__version__, ss.__file__)
import numpy as np
def make(EIP=ss.days(12)):
    Sh, Ih = ss.states(2)
    Sv, Ev, Iv = ss.states(3)
    N = (Sh + Ih).n
    return ss.Disease(infection=Sh >> Ih @ ss.perday(1e-9) * Iv.n / N, emergence=ss.null >> Sv @ ss.perday(5000),
                      biting=Sv >> Ev @ ss.perday(0.05), incubation=Ev >> Iv @ EIP,
                      mortality=Sv + Ev + Iv >> ss.null @ ss.perday(0.1), init=dict(Sh=1.0, Sv=33_333), name='mal')
for label, EIP in [('ss.days(12)', ss.days(12)), ('ss.constant(ss.days(12))', ss.constant(ss.days(12)))]:
    for solver in ['ode', 'binomial', 'abm']:
        pop = dict(n_agents=2000) if solver == 'abm' else dict(total_pop=2000)
        sim = ss.Sim(make(EIP), solver=solver, unit='day', start=0, stop=300, dt=1, verbose=1 if solver=='ode' else 0, **pop).run()
        print(f'EIP {label}, {solver}: Iv at day 300 = {sim.results.mal.n_Iv[-1]:,.0f}')
