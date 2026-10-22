import starsim as ss; print(ss.__version__, ss.__file__)
import malariasim as ms
sim = ms.Sim(human='ode', mosquito='eir_only', start=2000, stop=2005, dt=ss.days(1), verbose=0); sim.init()
yv = sim.t.yearvec
print('v3 yearvec[365], [730], [1825]:', yv[365], yv[730], yv[1825], 'dt_year', sim.t.dt_year)
