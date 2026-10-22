""" Repro: start=2000, stop=2002 with dt=ss.days(1). With the unit guessed as 'day' (from the typed values) the run is two years; saying unit='day', as describe() reports, makes it two days """
import io, contextlib
import numpy as np
import starsim as ss; print(ss.__version__, ss.__file__)
for kw in [dict(), dict(unit='day'), dict(unit='year')]:
    S, I = ss.states('S I')
    sis = ss.Disease(inf=S >> I @ ss.peryear(1.0), rec=I >> S @ ss.days(200), name='m')
    sim = ss.Sim(sis, solver='ode', total_pop=1000, start=2000, stop=2002, dt=ss.days(1), verbose=0, **kw).init()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.describe()
    head = [l for l in buf.getvalue().splitlines() if l.startswith('Sim:') or l.startswith('unit')]
    sim.run()
    print(f'{str(kw):18s}: {len(sim.results.timevec)} time points; describe: {head}')
