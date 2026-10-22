""" Print the order of module calls within one step (v3 or v4), for malariasim's SIS + EIR-only sim """
import starsim as ss; print(ss.__version__, ss.__file__)
import malariasim as ms
sim = ms.Sim(human='sis', mosquito='eir_only', start=2000, stop=2000.01, dt=ss.days(1), verbose=0) if ss.__version__.startswith('3') else None
if sim is not None:
    sim.init()
    for f in sim.loop.funcs[:40] if hasattr(sim.loop, 'funcs') else []: print(' ', getattr(f, '__qualname__', f))
