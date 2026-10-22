""" malariasim's PerformanceMonitor reads sim.loop.to_df()['cpu_time'] by module, inside a bare try/except; what does it get on v3 and v4? """
import starsim as ss; print(ss.__version__, ss.__file__)
if ss.__version__.startswith('4'):
    import malshim # noqa
import malariasim as ms
sim = ms.Sim(human='sis', mosquito='ode', nodes=ms.NodeGrid(populations=[2000]), start=2000, stop=2000.5, verbose=0)
mon = ms.PerformanceMonitor()
mon.start(); sim.run(); mon.stop(); mon.record(sim)
print('module_totals:', {k: round(v, 3) for k, v in mon._records[0]['module_totals'].items()})
try:
    df = sim.loop.to_df()
    print('loop.to_df() columns:', list(df.columns))
except Exception as e:
    print('loop.to_df() raised', type(e).__name__, e)
