import cProfile, pstats, sys
src = open('ab_fseis.py').read().split('make(0.1).run()')[0]
exec(src)
make(0.1).run()
sim = make(2); sim.init()
out = sys.argv[1]
cProfile.run('sim.run()', out)
pstats.Stats(out).sort_stats('tottime').print_stats(18)
