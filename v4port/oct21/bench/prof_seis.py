import cProfile, pstats, sys
sys.argv = ['x', 'none']
exec(open('ab_declared.py').read().split("if case == 'seis_abm'")[0])
seis(years=0.1).run()
sim = seis(years=2); sim.init()
cProfile.run('sim.run()', '/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/ab/seis.prof')
pstats.Stats('/tmp/claude-1000/-home-cliffk-idm-ngm-starsim-ngm/238c3afc-7d68-4645-9a99-83aa523bd3e8/scratchpad/ab/seis.prof').sort_stats('cumtime').print_stats(25)
