""" Run a port script on v4, loading the malariasim shim first if it uses malariasim """
import sys, runpy, os
path = sys.argv[1]
src = open(path).read()
if 'malariasim' in src and 'malshim' not in src:
    import malshim # noqa
sys.argv = sys.argv[1:]
runpy.run_path(path, run_name='__main__')
