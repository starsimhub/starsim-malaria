""" ss.diff(sim1, sim2) on two ms.Sims (v4 + shim): traceback; and on two plain sims with a coded module that has a function-valued parameter """
import traceback
import starsim as ss; print(ss.__version__, ss.__file__)
import malshim # noqa
import malariasim as ms
mk = lambda: ms.Sim(human='sis', mosquito='eir_only', nodes=ms.NodeGrid(populations=[500]), start=2000, stop=2000.1, verbose=0)
a, b = mk(), mk()
try:
    print(ss.diff(a, b))
except Exception:
    traceback.print_exc()
try:
    print(a.to_spec() is not None)
except Exception:
    traceback.print_exc()
