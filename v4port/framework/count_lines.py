""" Lines of code, counted the same way on both sides: no blank lines, no comment-only lines, no docstrings """
import ast, sys, io, tokenize

def sloc(path, names=None):
    src = open(path).read()
    tree = ast.parse(src)
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            skip.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    keep = None
    if names: # Only these top-level functions/classes
        keep = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                keep.update(range(node.lineno, node.end_lineno + 1))
    code_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            code_lines.update(range(tok.start[0], tok.end[0] + 1))
    lines = code_lines - skip
    if keep is not None:
        lines &= keep
    return len(lines)

R = '/home/cliffk/idm/ngm/starsim-ngm/malariasim/'
F = '/home/cliffk/idm/ngm/starsim-ngm/v4port/framework/'
rows = [
    ('malariasim: ms.Sim (sim.py)', sloc(R + 'sim.py')),
    ('malariasim: MalariaMultiSim (results.py)', sloc(R + 'results.py', ['MalariaMultiSim', '_plain_counts'])),
    ('malariasim: ResultsAnalyzer (results.py)', sloc(R + 'results.py', ['ResultsAnalyzer'])),
    ('malariasim: PerformanceMonitor (performance.py)', sloc(R + 'performance.py')),
    ('malariasim: eir_to_params (utils.py)', sloc(R + 'utils.py', ['eir_to_params'])),
    ('v4 port: malshim.py (smallest shim)', sloc(F + 'malshim.py')),
    ('v4 port: malshim_full.py (the rest, all 284 pass)', sloc(F + 'malshim_full.py')),
    ('v4 port: calib_eir.py ScaledMosODE (to make emergence fittable)', sloc(F + 'calib_eir.py', ['ScaledMosODE'])),
    ('v4 port: declared_seis.py make() (declared SEIS + mosquito counts)', sloc(F + 'declared_seis.py', ['make'])),
]
for k, v in rows:
    print(f'{v:5d}  {k}')
