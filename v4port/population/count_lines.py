""" Count code lines: not blank, not comment-only, not docstrings (the same rule for v3 and v4 files; as fpsim/v4port/population/count_lines.py) """
import ast, io, tokenize
def code_lines(path, funcs=None, start=None, stop=None):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            doc.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    keep = None
    if funcs:
        keep = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in funcs:
                keep.update(range(node.lineno, node.end_lineno + 1))
    if start:
        keep = set(range(start, stop + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            for l in range(tok.start[0], tok.end[0] + 1):
                if l not in doc and (keep is None or l in keep):
                    lines.add(l)
    return len(lines)
R = '/home/cliffk/idm/ngm/starsim-ngm/'
v3 = [
    ('node_grid.py (NodeGrid)', R + 'malariasim/node_grid.py', None),
    ('demographics.py (compute_mortality_schedule)', R + 'malariasim/demographics.py', None),
    ('defaults.py: make_steady_state_demographics, make_age_data, make_mortality_dataframe', R + 'malariasim/defaults.py', ['make_steady_state_demographics', 'make_age_data', 'make_mortality_dataframe']),
    ('FastBirths, FastDeaths, _death_draw_kernel (branch Interventions2-ck)', 'v3_ref/defaults_Interventions2-ck.py', ['FastBirths', 'FastDeaths', '_death_draw_kernel']),
    ('results.py: ResultsAnalyzer', R + 'malariasim/results.py', ['ResultsAnalyzer']),
    ('results.py: MalariaMultiSim (plotting)', R + 'malariasim/results.py', ['MalariaMultiSim']),
    ('run_population_stability.py', R + 'run_population_stability.py', None),
    ('human_base.py: node assignment (in _malaria_pre_step_state)', R + 'malariasim/human_base.py', None, 296, 330),
    ('mosquito_eir_only.py: per-node EIR loop in step()', R + 'malariasim/mosquito_eir_only.py', None, 129, 141),
]
v4 = [
    ('mal_v4.py: make_malaria (patch dimension, per-patch EIR and f_indoor; includes the SIS itself)', 'mal_v4.py', ['make_malaria', 'eir_series']),
    ('mal_v4.py: steady_state (declared and lifetable; the modules form is malariasim\'s, verbatim)', 'mal_v4.py', ['steady_state']),
    ('mal_v4.py: age_data, make_sim', 'mal_v4.py', ['age_data', 'make_sim']),
    ('common.py: compute_mortality_schedule (copied unchanged)', 'common.py', ['compute_mortality_schedule']),
    ('v4_stability.py (replaces run_population_stability.py; with plots)', 'v4_stability.py', None),
]
for heading, items in [('v3', v3), ('v4', v4)]:
    tot = 0
    print(heading)
    for item in items:
        label, path, funcs = item[:3]
        n = code_lines(path, funcs, *item[3:]) if len(item) > 3 else code_lines(path, funcs)
        tot += n; print(f'  {n:5d}  {label}')
    print(f'  {tot:5d}  total')
