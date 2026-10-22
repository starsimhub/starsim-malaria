""" Count lines of code the same way on both sides: non-blank lines that are not comments or docstrings (statements and their continuations) """
import ast, io, tokenize, sys
R = '/home/cliffk/idm/ngm/starsim-ngm/malariasim/'

def code_lines(path, funcs=None):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            doc |= set(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    keep = None
    if funcs is not None:
        keep = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in funcs:
                keep |= set(range(node.lineno, node.end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            lines.add(ln)
    lines -= doc
    if keep is not None:
        lines &= keep
    return len(lines)

v3_files = ['updater.py', 'updaters/__init__.py', 'updaters/waning.py', 'updaters/vaccine.py', 'updaters/bednet.py', 'updaters/drug.py',
            'distributors/__init__.py', 'distributors/eligibility.py', 'distributors/scheduled.py', 'campaign.py', 'outbreak.py']
human_base_funcs = ['define_intervention_states', 'initialize_intervention_states', 'reset_acquisition_probability', 'set_acquisition_probability', 'pre_administer',
                    'sync_vaccinated_fraction', 'get_vaccinated_weights', 'store_itn_fraction', 'get_itn_weights', 'aggregate_itn_to_node_state', 'set_drug_clearance',
                    'administer_outbreak', 'apply_coverage', 'validate_eligibility']
ode_funcs = human_base_funcs + ['_age_vaccinated_state']
mos_funcs = ['_apply_itn_killing']
total = 0
print('v3 (malariasim):')
for f in v3_files:
    n = code_lines(R + f); total += n; print(f'  {f:32s} {n:4d}')
for f, funcs in [('human_base.py', human_base_funcs), ('human_ode.py', ode_funcs), ('mosquito_agents.py', mos_funcs)]:
    n = code_lines(R + f, funcs); total += n; print(f'  {f:32s} {n:4d}  (intervention methods only: {len(funcs)} names)')
print(f'  total {total} (not counting the ITN lines inside the mosquito modules\' step(), NodeState\'s two fields, and ms.Sim\'s build_campaign window)')

print('v4 (this port):')
t4 = 0
for f, funcs in [('malaria_v4.py', ['box_exp', 'itn_waning', 'make_interventions']), ('malaria_rm_v4.py', ['itn_parts', 'node_means', 'make_interventions', 'itn_factors'])]:
    n = code_lines(f, funcs); t4 += n; print(f'  {f:32s} {n:4d}  ({", ".join(funcs)})')
print(f'  plus 3 rate terms (* br twice, * km) in the declared Ross-Macdonald disease; total {t4 + 3}')
