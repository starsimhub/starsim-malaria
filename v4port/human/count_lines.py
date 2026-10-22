""" Count lines of code (no blanks, comments or docstrings), the same way on both sides; whole files or chosen classes/methods """
import ast, io, tokenize

def code_lines(path, select=None, exclude=None):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            doc.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    code = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            code.update(range(tok.start[0], tok.end[0] + 1))
    code -= doc
    def lines_of(names):
        keep = set()
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name in names: keep.update(range(node.lineno, node.end_lineno + 1))
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and f'{node.name}.{sub.name}' in names:
                        keep.update(range(sub.lineno, sub.end_lineno + 1))
            elif isinstance(node, ast.FunctionDef) and node.name in names:
                keep.update(range(node.lineno, node.end_lineno + 1))
        return keep
    if select is not None: code &= lines_of(select)
    if exclude is not None: code -= lines_of(exclude)
    return len(code)

R = '/home/cliffk/idm/ngm/starsim-ngm/malariasim/'
iface_base = [f'MalariaHumanBase.{m}' for m in ['define_intervention_states', 'initialize_intervention_states', 'reset_acquisition_probability', 'set_acquisition_probability', 'pre_administer',
              'sync_vaccinated_fraction', 'get_vaccinated_weights', 'store_itn_fraction', 'get_itn_weights', 'aggregate_itn_to_node_state', 'set_drug_clearance', 'administer_outbreak', 'apply_coverage', 'validate_eligibility']]
iface_ode = [f'MalariaHumanODE.{m}' for m in ['apply_coverage', 'reset_acquisition_probability', 'set_acquisition_probability', 'pre_administer', 'sync_vaccinated_fraction', 'get_vaccinated_weights',
             'store_itn_fraction', 'get_itn_weights', 'aggregate_itn_to_node_state', 'set_drug_clearance', 'administer_outbreak', 'validate_eligibility', '_age_vaccinated_state']]
rows = [
    ('human_base.py, whole file', code_lines(R + 'human_base.py')),
    ('human_base.py, without the intervention interface', code_lines(R + 'human_base.py', exclude=iface_base)),
    ('human_sis.py', code_lines(R + 'human_sis.py')),
    ('human_seis.py', code_lines(R + 'human_seis.py')),
    ('human_ode.py, whole file', code_lines(R + 'human_ode.py')),
    ('human_ode.py, without the intervention interface', code_lines(R + 'human_ode.py', exclude=iface_ode)),
    ('mosquito_eir_only.py (the EIR input)', code_lines(R + 'mosquito_eir_only.py')),
    ('utils.interpolate_monthly', code_lines(R + 'utils.py', select=['interpolate_monthly'])),
]
v3_core = rows[1][1] + rows[2][1] + rows[3][1] + rows[5][1] + rows[6][1] + rows[7][1]
v3_all = rows[0][1] + rows[2][1] + rows[3][1] + rows[4][1] + rows[6][1] + rows[7][1]
for k,v in rows: print(f'{v:5d}  v3 {k}')
print(f'{v3_core:5d}  v3 total replaced (without the intervention interface)')
print(f'{v3_all:5d}  v3 total replaced (whole files)')
print(f'{code_lines("human_v4.py"):5d}  v4 human_v4.py (SIS and SEIS, agents and ODE, nodes, seasonal EIR, heterogeneity, age biting)')
