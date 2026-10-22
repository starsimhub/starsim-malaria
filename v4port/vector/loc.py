"""
Count lines of code the same way on both sides: physical lines that hold code, leaving out blank lines, comments and
docstrings. Optionally only the named top-level classes or functions of a file.
Usage: python loc.py file.py[:Name1,Name2] ...
"""
import ast, io, sys, tokenize
def code_lines(path, names=None):
    src = open(path).read()
    tree = ast.parse(src)
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant) and isinstance(node.body[0].value.value, str):
            doc.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    keep = None
    if names:
        keep = set()
        for node in tree.body:
            if getattr(node, 'name', None) in names:
                keep.update(range(node.lineno - len(getattr(node, 'decorator_list', [])), node.end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            lines.update(range(tok.start[0], tok.end[0] + 1))
    lines -= doc
    if keep is not None:
        lines &= keep
    return len(lines)
total = 0
for arg in sys.argv[1:]:
    path, _, names = arg.partition(':')
    n = code_lines(path, names.split(',') if names else None)
    total += n
    print(f'{n:5d}  {arg}')
print(f'{total:5d}  total')
