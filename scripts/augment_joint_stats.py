"""
Augment a jSFS or two-locus-SFS scenario config with the COSINE within-tree / cross-locus pairwise joint CDF *and*
PDF (the existing configs only test the un-wrapped pairwise CDF, which routes to de Hoog -- we test cosine). Run:

    python scripts/augment_joint_stats.py <config_name> ...

Operates on the ``jsfs`` / ``sfs2`` tolerance subtree: ensures a ``pairwise: {cosine: {cdf, pdf}}`` block exists
(adding it if absent, or replacing an un-wrapped pairwise cdf/pdf/quantile block), with placeholder tolerances that
tune_dist_tols.py tightens later.

NOTE: this uses a surgical *text* edit rather than a yaml round-trip on purpose -- jSFS configs carry
``!!python/tuple`` migration-rate keys that ruamel rewrites into plain (unhashable) lists, which breaks the
``yaml.full_load`` in ``Comparison.from_yaml``. The edit only rewrites/inserts the ``pairwise:`` block and leaves
the rest of the document (tuple keys included) byte-for-byte intact."""
import re
import sys

PLACEHOLDER = 0.5
# Un-wrapped pairwise block (children are bare cdf/pdf/quantile leaves -- the default-de-Hoog variants).
_UNWRAPPED = re.compile(r'(?P<ind>[ ]*)pairwise:\n(?:(?P=ind)[ ]{2}(?:cdf|pdf|quantile): [^\n]+\n)+')


def _cosine_block(ind: str) -> str:
    return f'{ind}pairwise:\n{ind}  cosine:\n{ind}    cdf: {PLACEHOLDER}\n{ind}    pdf: {PLACEHOLDER}\n'


def augment(path: str) -> bool:
    with open(path) as f:
        lines = f.readlines()
    out, i, touched = [], 0, False
    while i < len(lines):
        line = lines[i]
        m = re.match(r'(?P<ind>[ ]*)(?:jsfs|sfs2):[ ]*\n$', line)
        if not m:
            out.append(line); i += 1; continue
        node_ind = m.group('ind')
        child_ind = node_ind + '  '
        out.append(line); i += 1
        # collect the node body (lines more deeply indented than the node header)
        body = []
        while i < len(lines) and (lines[i].strip() == '' or lines[i].startswith(child_ind)):
            body.append(lines[i]); i += 1
        text = ''.join(body)
        if 'pairwise:' in text:
            new_text, n = _UNWRAPPED.subn(lambda mm: _cosine_block(mm.group('ind')), text)
            if n:
                text = new_text; touched = True
        else:
            text = _cosine_block(child_ind) + text; touched = True
        out.append(text)
    if touched:
        with open(path, 'w') as f:
            f.writelines(out)
    return touched


if __name__ == '__main__':
    for name in sys.argv[1:]:
        ok = augment(f"resources/configs/{name}.yaml")
        print(f"{'augmented' if ok else 'skipped  '} {name}")
