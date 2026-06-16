"""
Augment a jSFS or two-locus-SFS scenario config with the COSINE within-tree / cross-locus pairwise joint CDF *and*
PDF (the existing configs only test the un-wrapped pairwise CDF, which routes to de Hoog -- we test cosine). Run:

    python scripts/augment_joint_stats.py <config_name> ...

Operates on the ``jsfs`` / ``sfs2`` tolerance subtree: replaces any un-wrapped pairwise cdf/pdf with a
``cosine: {cdf, pdf}`` block (placeholder tolerances, tightened later by tune_dist_tols.py)."""
import sys
from ruamel.yaml import YAML

PLACEHOLDER = 0.5
yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096


def augment(path: str) -> bool:
    with open(path) as f:
        cfg = yaml.load(f)
    if cfg is None:
        return False
    tol = cfg.get('comparisons', {}).get('tolerance', {})
    touched = False
    for stat in ('jsfs', 'sfs2'):
        node = tol.get(stat)
        if node is None:
            continue
        # the generators always cache the pairwise joint ground truth, so a pairwise block can be added even if absent
        pw = node.setdefault('pairwise', {})
        for k in ('cdf', 'pdf', 'quantile'):  # drop un-wrapped (default de Hoog) variants
            pw.pop(k, None)
        cos = pw.setdefault('cosine', {})
        for k in ('cdf', 'pdf'):
            if k not in cos:
                cos[k] = PLACEHOLDER
        touched = True
    if touched:
        with open(path, 'w') as f:
            yaml.dump(cfg, f)
    return touched


if __name__ == '__main__':
    for name in sys.argv[1:]:
        ok = augment(f"resources/configs/{name}.yaml")
        print(f"{'augmented' if ok else 'skipped  '} {name}")
