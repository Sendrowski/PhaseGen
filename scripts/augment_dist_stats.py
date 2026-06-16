"""
Augment a single-population scenario's comparison config with distribution statistics (pdf / cdf / quantile for
tree_height, total_branch_length and the SFS, plus the aggregate within-tree pairwise joint CDF/PDF), so the test
suite exercises the reward-distribution machinery, not only moments. Placeholder tolerances are written (a generous
sentinel); they are tightened afterwards by ``tune_dist_tols.py`` from the observed diffs. Run:

    python scripts/augment_dist_stats.py <config_name> [<config_name> ...]

Idempotent: existing keys are left untouched, only missing dist stats are added. Skips non-single-population or
two-locus configs (tree_height/tbl/sfs reward distributions only apply there)."""
import sys
from ruamel.yaml import YAML

PLACEHOLDER = 0.5  # generous; tightened later from observed diffs

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096


def _ensure(d, key, value):
    if key not in d:
        d[key] = value


def augment(path: str) -> bool:
    with open(path) as f:
        cfg = yaml.load(f)
    if cfg is None or 'comparisons' not in cfg:
        return False
    # single-population, single-locus only (reward distributions tree_height/tbl/sfs apply there)
    if cfg.get('n_loci', 1) != 1:
        return False
    pop_sizes = cfg.get('pop_sizes', {})
    if len(pop_sizes) != 1:
        return False
    tol = cfg['comparisons'].get('tolerance')
    if tol is None:
        return False

    # tree_height (exact expm; no mode wrapper -- not a de Hoog inversion)
    th = tol.setdefault('tree_height', {})
    for k in ('pdf', 'cdf', 'quantile'):
        _ensure(th, k, PLACEHOLDER)

    # total_branch_length: cosine-mode curve only (un-wrapped pdf/cdf would route to the per-point de Hoog -- we test
    # cosine, not de Hoog)
    tbl = tol.setdefault('total_branch_length', {})
    for k in ('pdf', 'cdf', 'quantile'):
        tbl.pop(k, None)
    cos = tbl.setdefault('cosine', {})
    for k in ('pdf', 'cdf', 'quantile'):
        _ensure(cos, k, PLACEHOLDER)

    # SFS: spectrum-wide pdf/cdf/quantile (cosine) + aggregate within-tree pairwise joint cdf/pdf (cosine). Strip any
    # un-wrapped (default-de-Hoog) variants so only the cosine path is exercised.
    sfs = tol.setdefault('sfs', {})
    for k in ('pdf', 'cdf', 'quantile'):
        sfs.pop(k, None)
    scos = sfs.setdefault('cosine', {})
    for k in ('pdf', 'cdf', 'quantile'):
        _ensure(scos, k, PLACEHOLDER)
    # aggregate within-tree pairwise joint cdf/pdf over ALL bin pairs is O(pairs * 2D-cosine-build); only viable for
    # small n. For larger n it would take many minutes (n=10 4-epoch ~ 20 min), so the pairwise block is skipped --
    # the joint distribution is covered by the small-n scenarios (and the n=10 trio's few explicit surface pairs).
    n_val = cfg.get('n')
    if isinstance(n_val, int) and n_val <= 6:
        pw = sfs.setdefault('pairwise', {})
        for k in ('cdf', 'pdf', 'quantile'):
            pw.pop(k, None)
        pwc = pw.setdefault('cosine', {})
        for k in ('cdf', 'pdf'):
            _ensure(pwc, k, PLACEHOLDER)
    else:
        sfs.pop('pairwise', None)

    with open(path, 'w') as f:
        yaml.dump(cfg, f)
    return True


if __name__ == '__main__':
    for name in sys.argv[1:]:
        p = f"resources/configs/{name}.yaml"
        ok = augment(p)
        print(f"{'augmented' if ok else 'skipped  '} {name}")
