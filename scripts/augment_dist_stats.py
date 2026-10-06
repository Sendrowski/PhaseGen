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

    # number of lineages: a scalar ``n`` or the single-population dict form ``n: {pop_0: k}``
    n_raw = cfg.get('n')
    if isinstance(n_raw, int):
        n_lineages = n_raw
    elif isinstance(n_raw, dict) and len(n_raw) == 1:
        n_lineages = int(next(iter(n_raw.values())))
    else:
        n_lineages = None

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

    # SFS 1D curves: cosine pdf/cdf/quantile on 2-3 *representative* bins (low / mid / high frequency) via the per-bin
    # path, rather than the spectrum-wide curve over every bin -- same representative-selection rationale as the
    # pairwise joint, and it scales to large n. Drop any spectrum-wide (un-wrapped or cosine) variants and any prior
    # bin selection. The spectrum-wide moments (mean/cov/corr) are kept (cheap, comprehensive).
    sfs = tol.setdefault('sfs', {})
    # a manual representative-bin selection (a '[...]' bin-list key) is left untouched; only the spectrum-wide curve
    # variants are stripped and converted
    has_bin_sel = any(isinstance(k, str) and k.startswith('[') for k in sfs)
    for k in ('pdf', 'cdf', 'quantile', 'cosine'):
        sfs.pop(k, None)
    hi = (n_lineages - 1) if n_lineages else 0
    sfs_bins = sorted({b for b in (1, max(2, hi // 2), hi) if 1 <= b <= hi})
    if sfs_bins and not has_bin_sel:
        bkey = '[' + ', '.join(str(b) for b in sfs_bins) + ']'
        sfs[bkey] = {'cosine': {'pdf': PLACEHOLDER, 'cdf': PLACEHOLDER, 'quantile': PLACEHOLDER}}
    # within-tree pairwise joint cdf/pdf for 2-3 *representative* bin pairs (a per-pair full-grid surface comparison),
    # rather than the aggregate over ALL O(n^2) pairs -- the latter is wasteful for n > 3 and does not scale, while a
    # fixed handful of pairs is cheap at any n (so we can exercise the joint even for large n). The pairs span the
    # informative regimes: (1, 2) low-low (most branch length, most correlated), (1, n-1) low-high (the anti-correlated
    # extremes) and (2, n-1) mid-high. The list key broadcasts the {cdf, pdf} tolerance over each pair.
    if n_lineages and n_lineages >= 3:
        pairs = [(1, 2)]
        if n_lineages - 1 >= 3:  # enough polymorphic bins for the low-high / mid-high extremes to be distinct
            pairs += [(1, n_lineages - 1), (2, n_lineages - 1)]
        pw = sfs.setdefault('pairwise', {})
        for k in ('cdf', 'pdf', 'quantile'):  # drop any aggregate (all-pairs) leaves
            pw.pop(k, None)
        pwc = pw.setdefault('cosine', {})
        for k in ('cdf', 'pdf'):
            pwc.pop(k, None)
        key = '[' + ', '.join(f'({i}, {j})' for i, j in pairs) + ']'  # broadcast tol over the representative pairs
        if not any(k not in ('cdf', 'pdf') for k in pwc):  # only add if no explicit pair keys already present
            pwc[key] = {'cdf': PLACEHOLDER, 'pdf': PLACEHOLDER}
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
