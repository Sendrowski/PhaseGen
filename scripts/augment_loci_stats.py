"""
Augment a two-locus *branch-length* scenario config (``n_loci: 2``, e.g. ``1_epoch_2_loci_n_4_r_1``) with the COSINE
cross-locus pairwise joint CDF and PDF of the per-locus tree height and total branch length -- the distributional
extension of the existing ``loci: {cov, corr}`` cross-locus moments. Run:

    python scripts/augment_loci_stats.py <config_name> ...

For each ``tree_height`` / ``total_branch_length`` tolerance block that already carries a ``loci:`` sub-block, insert
a ``pairwise: {cosine: {cdf, pdf}}`` entry (placeholder tolerances tightened later by tune_dist_tols.py). Uses a
surgical text edit so the ``!!python/tuple`` migration keys of multi-population two-locus configs are preserved
(see the ruamel tuple-key hazard); pass ``--cdf-only`` for fully-linked (r = 0) scenarios whose near-diagonal joint
density the 2D cosine cannot resolve."""
import sys

PLACEHOLDER = 0.5


def augment(path: str, cdf_only: bool = False) -> bool:
    with open(path) as f:
        lines = f.readlines()
    out, touched = [], False
    for line in lines:
        out.append(line)
        # a ``loci:`` mapping header (under tree_height / total_branch_length, at 6-space indent); insert the pairwise
        # joint block as its first child at child indent. Skip if the next non-appended insert would duplicate.
        stripped = line.rstrip('\n')
        if stripped.endswith('loci:') and stripped.lstrip() == 'loci:':
            ind = line[:len(line) - len(line.lstrip())]
            child = ind + '  '
            block = [f'{child}pairwise:\n', f'{child}  cosine:\n', f'{child}    cdf: {PLACEHOLDER}\n']
            if not cdf_only:
                block.append(f'{child}    pdf: {PLACEHOLDER}\n')
            out.extend(block)
            touched = True
    # idempotency: if a pairwise block already existed, the file would now have two -- guard by refusing when present
    if touched and any('pairwise:' in l for l in lines):
        return False
    if touched:
        with open(path, 'w') as f:
            f.writelines(out)
    return touched


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--cdf-only']
    cdf_only = '--cdf-only' in sys.argv
    for name in args:
        ok = augment(f"resources/configs/{name}.yaml", cdf_only=cdf_only)
        print(f"{'augmented' if ok else 'skipped  '} {name}")
