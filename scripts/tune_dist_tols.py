"""
Retune the comparison tolerances of one or more scenario configs, by running the comparison against the (already
regenerated) fixture and rewriting each tolerance leaf to

    max( 1.5x observed diff,  SIGMA_FLOOR x the msprime reference's own standard error )

    python scripts/tune_dist_tols.py [--allow-loosen] [--only=kind,kind] <config_name> [<config_name> ...]

phasegen is deterministic and the fixture is fixed, so the observed diff is reproducible and 1.5x gives modest
headroom for library drift. But the *reference* is a finite sample, and a tolerance below its sampling error is not a
bound at all -- it records that one random draw happened to land close. Regenerate the fixture and the same
distribution yields a different realisation, which then fails for no reason. Hence the floor (see :func:`noise_floor`).

Each leaf is otherwise clamped to ``min(current, 1.5x observed)``, so a re-tune cannot walk a bound outward;
``--allow-loosen`` lifts that clamp (needed when a comparison's *metric* changed, not just its value). The floor
applies either way, and is the one thing that may raise a bound in tighten-only mode.

Leaves with no matching observed diff are left untouched. Run AFTER regenerating the fixture(s)."""
import ast
import logging
import math
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
from ruamel.yaml import YAML

from phasegen.comparison import Comparison

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096
# jSFS/multi-pop configs carry ``!!python/tuple`` migration-rate keys; without explicit round-trip handling ruamel
# rewrites them as plain (unhashable) lists, which breaks the ``yaml.full_load`` in ``Comparison.from_yaml``. Register
# the tag so load/dump preserves the tuple keys verbatim. ``flow_style=True`` keeps the compact inline form
# (``!!python/tuple ['pop_0', 'pop_1']:``) so a complex (tuple) mapping key is not blown up into the verbose
# block ``? ... :`` form, which would reformat every migration-rate line on an unrelated tolerance edit.
yaml.constructor.add_constructor(
    'tag:yaml.org,2002:python/tuple', lambda loader, node: tuple(loader.construct_sequence(node)))
yaml.representer.add_representer(
    tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:python/tuple', list(data),
                                                          flow_style=True))

KINDS = {'pdf', 'cdf', 'quantile', 'mean', 'var', 'std', 'cov', 'corr', 'm3', 'm4',
         'theta_pi', 'theta_w', 'tajimas_d', 'mutation_configs', 'mass'}
MODES = {'cosine', 'de_hoog'}


def nudge(d: float) -> float:
    if d <= 0:
        return 0.0005
    t = d * 1.5
    e = math.floor(math.log10(t))
    f = 10 ** (e - 1)
    return round(math.ceil(t / f) * f, 6)


#: Standard errors of the reference that a tolerance must clear, as a multiple. A tolerance below the noise of the very
#: sample it is compared against is not a bound at all: it records that one random draw happened to land close, and the
#: next regeneration of the fixture -- a different realisation of the *same* distribution -- breaks it for no reason.
#:
#: 4 sigma, because a scenario asserts dozens of statistics and the suite must not flake: the worst of that many draws
#: sits well beyond 2 sigma even when everything is correct.
SIGMA_FLOOR = 4.0


def noise_floor(comp, path: list) -> float:
    """The reference's own sampling error for one tolerance leaf, expressed on the scale its comparison metric is
    measured on. A tolerance is never set below :attr:`SIGMA_FLOOR` times this.

    The moment statistics read the standard errors the reference cached before dropping its samples
    (:meth:`~phasegen.distributions.empirical.EmpiricalDistribution.cache_standard_errors`) and divide by the value, as
    the metric itself does (``rel_diff`` is the worst *relative* difference over the bins, so the worst *relative*
    standard error over the bins is the matching scale). The remaining metrics get an analytic scale: a CDF is compared
    absolutely, and its binomial standard error peaks at ``0.5 / sqrt(n)``.

    The pdf (total variation) and quantile (Wasserstein) metrics integrate over the whole grid, so their noise is
    concentrated -- unlike a signed moment difference, an observed value cannot land far below it by luck -- and 1.5x
    the observed diff is already a reproducible bound. They take the generic ``1 / sqrt(n)`` scale.

    :param comp: The comparison, carrying the cached ground truth.
    :param path: The tolerance leaf's YAML key path, whose head selects the reference: an ``empirical`` block is
        compared against phasegen's own sampler (with its own, much smaller, replicate count), everything else against
        msprime.
    :return: The reference's standard error on the metric's own scale, or 0 if it cannot be determined.
    """
    operand = comp.empirical if path[0] == 'empirical' else comp.ms
    path = path[1:] if path[0] == 'empirical' else path

    ref = getattr(operand, path[0], None)
    if ref is None:
        return 0.0

    if 'atom' in path:
        return atom_noise_floor(ref, path)

    n = getattr(ref, 'n_samples', None)
    if not n:
        return 0.0
    root = np.sqrt(float(n))

    kind, factor = path[-1], 1.0
    if kind == 'std':
        kind, factor = 'var', 0.5  # SE[std] / std is half the relative error of the variance

    errors = getattr(ref, 'standard_errors', None) or {}

    # a deme/locus cov/corr leaf (e.g. tree_height/loci/cov) compares a *matrix*, so its noise floor is the block SE
    # of that matrix (keyed "demes.cov" etc.) over the matrix itself, not the scalar total's variance error
    container = path[-2] if len(path) >= 3 and path[-2] in ('demes', 'loci') else None
    if container and kind in ('cov', 'corr'):
        se_key = f"{container}.{kind}"
        attr = {'demes.cov': 'pops_cov', 'demes.corr': 'pops_corr',
                'loci.cov': 'loci_cov', 'loci.corr': 'loci_corr'}[se_key]
        matrix = getattr(ref, attr, None)
        if se_key in errors and matrix is not None:
            value = np.abs(np.asarray(matrix, dtype=float))
            rel = np.divide(np.asarray(errors[se_key], dtype=float), value, out=np.zeros_like(value), where=value > 0)
            return factor * float(np.max(rel[np.isfinite(rel)], initial=0.0))
        return 1.0 / root

    if kind in errors:
        value = np.abs(np.asarray(getattr(ref, kind), dtype=float))
        # the same denominator the metric uses; a vanishing entry (an off-diagonal covariance) is left out rather than
        # dividing by zero -- the metric blows up there too, and such a leaf is tuned from its observed diff alone
        rel = np.divide(np.asarray(errors[kind], dtype=float), value, out=np.zeros_like(value), where=value > 0)
        return factor * float(np.max(rel[np.isfinite(rel)], initial=0.0))

    if kind == 'cdf':
        return 0.5 / root

    return 1.0 / root


def atom_noise_floor(ref, path: list) -> float:
    """The noise floor of an atom-conditional leaf. Its reference is not the full sample but the sub-sample on which
    the conditioning reward vanishes, ``{R_on = 0}``, which is smaller by the atom's mass and correspondingly noisier;
    an atom on a rare bin can leave a few thousand replicates out of a million. The leaf covers both conditioning axes,
    so the worst of the two applies.

    The atom mass is a proportion of the *full* sample, and is compared absolutely, so its error is binomial.

    :param ref: The reference distribution, carrying the cached atom-conditional ground truth.
    :param path: The tolerance leaf's key path (below any ``empirical`` head).
    :return: The reference's standard error on the metric's own scale, or 0 if it cannot be determined.
    """
    pairs = next((_collection(p) for p in path if _collection(p) is not None), None) or []
    kind = path[-1]

    floors = [0.0]
    for i, j, on, mass, dist in getattr(ref, '_atom_conditional', []):
        if (i, j) not in pairs:
            continue

        if kind == 'mass':
            floors.append(float(np.sqrt(max(mass, 0.0) * (1.0 - mass) / ref.n_samples)))
            continue

        # an axis whose reward never vanishes has no atom and no conditional sample, and asserts only a zero mass
        if dist is None or not dist.n_samples:
            continue

        errors = getattr(dist, 'standard_errors', None) or {}
        if kind in errors:
            value = abs(float(np.asarray(getattr(dist, kind), dtype=float)))
            floors.append(float(errors[kind]) / value if value > 0 else 0.0)
        elif kind == 'cdf':
            floors.append(0.5 / np.sqrt(dist.n_samples))
        else:
            floors.append(1.0 / np.sqrt(dist.n_samples))

    return max(floors)


def observed(name: str) -> dict:
    """Run the comparison and return {title-path-without-scenario-prefix: observed diff}."""
    obs = {}
    pat = re.compile(r'#\d+ [^:]+: (.+?): ([\d.]+(?:e-?\d+)?) [<>]=?')

    class H(logging.Handler):
        def emit(self, r):
            m = pat.search(r.getMessage())
            if m:
                obs[m.group(1).strip()] = float(m.group(2))

    for nm in ('JointRewardDistribution', 'RewardDistribution', 'Reward'):
        logging.getLogger(nm).setLevel(logging.ERROR)
    comp = Comparison.from_file(f"results/comparisons/serialized/{name}.json")
    comp.comparisons = Comparison.from_yaml(f"resources/configs/{name}.yaml").comparisons
    comp.do_assertion = False
    comp.visualize = False
    h = H()
    comp.logger.addHandler(h)
    comp.logger.setLevel(logging.INFO)
    comp.compare(name)
    comp.logger.removeHandler(h)
    return obs


def _collection(key: str):
    """Elements broadcast by a YAML collection key, else ``None``. A *list* literal broadcasts over its elements -- an
    SFS bin list ``[1, 4, 9]`` or a pairwise pair list ``[(1, 2), (1, 9)]``. A bare *tuple* literal is a single
    pairwise pair, returned as a one-element list -- an SFS bin pair ``(1, 2)`` or a jSFS config pair
    ``((1, 0), (0, 1))`` -- so a per-index (single-pair) tolerance is matched and tuned too."""
    if not (isinstance(key, str) and key[:1] in '[('):
        return None
    try:
        v = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return None
    if isinstance(v, list):
        return list(v)
    if isinstance(v, tuple):
        return [v]  # a single pair (SFS (i, j) or jSFS ((..), (..)))
    return None


def title_for(path: list) -> list:
    """Map a YAML tolerance-leaf key path to the comparison title(s) it covers (sans scenario prefix). Returns a
    *list*: a collection key (an SFS bin list or an explicit pairwise pair list) expands to one title per element, all
    sharing the single tolerance leaf (the tuner then takes the worst observed across them)."""
    path = [str(p) for p in path]
    kind = path[-1]
    mode = next((p for p in path if p in MODES), None)
    coll = next((_collection(p) for p in path if _collection(p) is not None), None)

    if 'atom' in path:
        # ``<dist>: conditional: (i, j): atom: <kind>`` is logged once per conditioning axis, both sharing the leaf
        return [f"{path[0]}: conditional ({p[0]}, {p[1]}) atom on {axis}: {kind}"
                for p in (coll or []) for axis in ('a', 'b')]

    if 'pairwise' in path:
        i = path.index('pairwise')
        stat = ': '.join(path[:i])
        md = (mode + ': ') if mode else ''
        if coll is not None:
            # an explicit pairwise pair list is logged per pair as ``stat: mode: pairwise (i, j) kind``
            return [f"{stat}: {md}pairwise ({p[0]}, {p[1]}) {kind}" for p in coll]
        # aggregate pairwise: ``stat: mode: {loci_,}pairwise_kind`` (cross-locus joints log ``loci_pairwise_*``)
        prefix = 'loci_pairwise' if 'loci' in path[:i] else 'pairwise'
        mid = (': ' + mode) if mode else ''
        return [f"{stat}{mid}: {prefix}_{kind}"]

    if coll is not None:
        # an SFS bin list ``sfs: [1, 4, 9]: mode: kind`` is logged per bin as ``sfs: mode: bin: kind``
        root = path[0]
        md = (mode + ': ') if mode else ''
        return [f"{root}: {md}{b}: {kind}" for b in coll]

    return [': '.join(path)]


def retune(name: str, tighten_only: bool = True, only: set = None) -> int:
    """Rewrite each matched tolerance leaf to ``nudge(observed)`` (1.5x the observed diff), but never below the
    reference's own sampling error (:attr:`SIGMA_FLOOR` times :func:`noise_floor`), and clamped to
    ``min(current, ...)`` unless ``tighten_only`` is off: the 1.5x rule re-derives a leaf from whatever the current
    fixture yields, so without the clamp a re-tune after a fixture regeneration loosens every bound whose fresh
    observation is a little worse.

    The floor is what makes a tolerance a *bound* rather than a record of one lucky draw. 1.5x an observed diff that
    happens to land far below the sampling noise is not reproducible: regenerate the fixture and the same distribution
    yields a different realisation, which then fails.

    With ``only`` (a set of statistic kinds) only those kinds are retuned, leaving every other leaf untouched -- used
    to retune a single metric (e.g. after changing the difference metric for ``mutation_configs``)."""
    obs = observed(name)
    comp = Comparison.from_file(f"results/comparisons/serialized/{name}.json")
    p = f"resources/configs/{name}.yaml"
    with open(p) as f:
        orig = f.read()
    cfg = yaml.load(orig)
    tol = cfg.get('comparisons', {}).get('tolerance')
    if tol is None:
        return 0
    changed = [0]

    def walk(node, path):
        for k, v in list(node.items()):
            kp = path + [str(k)]
            if hasattr(v, 'items'):
                walk(v, kp)
            elif str(k) in KINDS and (only is None or str(k) in only):
                matched = [obs[t] for t in title_for(kp) if t in obs]
                if matched:
                    new = nudge(max(matched))  # one leaf can cover several titles (bin/pair list) -> worst observed
                    if tighten_only and isinstance(v, (int, float)):
                        new = min(float(v), new)
                    # never tighter than the noise of the sample being compared against, however lucky this draw was
                    new = max(new, nudge(SIGMA_FLOOR * noise_floor(comp, kp) / 1.5))
                    if new != v:
                        node[k] = new
                        changed[0] += 1

    walk(tol, [])
    import io
    buf = io.StringIO()
    yaml.dump(cfg, buf)
    # ruamel round-trips every line but cannot reproduce the hand-written formatting of the ``!!python/tuple``
    # migration-rate keys (the only complex/tagged keys), so it would reformat that block on an unrelated tolerance
    # edit. Splice the original ``migration_rates`` block back verbatim -- the tolerance subtree is what changed.
    new_text = _restore_top_level_block(orig, buf.getvalue(), 'migration_rates')
    with open(p, 'w') as f:
        f.write(new_text)
    return changed[0]


def _restore_top_level_block(orig: str, new: str, key: str) -> str:
    """Return ``new`` with its top-level ``key:`` block (the ``key:`` line plus the following indented/blank lines,
    until the next top-level key) replaced by the one from ``orig`` -- so a round-trip that reformats only that block
    leaves it byte-identical to the original."""
    def block(text):
        lines = text.splitlines(keepends=True)
        start = next((i for i, l in enumerate(lines) if l.startswith(key + ':')), None)
        if start is None:
            return None
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in ' \t'):
            end += 1
        return lines, start, end

    ob, nb = block(orig), block(new)
    if ob is None or nb is None:
        return new
    olines, os_, oe = ob
    nlines, ns, ne = nb
    return ''.join(nlines[:ns] + olines[os_:oe] + nlines[ne:])


if __name__ == '__main__':
    args = sys.argv[1:]
    tighten_only = '--allow-loosen' not in args
    only = next((set(a.split('=', 1)[1].split(',')) for a in args if a.startswith('--only=')), None)
    for name in [a for a in args if not a.startswith('--')]:
        # one config whose fixture is incomplete (e.g. an uncached pairwise surface) must not abort a whole batch
        try:
            n = retune(name, tighten_only=tighten_only, only=only)
            print(f"{'tightened' if tighten_only else 'tuned (loosening allowed)'} {n:>3} tolerances in {name}"
                  + (f" (only {', '.join(sorted(only))})" if only else ""))
        except Exception as e:
            print(f"SKIPPED {name}: {type(e).__name__}: {e}")
