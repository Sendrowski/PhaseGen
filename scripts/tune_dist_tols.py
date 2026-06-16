"""
Tighten ("time") the comparison tolerances of one or more scenario configs to ~1.5x their observed diffs, by running
the comparison against the (already regenerated) fixture and rewriting each tolerance leaf. The fixtures are fixed and
phasegen is deterministic, so the observed diff is reproducible -- 1.5x gives modest headroom for library drift.

    python scripts/tune_dist_tols.py <config_name> [<config_name> ...]

Leaves with no matching observed diff are left untouched. Run AFTER regenerating the fixture(s)."""
import ast
import logging
import math
import re
import sys
import warnings

warnings.filterwarnings('ignore')
from ruamel.yaml import YAML

from phasegen.comparison import Comparison

yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 4096
# jSFS/multi-pop configs carry ``!!python/tuple`` migration-rate keys; without explicit round-trip handling ruamel
# rewrites them as plain (unhashable) lists, which breaks the ``yaml.full_load`` in ``Comparison.from_yaml``. Register
# the tag so load/dump preserves the tuple keys verbatim.
yaml.constructor.add_constructor(
    'tag:yaml.org,2002:python/tuple', lambda loader, node: tuple(loader.construct_sequence(node)))
yaml.representer.add_representer(
    tuple, lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:python/tuple', list(data)))

KINDS = {'pdf', 'cdf', 'quantile', 'mean', 'var', 'std', 'cov', 'corr', 'm3', 'm4',
         'theta_pi', 'theta_w', 'tajimas_d', 'mutation_configs'}
MODES = {'cosine', 'de_hoog'}


def nudge(d: float) -> float:
    if d <= 0:
        return 0.0005
    t = d * 1.5
    e = math.floor(math.log10(t))
    f = 10 ** (e - 1)
    return round(math.ceil(t / f) * f, 6)


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
    """If a YAML key is a bracketed collection literal (an SFS bin list ``[1, 4, 9]`` or a pairwise pair list
    ``[(1, 2), (1, 9)]``), return its elements, else ``None``."""
    if not (isinstance(key, str) and key.startswith('[') and key.endswith(']')):
        return None
    try:
        v = ast.literal_eval(key)
        return list(v) if isinstance(v, (list, tuple)) else None
    except (ValueError, SyntaxError):
        return None


def title_for(path: list) -> list:
    """Map a YAML tolerance-leaf key path to the comparison title(s) it covers (sans scenario prefix). Returns a
    *list*: a collection key (an SFS bin list or an explicit pairwise pair list) expands to one title per element, all
    sharing the single tolerance leaf (the tuner then takes the worst observed across them)."""
    path = [str(p) for p in path]
    kind = path[-1]
    mode = next((p for p in path if p in MODES), None)
    coll = next((_collection(p) for p in path if _collection(p) is not None), None)

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


def retune(name: str, tighten_only: bool = False) -> int:
    """Rewrite each matched tolerance leaf to ``nudge(observed)`` (1.5x the observed diff). With ``tighten_only`` the
    leaf is set to ``min(current, nudge(observed))`` so a tolerance can only ever be tightened, never loosened -- used
    to drive down stale/degenerate (over-loose) tolerances without risking a currently-tight, barely-passing leaf."""
    obs = observed(name)
    p = f"resources/configs/{name}.yaml"
    with open(p) as f:
        cfg = yaml.load(f)
    tol = cfg.get('comparisons', {}).get('tolerance')
    if tol is None:
        return 0
    changed = [0]

    def walk(node, path):
        for k, v in list(node.items()):
            kp = path + [str(k)]
            if hasattr(v, 'items'):
                walk(v, kp)
            elif str(k) in KINDS:
                matched = [obs[t] for t in title_for(kp) if t in obs]
                if matched:
                    new = nudge(max(matched))  # one leaf can cover several titles (bin/pair list) -> worst observed
                    if tighten_only and isinstance(v, (int, float)):
                        new = min(float(v), new)
                    if new != v:
                        node[k] = new
                        changed[0] += 1

    walk(tol, [])
    with open(p, 'w') as f:
        yaml.dump(cfg, f)
    return changed[0]


if __name__ == '__main__':
    args = sys.argv[1:]
    tighten_only = '--tighten-only' in args
    for name in [a for a in args if not a.startswith('--')]:
        n = retune(name, tighten_only=tighten_only)
        print(f"{'tightened' if tighten_only else 'tuned'} {n:>3} tolerances in {name}")
