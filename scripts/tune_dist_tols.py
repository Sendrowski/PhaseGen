"""
Tighten ("time") the comparison tolerances of one or more scenario configs to ~1.5x their observed diffs, by running
the comparison against the (already regenerated) fixture and rewriting each tolerance leaf. The fixtures are fixed and
phasegen is deterministic, so the observed diff is reproducible -- 1.5x gives modest headroom for library drift.

    python scripts/tune_dist_tols.py <config_name> [<config_name> ...]

Leaves with no matching observed diff are left untouched. Run AFTER regenerating the fixture(s)."""
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


def title_for(path: list) -> str:
    """Map a YAML tolerance-leaf key path to the comparison's title (sans scenario prefix)."""
    if 'pairwise' in path:
        i = path.index('pairwise')
        stat = ': '.join(path[:i])
        kind = path[-1]
        modes = [p for p in path[i + 1:-1] if p in MODES]
        mid = (': ' + ': '.join(modes)) if modes else ''
        return f"{stat}{mid}: pairwise_{kind}"
    return ': '.join(path)


def retune(name: str) -> int:
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
                t = title_for(kp)
                if t in obs:
                    node[k] = nudge(obs[t])
                    changed[0] += 1

    walk(tol, [])
    with open(p, 'w') as f:
        yaml.dump(cfg, f)
    return changed[0]


if __name__ == '__main__':
    for name in sys.argv[1:]:
        n = retune(name)
        print(f"tuned {n:>3} tolerances in {name}")
