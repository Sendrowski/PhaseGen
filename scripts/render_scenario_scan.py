"""Render every comparison's diff plot for the non-slow scenario suite and write a manifest tying
each comparison (config, stat, metric, diff, tol, ...) to its PNG.

The non-slow set is the single source of truth in ``testing.test_scenarios`` (all ``configs`` minus
``slow_configs``). Parallel; Agg backend; low DPI to keep the on-disk footprint small. Driven by the
``render_scenario_scan`` Snakemake rule, which passes the output directory as the sole argument.
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import re
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from testing.test_scenarios import configs, slow_configs

OUTDIR = sys.argv[1]  # plots + manifest go here
CONFIGS = [c for c in configs if c not in slow_configs]
_PAT = re.compile(r'^#\d+\s+(.*?):\s+\S+\s+<=\s+\S+\s+\((.*?),\s+([\d.]+)s\)\s*$')


def run_one(config):
    import matplotlib
    matplotlib.use("Agg")
    from phasegen.comparison import Comparison
    recs = []
    cur = {"name": None}
    orig = Comparison._save_and_show

    def scap(self, name, pad=2, extra_right=0.0):
        cur["name"] = name
        return orig(self, name, pad, extra_right)

    def lcap(self, msg, diff, tol):
        recs.append((msg, float(diff), float(tol), cur["name"]))
        cur["name"] = None

    Comparison._save_and_show = scap
    Comparison._log_result = lcap
    try:
        c = Comparison.from_file(f"results/comparisons/serialized/{config}.json")
        c.do_assertion = False
        c.visualize = True
        c.show_title = True
        c.dpi = 72
        c.figure_path = os.path.join(OUTDIR, config)
        c.compare(title=config)
    except Exception as e:
        recs.append((f"#0 {config}: ERROR {type(e).__name__}: {e} <= x (error, 0.0s)", float("nan"), float("nan"), None))
    return config, recs


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = []
    n = len(CONFIGS)
    done = 0
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, 10)) as ex:
        futs = {ex.submit(run_one, c): c for c in CONFIGS}
        for fut in as_completed(futs):
            config, recs = fut.result()
            done += 1
            nplot = 0
            for msg, diff, tol, name in recs:
                m = _PAT.match(msg)
                title, metric, rt = (m.group(1), m.group(2), float(m.group(3))) if m else (msg, "?", 0.0)
                stat = title[len(config) + 2:] if title.startswith(config + ":") else title
                ratio = (diff / tol) if (tol == tol and tol > 0) else None
                rel = f"{config}/{name}.png" if name and os.path.exists(os.path.join(OUTDIR, config, f"{name}.png")) else None
                if rel:
                    nplot += 1
                # the candidate operand rides in the stat path: a `...: empirical: ...` row was compared against
                # PhaseGen's own sampler (self-consistency), the rest against msprime (external ground truth)
                is_empirical = any(seg.strip() == 'empirical' for seg in stat.split(':'))
                manifest.append(dict(config=config, stat=stat, metric=metric, diff=diff, tol=tol,
                                     ratio=ratio, runtime=rt, plot=rel, empirical=is_empirical))
            print(f"[{done}/{n}] {config}: {len(recs)} rows, {nplot} plots", flush=True)

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    withp = sum(1 for r in manifest if r["plot"])
    print(f"TOTAL {len(manifest)} rows, {withp} with plots -> {OUTDIR}/manifest.json")
