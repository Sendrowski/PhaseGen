"""
Drive the full dist-stat pipeline for a wave of (already augmented) scenario configs:
  regenerate fixture -> tune tolerances to ~1.5x observed -> sync into fixture -> verify (assertions pass).

    python scripts/process_wave.py [--regen] [--jobs N] <config_name> ...

``--regen`` forces fixture re-simulation (needed when dist stats were just added); omit it to only re-tune/sync/verify
against an existing fixture. Prints a per-config PASS/FAIL summary at the end."""
import subprocess
import sys
import warnings

warnings.filterwarnings('ignore')

SNAKE = "/Users/janek/miniforge3/bin/snakemake"
SER = "results/comparisons/serialized"


def snake(targets, jobs=1):
    cmd = [SNAKE, "--use-conda", "--scheduler", "greedy", "-j", str(jobs), "-f", *targets]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def main(argv):
    regen = "--regen" in argv
    argv = [a for a in argv if a != "--regen"]
    jobs = 4
    if "--jobs" in argv:
        i = argv.index("--jobs"); jobs = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    names = argv

    if regen:
        print(f"[regen] {len(names)} fixtures (jobs={jobs}) ...", flush=True)
        rc = snake([f"{SER}/{n}.json" for n in names], jobs)
        print(f"[regen] snakemake rc={rc}", flush=True)

    import importlib
    tune = importlib.import_module("tune_dist_tols")
    from phasegen.comparison import Comparison
    # NB: do NOT globally disable logging -- the tuner reads the comparison's INFO log lines to learn the observed
    # diffs. Noise is suppressed via warnings filtering + per-logger ERROR levels inside the tuner instead.

    results = {}
    for n in names:
        try:
            k = tune.retune(n)
        except Exception as e:
            results[n] = f"TUNE-ERR {type(e).__name__}: {e}"; print(f"  {n}: {results[n]}", flush=True); continue
        # sync yaml tolerances into the fixture
        subprocess.run(["rm", "-f", f"{SER}/.{n}.tolerances_synced"])
        rc = snake([f"{SER}/.{n}.tolerances_synced"], 1)
        # verify
        try:
            c = Comparison.from_file(f"{SER}/{n}.json")
            c.do_assertion = True; c.visualize = False
            c.compare(n)
            results[n] = f"PASS ({c.n_assertions} asserts, tuned {k})"
        except AssertionError as e:
            results[n] = f"FAIL {str(e)[:80]}"
        except Exception as e:
            results[n] = f"VERIFY-ERR {type(e).__name__}: {str(e)[:80]}"
        print(f"  {n}: {results[n]}", flush=True)

    print("\n===== WAVE SUMMARY =====")
    for n, r in results.items():
        print(f"  {'OK ' if r.startswith('PASS') else 'XX '} {n}: {r}")
    bad = [n for n, r in results.items() if not r.startswith('PASS')]
    print(f"{len(names) - len(bad)}/{len(names)} passed" + (f"; FAILED: {bad}" if bad else ""))


if __name__ == '__main__':
    sys.path.insert(0, "scripts")
    main(sys.argv[1:])
