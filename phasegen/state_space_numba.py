"""
Numba-accelerated kernels for single-locus state-space construction.

This module is imported behind a guard (:data:`HAS_NUMBA`); when numba is unavailable the public classes fall back
to the pure-Python construction in :mod:`phasegen.state_space`. The kernels operate on integer state rows (the
flattened ``lineages`` array of shape ``(n_demes, n_blocks)`` for a single locus) and build the rate matrix directly.

Coalescent rates are reproduced from the model formulae (exact ``comb`` via an integer loop, the Euler beta via
``math.lgamma``, and the binomial pmf via ``comb`` and powers), parameterised by a ``model_id`` (0 standard,
1 beta, 2 dirac) plus ``alpha``/``psi``/``c``. Per-deme timescales and the migration-rate matrix are precomputed in
Python and passed in, so no transcendental model code other than the rates lives here.

States are numbered in discovery order, which differs from the pure-Python enumeration; this is intentional and
validated by permutation-invariant parity tests.
"""

import math

import numpy as np

try:
    from numba import njit
    from numba.typed import Dict, List
    from numba.core import types

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba is absent
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op ``njit`` shim so the kernels remain importable without numba (the Python fallback is used)."""
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------------------------------------------------
# rate math (ports of the CoalescentModel formulae)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _comb(n, k):
    """Binomial coefficient C(n, k) as a float."""
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    if k > n - k:
        k = n - k
    r = 1.0
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


@njit(cache=True)
def _beta(a, b):
    """Euler beta function via log-gamma."""
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


@njit(cache=True)
def _binom_pmf(k, n, p):
    """Binomial pmf P(X = k) for X ~ Binom(n, p)."""
    if k < 0 or k > n:
        return 0.0
    return _comb(n, k) * p ** k * (1.0 - p) ** (n - k)


@njit(cache=True)
def _rate_pairwise(model_id, alpha, psi, c, b, k):
    """Reproduce ``CoalescentModel._get_rate(b, k)`` (lineage-counting merger rate)."""
    if model_id == 0:  # standard
        if k == 2:
            return b * (b - 1) / 2.0
        return 0.0

    if model_id == 1:  # beta
        if k < 1 or k > b:
            return 0.0
        base = _beta(k - alpha, b - k + alpha) / _beta(alpha, 2.0 - alpha)
        return _comb(b, k) * base

    # dirac
    rate_binary = b * (b - 1) / 2.0 if k == 2 else 0.0
    return rate_binary + _binom_pmf(k, b, psi) * c


@njit(cache=True)
def _rate_block(model_id, alpha, psi, c, n, b_arr, k_arr):
    """Reproduce ``CoalescentModel._get_rate_block_counting(n, b, k)`` for a merger touching ``len(b_arr)`` blocks."""
    m = b_arr.shape[0]

    if model_id == 0:  # standard
        if m == 1:
            return _rate_pairwise(0, alpha, psi, c, b_arr[0], k_arr[0])
        if m == 2 and k_arr[0] == 1 and k_arr[1] == 1:
            return float(b_arr[0] * b_arr[1])
        return 0.0

    sum_k = 0
    sum_b = 0
    for i in range(m):
        sum_k += k_arr[i]
        sum_b += b_arr[i]

    if model_id == 1:  # beta
        combs = 1.0
        for i in range(m):
            combs *= _comb(b_arr[i], k_arr[i])
        base = _beta(sum_k - alpha, n - sum_k + alpha) / _beta(alpha, 2.0 - alpha)
        return combs * base

    # dirac
    if m == 1:
        rate_binary = _rate_pairwise(0, alpha, psi, c, b_arr[0], k_arr[0])
    elif m == 2 and k_arr[0] == 1 and k_arr[1] == 1:
        rate_binary = float(b_arr[0] * b_arr[1])
    else:
        rate_binary = 0.0

    p_psi = 1.0
    for i in range(m):
        p_psi *= _binom_pmf(k_arr[i], b_arr[i], psi)
    if sum_b < n:
        p_psi *= _binom_pmf(0, n - sum_b, psi)

    return rate_binary + p_psi * c


# ---------------------------------------------------------------------------------------------------------------------
# collision-safe hash map over integer state rows (separate chaining via a typed.List)
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _hash_row(row):
    """FNV-1a hash of a non-negative integer row, returned as int64."""
    h = np.uint64(14695981039346656037)
    for i in range(row.shape[0]):
        h = (h ^ np.uint64(row[i] + 1)) * np.uint64(1099511628211)
    return np.int64(h)


@njit(cache=True)
def _rows_equal(a, b):
    """Whether two integer rows are element-wise equal."""
    for i in range(a.shape[0]):
        if a[i] != b[i]:
            return False
    return True


@njit(cache=True)
def _find_or_add(rows, chain_next, head, target):
    """Return the index of ``target`` in ``rows``, appending it (and updating the hash chains) if new."""
    h = _hash_row(target)
    if h in head:
        i = head[h]
        while i != -1:
            if _rows_equal(rows[i], target):
                return i
            i = chain_next[i]

    idx = len(rows)
    rows.append(target.copy())
    chain_next.append(head[h] if h in head else -1)
    head[h] = idx
    return idx


# ---------------------------------------------------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------------------------------------------------

@njit(cache=True)
def _build(initial, kind, n_demes, n_blocks, mig, timescales, model_id, alpha, psi, c, block_vectors):
    """
    Build the single-locus state graph by BFS over integer ``lineages`` rows.

    :param initial: Flattened initial lineage row (length ``n_demes * n_blocks``).
    :param kind: 0 lineage-counting, 1 block-/joint-counting.
    :param mig: ``(n_demes, n_demes)`` migration-rate matrix.
    :param timescales: per-deme timescale by which coalescence rates are divided.
    :param block_vectors: ``(n_blocks, vdim)`` block labels (descendant vectors / size classes); the merged block of a
        merger is the one whose label equals the summed label of the merging blocks (found by linear search).
    :return: ``(rows_arr, src_arr, dst_arr, rate_arr)`` — the state rows and the COO transitions.
    """
    dim = n_demes * n_blocks
    vdim = block_vectors.shape[1]

    rows = List()
    rows.append(initial.copy())
    chain_next = List()
    chain_next.append(np.int64(-1))
    head = Dict.empty(key_type=types.int64, value_type=types.int64)
    head[_hash_row(initial)] = 0

    src = List.empty_list(types.int64)
    dst = List.empty_list(types.int64)
    rate = List.empty_list(types.float64)

    cur = 0
    while cur < len(rows):
        source = rows[cur].copy()

        total = 0
        for x in range(dim):
            total += source[x]

        # --- migration: move one lineage of each block between demes, rate scaled by source count ---
        # the target states are discovered for every ordered deme pair regardless of the migration rate (matching
        # the pure-Python construction, so the state set is epoch-independent); only the edge is rate-gated
        for d1 in range(n_demes):
            for d2 in range(n_demes):
                if d1 == d2:
                    continue
                for blk in range(n_blocks):
                    cnt = source[d1 * n_blocks + blk]
                    if cnt > 0:
                        target = source.copy()
                        target[d1 * n_blocks + blk] -= 1
                        target[d2 * n_blocks + blk] += 1
                        tidx = _find_or_add(rows, chain_next, head, target)
                        if mig[d1, d2] != 0.0:
                            src.append(np.int64(cur))
                            dst.append(np.int64(tidx))
                            rate.append(mig[d1, d2] * cnt)

        # --- coalescence (only when more than one lineage remains) ---
        if total > 1:
            for deme in range(n_demes):
                base = deme * n_blocks

                deme_total = 0
                for blk in range(n_blocks):
                    deme_total += source[base + blk]
                if deme_total < 2:
                    continue

                ts = timescales[deme]

                if kind == 0:
                    # lineage-counting: merge k of the deme's lineages into one (count decreases by k - 1)
                    cnt = source[base]
                    for k in range(2, cnt + 1):
                        r = _rate_pairwise(model_id, alpha, psi, c, cnt, k)
                        if r == 0.0:
                            continue
                        target = source.copy()
                        target[base] = cnt - (k - 1)
                        tidx = _find_or_add(rows, chain_next, head, target)
                        src.append(np.int64(cur))
                        dst.append(np.int64(tidx))
                        rate.append(r / ts)
                    continue

                # block-/joint-counting: enumerate merger combinations over present blocks via an odometer
                present = List.empty_list(types.int64)
                for blk in range(n_blocks):
                    if source[base + blk] > 0:
                        present.append(np.int64(blk))
                n_present = len(present)

                comb = np.zeros(n_present, dtype=np.int64)
                while True:
                    # advance odometer: comb[j] in 0..count[present[j]]
                    j = 0
                    while j < n_present:
                        comb[j] += 1
                        if comb[j] <= source[base + present[j]]:
                            break
                        comb[j] = 0
                        j += 1
                    if j == n_present:
                        break

                    sum_comb = 0
                    for x in range(n_present):
                        sum_comb += comb[x]
                    if sum_comb < 2:
                        continue

                    # merging blocks: gather b (current counts) and k (comb), and the summed label
                    m = 0
                    for x in range(n_present):
                        if comb[x] > 0:
                            m += 1
                    bb = np.empty(m, dtype=np.int64)
                    kk = np.empty(m, dtype=np.int64)
                    label = np.zeros(vdim, dtype=np.int64)
                    idx = 0
                    for x in range(n_present):
                        if comb[x] > 0:
                            blk = present[x]
                            bb[idx] = source[base + blk]
                            kk[idx] = comb[x]
                            for v in range(vdim):
                                label[v] += comb[x] * block_vectors[blk, v]
                            idx += 1

                    r = _rate_block(model_id, alpha, psi, c, deme_total, bb, kk)
                    if r == 0.0:
                        continue

                    # find merged block by matching the summed label
                    merged_idx = -1
                    for blk in range(n_blocks):
                        ok = True
                        for v in range(vdim):
                            if block_vectors[blk, v] != label[v]:
                                ok = False
                                break
                        if ok:
                            merged_idx = blk
                            break

                    target = source.copy()
                    for x in range(n_present):
                        if comb[x] > 0:
                            target[base + present[x]] -= comb[x]
                    target[base + merged_idx] += 1

                    tidx = _find_or_add(rows, chain_next, head, target)
                    src.append(np.int64(cur))
                    dst.append(np.int64(tidx))
                    rate.append(r / ts)

        cur += 1

    # convert to arrays
    n_states = len(rows)
    rows_arr = np.empty((n_states, dim), dtype=np.int64)
    for i in range(n_states):
        rows_arr[i] = rows[i]

    n_edges = len(src)
    src_arr = np.empty(n_edges, dtype=np.int64)
    dst_arr = np.empty(n_edges, dtype=np.int64)
    rate_arr = np.empty(n_edges, dtype=np.float64)
    for i in range(n_edges):
        src_arr[i] = src[i]
        dst_arr[i] = dst[i]
        rate_arr[i] = rate[i]

    return rows_arr, src_arr, dst_arr, rate_arr


def build_rate_matrix(
        initial: np.ndarray,
        kind: int,
        n_demes: int,
        n_blocks: int,
        mig: np.ndarray,
        timescales: np.ndarray,
        model_id: int,
        alpha: float,
        psi: float,
        c: float,
        block_vectors: np.ndarray,
):
    """
    Python entry point: build the single-locus state rows and dense rate matrix via the numba kernel.

    :return: ``(rows, S)`` where ``rows`` is ``(n_states, n_demes * n_blocks)`` integer lineage rows (discovery
        order) and ``S`` is the dense intensity matrix (diagonal filled with the negative row sums).
    """
    rows, src, dst, rate = _build(
        initial.astype(np.int64), kind, n_demes, n_blocks,
        mig.astype(np.float64), timescales.astype(np.float64),
        model_id, float(alpha), float(psi), float(c),
        block_vectors.astype(np.int64),
    )

    n = rows.shape[0]
    S = np.zeros((n, n))
    # accumulate (a target may be reached by several transitions, e.g. multiple merger paths)
    np.add.at(S, (src, dst), rate)
    S[np.diag_indices_from(S)] = -S.sum(axis=1)

    return rows, S
