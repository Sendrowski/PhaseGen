"""
Exact gradients of first moments (:class:`phasegen.distributions._gradient.MeanGradient`) and of inference losses
(:class:`phasegen.gradient.Gradient`), against Richardson-extrapolated central differences. The steps are
relative 1e-3: the adaptive truncation of the sparse action makes the computed moments non-smooth at about 1e-13
relative, which smaller steps amplify.
"""
from typing import Callable, Dict

import numpy as np
import pytest

import phasegen as pg
from phasegen.gradient import Gradient
from phasegen.distributions._gradient import MeanGradient
from phasegen.inference import Inference
from phasegen.settings import Settings

PAIRS = [('pop_0', 'pop_1'), ('pop_1', 'pop_0')]


def _richardson(f: Callable[[float], float], x: float) -> float:
    """Richardson-extrapolated central difference of ``f`` at ``x``."""
    h = 1e-3 * max(abs(x), 0.1)
    d1 = (f(x + h) - f(x - h)) / (2 * h)
    d2 = (f(x + h / 2) - f(x - h / 2)) / h
    return (4 * d2 - d1) / 3


def _quantities(n_pops: int, n_epochs: int, seed: int) -> Dict[str, np.ndarray]:
    """Random population sizes, migration rates and epoch start times."""
    rng = np.random.default_rng(seed)
    return dict(
        N=np.exp(rng.normal(0, 0.5, (n_epochs, n_pops))),
        m=np.exp(rng.normal(-0.5, 0.5, (n_epochs, n_pops * (n_pops - 1)))),
        t=np.concatenate([[0], np.sort(rng.uniform(0.05, 2, n_epochs - 1))])
    )


def _coalescent(n, q: Dict[str, np.ndarray], model=None) -> pg.Coalescent:
    """The coalescent of the quantities."""
    pops = list(n) if isinstance(n, dict) else ['pop_0']
    pairs = [(a, b) for a in pops for b in pops if a != b]
    t = q['t']
    demography = pg.Demography(
        pop_sizes={p: {float(t[e]): float(q['N'][e, d]) for e in range(len(t))} for d, p in enumerate(pops)},
        migration_rates={pr: {float(t[e]): float(q['m'][e, j]) for e in range(len(t))} for j, pr in enumerate(pairs)}
    )
    return pg.Coalescent(n=n, demography=demography, **({} if model is None else dict(model=model)))


def _assert_mean_gradient(n, stat: str, model=None, n_epochs: int = 3, seed: int = 1, entry=None):
    """Compare the gradient of a mean, or of one entry of a spectrum mean, with central differences."""
    q = _quantities(len(n) if isinstance(n, dict) else 1, n_epochs, seed)

    def value(q):
        mean = getattr(_coalescent(n, q, model), stat).mean
        return float(mean) if entry is None else float(mean.data[entry])

    dist = getattr(_coalescent(n, q, model), stat)
    host, R, positions = dist._mean_rewards()
    reward = R[:, 0] if entry is None else R[:, positions.index(entry)]
    result, grad = MeanGradient(host).compute(reward)

    assert result == pytest.approx(value(q), rel=1e-12)

    used = [e.index for e in grad.epochs]
    exact, numeric = [], []
    for key, g in [('N', grad.pop_sizes), ('m', grad.migration_rates), ('t', grad.start_times[:, None])]:
        arr = q[key] if key != 't' else q[key][:, None]
        for idx in np.ndindex(arr.shape):
            if key == 't' and idx[0] == 0 or idx[0] not in used:
                continue

            def f(x, key=key, idx=idx):
                q2 = {k: v.copy() for k, v in q.items()}
                if key == 't':
                    q2['t'][idx[0]] = x
                else:
                    q2[key][idx] = x
                return value(q2)

            exact.append(g[(used.index(idx[0]),) + idx[1:]])
            numeric.append(_richardson(f, arr[idx]))

    exact, numeric = np.array(exact), np.array(numeric)
    np.testing.assert_allclose(exact, numeric, atol=1e-7 * np.abs(numeric).max())


@pytest.mark.parametrize('n, stat, model', [
    ({'pop_0': 2, 'pop_1': 2}, 'tree_height', None),
    ({'pop_0': 3, 'pop_1': 2}, 'total_branch_length', None),
    (6, 'tree_height', pg.BetaCoalescent(alpha=1.5)),
    (5, 'total_branch_length', pg.DiracCoalescent(psi=0.5, c=2)),
])
@pytest.mark.parametrize('action', [False, True])
def test_mean_gradient_matches_central_differences(n, stat, model, action):
    """The gradient over sizes, migration rates and epoch start times, on the dense and on the sparse-action path."""
    if action:
        Settings.expm_action_min_dim = 0
    _assert_mean_gradient(n, stat, model)


@pytest.mark.parametrize('n, stat, entry', [
    (6, 'sfs', (2,)),
    ({'pop_0': 3, 'pop_1': 2}, 'sfs', (1,)),
    ({'pop_0': 2, 'pop_1': 2}, 'jsfs', (1, 1)),
    ({'pop_0': 3, 'pop_1': 3}, 'fsfs', (2,)),
])
def test_spectrum_entry_gradient_matches_central_differences(n, stat, entry):
    """The gradient of a spectrum entry, on the flattened single-deme and on the block-counting spaces."""
    _assert_mean_gradient(n, stat, entry=entry)


def test_mean_gradient_two_loci():
    """The recombination part of the generator depends on no demographic quantity."""
    def mean(N):
        return pg.Coalescent(n=3, loci=2, recombination_rate=0.7,
                             demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: N}})).tree_height.mean

    th = pg.Coalescent(n=3, loci=2, recombination_rate=0.7,
                       demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 1.7}})).tree_height
    host, R, _ = th._mean_rewards()
    value, grad = MeanGradient(host).compute(R[:, 0])

    assert value == pytest.approx(th.mean, rel=1e-12)
    assert grad.pop_sizes[1, 0] == pytest.approx(_richardson(mean, 1.7), rel=1e-7)


@pytest.mark.parametrize('action', [False, True])
def test_mean_gradient_at_zero_migration_rate(action):
    """The derivative by a migration rate of zero, from the right. Regression: the gradient was taken on the sparsity
    pattern of the generator of the epoch, which lacks the migration entries at a rate of zero, about 70 times off."""
    if action:
        Settings.expm_action_min_dim = 0

    def tree_height(m):
        return pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': 1.5},
            migration_rates={('pop_0', 'pop_1'): {0: m, 0.5: 0.7}, ('pop_1', 'pop_0'): {0: 0.4, 0.5: 0.7}}
        )).tree_height

    th = tree_height(0.0)
    host, R, _ = th._mean_rewards()
    _, grad = MeanGradient(host).compute(R[:, 0])

    h = 1e-5
    one_sided = (-3 * th.mean + 4 * tree_height(h).mean - tree_height(2 * h).mean) / (2 * h)
    assert grad.migration_rates[0, 0] == pytest.approx(one_sided, rel=1e-6)


def test_generator_basis_failure_restores_epoch(monkeypatch):
    """A state space without a basis is returned to its epoch, and the failure is kept on it."""
    from phasegen.distributions._gradient import _GeneratorBasis

    def fail(self, *args):
        raise ValueError('not affine')

    monkeypatch.setattr(_GeneratorBasis, '_check', fail)
    th = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}})).tree_height
    ss = th.state_space
    epoch = ss.epoch

    with pytest.raises(ValueError, match='not affine'):
        _GeneratorBasis(ss)
    assert ss.epoch == epoch

    for _ in range(2):
        with pytest.raises(ValueError, match='not affine'):
            MeanGradient(th).compute(th.reward._get(ss))
    assert isinstance(ss.__dict__['_generator_basis'], ValueError)


def test_mean_gradient_undefined_without_certain_absorption():
    """Without certain absorption in the last epoch, the occupation has no closed form and no gradient."""
    coal = pg.Coalescent(n={'pop_0': 1, 'pop_1': 1}, demography=pg.Demography(pop_sizes={'pop_0': 1, 'pop_1': 1}))
    th = coal.tree_height
    assert MeanGradient(th).compute(th.reward._get(th.state_space)) is None


def _inference(coal: Callable, loss: Callable, x0: dict, bounds: dict, observation, **kwargs) -> Inference:
    return Inference(coal=coal, loss=loss, x0=x0, bounds=bounds, observation=observation, pbar=False, n_runs=1,
                     **kwargs)


def _sfs_loss(coal, obs):
    mean = coal.sfs.mean
    return pg.PoissonLikelihood().compute(obs.polymorphic, mean.polymorphic * obs.n_polymorphic / mean.n_polymorphic)


def _two_deme(**p) -> pg.Coalescent:
    times = [0, 0.3, 0.8]
    return pg.Coalescent(n={'pop_0': 3, 'pop_1': 2}, demography=pg.Demography(
        pop_sizes={'pop_0': {t: p[f'a{e}'] for e, t in enumerate(times)},
                   'pop_1': {t: p[f'b{e}'] for e, t in enumerate(times)}},
        migration_rates={('pop_0', 'pop_1'): p['m'], ('pop_1', 'pop_0'): p['m']}))


def _assert_loss_gradient(coal, loss, x0, bounds, observation, rtol=1e-7):
    """The gradient of ``Inference`` against central differences of its loss."""
    inf = _inference(coal, loss, x0, bounds, observation)
    f = Inference._get_loss_function(observation, x0, None, inf.get_coal, loss)
    evaluated = []

    def counted(c):
        if isinstance(c, pg.Coalescent):
            evaluated.append(1)
        return loss(c, observation)

    out = Gradient(inf.get_coal, counted)._exact(x0, bounds)
    assert out is not None

    # every parameter is differentiated through the moments, none by a difference of the loss
    assert not evaluated

    value, grad = out
    x = np.array(list(x0.values()))
    assert value == pytest.approx(f(x), rel=1e-12)

    numeric = np.array([_richardson(lambda y, i=i: f(np.where(np.arange(len(x)) == i, y, x)), x[i])
                        for i in range(len(x))])
    np.testing.assert_allclose(grad, numeric, atol=rtol * np.abs(numeric).max())


def test_loss_gradient_two_demes():
    """A Poisson likelihood of a two-deme SFS over per-epoch sizes and a shared migration rate."""
    x0 = {'a0': 1.2, 'a1': 0.6, 'a2': 1.5, 'b0': 0.8, 'b1': 1.1, 'b2': 2.0, 'm': 0.4}
    observation = pg.SFS(np.round(1000 * _two_deme(**{k: v * 0.8 for k, v in x0.items()}).sfs.mean.normalize().data))
    _assert_loss_gradient(_two_deme, _sfs_loss, x0, {k: (0.05, 5) for k in x0}, observation)


def test_loss_gradient_epoch_time():
    """A parameter that moves an epoch boundary enters through the start time of the epoch."""
    coal = lambda t, Ne: pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}}))
    observation = pg.SFS(np.round(1000 * coal(0.4, 0.3).sfs.mean.normalize().data))
    _assert_loss_gradient(coal, _sfs_loss, {'t': 0.7, 'Ne': 0.5}, {'t': (0.01, 4), 'Ne': (0.05, 5)}, observation)


def test_loss_gradient_model_parameter():
    """A parameter of the coalescent model is differentiated through the generator, with the demographic ones."""
    coal = lambda alpha, N: pg.Coalescent(n=6, model=pg.BetaCoalescent(alpha=alpha),
                                          demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    observation = pg.SFS(np.round(1000 * coal(1.5, 2).sfs.mean.normalize().data))
    _assert_loss_gradient(coal, _sfs_loss, {'alpha': 1.6, 'N': 1.5}, {'alpha': (1.01, 1.99), 'N': (0.1, 5)},
                          observation, rtol=1e-5)


@pytest.mark.parametrize('loss', [
    lambda coal, obs: float(np.sum(coal.sfs.var.polymorphic)),
    lambda coal, obs: float(coal.demography.get_epoch(0).pop_sizes['pop_0']),
    lambda coal, obs: 1.0,
])
def test_loss_gradient_unsupported_loss(loss):
    """A loss reading anything but a supported mean, or no mean at all, has no exact gradient."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_loss_gradient_unsupported_read_caught_by_loss():
    """A loss that catches every exception still has an unsupported read detected. Regression: a broad ``except``
    absorbed the signal of the unsupported read, and the gradient was taken of the value of the handler."""
    def loss(coal, obs):
        try:
            return float(coal.sfs.mean.polymorphic.sum() + coal.sfs.var.polymorphic.sum())
        except Exception:
            return 1e10

    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_loss_gradient_unsupported_read_raised_as_other_error():
    """A loss from another language, such as R, raises its own error in place of the signal of an unsupported read,
    which is still detected."""
    def loss(coal, obs):
        try:
            return float(coal.sfs.var.data[1])
        except BaseException as e:
            raise RuntimeError(f'Error in py_get_attr: {e}')

    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    gradient = Gradient(coal, lambda c: loss(c, None))

    assert gradient._exact({'N': 1.0}, {'N': (0.1, 5)}) is None
    assert not gradient.supported({'N': 1.0})


def test_loss_gradient_unsupported_read_at_perturbed_mean():
    """A loss that reads an unsupported quantity only at a perturbed mean has no exact gradient. Regression: the
    signal escaped every handler and aborted the optimization."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    reference = float(coal(1.0).sfs.mean.data[1])

    def loss(c, obs):
        mean = c.sfs.mean
        if mean.data[1] > reference:
            return float(c.sfs.var.data[1])
        return float(mean.data[1])

    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_loss_gradient_first_moment_of_the_sfs():
    """A loss reading the SFS through moment(1) has the gradient of the same loss through the mean."""
    x0 = {'a0': 1.2, 'a1': 0.6, 'a2': 1.5, 'b0': 0.8, 'b1': 1.1, 'b2': 2.0, 'm': 0.4}
    observation = pg.SFS(np.round(1000 * _two_deme(**{k: v * 0.8 for k, v in x0.items()}).sfs.mean.normalize().data))

    def loss(coal, obs):
        mean = coal.sfs.moment(1)
        return pg.PoissonLikelihood().compute(obs.polymorphic,
                                              mean.polymorphic * obs.n_polymorphic / mean.n_polymorphic)

    _assert_loss_gradient(_two_deme, loss, x0, {k: (0.05, 5) for k in x0}, observation)


def test_loss_gradient_first_moments_of_custom_rewards():
    """First moments of rewards other than those of the distributions, read through Coalescent.moment and the moment
    of a distribution, with reward objects created anew in each evaluation of the loss."""
    x0 = {'a0': 1.2, 'a1': 0.6, 'a2': 1.5, 'b0': 0.8, 'b1': 1.1, 'b2': 2.0, 'm': 0.4}

    def loss(coal, obs):
        length = coal.moment(1, rewards=(pg.TotalBranchLengthReward(),))
        height = coal.tree_height.moment(1, rewards=(pg.TreeHeightReward(),))
        return float((length - 4) ** 2 + np.log(height) + coal.jsfs.moment(1).data[1, 0])

    _assert_loss_gradient(_two_deme, loss, x0, {k: (0.05, 5) for k in x0}, None)


@pytest.mark.parametrize('loss', [
    lambda coal, obs: float(coal.tree_height.moment(2)),
    lambda coal, obs: float(coal.moment(1, end_time=1.0)),
    lambda coal, obs: float(coal.moment(2, rewards=(pg.TreeHeightReward(), pg.TotalBranchLengthReward()))),
])
def test_loss_gradient_unsupported_moments(loss):
    """Moments of higher order and moments in a time window have no exact gradient."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_loss_gradient_value_dependent_reads_unsupported():
    """A loss that reads a further moment depending on the value of another has no exact gradient, since its reads
    cannot be matched across evaluations."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    reference = float(coal(1.0).tree_height.mean)

    def loss(c, obs):
        height = c.tree_height.mean
        return float(height + (c.total_branch_length.mean if height > reference else 0))

    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_loss_gradient_shared_host_one_pass(monkeypatch):
    """Means on one state space share one adjoint pass."""
    calls = []
    compute = MeanGradient.compute_columns
    monkeypatch.setattr(MeanGradient, 'compute_columns', lambda self, R: calls.append(1) or compute(self, R))

    coal = lambda N: pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    loss = lambda c, obs: float(c.sfs.mean.data[1] * c.tree_height.mean)
    out = Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.3}, {'N': (0.1, 5)})

    assert len(calls) == 1
    assert out[1][0] == pytest.approx(_richardson(lambda N: loss(coal(N), None), 1.3), rel=1e-7)


def test_inference_unsupported_loss_uses_optimizer_differences():
    """A loss the exact gradient does not support is optimized exactly as with finite differences."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    loss = lambda c, obs: float((c.sfs.var.polymorphic.sum() - 3) ** 2)

    results = []
    for gradient in ['exact', 'finite_differences']:
        inf = _inference(coal, loss, {'N': 1.0}, {'N': (0.1, 5)}, None, gradient=gradient)
        inf.run()
        results.append((inf.loss_inferred, inf.result.nfev))

    assert results[0] == results[1]


def test_loss_gradient_windowed_mean_unsupported():
    """A mean accumulated from a positive start time has no exact gradient."""
    coal = lambda N: pg.Coalescent(n=4, start_time=0.2, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    loss = lambda c, obs: float(c.tree_height.mean)
    assert Gradient(coal, lambda c: loss(c, None))._exact({'N': 1.0}, {'N': (0.1, 5)}) is None


def test_inference_exact_gradient_reaches_finite_difference_optimum():
    """Both gradients reach the same optimum, the exact one with a fraction of the loss evaluations."""
    x0 = {'a0': 1.2, 'a1': 0.6, 'a2': 1.5, 'b0': 0.8, 'b1': 1.1, 'b2': 2.0, 'm': 0.4}
    observation = pg.SFS(np.round(1000 * _two_deme(**{k: v * 0.8 for k, v in x0.items()}).sfs.mean.normalize().data))

    results = {}
    for gradient in ['exact', 'finite_differences']:
        inf = _inference(_two_deme, _sfs_loss, x0, {k: (0.05, 5) for k in x0}, observation, gradient=gradient)
        inf.run()
        results[gradient] = inf

    assert results['exact'].loss_inferred == pytest.approx(results['finite_differences'].loss_inferred, rel=1e-6)
    assert results['exact'].result.nfev < results['finite_differences'].result.nfev / 3


def test_inference_logs_exact_gradient(caplog):
    """A supported loss is announced at INFO level, without a warning."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    with caplog.at_level('INFO'):
        _inference(coal, lambda c, obs: float(c.tree_height.mean), {'N': 1.0}, {'N': (0.1, 5)}, None).run()

    assert any(r.levelname == 'INFO' and 'exact gradient' in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelname == 'WARNING']


def test_inference_warns_on_unsupported_loss(caplog):
    """An unsupported loss is warned about once per run."""
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    with caplog.at_level('WARNING'):
        _inference(coal, lambda c, obs: float(c.sfs.var.data[1]), {'N': 1.0}, {'N': (0.1, 5)}, None).run()

    assert len([r for r in caplog.records if 'finite differences' in r.getMessage()]) == 1


def test_inference_warns_after_repeated_gradient_failures(caplog, monkeypatch):
    """Consecutive evaluations without an exact gradient switch the optimization to finite differences, with one
    warning."""
    monkeypatch.setattr(Gradient, '_exact', lambda self, params, bounds: None)
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    with caplog.at_level('WARNING'):
        inf = _inference(coal, lambda c, obs: (float(c.tree_height.mean) - 2.0) ** 2, {'N': 1.0}, {'N': (0.1, 5)},
                         None)
        inf.run()

    assert len([r for r in caplog.records if 'consecutive evaluations' in r.getMessage()]) == 1
    assert inf.loss_inferred == pytest.approx(0, abs=1e-8)


def test_inference_rejects_unknown_gradient():
    with pytest.raises(ValueError, match='gradient'):
        _inference(lambda N: pg.Coalescent(n=3), _sfs_loss, {'N': 1}, {'N': (0.1, 2)}, None, gradient='adjoint')


def _generator(scale: float) -> 'sp.csr_matrix':
    """The rate matrix of the block-counting state space of a sample of 12, scaled."""
    import scipy.sparse as sp
    return sp.csr_matrix(pg.Coalescent(n=12).sfs.state_space.S * scale)


@pytest.mark.parametrize('scale, shift, steps', [
    (0.01, 0, 1),  # a single step
    (0.1, 0, 2),  # Taylor series truncated early
    (20, 0, 100),  # many steps
    (1, -50, 8),  # large diagonal shift
])
def test_taped_action_matches_scipy_expm_multiply(scale, shift, steps):
    """The taped action equals scipy's expm_multiply, whose parameters it takes, from a single step to many."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    from phasegen.distributions._gradient import _TapedAction

    M = (_generator(scale) + shift * sp.identity(77)).tocsr()
    x = np.random.default_rng(1).uniform(0, 1, 77)
    action = _TapedAction(M)
    y, checkpoints = action.apply(x)

    assert action.steps >= steps
    np.testing.assert_allclose(y, expm_multiply(M, x), rtol=1e-12, atol=1e-14 * np.abs(y).max())

    if scale == 0.1:
        assert any(n_terms < action.degree + 1 for _, n_terms in checkpoints)


@pytest.mark.parametrize('scale, shift', [(0.01, 0), (0.1, 0), (5, 0), (1, -50)])
def test_taped_action_reverse_is_the_frechet_derivative(scale, shift):
    """The reverse pass gives the adjoint of the input, the transposed action, and the derivative by every entry of
    the matrix, the Frechet derivative of the exponential, both for a contraction with a vector."""
    import scipy.sparse as sp
    from scipy.linalg import expm, expm_frechet
    from phasegen.distributions._gradient import _TapedAction

    M = (_generator(scale) + shift * sp.identity(77)).tocsr()
    rng = np.random.default_rng(2)
    x, lam = rng.uniform(0, 1, 77), rng.uniform(-1, 1, 77)
    coo = M.tocoo()
    pick = rng.choice(len(coo.row), 25, replace=False)
    rows, cols = coo.row[pick], coo.col[pick]

    action = _TapedAction(M)
    _, checkpoints = action.apply(x)
    dx, dM = action.reverse(lam, checkpoints, rows, cols)

    dense = M.toarray()
    np.testing.assert_allclose(dx, expm(dense).T @ lam, rtol=1e-10, atol=1e-12 * np.abs(dx).max())

    frechet = []
    for r, c in zip(rows, cols):
        E = np.zeros_like(dense)
        E[r, c] = 1
        frechet.append(lam @ expm_frechet(dense, E, compute_expm=False) @ x)
    frechet = np.array(frechet)
    np.testing.assert_allclose(dM, frechet, rtol=1e-9, atol=1e-11 * np.abs(frechet).max())


def test_taped_action_of_a_multiple_of_the_identity():
    """A multiple of the identity, whose shifted matrix vanishes, is exponentiated without Taylor terms."""
    import scipy.sparse as sp
    from phasegen.distributions._gradient import _TapedAction

    action = _TapedAction(sp.identity(5, format='csr') * -0.7)
    x = np.arange(1.0, 6.0)
    y, checkpoints = action.apply(x)
    dx, dM = action.reverse(np.ones(5), checkpoints, np.arange(5), np.arange(5))

    assert action.degree == 0
    np.testing.assert_allclose(y, np.exp(-0.7) * x, rtol=1e-15)
    np.testing.assert_allclose(dx, np.exp(-0.7) * np.ones(5), rtol=1e-15)


def _state_spaces():
    """State spaces of every type, under each coalescent model where it applies."""
    two_demes = dict(pop_sizes={'a': 1, 'b': 1}, migration_rates={('a', 'b'): 0.5, ('b', 'a'): 0.5})
    out = []
    for name, model in [('kingman', None), ('beta', pg.BetaCoalescent(alpha=1.5)),
                        ('dirac', pg.DiracCoalescent(psi=0.5, c=2))]:
        kw = {} if model is None else dict(model=model)
        out += [
            pytest.param(lambda kw=kw: pg.Coalescent(n=6, **kw).tree_height.state_space, id=f'{name}-lineages'),
            pytest.param(lambda kw=kw: pg.Coalescent(n=6, **kw).sfs.state_space, id=f'{name}-blocks'),
            pytest.param(lambda kw=kw: pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(**two_demes),
                                                     **kw).jsfs.state_space, id=f'{name}-joint'),
        ]

    out.append(pytest.param(
        lambda: pg.Coalescent(n=4, loci=2, recombination_rate=0.7).tree_height.state_space, id='kingman-two-loci'))

    return out


@pytest.mark.parametrize('get_state_space', _state_spaces())
def test_generator_basis_reproduces_the_generator(get_state_space):
    """The affine basis rebuilds the generator at random population sizes and migration rates, for every coalescent
    model and state space, and returns the state space to its epoch."""
    from phasegen.demography import Epoch
    from phasegen.distributions._gradient import _GeneratorBasis

    ss = get_state_space()
    epoch = ss.epoch
    basis = _GeneratorBasis(ss)
    model = ss.model
    rng = np.random.default_rng(3)

    assert ss.epoch == epoch

    try:
        for _ in range(5):
            sizes = {p: float(np.exp(rng.normal(0, 1))) for p in basis.pops}
            rates = {pair: float(np.exp(rng.normal(-1, 1))) for pair in basis.pairs}
            ss.update_epoch(Epoch(pop_sizes=sizes, migration_rates=rates))
            S = np.asarray(ss.S.todense() if hasattr(ss.S, 'todense') else ss.S)

            rebuilt = basis.K.toarray()
            for p in basis.pops:
                rebuilt = rebuilt + basis.C[p].toarray() / model._get_timescale(sizes[p])
            for pair in basis.pairs:
                rebuilt = rebuilt + rates[pair] * basis.M[pair].toarray()

            np.testing.assert_allclose(rebuilt, S, rtol=0, atol=1e-12 * np.abs(S).max())
    finally:
        ss.update_epoch(epoch)


@pytest.mark.parametrize('model', [None, pg.BetaCoalescent(alpha=1.5), pg.DiracCoalescent(psi=0.5, c=2)],
                         ids=['kingman', 'beta', 'dirac'])
def test_generator_basis_size_derivative(model):
    """The derivative of the generator by a population size, the coefficient times the coalescence part, against
    central differences of the generator."""
    from phasegen.demography import Epoch
    from phasegen.distributions._gradient import _GeneratorBasis

    ss = pg.Coalescent(n=5, **({} if model is None else dict(model=model))).sfs.state_space
    epoch = ss.epoch
    basis = _GeneratorBasis(ss)

    def S(N):
        ss.update_epoch(Epoch(pop_sizes={'pop_0': N}))
        return np.asarray(ss.S, dtype=float).copy()

    try:
        N = 1.7
        numeric = np.vectorize(lambda i, j: _richardson(lambda n: S(n)[i, j], N))(*np.indices(S(N).shape))
        exact = basis.coefficients(Epoch(pop_sizes={'pop_0': N}))['pop_0'] * basis.C['pop_0'].toarray()
    finally:
        ss.update_epoch(epoch)

    np.testing.assert_allclose(exact, numeric, rtol=0, atol=1e-8 * np.abs(numeric).max())


def test_generator_basis_pattern_holds_migration_at_rate_zero():
    """The union pattern of the transient parts holds the migration entries, which a generator at a migration rate of
    zero lacks."""
    from phasegen.demography import Epoch
    from phasegen.distributions._gradient import _GeneratorBasis

    ss = pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(pop_sizes={'a': 1, 'b': 1})).tree_height.state_space
    basis = _GeneratorBasis(ss)
    idx_t = np.arange(ss.S.shape[0] - 1)
    _, M, rows, cols = basis.transient(idx_t)
    pattern = set(zip(rows.tolist(), cols.tolist()))

    ss.update_epoch(Epoch(pop_sizes={'a': 1, 'b': 1}))
    without = set(zip(*np.nonzero(np.asarray(ss.S)[np.ix_(idx_t, idx_t)])))

    for X in M.values():
        entries = set(zip(*[a.tolist() for a in X.nonzero()]))
        assert entries <= pattern
        assert entries - without


@pytest.mark.parametrize('stat', ['tree_height', 'total_branch_length', 'sfs', 'fsfs'])
def test_distribution_gradient_matches_central_differences(stat):
    """The public gradient of a distribution, of every bin for a spectrum, against central differences of the mean by
    the population size of the second epoch."""
    def dist(N):
        return getattr(pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}})), stat)

    grad = dist(1.7).gradient()
    numeric = _richardson(lambda N: np.asarray(getattr(dist(N).mean, 'data', dist(N).mean), dtype=float), 1.7)

    np.testing.assert_allclose(grad.pop_sizes[..., 1, 0], numeric, rtol=1e-7, atol=1e-10)


def test_joint_sfs_gradient_by_migration_rate():
    """The gradient of every joint SFS bin by a migration rate, against central differences."""
    def jsfs(m):
        return pg.Coalescent(n={'a': 2, 'b': 2}, demography=pg.Demography(
            pop_sizes={'a': 1, 'b': 1.5}, migration_rates={('a', 'b'): m, ('b', 'a'): 0.4})).jsfs

    grad = jsfs(0.6).gradient()
    numeric = _richardson(lambda m: np.asarray(jsfs(m).mean.data, dtype=float), 0.6)

    assert grad.migration_rates.shape == (3, 3, 1, 2)
    np.testing.assert_allclose(grad.migration_rates[..., 0, grad.pairs.index(('a', 'b'))], numeric,
                               rtol=1e-7, atol=1e-10)


def test_coalescent_gradient_of_a_reward():
    """The gradient of Coalescent.moment of order one for a reward equals that of the distribution of the reward."""
    coal = pg.Coalescent(n=5, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))

    np.testing.assert_allclose(coal.gradient(1, rewards=(pg.TotalBranchLengthReward(),)).pop_sizes,
                               coal.total_branch_length.gradient().pop_sizes, rtol=1e-12)


def test_gradient_of_higher_moments_and_windows_raises():
    """Gradients are available for first moments accumulated until absorption only."""
    coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))

    with pytest.raises(NotImplementedError, match='first moments'):
        coal.tree_height.gradient(2)
    with pytest.raises(NotImplementedError, match='first moments'):
        coal.gradient(2)
    with pytest.raises(NotImplementedError, match='until absorption'):
        pg.Coalescent(n=4, end_time=1.0).tree_height.gradient()


def test_parameter_jacobian_matches_central_differences():
    """The Jacobian of a normalized SFS by the parameters of a callback, against central differences."""
    gradient = pg.Gradient(
        coal=lambda t, Ne: pg.Coalescent(n=8, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: Ne}})),
        statistic=lambda coal: coal.sfs.mean.normalize().polymorphic
    )
    params = dict(t=0.4, Ne=0.6)
    value, jacobian = gradient.compute(params)

    for i, key in enumerate(params):
        numeric = _richardson(lambda x: gradient._entries(gradient.coal(**(params | {key: x}))), params[key])
        np.testing.assert_allclose(jacobian[:, i], numeric, rtol=1e-6, atol=1e-9)

    assert value.shape == (7,)


def test_parameter_jacobian_of_an_unsupported_statistic_uses_differences(caplog):
    """A statistic reading a variance is differentiated by central differences, as noted in the log."""
    import logging

    gradient = pg.Gradient(
        coal=lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}})),
        statistic=lambda coal: coal.tree_height.var
    )

    with caplog.at_level(logging.INFO, logger='phasegen'):
        value, jacobian = gradient.compute(dict(N=1.3))

    numeric = _richardson(lambda N: float(gradient.coal(N=N).tree_height.var), 1.3)
    assert jacobian[0] == pytest.approx(numeric, rel=1e-6)
    assert any('no exact gradient' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('model, name, x', [
    (lambda x: pg.BetaCoalescent(alpha=x), 'alpha', 1.5),
    (lambda x: pg.DiracCoalescent(psi=x, c=2), 'psi', 0.5),
    (lambda x: pg.DiracCoalescent(psi=0.5, c=x), 'c', 2.0),
])
@pytest.mark.parametrize('stat', ['tree_height', 'sfs'])
def test_gradient_by_model_parameter(model, name, x, stat):
    """The derivative of a mean, of every bin for a spectrum, by a parameter of the coalescent model, against central
    differences, over two epochs."""
    def dist(y):
        return getattr(pg.Coalescent(n=5, model=model(y), demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.5: 2}})), stat)

    numeric = _richardson(lambda y: np.asarray(getattr(dist(y).mean, 'data', dist(y).mean), dtype=float), x)
    np.testing.assert_allclose(dist(x).gradient().parameters[name], numeric, rtol=1e-7, atol=1e-10)


def test_gradient_by_model_parameter_near_a_bound():
    """Just below psi = 1, which a Dirac coalescent excludes, the derivative is one-sided."""
    def mean(psi):
        return pg.Coalescent(n=4, model=pg.DiracCoalescent(psi=psi, c=2)).tree_height.mean

    x = 1 - 1e-7
    h = 1e-4
    one_sided = (3 * mean(x) - 4 * mean(x - h) + mean(x - 2 * h)) / (2 * h)
    th = pg.Coalescent(n=4, model=pg.DiracCoalescent(psi=x, c=2)).tree_height

    assert th.gradient().parameters['psi'] == pytest.approx(one_sided, rel=1e-6)


def test_gradient_by_recombination_rate():
    """The derivative of the two-locus tree height by the recombination rate, against central differences."""
    def mean(r):
        return pg.Coalescent(n=3, loci=2, recombination_rate=r, demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.4: 1.7}})).tree_height.mean

    th = pg.Coalescent(n=3, loci=2, recombination_rate=0.7,
                       demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 1.7}})).tree_height

    assert th.gradient().parameters['recombination_rate'] == pytest.approx(_richardson(mean, 0.7), rel=1e-7)


def _two_locus(n: int, r: float) -> pg.Coalescent:
    """A two-locus coalescent with a size change, the setup of the recombination-rate references."""
    return pg.Coalescent(n=n, loci=2, recombination_rate=r,
                         demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 1.3}}))


@pytest.mark.parametrize('n, r, action, expected', [
    # references from a fourth-order forward difference with h = 1e-3, Richardson-extrapolated against h = 2e-3
    (3, 0.0, True, 1.3230734390),
    (5, 0.0, False, 1.5081653871),
    (3, 1e-9, False, 1.3230734345),
])
def test_gradient_by_recombination_rate_near_zero(n, r, action, expected):
    """At a recombination rate of zero, the generator has no recombination entries, which the sparsity pattern of the
    reverse pass of the action and of the sparse last epoch then lacked, so the derivative came out as -0.480 for n=3
    on the action path and -1.418 for n=5 on the sparse last epoch. At r=1e-9, the central difference of the generator
    shrank its step to 1e-12 and lost three digits to cancellation."""
    Settings.expm_action_min_dim = 0 if action else 10 ** 9

    assert _two_locus(n, r).tree_height.gradient().parameters['recombination_rate'] == pytest.approx(expected,
                                                                                                    rel=1e-8)


def test_parameter_jacobian_by_recombination_rate_at_zero():
    """The Jacobian by a recombination rate on its lower bound of zero has the sign of the one-sided difference."""
    gradient = pg.Gradient(lambda r, N: pg.Coalescent(n=5, loci=2, recombination_rate=r, demography=pg.Demography(
        pop_sizes={'pop_0': {0: 1, 0.4: N}})), lambda c: c.tree_height.mean)

    _, jacobian = gradient.compute(dict(r=0.0, N=1.3), bounds=dict(r=(0, 5), N=(0.1, 10)))

    assert np.ravel(jacobian)[0] == pytest.approx(1.5081653871, rel=1e-8)


def test_loss_gradient_model_and_recombination_parameters_without_loss_differences():
    """A loss over a Beta coalescent parameter, a recombination rate and a population size is differentiated exactly,
    without evaluating the loss at perturbed parameters."""
    calls = []

    def loss(coal, obs):
        if isinstance(coal, pg.Coalescent):
            calls.append(1)
        return float(np.log(coal.tree_height.mean))

    coal = lambda alpha, r, N: pg.Coalescent(n=3, loci=2, recombination_rate=r, model=pg.BetaCoalescent(alpha=alpha),
                                             demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))
    x0 = {'alpha': 1.6, 'r': 0.5, 'N': 1.4}
    bounds = {'alpha': (1.01, 1.99), 'r': (0, 5), 'N': (0.1, 5)}

    _assert_loss_gradient(coal, loss, x0, bounds, None)
    calls.clear()
    Gradient(coal, lambda c: loss(c, None))._exact(x0, bounds)

    assert not calls


def _epochs(n_epochs: int, scale: float, step: float = 0.15) -> Dict[float, float]:
    """Population sizes over epochs of equal length, scaled."""
    return {e * step: scale * (1 + 0.5 * np.sin(e)) for e in range(n_epochs)}


#: Configurations of the regimes the exact gradient reaches: a coalescent callback, the parameters, the statistic, and
#: whether every parameter is differentiated exactly
_REGIMES = {
    '10 epochs, SFS': (
        lambda a, b: pg.Coalescent(n=5, demography=pg.Demography(
            pop_sizes={'pop_0': {t: (a if i % 2 else b) * N for i, (t, N) in enumerate(_epochs(10, 1).items())}})),
        dict(a=1.3, b=0.8), lambda c: c.sfs.mean.polymorphic),
    '20 epochs, two demes': (
        lambda a, b, m: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': _epochs(20, a, 0.08), 'pop_1': _epochs(20, b, 0.08)},
            migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): 0.5})),
        dict(a=1.2, b=0.7, m=0.4), lambda c: c.tree_height.mean),
    'exponential growth': (
        lambda N, r: pg.Coalescent(n=5, demography=pg.Demography(
            pop_sizes={'pop_0': {0.3: 0.5}},
            events=[pg.ExponentialPopSizeChanges(initial_size={'pop_0': N}, growth_rate={'pop_0': r},
                                                 start_time={'pop_0': 0}, end_time={'pop_0': 0.3}, step_size=0.05)])),
        dict(N=2.0, r=3.0), lambda c: c.sfs.mean.polymorphic),
    'pulse': (
        lambda p, N: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': 1, 'pop_1': N}, migration_rates={('pop_0', 'pop_1'): 0.2, ('pop_1', 'pop_0'): 0.2},
            events=[pg.Pulse(time=0.3, source='pop_0', dest='pop_1', proportion=p)])),
        dict(p=0.3, N=1.5), lambda c: c.jsfs.mean.data.ravel()[1:-1]),
    'split': (
        lambda N, M: pg.Coalescent(n={'pop_0': 2, 'pop_1': 2}, demography=pg.Demography(
            pop_sizes={'pop_0': N, 'pop_1': M}, events=[pg.PopulationSplit(time=0.5, derived='pop_1', ancestral='pop_0')])),
        dict(N=1.2, M=0.6), lambda c: c.total_branch_length.mean),
    'initial distribution': (
        lambda N, m: pg.Coalescent(
            n=pg.InitialDistribution([(1, {'pop_0': 2, 'pop_1': 1}), (3, {'pop_0': 1, 'pop_1': 2})]),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: N}, 'pop_1': 1},
                                     migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): m})),
        dict(N=1.5, m=0.6), lambda c: c.tree_height.mean),
    'sparse action, 268 states': (
        lambda N, m: pg.Coalescent(n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.3: N, 0.9: 1.2}, 'pop_1': {0: 0.7, 0.3: 1.1, 0.9: N}},
            migration_rates={('pop_0', 'pop_1'): m, ('pop_1', 'pop_0'): 0.5})),
        dict(N=1.4, m=0.3), lambda c: c.jsfs.mean.data[1, 1]),
    'beta coalescent over 5 epochs': (
        lambda alpha, N: pg.Coalescent(n=5, model=pg.BetaCoalescent(alpha=alpha), demography=pg.Demography(
            pop_sizes={'pop_0': {t: N * v for t, v in _epochs(5, 1).items()}})),
        dict(alpha=1.4, N=1.3), lambda c: c.sfs.mean.polymorphic),
    'two loci over 3 epochs': (
        lambda r, N: pg.Coalescent(n=3, loci=2, recombination_rate=r, demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.3: N, 0.8: 0.6}})),
        dict(r=0.8, N=1.6), lambda c: c.tree_height.mean),
}


@pytest.mark.parametrize('closed_form', [True, False])
@pytest.mark.parametrize('regime', list(_REGIMES))
def test_parameter_jacobian_across_regimes(regime, closed_form, caplog):
    """The exact Jacobian against Richardson-extrapolated central differences through the callback, over many epochs,
    event-built demographies, an initial distribution, a sparse-action space, the model parameters and the
    recombination rate, with and without the closed form of the last epoch."""
    import logging

    Settings.closed_form_last_epoch = closed_form
    coal, params, statistic = _REGIMES[regime]
    evaluated = []

    def counted(c):
        if isinstance(c, pg.Coalescent):
            evaluated.append(1)
        return statistic(c)

    gradient = pg.Gradient(coal, counted)

    with caplog.at_level(logging.INFO, logger='phasegen'):
        value, jacobian = gradient.compute(params)

    # every parameter is differentiated through the moments, none by a difference of the statistic
    assert not evaluated
    assert not any('no exact gradient' in r.getMessage() for r in caplog.records)

    for i, key in enumerate(params):
        numeric = _richardson(lambda x: gradient._entries(coal(**(params | {key: x}))), params[key])
        np.testing.assert_allclose(jacobian[..., i], numeric, rtol=1e-6, atol=1e-8 * np.abs(numeric).max())


@pytest.mark.parametrize('case', ['lineage weight', 'locus weight', 'custom model'])
def test_parameter_jacobian_of_parameters_outside_the_rates(case):
    """A parameter that enters neither the demography nor the rate parameters, such as a weight of an initial
    distribution or the argument of a custom model, is differentiated by a difference of the statistic. Regression:
    such a parameter got a derivative of exactly zero."""
    class Custom(pg.BetaCoalescent):
        def __init__(self, a):
            super().__init__(alpha=a)

    demography = pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: 1.5}, 'pop_1': 1},
                               migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5})
    coal, x = {
        'lineage weight': (lambda w, N: pg.Coalescent(
            n=pg.InitialDistribution([(w, {'pop_0': 2, 'pop_1': 1}), (1, {'pop_0': 1, 'pop_1': 2})]),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: N}, 'pop_1': 1},
                                     migration_rates={('pop_0', 'pop_1'): 0.5, ('pop_1', 'pop_0'): 0.5})), 0.5),
        'locus weight': (lambda w, N: pg.Coalescent(
            n=3, loci=pg.InitialDistribution([(w, pg.LocusConfig(n=2, recombination_rate=0.5)),
                                              (1, pg.LocusConfig(n=2, n_unlinked=1, recombination_rate=0.5))]),
            demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.4: N}})), 0.5),
        'custom model': (lambda w, N: pg.Coalescent(n=4, model=Custom(w), demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.4: N}})), 1.5),
    }[case]
    gradient = pg.Gradient(coal, lambda c: c.tree_height.mean)
    params = dict(w=x, N=1.5)
    _, jacobian = gradient.compute(params)

    for i, key in enumerate(params):
        numeric = _richardson(lambda y: float(coal(**(params | {key: y})).tree_height.mean), params[key])
        assert jacobian[i] == pytest.approx(numeric, rel=1e-5, abs=1e-9)
    assert abs(jacobian[0]) > 1e-3


def test_parameter_jacobian_of_a_fixed_parameter_is_zero():
    """A parameter whose bounds coincide has a derivative of zero, exact and by differences. Regression: the exact
    gradient divided by its zero step and gave NaN."""
    gradient = pg.Gradient(lambda t, N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, t: N}})),
                           lambda c: c.tree_height.mean)
    bounds = dict(t=(0.3, 0.3), N=(0.1, 10))

    for exact in (True, False):
        _, jacobian = gradient.compute(dict(t=0.3, N=1.5), bounds, exact=exact)
        assert jacobian[0] == 0
        assert np.isfinite(jacobian[1]) and jacobian[1] != 0


def test_gradient_by_beta_alpha_close_to_two():
    """The derivative by alpha just below 2, where the generator varies on the scale of 2 - alpha, against a local
    central difference. Regression: the step of the difference of the generator was fixed, about 70% off at
    2 - alpha = 1e-6."""
    def mean(alpha):
        return pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=alpha), demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.5: 2}})).tree_height.mean

    for distance in (1e-4, 1e-6):
        alpha = 2 - distance
        h = distance / 100
        th = pg.Coalescent(n=4, model=pg.BetaCoalescent(alpha=alpha), demography=pg.Demography(
            pop_sizes={'pop_0': {0: 1, 0.5: 2}})).tree_height
        assert th.gradient().parameters['alpha'] == pytest.approx((mean(alpha + h) - mean(alpha - h)) / (2 * h),
                                                                  rel=1e-5)


def test_gradient_returns_a_copy_and_accepts_a_list_of_rewards():
    """The cached gradient is returned as a copy, so that changing the result leaves later calls intact, and the
    rewards may be a list. Regression: the cached object itself was returned, and a list raised a TypeError."""
    coal = pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: 2}}))
    first = coal.tree_height.gradient()
    first.pop_sizes += 1

    np.testing.assert_array_equal(coal.tree_height.gradient().pop_sizes, first.pop_sizes - 1)

    by_list = coal.gradient(1, rewards=[pg.TotalBranchLengthReward()])
    by_list.pop_sizes += 1
    np.testing.assert_array_equal(coal.gradient(1, rewards=[pg.TotalBranchLengthReward()]).pop_sizes,
                                  by_list.pop_sizes - 1)


def test_parameter_jacobian_value_mismatch_falls_back(monkeypatch, caplog):
    """A backward pass whose value disagrees with the moments gives no exact Jacobian, with a warning."""
    import logging

    columns = MeanGradient.compute_columns

    def shifted(self, R):
        return [(v * 1.01, g) for v, g in columns(self, R)]

    monkeypatch.setattr(MeanGradient, 'compute_columns', shifted)
    gradient = pg.Gradient(lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}})),
                           lambda c: c.tree_height.mean)

    with caplog.at_level(logging.WARNING, logger='phasegen'):
        assert gradient._exact(dict(N=1.5), {}) is None
    assert any('backward pass differs' in r.getMessage() for r in caplog.records)


def test_parameter_jacobian_of_a_non_finite_statistic_is_not_exact():
    """A statistic that is not finite has no exact Jacobian."""
    gradient = pg.Gradient(lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}})),
                           lambda c: np.log(c.tree_height.mean - 1e6))

    assert gradient._exact(dict(N=1.5), {}) is None


def test_support_is_assumed_where_the_callback_raises():
    """Where the model cannot be evaluated, support is left open in favour of the exact Jacobian."""
    def coal(N):
        raise pg.ModelError('invalid')

    assert pg.Gradient(coal, lambda c: c.tree_height.mean).supported(dict(N=1.0))


def test_parameter_jacobian_by_differences_stays_within_the_bounds():
    """At a parameter on its lower bound, the differences are one-sided and evaluate the callback inside the bounds."""
    seen = []

    def coal(N):
        seen.append(N)
        return pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))

    gradient = pg.Gradient(coal, lambda c: c.tree_height.var)
    _, jacobian = gradient.compute(dict(N=0.1), dict(N=(0.1, 10)))

    assert min(seen) >= 0.1
    h = 1e-4
    one_sided = (-3 * coal(0.1).tree_height.var + 4 * coal(0.1 + h).tree_height.var
                 - coal(0.1 + 2 * h).tree_height.var) / (2 * h)
    assert jacobian[0] == pytest.approx(one_sided, rel=1e-4)


def test_inference_failure_counter_resets_after_an_exact_gradient(caplog, monkeypatch):
    """Failures of the exact gradient interrupted by a success do not switch the optimization to finite differences;
    a raising or non-finite exact gradient counts as a failure."""
    import logging

    exact = Gradient._exact
    sequence = iter(['none', 'raise', 'ok', 'nan', 'none'] + ['ok'] * 1000)

    def flaky(self, params, bounds):
        kind = next(sequence)
        if kind == 'raise':
            raise RuntimeError('unavailable')
        out = exact(self, params, bounds)
        if kind == 'none':
            return None
        if kind == 'nan':
            return out[0], np.full_like(out[1], np.nan)
        return out

    monkeypatch.setattr(Gradient, '_exact', flaky)
    coal = lambda N: pg.Coalescent(n=4, demography=pg.Demography(pop_sizes={'pop_0': {0: 1, 0.5: N}}))

    with caplog.at_level(logging.WARNING, logger='phasegen'):
        inf = _inference(coal, lambda c, obs: (float(c.tree_height.mean) - 2.0) ** 2, {'N': 1.0}, {'N': (0.1, 5)},
                         None)
        inf.run()

    assert not [r for r in caplog.records if 'consecutive evaluations' in r.getMessage()]
    assert inf.loss_inferred == pytest.approx(0, abs=1e-8)


def test_inference_with_a_fixed_parameter_converges():
    """A parameter whose bounds coincide does not stop an optimization with the exact gradient. Regression: its
    gradient was NaN and L-BFGS-B returned the start point."""
    observation = pg.SFS(np.round(1000 * pg.Coalescent(n=6, demography=pg.Demography(
        pop_sizes={'pop_0': {0: 2, 0.3: 1}})).sfs.mean.normalize().data))
    coal = lambda N, t: pg.Coalescent(n=6, demography=pg.Demography(pop_sizes={'pop_0': {0: N, t: 1}}))

    results = []
    for gradient in ('exact', 'finite_differences'):
        inf = _inference(coal, _sfs_loss, {'N': 1.0, 't': 0.3}, {'N': (0.1, 10), 't': (0.3, 0.3)}, observation,
                         gradient=gradient)
        inf.run()
        results.append(inf.params_inferred['N'])

    assert results[0] == pytest.approx(results[1], rel=1e-3)
    assert results[0] != 1.0
