import phasegen as pg

coal = pg.Coalescent(
    n=20,
    model=pg.BetaCoalescent(alpha=1.3),
)

corr = coal.fsfs.corr

corr.plot(max_abs=1)

pass