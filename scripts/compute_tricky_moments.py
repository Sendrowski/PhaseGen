import phasegen as pg

pg.logger.setLevel('DEBUG')

pg.Coalescent(n=400, demography=pg.Demography(pop_sizes={0: 1})).sfs.mean.plot()

pg.Coalescent(n=100, demography=pg.Demography(pop_sizes={0: 1, 1: 2})).sfs.mean.plot()