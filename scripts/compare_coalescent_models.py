from matplotlib import pyplot as plt

import phasegen as pg

# construct coalescent object
kingman = pg.Coalescent(n=10, model=pg.StandardCoalescent())
beta = pg.Coalescent(n=10, model=pg.BetaCoalescent(alpha=1.5))

_, axs = plt.subplots(ncols=2, figsize=(8, 4), subplot_kw={"projection": "3d"})

kingman.sfs.cov.plot(ax=axs[0], show=False, title='Kingman')
beta.sfs.cov.plot(ax=axs[1], title='Beta')
