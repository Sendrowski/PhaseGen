import phasegen as pg

pg.logger.setLevel('DEBUG')

# flattening + last epoch closed form
print('n = 400, 1 epoch, mean SFS')
pg.Coalescent(n=400, demography=pg.Demography(pop_sizes={0: 1})).sfs.mean.plot()

# flattening + dense LU
print('n = 100, 2 epochs, mean SFS')
pg.Coalescent(n=100, demography=pg.Demography(pop_sizes={0: 1, 1: 2})).sfs.mean.plot()

# batched (shared two-point occupation)
print('n = 20, 1 epoch, 2-SFS')
pg.Coalescent(n=20).sfs.corr.plot()

    # batched occupation times, sparse factorization
print("n = 10, two demes, 1 epoch, joint SFS")
pg.Coalescent(
    n={'pop_0': 5, 'pop_1': 5}, demography=pg.Demography(migration_rates={('pop_0', 'pop_1'): 1})
).jsfs.mean.plot()

# dense Van Loan
print('n = 5, 2 loci, 2-locus SFS')
pg.Coalescent(n=5, loci=2).sfs2.mean.plot()

