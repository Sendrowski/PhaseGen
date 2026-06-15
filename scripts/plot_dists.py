
import phasegen as pg

pg.logger.setLevel('DEBUG')

from phasegen.distributions.empirical import MsprimeCoalescent

coal = pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1}))

ms = MsprimeCoalescent(parallelize=False, num_replicates=10**6, n=10, demography=pg.Demography(pop_sizes={0: 1}))

for c in [coal, ms]:
    c.tree_height.pdf.plot()

pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1})).tree_height.pdf.plot()
MsprimeCoalescent(parallelize=True, num_replicates=10**6, n=10, demography=pg.Demography(pop_sizes={0: 1})).tree_height.pdf.plot()

pg.Coalescent(n=10, demography=pg.Demography(pop_sizes={0: 1})).sfs.pdf.plot()
MsprimeCoalescent(parallelize=False, n=10, demography=pg.Demography(pop_sizes={0: 1})).sfs.pdf.plot()
exit()
pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(migration_rates={('pop_0', 'pop_1'): 1})
).jsfs.pdf.plot()

pg.Coalescent(
    n={'pop_0': 3, 'pop_1': 3}, demography=pg.Demography(
        pop_sizes={0: 1, 1: 2}, migration_rates={('pop_0', 'pop_1'): 1}
    )
).jsfs.joint_distribution((0, 3), (3, 0)).pdf.plot()

# batched (shared two-point occupation)
print('n = 20, 1 epoch, 2-SFS')
pg.Coalescent(n=20).sfs.corr.plot()

    # batched occupation times, sparse factorization
print("n = 10, two demes, 1 epoch, joint SFS")


# dense Van Loan
print('n = 5, 2 loci, 2-locus SFS')
pg.Coalescent(n=5, loci=2).sfs2.mean.plot()

