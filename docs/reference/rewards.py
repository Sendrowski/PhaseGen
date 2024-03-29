import phasegen as pg

coal = pg.Coalescent(
    n={'pop_0': 1, 'pop_1': 2, 'pop_2': 2},
    demography=pg.Demography(
        pop_sizes={'pop_0': 3, 'pop_1': 2, 'pop_2': 1},
        events=[
            pg.SymmetricMigrationRateChanges(
                pops=['pop_0', 'pop_1', 'pop_2'],
                migration_rates=1
            )
        ]
    )
)

sfs = coal.sfs.moment(1, (pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')]),))
sfs.plot()

pg.Spectra({d: coal.sfs.demes[d].mean for d in coal.demography.pop_names}).plot()

demes = pg.SumReward([pg.DemeReward('pop_0'), pg.DemeReward('pop_1')])
sfs_bin = pg.UnfoldedSFSReward(2)

assert sfs.data[2] == coal.moment(1, (pg.ProductReward([demes, sfs_bin]),))

mean1 = pg.Coalescent(n=10, end_time=2).tree_height.mean
mean2 = pg.Coalescent(n=10).tree_height.moment(1, end_time=2)

pass
