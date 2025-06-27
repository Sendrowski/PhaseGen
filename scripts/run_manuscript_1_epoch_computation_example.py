"""
Run manuscript 1-epoch computation example.
"""
import phasegen as pg

pg.Settings.regularize = False

coal = pg.Coalescent(n=3, end_time=3)

m = coal.moment(k=1, rewards=[pg.UnfoldedSFSReward(1)])
tree_height = coal.tree_height.mean

pass
