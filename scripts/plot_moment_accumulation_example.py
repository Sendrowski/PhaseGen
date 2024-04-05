"""
Plot moment accumulation example.
"""
import phasegen as pg
import matplotlib.pyplot as plt

coal = pg.Coalescent(n=5)

fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.9))

for k in range(1, 6):
    coal.tree_height.plot_accumulation(
        k=k,
        ax=ax,
        title="Moment accumulation",
        show=False,
        label=f"$\mu_{{{k}}}$"
    )

for line in plt.gca().lines:
    line.set_linewidth(3)

plt.margins(x=0.1, y=0.1)
plt.legend().remove()
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.show()

pass