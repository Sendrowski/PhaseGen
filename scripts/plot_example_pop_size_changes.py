"""
Plot an example demographic trajectory with changes in population size.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Increase all text sizes
mpl.rcParams['font.size'] = 14

# Define the times and population sizes
times = [0, 3, 6, 8, 10]
pop_sizes = [1, 3, 0.4, 2, 2]  # Extra element to extend the last size to time=10

# Create the plot
plt.step(times, pop_sizes, where='post', linewidth=3)

# Label the axes
plt.xlabel('t')
plt.ylabel('$N$', rotation=0, labelpad=12)

# Set the y-axis limits
plt.ylim(0, 4)  # Set according to the range of your data

# Remove x margin
plt.margins(x=0)

# Add title
plt.title('population size trajectory', pad=25)

# Add vertical dashed lines at each change in population size
for i, t in enumerate(times[:-1]):  # exclude the last time as it is only for extending the plot
    plt.vlines(t, 0, 4, colors='red', linestyles='dashed', alpha=0.5)
    plt.text(t, 4 + 0.025, f'$t_{i}$', ha='center', va='bottom')

# Add horizontal dashed lines at each change in population size
for i, N in enumerate(pop_sizes[:-1]):  # exclude the last population size as it is only for extending the plot
    plt.hlines(N, 0, 10, colors='green', linestyles='dashed', alpha=0.5)
    plt.text(10 + 0.1, N, f'$N_{i}$', ha='left', va='center')

plt.savefig("scratch/pop_size_changes_example.png", dpi=500)

# Show the plot
plt.show()
