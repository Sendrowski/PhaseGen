- Discretized methods work in principle, but it's unclear how to discretize the Van Loan matrix, which is required for computing higher-order moments.

- Discretizing first order-moment provides fast results with an error rate lower than 1e-2, but is not suitable for gradient-based optimization.

- Sparse matrix multiplication is efficient when discretizing time.

- Matrix inversion only applies to time-homogeneous cases; the matrix becomes singular when recursions exist (i.e., in models with multiple populations).

- State space reduction by removing zero-reward states rescales time and thus breaks down in multi-epoch settings.

- Graph-based methods fail for multiple epochs because they require conditioning on the remaining time in the current epoch, which involves integrating over the density of previous states.

- Single-precision matrix exponentiation is not significantly faster than double-precision.

- Matrix exponentiation via Taylor series expansion is inherently unstable, even for moderately sized matrices.

- Sparse matrix exponentiation is generally slow and only outperforms dense exponentiation for extremely large matrices (tens of thousands of states).

- Collapsing the block-counting state space into the lineage-counting state space is feasible for one-population, one-locus Kingman coalescent models and significantly improves computational efficiency. It doesn't for MMCs, however.