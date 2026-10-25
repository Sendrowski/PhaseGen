"""
Block-size kernel of the Kingman coalescent, which reduces the second moments of the site-frequency spectrum of a
single deme to the times spent with each number of lineages.
"""
import numpy as np
from scipy.special import binom


class KingmanBlockKernel:
    r"""
    Contraction of the ordered level occupation of the lineage-counting state space with the block-size kernel
    :math:`W_{ijkl} = \mathbb{E}[B_{k,i} B_{l,j}]`, the expected product of the number :math:`B_{k,i}` of the
    :math:`k` lineages that subtend :math:`i` of the :math:`n` samples and the number :math:`B_{l,j}` of the
    :math:`l \le k` lineages that subtend :math:`j`. Level :math:`l` is ancestral to level :math:`k`. With
    :math:`d` the number of the :math:`k` lineages that descend from a given one of the :math:`l` lineages and
    :math:`c(m, p) = \binom{m - 1}{p - 1}` the number of compositions of :math:`m` into :math:`p` positive parts,

    .. math::

        W_{ijkl} = l \sum_{d} \frac{c(k - d, l - 1)}{c(k, l)} \cdot
        \frac{d\, c(j - i, d - 1)\, c(n - j, k - d) + (k - d)\, c(j, d)\, c(n - j - i, k - d - 1)}{c(n, k)},

    The occupation is contracted over :math:`l` first, without forming the kernel.
    """

    def __init__(self, n: int) -> None:
        """
        :param n: The number of samples :math:`n`, at least 2.
        """
        self.n = n

    def contract(self, occupation: np.ndarray) -> np.ndarray:
        r"""
        The ordered cross-moments :math:`X_{ij} = \sum_{k \ge l} M_{kl} W_{ijkl}`, where :math:`M_{kl}` is the expected
        measure of the times :math:`s < t` with :math:`k` lineages at :math:`s` and :math:`l` at :math:`t`, so that
        :math:`\mathbb{E}[L_i L_j] = X_{ij} + X_{ji}` for the branch length :math:`L_i` subtending :math:`i` samples.

        :param occupation: The ordered occupation over the transient states of the lineage-counting state space, entry
            ``[n - k, n - l]`` holding :math:`M_{kl}`, of shape ``(n - 1, n - 1)``.
        :return: :math:`X_{ij}` at ``[i - 1, j - 1]`` for :math:`i, j = 1, \dots, n - 1`.
        """
        n = self.n
        c = self._compositions()

        # the pairs (k, d) of a level k and a number d = 1, ..., k - 1 of its lineages descending from a lineage of a
        # level l <= k
        k = np.repeat(np.arange(2, n + 1), np.arange(1, n))
        d = np.concatenate([np.arange(1, m) for m in range(2, n + 1)])

        # weights over l = 2, ..., n of each pair: l c(k - d, l - 1) / c(k, l) M_kl, zero for l > k
        l = np.arange(2, n + 1)
        valid = l[None, :] <= k[:, None]
        share = np.where(valid, l * c[k - d][:, l - 1] / np.where(valid, c[k][:, l], 1.0), 0.0)
        levels = np.asarray(occupation)[np.ix_(n - np.arange(2, n + 1), n - l)]
        weight = (share * levels[k - 2]).sum(axis=1) / c[n, k]

        # the descendants (first term) and the other lineages (second term) of size i, as products over the pairs of a
        # factor in the size j and one in j - i or n - j - i
        j = np.arange(1, n)[:, None]
        q = np.arange(n)[:, None]
        first = (weight * d * c[n - j, k - d]) @ c[q, d - 1].T
        second = (weight * (k - d) * c[j, d]) @ c[q, k - d - 1].T

        i = np.arange(1, n)
        diff = j - i[None, :]
        rest = n - j - i[None, :]
        cross = np.where(diff >= 0, np.take_along_axis(first, np.maximum(diff, 0), axis=1), 0.0) + \
            np.where(rest >= 0, np.take_along_axis(second, np.maximum(rest, 0), axis=1), 0.0)

        # entry [j - 1, i - 1] holds X_ij
        return cross.T

    def _compositions(self) -> np.ndarray:
        r"""
        The numbers of compositions :math:`c(m, p)` for :math:`m, p = 0, \dots, n`, with :math:`c(0, 0) = 1`.

        :return: Array of shape ``(n + 1, n + 1)``.
        """
        m = np.arange(self.n + 1)[:, None]
        p = np.arange(self.n + 1)[None, :]
        c = np.where((m >= p) & (p >= 1), binom(np.maximum(m - 1, 0), np.maximum(p - 1, 0)), 0.0)
        c[0, 0] = 1.0

        return c
