"""
Matrix exponential of block upper triangular matrices whose diagonal blocks are scalar multiples of the identity or
one shared matrix, the structure of the Van Loan matrices and of the augmented occupation generator.
"""
from typing import Sequence, Tuple

import numpy as np
import scipy.linalg

#: Padé coefficients of degree 3, 5, 7, 9 and 13 (Higham 2005, Table 2.2).
_PADE = {
    3: (120., 60., 12., 1.),
    5: (30240., 15120., 3360., 420., 30., 1.),
    7: (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.),
    9: (17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.),
    13: (64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
         10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.),
}

#: Largest :math:`\eta` admitted by each Padé degree (Al-Mohy & Higham 2009, Table 3.1, with 4.25 for degree 13).
_THETA = {3: 1.495585217958292e-2, 5: 2.539398330063230e-1, 7: 9.504178996162932e-1, 9: 2.097847961257068, 13: 4.25}

#: Reciprocal moduli of the leading coefficients of the Padé error series (Al-Mohy & Higham 2009, eq. 5.3).
_ELL_C = {3: 100800., 5: 10059033600., 7: 4487938430976000., 9: 5914384781877411840000.,
          13: 113250775606021113483283660800000000.}

#: Unit roundoff of double precision.
_U = 2.0 ** -53

#: Largest 1-norm at which the powers are formed unscaled. A larger argument is first halved until it is below, the
#: halvings undone by squarings, which keeps the sixth power finite.
_MAX_NORM = 1e36


class BlockTriangularExpm:
    r"""
    Matrix exponential of a :math:`K n \times K n` block upper triangular matrix :math:`\mathbf{A}` with blocks
    :math:`\mathbf{A}_{ij}` of size :math:`n`, :math:`\mathbf{A}_{ij} = 0` for :math:`i > j`, whose diagonal blocks are
    scalar multiples :math:`c_i \mathbf{I}` of the identity or one shared matrix :math:`\mathbf{D}`. Polynomials in
    :math:`\mathbf{A}` and their quotients keep this structure, so the scaling and squaring with Padé approximants of
    Al-Mohy & Higham (2009) multiplies only the upper blocks, block column :math:`l` of the left factor with block row
    :math:`l` of the right factor, and the Padé solve is a block back substitution with one LU factorization of
    :math:`\mathbf{D}`. A product costs :math:`\sum_{l=1}^{K} l (K - l + 1)` products of size :math:`n`, and none for the
    scalar diagonal blocks, instead of :math:`K^3`.

    Example, the occupation integral :math:`\int_0^t e^{\mathbf{S} s} ds` of a generator :math:`\mathbf{S}` of size
    :math:`n`::

        aug = np.block([[S * t, np.eye(n) * t], [np.zeros((n, n)), np.zeros((n, n))]])
        occupation = BlockTriangularExpm(n, scalar=[False, True]).compute(aug)[:n, n:]

    Al-Mohy, A. H., & Higham, N. J. (2009). A new scaling and squaring algorithm for the matrix exponential. SIAM
    Journal on Matrix Analysis and Applications, 31(3), 970-989.
    """

    def __init__(self, n: int, scalar: Sequence[bool]) -> None:
        """
        Initialize the structure.

        :param n: The size of one block.
        :param scalar: Whether each diagonal block is a scalar multiple of the identity, of length :math:`K`. The
            remaining diagonal blocks must be equal.
        """
        #: The size of one block
        self.n: int = n

        #: Whether each diagonal block is a scalar multiple of the identity
        self.scalar: np.ndarray = np.asarray(scalar, dtype=bool)

        #: The number of block rows
        self.K: int = len(self.scalar)

    def compute(self, A: np.ndarray) -> np.ndarray:
        """
        The exponential of ``A``.

        :param A: Block upper triangular matrix of the configured structure, of size :math:`K n`.
        :return: The exponential of ``A``, with exact zeros below the diagonal blocks.
        """
        A = np.array(A, dtype=float)
        norm = self._norm(A)
        if not np.isfinite(norm):
            return scipy.linalg.expm(A)
        if norm == 0:
            return np.eye(len(A))

        # the diagonal and superdiagonal of a triangular matrix, whose exponentials are known exactly
        triangular = not np.tril(A, -1).any()
        d, sd = np.diag(A).copy(), np.diag(A, 1).copy()

        # extreme norms are halved first, the halvings undone by as many extra squarings
        s0 = int(np.ceil(np.log2(norm / _MAX_NORM))) if norm > _MAX_NORM else 0
        if s0:
            A /= 2.0 ** s0
            norm /= 2.0 ** s0

        X, s = self._pade(A, norm)
        for i in range(s + s0, -1, -1):
            if i < s + s0:
                X = self._mul(X, X)
            if triangular:
                self._set_exact_diagonals(X, d * 2.0 ** -i, sd * 2.0 ** -i)

        return X

    @staticmethod
    def _set_exact_diagonals(X: np.ndarray, d: np.ndarray, sd: np.ndarray) -> None:
        r"""
        Overwrite the diagonal and the first superdiagonal of the exponential ``X`` of an upper triangular matrix
        with diagonal ``d`` and superdiagonal ``sd`` by their exact values :math:`e^{d_i}` and
        :math:`s_i (e^{d_{i+1}} - e^{d_i}) / (d_{i+1} - d_i)` (Al-Mohy & Higham 2009, Code Fragment 2.1).

        :param X: The exponential, modified in place.
        :param d: The diagonal of the matrix.
        :param sd: The first superdiagonal of the matrix.
        """
        e = np.exp(d)
        diff, dd = np.diff(e), np.diff(d)
        z = dd == 0
        diff[~z] /= dd[~z]
        diff[z] = e[:-1][z]
        np.einsum('ii->i', X)[:] = e
        np.einsum('ii->i', X[:-1, 1:])[:] = diff * sd

    def _pade(self, A: np.ndarray, norm: float) -> Tuple[np.ndarray, int]:
        """
        The Padé approximant of the exponential of ``A / 2^s`` and the number of squarings ``s``, chosen by
        Algorithm 5.1 of Al-Mohy & Higham (2009) with the exact norms of the second, fourth and sixth power and upper
        bounds on those of the eighth and tenth.

        :param A: The matrix.
        :param norm: The 1-norm of ``A``.
        :return: The approximant and the number of squarings.
        """
        A2 = self._mul(A, A)
        A4 = self._mul(A2, A2)
        A6 = self._mul(A2, A4)
        n2, n4, n6 = self._norm(A2), self._norm(A4), self._norm(A6)
        d4, d6 = n4 ** 0.25, n6 ** (1 / 6)
        d8 = min(d4, (n2 * n6) ** 0.125)
        d10 = (n4 * n6) ** 0.1

        eta1 = max(d4, d6)
        if eta1 <= _THETA[3] and self._ell(A, norm, 3) == 0:
            return self._solve(*self._uv(A, [A2], 3)), 0
        if eta1 <= _THETA[5] and self._ell(A, norm, 5) == 0:
            return self._solve(*self._uv(A, [A2, A4], 5)), 0
        eta3 = max(d6, d8)
        if eta3 <= _THETA[7] and self._ell(A, norm, 7) == 0:
            return self._solve(*self._uv(A, [A2, A4, A6], 7)), 0
        if eta3 <= _THETA[9] and self._ell(A, norm, 9) == 0:
            return self._solve(*self._uv(A, [A2, A4, A6, self._mul(A4, A4)], 9)), 0

        eta5 = min(eta3, max(d8, d10))
        s = max(int(np.ceil(np.log2(eta5 / _THETA[13]))), 0) if eta5 > 0 else 0
        s += self._ell(A / 2.0 ** s, norm / 2.0 ** s, 13)

        c = [2.0 ** (-s * p) for p in (1, 2, 4, 6)]
        A, A2, A4, A6 = A * c[0], A2 * c[1], A4 * c[2], A6 * c[3]
        b = _PADE[13]
        I = np.eye(len(A))
        U = self._mul(A, self._mul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
                      + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
        V = self._mul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I

        return self._solve(U, V), s

    def _uv(self, A: np.ndarray, powers: Sequence[np.ndarray], m: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The odd and even parts ``U`` and ``V`` of the Padé approximant of degree ``m`` below 13.

        :param A: The matrix.
        :param powers: The even powers 2, 4, ..., m - 1 of ``A``.
        :param m: The degree.
        :return: ``(U, V)``.
        """
        b = _PADE[m]
        I = np.eye(len(A))
        u, v = b[1] * I, b[0] * I
        for j, P in enumerate(powers, start=1):
            u = u + b[2 * j + 1] * P
            v = v + b[2 * j] * P

        return self._mul(A, u), v

    def _ell(self, A: np.ndarray, norm: float, m: int) -> int:
        r"""
        The extra squarings of Al-Mohy & Higham (2009, eq. 5.1) that bound the relative backward error of the degree
        ``m`` approximant by the unit roundoff, from the exact 1-norm of :math:`|\mathbf{A}|^{2m + 1}`.

        :param A: The matrix.
        :param norm: The 1-norm of ``A``.
        :param m: The degree.
        :return: The number of extra squarings.
        """
        B = np.abs(A)
        v = np.ones(len(A))
        for _ in range(2 * m + 1):
            v = v @ B
        p = float(v.max())
        if p == 0:
            return 0

        return max(int(np.ceil(np.log2(p / (norm * _ELL_C[m] * _U)) / (2 * m))), 0)

    @staticmethod
    def _norm(A: np.ndarray) -> float:
        """
        The 1-norm, the largest column sum of the moduli.

        :param A: The matrix.
        :return: The 1-norm.
        """
        return float(np.abs(A).sum(axis=0).max())

    def _mul(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        The product of two matrices of the configured structure.

        :param X: The left factor.
        :param Y: The right factor.
        :return: The product.
        """
        n = self.n
        Z = np.zeros_like(X)
        for l in range(self.K):
            a, b = l * n, (l + 1) * n
            if self.scalar[l]:
                # the diagonal block l is a multiple of the identity in both factors
                Z[:a, a:b] += X[:a, a:b] * Y[a, a]
                Z[a:b, b:] += X[a, a] * Y[a:b, b:]
                Z[a:b, a:b] = X[a, a] * Y[a:b, a:b]
            else:
                Z[:b, a:] += X[:b, a:b] @ Y[a:b, a:]

        return Z

    def _solve(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        The solution ``X`` of ``(V - U) X = V + U`` by block back substitution, with one LU factorization of the
        shared diagonal block.

        :param U: The odd part.
        :param V: The even part.
        :return: ``X``.
        """
        n, K = self.n, self.K
        Q, X = V - U, V + U
        lu = None
        for i in range(K - 1, -1, -1):
            a, b = i * n, (i + 1) * n
            if b < K * n:
                X[a:b, a:] -= Q[a:b, b:] @ X[b:, a:]
            if self.scalar[i]:
                X[a:b, a:] /= Q[a, a]
            else:
                if lu is None:
                    lu = scipy.linalg.lu_factor(Q[a:b, a:b])
                X[a:b, a:] = scipy.linalg.lu_solve(lu, X[a:b, a:])

        return X
