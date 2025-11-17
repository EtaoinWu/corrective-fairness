import chex
import jax.random as jr
import numpy as np

from fairsim.util import convolve_clipped


def mock_convolve_clipped(a, b):
    a = np.array(a)
    b = np.array(b)
    n = a.shape[0]
    m = b.shape[0]
    t = (m - 1) // 2
    c = np.zeros_like(a)
    for i in range(n):
        for j in range(m):
            k = min(max(i + j - t, 0), n - 1)
            c[k] += a[i] * b[j]
    return c


def test_convolve_clipped():
    k = jr.key(1)
    ks = jr.split(k, 10)
    for key in ks:
        a = jr.normal(key, (20,))
        b = jr.normal(key, (7,))
        c1 = mock_convolve_clipped(a, b)
        c2 = convolve_clipped(a, b)
        chex.assert_trees_all_close(c1, c2)
