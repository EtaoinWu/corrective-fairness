from typing import cast

import jax
import jax.numpy as jnp
from beartype.typing import Callable, Literal
from jaxtyping import Array, Float, Scalar, ScalarLike
from mpax import create_lp, r2HPDHG, raPDHG
from typinox import Vmapped

from fairsim.util import typed

from .mdp import CreditDistribution, Policy, Reward, SuccessProb
from .myopic import _generalized_threshold_match


@typed
def prepend_zero(x: Float[Array, " n"]) -> Float[Array, " (n+1)"]:
    return jnp.concatenate([jnp.array([0.0]), x])


_alpha = (3 - jnp.sqrt(5)) / 2


def ternary_split_phi(l, r):
    dif = (r - l) * _alpha
    return l + dif, r - dif


@typed
def ternary_maximize_1d(
    func: Callable[[Float[Scalar, ""]], Float[Scalar, ""]],
    l: Float[ScalarLike, ""],
    r: Float[ScalarLike, ""],
    n_iter: int = 43,
) -> tuple[Float[Scalar, ""], Float[Scalar, ""]]:
    """
    Maximize a concave function on a 1-dimensional interval using ternary search.

    Parameters
    ----------
    func: float->float
        The function to maximize. Must be concave.
    l, r: float
        The left and right boundaries of the interval.
    n_iter: int, optional
        The number of iterations to perform. The primal accuracy is approximately 0.618^n_iter.
        The default value of 43 gives an accuracy of 1e-9.
    """
    l, r = jnp.array(l), jnp.array(r)
    l1, r1 = ternary_split_phi(l, r)
    fl1, fr1 = func(l1), func(r1)

    def iter(state, _):
        l, r, fl1, fr1 = state
        l1, r1 = ternary_split_phi(l, r)
        return (
            jax.lax.cond(
                fl1 < fr1,
                lambda: (l1, r, fr1, func(r - (r1 - l1))),
                lambda: (l, r1, func(l + (r1 - l1)), fl1),
            ),
            None,
        )

    state, _ = jax.lax.scan(iter, (l, r, fl1, fr1), None, length=n_iter)
    l, r, fl1, fr1 = state
    l1, r1 = ternary_split_phi(l, r)
    return jax.lax.cond(
        fl1 > fr1,
        lambda: (l1, fl1),
        lambda: (r1, fr1),
    )


@jax.jit
@typed
def joint_opt_demographic_parity(
    dis: Vmapped[CreditDistribution, " n_groups"],
    weights: Float[Array, " n_groups"],
    rew: Reward,
) -> tuple[Float[Scalar, ""], Vmapped[Policy, " n_groups"]]:
    """Jointly optimize myopic rewards, given that the choice probability is the same.

    Parameters
    ----------
    dis : (float[n_states], sum to 1)[n_groups]
        The score distribution for each group.
    weights : float[n_groups]
        The population of each group.
    rew : float[n_states], monotonic
        The reward for every score.

    Returns
    -------
    threshold : float
        The optimal choice ratio.
    policies : (float[n_states])[n_groups]
        The corresponding policies for each group.
    """

    rdis = dis[:, ::-1]
    cdfs = jax.vmap(lambda d: d.cumsum())(rdis)
    rew = rew[::-1]
    duals = rdis * rew[None, :]
    cduals = jax.vmap(lambda x: prepend_zero(x.cumsum()))(duals)

    # Step 1: Evaluate thresholds
    @typed
    def eval_threshold(th: Float[Scalar, ""]):
        @typed
        def eval_single_cdf(
            rd: Float[Array, " n_states"],
            cdf: Float[Array, " n_states"],
            dual: Float[Array, " n_states"],
            cdual: Float[Array, " (n_states+1)"],
        ):
            return _generalized_threshold_match(
                rd, th, inverse=False, cdf=cdf, dual=dual, dual_cum=cdual
            )

        return jnp.dot(
            weights, jax.vmap(eval_single_cdf)(rdis, cdfs, duals, cduals)
        )

    # Step 2: Find the optimal threshold using ternary search
    best_th, _ = ternary_maximize_1d(eval_threshold, 0.0, 1.0)

    # Step 3: Construct the policy for the optimal threshold
    @typed
    def single_threshold(
        rd: Float[Array, " n_states"],
        cdf: Float[Array, " n_states"],
    ):
        return _generalized_threshold_match(
            rd, best_th, inverse=False, cdf=cdf
        )[::-1]

    return best_th, jax.vmap(single_threshold)(rdis, cdfs)


@jax.jit
@typed
def joint_opt_equal_opportunity(
    dis: Vmapped[CreditDistribution, " n_groups"],
    weights: Float[Array, " n_groups"],
    suc: SuccessProb,
    rew: Reward,
) -> tuple[Float[Scalar, ""], Vmapped[Policy, " n_groups"]]:
    """Jointly optimize myopic rewards, given that the TPR is the same.

    Parameters
    ----------
    dis : (float[n_states], sum to 1)[n_groups]
        The score distribution for each group.
    weights : float[n_groups]
        The population of each group.
    suc : float[n_states], monotonic
        The success probability for every score.
    rew : float[n_states], monotonic
        The reward for every score.

    Returns
    -------
    threshold : float
        The optimal TPR.
    policies : (float[n_states])[n_groups]
        The corresponding policies for each group.
    """
    caps = (dis * suc[None, :])[:, ::-1]
    capsums = jax.vmap(jnp.sum)(caps)
    ccaps = jax.vmap(jnp.cumsum)(caps)
    
    rdis = dis[:, ::-1]
    rew = rew[::-1]
    duals = rdis * rew[None, :]
    cduals = jax.vmap(lambda x: prepend_zero(x.cumsum()))(duals)

    # Step 1: Evaluate each TPR
    @typed
    def eval_tpr(tpr: Float[Scalar, ""]):
        @typed
        def eval_single_cdf(
            cap: Float[Array, " n_states"],
            capsum: Float[Scalar, ""],
            ccap: Float[Array, " n_states"],
            dual: Float[Array, " n_states"],
            cdual: Float[Array, " (n_states+1)"],
        ):
            return _generalized_threshold_match(
                cap, tpr * capsum, inverse=False, cdf=ccap, dual=dual, dual_cum=cdual
            )

        return jnp.dot(
            weights, jax.vmap(eval_single_cdf)(caps, capsums, ccaps, duals, cduals)
        )

    # Step 2: Find the optimal TPR using ternary search
    best_tpr, _ = ternary_maximize_1d(eval_tpr, 0.0, 1.0)

    # Step 3: Construct the policy for the optimal TPR
    @typed
    def single_tpr(
        cap: Float[Array, " n_states"],
        capsum: Float[Scalar, ""],
        ccap: Float[Array, " n_states"],
    ):
        return _generalized_threshold_match(
            cap, best_tpr * capsum, inverse=False, cdf=ccap
        )[::-1]
    
    return best_tpr, jax.vmap(single_tpr)(caps, capsums, ccaps)


solvers = {
    'r2HPDHG': r2HPDHG(eps_abs=1e-5, eps_rel=1e-5, verbose=False),
    'raPDHG': raPDHG(eps_abs=1e-5, eps_rel=1e-5, verbose=False),
}


@jax.jit
@typed
def joint_opt_equalized_odds(
    dis: Vmapped[CreditDistribution, " n_groups"],
    weights: Float[Array, " n_groups"],
    suc: SuccessProb,
    rew: Reward,
    solver: Literal['r2HPDHG', 'raPDHG'] = 'r2HPDHG',
) -> tuple[Float[Scalar, ""], Vmapped[Policy, " n_groups"]]:
    """Jointly optimize myopic rewards, given that the TPR and TNR are the same.

    Parameters
    ----------
    dis : (float[n_states], sum to 1)[n_groups]
        The score distribution for each group.
    weights : float[n_groups]
        The population of each group.
    suc : float[n_states], monotonic
        The success probability for every score.
    rew : float[n_states], monotonic
        The reward for every score.

    Returns
    -------
    threshold : float
        The optimal TPR.
    policies : (float[n_states])[n_groups]
        The corresponding policies for each group.
    """
    l, u = 0, 1
    c = (dis * weights[:, None] * rew[None, :]).flatten()
    n = suc.shape[0]
    a0l = dis[0] * suc
    a0r = dis[1] * suc
    a1l = dis[0] * (1 - suc)
    a1r = dis[1] * (1 - suc)
    a0l, a0r, a1l, a1r = map(lambda x: x / x.sum(), (a0l, a0r, a1l, a1r))
    a = jnp.block([[a0l, -a0r], [a1l, -a1r]])
    b = jnp.array([0.0, 0.0])
    g = jnp.zeros((0, n * 2))
    h = jnp.array([])
    lp = create_lp(-c, a, b, g, h, l, u, use_sparse_matrix=False)
    result = solvers[solver].optimize(lp)
    return jnp.array(0.0), result.primal_solution.reshape((2, n))
