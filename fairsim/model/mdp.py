"""Model for the creditworthiness MDP model."""

from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real, Scalar

from ..util import convolve_clipped, typed

type CreditDistribution = Float[Array, " n_states"]  # Sums to 1
type Policy = Float[Array, " n_states"]  # between 0 and 1
type Reward = Float[Array, " n_states"]  # monotonic
type SuccessProb = Float[Array, " n_states"]  # monotonic

# each innermost row sums to 1
# kernel[i, j]: the distribution of score change for action i and outcome j
type TransitionKernel = Float[Array, "2 2 n_kernel"]


@typed
def transition_kernel_construct(
    repay: Real[Array, " n_transition"],
    default: Real[Array, " n_transition"],
    reject: Real[Array, " n_transition"],
    noise: Real[Array, " n_noise"],
) -> Float[Array, "2 2 (n_noise+n_transition-1)"]:
    repay, default, reject, noise = [
        z / z.sum() for z in [repay, default, reject, noise]
    ]
    rt, dt, ut = [
        jnp.convolve(t, noise, mode="full") for t in [repay, default, reject]
    ]
    return jnp.array([[rt, dt], [ut, ut]])


@jax.jit
@typed
def transition(
    dis: CreditDistribution,
    pol: Policy,
    suc: SuccessProb,
    kern: TransitionKernel,
) -> CreditDistribution:
    """
    Transition of the credit distribution given a policy.

    Parameters
    ----------
    dis : float[n_states], sum to 1
        The current credit distribution.
    pol : float[n_states], between 0 and 1
        The policy, representing the probability of approval.
    suc : float[n_states], monotonic
        The success probability for each state.
    kern : float[2 2 n_kernel], each innermost row sums to 1
        The transition kernel for each action.

    Returns
    -------
    float[n_states], sum to 1
        The next credit distribution.
    """

    # action_dis[i, j, k] = Pr[action=i, outcome=j, state=k]
    action_dis = (
        dis[None, None, :]
        * jnp.stack([pol, 1 - pol])[:, None, :]
        * jnp.stack([suc, 1 - suc])[None, :, :]
    )
    next_dis = jax.vmap(jax.vmap(convolve_clipped))(action_dis, kern)
    return next_dis.sum(axis=0).sum(axis=0)


@jax.jit
@typed
def opt_unconstrained(dis: CreditDistribution, rew: Reward) -> Policy:
    """Myopic optimal unconstrained policy.
    maximize sum_s pol[s] * rew[s]

    Parameters
    ----------
    dis : float[n_states], sum to 1
        The current credit distribution.
    rew : float[n_states], monotonic
        The reward for each state.

    Returns
    -------
    float[n_states]
        The optimal policy.
    """
    return (rew > 0).astype(float)


@typed
def _generalized_threshold_match(
    a: Float[Array, " n"],
    w: Float[Scalar, ""],
) -> Float[Array, " n"]:
    """
    Find a generalized threshold function with an inner product constraint.
    A generalized threshold function looks like [0, ..., 0, c, 1, ..., 1].
    This function solves for x such that <a, x> = w.

    Parameters
    ----------
    a : float[n]
        Array of positive values.
    w : float
        Inner product constraint.

    Returns
    -------
    float[n]
        Generalized threshold function.
    """
    a = a[::-1]
    cdf = a.cumsum()
    idx = jnp.searchsorted(cdf, w, side="left")
    x = (jnp.arange(a.shape[0]) < idx).astype(float)
    below = cast(
        Float[Scalar, ""],
        jax.lax.cond(idx > 0, lambda: cdf[idx - 1], lambda: 0.0),
    )
    x = x.at[idx].add((w - below) / a[idx])
    return x[::-1]


@jax.jit
@typed
def opt_choice_prob(
    dis: CreditDistribution, suc: SuccessProb, alpha: Float[Scalar, ""]
) -> Policy:
    """Myopic optimal policy given a choice probability parameter.
    The constraint is given as:
        sum_s pol[s] dis[s] = alpha.

    Parameters
    ----------
    dis : float[n_states], sum to 1
        The current credit distribution.
    suc : float[n_states], monotonic
        The success probability for each state.
    alpha : float, between 0 and 1
        The required choice probability constraint.

    Returns
    -------
    float[n_states]
        The optimal policy.
    """

    return _generalized_threshold_match(dis, alpha)


@jax.jit
@typed
def opt_tpr(
    dis: CreditDistribution,
    suc: SuccessProb,
    beta: Float[Scalar, ""],
) -> Policy:
    """Myopic optimal policy given a true positive rate constraint.
    The constraint is given as:
        sum_s pol[s] dis[s] suc[s] / sum_s dis[s] suc[s] = beta.
    We assume that
        rew[s] = suc[s] * rew_success + (1 - suc[s]) * rew_failure,
        where rew_success > rew_failure and rew_failure < 0.

    Parameters
    ----------
    dis : float[n_states], sum to 1
        The current credit distribution.
    suc : float[n_states], monotonic
        The success probability for each state.
    beta : float
        The required true positive rate constraint.

    Returns
    -------
    float[n_states]
        The optimal policy.
    """

    # Rewrite the constraint:
    # sum_s pol[s] dis[s] suc[s] = rhs
    cap = dis * suc
    rhs = beta * cap.sum()

    # Rewrite the objective:
    # min_pol sum_s dis[s] pol[s]
    # from KKT, the solution is a generalized threshold function.
    return _generalized_threshold_match(cap, rhs)
