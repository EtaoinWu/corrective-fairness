"""Model for the creditworthiness MDP model."""

import functools as ft

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real

from ..util import convolve_clipped, typed

type CreditDistribution = Float[Array, " n_states"]  # Sums to 1
type Policy = Float[Array, " n_states"]  # between 0 and 1
type Reward = Float[Array, " n_states"]  # monotonic
type SuccessProb = Float[Array, " n_states"]  # monotonic

# each innermost row sums to 1
# kernel[i, j]: the distribution of score change for action i and outcome j
type TransitionKernel = Float[Array, "2 2 n_kernel"]
type NoiseKernel = Float[Array, " n_noise"]  # Sums to 1
type Transition = tuple[TransitionKernel, NoiseKernel]


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


@typed
def construct_transition(
    repay: Real[Array, " n_transition"],
    default: Real[Array, " n_transition"],
    reject: Real[Array, " n_transition"],
    noise: Real[Array, " n_noise"],
) -> Transition:
    repay, default, reject, noise = [
        z / z.sum() for z in [repay, default, reject, noise]
    ]
    return (jnp.array([[repay, default], [reject, reject]]), noise)


@jax.jit
@typed
def transition(
    dis: CreditDistribution,
    pol: Policy,
    suc: SuccessProb,
    kern: TransitionKernel | Transition,
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
    if isinstance(kern, tuple):
        mat, noise = kern
        next_dis = jax.vmap(jax.vmap(convolve_clipped))(action_dis, mat)
        next_dis = jax.vmap(jax.vmap(ft.partial(convolve_clipped, b=noise)))(next_dis)
        return next_dis.sum(axis=0).sum(axis=0)
    else:
        next_dis = jax.vmap(jax.vmap(convolve_clipped))(action_dis, kern)
        return next_dis.sum(axis=0).sum(axis=0)

