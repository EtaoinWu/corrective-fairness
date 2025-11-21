"""ML model for deciding the policy for the institution."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import typinox as tpx
from beartype.typing import Callable
from jaxtyping import Array, Float, Key, Scalar, ScalarLike
from typinox import Vmapped

from ..model.mdp import CreditDistribution, Policy, Setting
from ..util import KeyGen, typed

type DecisionInput = Vmapped[CreditDistribution, " n_groups"]
type DecisionOutput = Vmapped[Policy, " n_groups"]
type ConstraintType = Literal["unconstrained", "eo", "dp"]


class DecisionUnnet(tpx.Module):
    values: Float[Array, " n_states"]
    def __init__(
        self,
        n_states: int,
        n_groups: int = 2,
        *,
        key: Key[Scalar, ""],
    ):
        self.values = jax.random.normal(key, (n_states,))

    def __call__(
        self,
        x: DecisionInput,
    ) -> DecisionOutput:
        policy = jax.vmap(jax.nn.sigmoid)(self.values)
        return jax.vmap(lambda _: policy)(x)


class DecisionNet(tpx.Module):
    """A simple feedforward neural network for decision making."""

    layers: list[eqx.nn.Linear | eqx.nn.PReLU]

    def __init__(
        self,
        n_states: int,
        n_groups: int = 2,
        n_hidden: int = 3,
        w_hidden: int = -1,
        *,
        key: Key[Scalar, ""],
    ):
        n_input = n_states * n_groups
        n_output = n_input
        if w_hidden == -1:
            w_hidden = n_input * 4
        kg = KeyGen(key=key)
        layers = []
        for i in range(n_hidden + 1):
            n_in = n_input if i == 0 else w_hidden
            n_out = n_output if i == n_hidden else w_hidden
            layers.append(eqx.nn.Linear(n_in, n_out, key=kg()))
            if i < n_hidden:
                layers.append(eqx.nn.PReLU(init_alpha=0.25))
        self.layers = layers

    def __call__(
        self,
        x: DecisionInput,
    ) -> DecisionOutput:
        """Forward pass of the network.

        Parameters
        ----------
        x : float[n_groups, n_states]
            The input credit distributions for each group.

        Returns
        -------
        float[n_groups, n_states]
            The output policies for each group.
        """
        n_groups, n_states = x.shape
        x = x.flatten()
        for layer in self.layers:
            x = layer(x)
        x = jax.scipy.stats.norm.cdf(x)
        x = x.reshape((n_groups, n_states))
        return x


@typed
def policy_and_loss(
    net: Callable[[DecisionInput], DecisionOutput],
    distrib: DecisionInput,
    weights: Float[Array, " n_groups"],
    setting: Setting,
    constraint_type: ConstraintType = "unconstrained",
    constraint_alpha: Float[ScalarLike, ""] = 1.0,
) -> tuple[DecisionOutput, Float[Scalar, ""], Float[Scalar, ""]]:
    policies = net(distrib)
    constraint_loss = jnp.array(0.0)
    if constraint_type == "dp":
        choice_probs = jnp.sum(distrib * policies, axis=1)
        constraint_loss += jnp.var(jnp.log(choice_probs))
        dp_normalizer = jnp.min(choice_probs) / choice_probs
        policies = policies * dp_normalizer[:, None]
    elif constraint_type == "eo":
        tprs = jnp.sum(
            distrib * policies * setting.success_prob[None, :], axis=1
        ) / jnp.sum(distrib * setting.success_prob[None, :], axis=1)
        constraint_loss += jnp.var(jnp.log(tprs))
        eo_normalizer = jnp.min(tprs) / tprs
        policies = policies * eo_normalizer[:, None]

    expected_reward = jnp.sum(
        distrib * policies * setting.reward[None, :] * weights[:, None]
    )
    loss = -expected_reward + constraint_alpha * constraint_loss
    return policies, loss, constraint_loss


@typed
def fair_loss(
    net: Callable[[DecisionInput], DecisionOutput],
    distrib: DecisionInput,
    weights: Float[Array, " n_groups"],
    setting: Setting,
    constraint: ConstraintType = "unconstrained",
    constraint_alpha: Float[ScalarLike, ""] = 1.0,
) -> Float[Scalar, ""]:
    return policy_and_loss(
        net,
        distrib,
        weights,
        setting,
        constraint,
        constraint_alpha,
    )[1]
