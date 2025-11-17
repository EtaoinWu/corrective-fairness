import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

from fairsim.model.joint import (
    joint_opt_demographic_parity,
    joint_opt_equal_opportunity,
)
from fairsim.model.myopic import (
    opt_choice_prob,
    opt_tpr,
)
from fairsim.util import KeyGen, tree_stack


@pytest.mark.parametrize("n_scores", [3, 5, 10, 42])
def test_joint_opt_demographic_parity(n_scores):
    for seed in range(100):
        kg = KeyGen(seed=seed)
        succ_prob = jr.uniform(kg(), shape=(n_scores,))
        succ_prob = jnp.sort(succ_prob)
        reward_threshold: Array = jr.uniform(
            kg(), shape=(), minval=succ_prob.min(), maxval=succ_prob.max()
        )
        reward = succ_prob - reward_threshold
        distrib1 = jr.dirichlet(kg(), jnp.ones(n_scores))
        distrib2 = jr.dirichlet(kg(), jnp.ones(n_scores))
        weights = jr.dirichlet(kg(), jnp.ones(2))
        distribs = tree_stack([distrib1, distrib2])
        th, policies = joint_opt_demographic_parity(distribs, weights, reward)

        def eval_pol(pol):
            return (
                pol[0].dot(reward * distrib1) * weights[0]
                + pol[1].dot(reward * distrib2) * weights[1]
            )

        def eval_th(th):
            return eval_pol(
                [
                    opt_choice_prob(distrib1, succ_prob, th),
                    opt_choice_prob(distrib2, succ_prob, th),
                ]
            )

        # Check that the threshold found by the joint optimization matches the policies
        assert jnp.isclose(eval_th(th), eval_pol(policies))

        ths = jnp.linspace(0, 1, 10001)
        vals = jax.vmap(eval_th)(ths)
        # Check that the threshold found by the joint optimization is optimal
        assert vals.max() <= eval_pol(policies) + 1e-7


@pytest.mark.parametrize("n_scores", [3, 5, 10, 42])
def test_joint_opt_equal_opportunity(n_scores):
    for seed in range(100):
        kg = KeyGen(seed=seed)
        succ_prob = jr.uniform(kg(), shape=(n_scores,))
        succ_prob = jnp.sort(succ_prob)
        reward_threshold: Array = jr.uniform(
            kg(), shape=(), minval=succ_prob.min(), maxval=succ_prob.max()
        )
        reward = succ_prob - reward_threshold
        distrib1 = jr.dirichlet(kg(), jnp.ones(n_scores))
        distrib2 = jr.dirichlet(kg(), jnp.ones(n_scores))
        weights = jr.dirichlet(kg(), jnp.ones(2))
        distribs = tree_stack([distrib1, distrib2])
        th, policies = joint_opt_equal_opportunity(distribs, weights, succ_prob, reward)

        def eval_pol(pol):
            return (
                pol[0].dot(reward * distrib1) * weights[0]
                + pol[1].dot(reward * distrib2) * weights[1]
            )

        def eval_th(th):
            return eval_pol(
                [
                    opt_tpr(distrib1, succ_prob, th),
                    opt_tpr(distrib2, succ_prob, th),
                ]
            )

        # Check that the threshold found by the joint optimization matches the policies
        assert jnp.isclose(eval_th(th), eval_pol(policies))

        ths = jnp.linspace(0, 1, 10001)
        vals = jax.vmap(eval_th)(ths)
        # Check that the threshold found by the joint optimization is optimal
        assert vals.max() <= eval_pol(policies) + 1e-7
