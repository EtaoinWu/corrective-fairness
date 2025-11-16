"""Model for the individual creditworthiness model with continuous parameters.
"""

import jax.numpy as jnp
import typinox as tpx
from jaxtyping import Array, Float, Scalar, ScalarLike

from ..util import float_ife


class IndividualParams(tpx.TypedModule):
    """Parameters of the individual model.

    Parameters
    ----------
    decline_credit_penalty : float
        The penalty applied to the credit score when a loan is declined.
        Represents the harm of a hard pull on the individual's credit.
    score_alpha : float
        The exponential moving average parameter for the credit score.
        A smaller alpha means the score updates more slowly.
    decline_latent_penalty : float
        The penalty to the latent creditworthiness when a loan is declined.
    default_latent_penalty : float
        The penalty to the latent creditworthiness when a loan is approved but defaulted.
    repay_latent_gain : float
        The gain to the latent creditworthiness when a loan is approved and repaid.
    repay_rate_min : float
    repay_rate_max : float
        The minimum and maximum repayment rates.
    """

    decline_credit_penalty: Float[ScalarLike, ""]
    score_alpha: Float[ScalarLike, ""]

    decline_latent_penalty: Float[ScalarLike, ""]
    default_latent_penalty: Float[ScalarLike, ""]
    repay_latent_gain: Float[ScalarLike, ""]

    latent_noise_scale: Float[ScalarLike, ""]
    score_noise_scale: Float[ScalarLike, ""]


class Individual(tpx.TypedModule):
    """Model for an individual's latent creditworthiness and credit score.

    Parameters
    ----------
    latent : float
        The latent creditworthiness of the individual. Their non-default rate is
        determined as the probit of their latent score.
    score : float
        The credit score of the individual.
        Updates as an exponential moving average of the repayment behavior.
    """

    latent: Float[Scalar, ""]
    score: Float[Scalar, ""]

    params: IndividualParams = tpx.field(static=True)

    @property
    def repay_probability(self) -> Float[Scalar, ""]:
        """The probability that the individual will repay a loan."""
        return self.latent

    def update(self, approval: Float[Scalar, ""], repay: Float[Scalar, ""], noise: Float[Array, "2"]):
        # If the loan is approved, update the score as an exponential moving average of repayment.
        # If the loan is declined, apply a decline penalty to the score to reflect the harm of a hard pull.
        new_score = float_ife(
            approval,
            float_ife(self.params.score_alpha, repay, self.score),
            self.score * (1 - self.params.decline_credit_penalty),
        )
        new_score = new_score + noise[0] * self.params.score_noise_scale
        new_score = jnp.clip(new_score, 0, 1)
        new_latent = float_ife(
            approval,
            float_ife(
                repay,
                self.latent + self.params.repay_latent_gain,
                self.latent - self.params.default_latent_penalty,
            ),
            self.latent - self.params.decline_latent_penalty,
        )
        new_latent = jnp.clip(new_latent, 0, 1)
        new_latent = new_latent + noise[1] * self.params.latent_noise_scale
        new_latent = jnp.clip(new_latent, 0, 1)
        return self.__replace__(latent=new_latent, score=new_score)
