import warnings

from jax import config

warnings.filterwarnings(
    "ignore", message="As of jaxtyping version 0.2.24"
)

config.update("jax_enable_x64", True)
config.update("jax_numpy_rank_promotion", "warn")
