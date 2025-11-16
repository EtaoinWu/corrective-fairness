from beartype import beartype
from beartype.typing import Callable, overload
from jax import (
    numpy as jnp,
    random as jr,
)
from jaxtyping import Array, ArrayLike, Float, Key, Scalar, ScalarLike, jaxtyped


def typed(func: Callable) -> Callable:
    return jaxtyped(func, typechecker=beartype)


@typed
def float_ife(
    cond: Float[ScalarLike, ""],
    yes: Float[ArrayLike, "*dim"],
    no: Float[ArrayLike, "*dim"],
) -> Float[Array, "*dim"]:
    return jnp.array(cond * yes + (1 - cond) * no)


@beartype
class KeyGen:
    """
    A stateful key generator that can be used to generate subkeys.
    """

    key: Key[Scalar, ""]

    @overload
    def __init__(self, seed: int): ...

    @overload
    def __init__(self, *, key: Key[Scalar, ""]): ...

    def __init__(
        self,
        seed: int | None = None,
        *,
        key: Key[Scalar, ""] | None = None,
    ):
        if seed is not None:
            if key is not None:
                raise ValueError(
                    "Either seed or key must be provided, not both."
                )
            self.key = jr.key(seed)
        elif key is not None:
            self.key = key
        else:
            raise ValueError("Either seed or key must be provided.")

    def __call__(
        self, n: int | tuple[int, ...] | None = None
    ) -> Key[Array, "..."]:
        """
        Generate a subkey or an array of subkeys.

        Parameters
        ----------
        n : shape, optional
            The shape of the array of subkeys to generate.

        Returns
        -------
        key[...]
            The subkey if n is None; or an array of subkeys, shaped as n.
        """
        self.key, subkey = jr.split(self.key)
        if n is None:
            return subkey
        return jr.split(subkey, n)
