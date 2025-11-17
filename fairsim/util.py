from beartype import beartype
from beartype.typing import Callable, overload
from jax import (
    numpy as jnp,
    random as jr,
    tree as jt,
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


@typed
def convolve_clipped(a: Float[Array, " n"], b: Float[Array, " m"]) -> Float[Array, " n"]:
  """
    Convolve two arrays using the following clipped formula:
    Let 2t+1 = m. Conceptually, let b = [b[-t], ..., b[0], ..., b[t]].
    for i: 
      for j: 
        c[clip(i+j, 0, n-1)] += a[i] * b[j]
    The result is c, which has the same length as a.

    Parameters
    ----------
    a : float[n]
      The array to be convolved.
    b : float[m]
      The kernel array, where m is odd.

    Returns
    -------
    float[n]
      The result of the clipped convolution.
  """

  n = a.shape[0]
  m = b.shape[0]
  t = (m - 1) // 2
  c_ = jnp.convolve(a, b, mode='full')
  c = c_[t:t+n]
  c = c.at[0].add(c_[:t].sum())
  c = c.at[-1].add(c_[t+n:].sum())
  return c


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

def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jt.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    # leaves, treedef = jt.flatten(tree)
    # n_trees = leaves[0].shape[0]
    # for i in range(n_trees):
    #     new_leaves = [leaf[i] for leaf in leaves]
    #     yield treedef.unflatten(new_leaves)
    return Unstacked(tree)


class UnstackedIter:
    def __init__(self, unstacked):
        self.unstacked = unstacked
        self.index = 0

    def __next__(self):
        if self.index >= len(self.unstacked):
            raise StopIteration
        item = self.unstacked[self.index]
        self.index += 1
        return item

    def __iter__(self):
        return self

class Unstacked:
    def __init__(self, tree):
        self.tree = tree
        self.leaves, self.treedef = jt.flatten(tree)
        self.n_trees = self.leaves[0].shape[0]
    
    def __getitem__(self, index):
        new_leaves = [leaf[index] for leaf in self.leaves]
        return self.treedef.unflatten(new_leaves)
    
    def __len__(self):
        return self.n_trees
    
    def __iter__(self):
        return UnstackedIter(self)
    
