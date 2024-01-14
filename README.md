# jax_tensor_puzzles

Solving Sasha Rush's [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) with JAX.

[JAX](https://jax.readthedocs.io/en/latest/index.html) is Google's Python array library for GPU computing and autodifferentiation.

## Intro

My goal was to implement standard `jax.numpy` (`jnp`) functions only using broadcasting, math, and simpler `jnp` functions.
Unlike Sasha's puzzles, my solutions use JAX arrays instead of PyTorch tensors.

Implementing the function completes the puzzle, and to start, you're only allowed to use the `jnp.arange()` constructor, multiplication, `@`,  `.shape`, and broadcasting and indexing. After solving a puzzle, you can use its implementation in the standard `jnp` lib for all the later puzzles.

For these challenges I'm using JAX metal on my M2 MacBook.

Note:  The original Tensor Puzzles require using TorchTyping for all of the solutions.  
These are not just type hints; the solution checker uses the expected types returned to certify that only tensor operations are being used.  
There is a jax.typing module but I'll be ignoring it for this, focusing on the array solutions.

## Puzzle 0:  Implement `jnp.where`

This function replaces an if statement.   An if statement can't be used in `vmap` or `@jit` blocks (see [here](https://github.com/google/jax/discussions/4951)), so this function (or better, its library implementation `jnp.where`) is of fundamental value.

#### Solution: 

```python
def where(q, a, b):
    return (q * a) + (~q) * b
```

## Puzzle 1:  Implement `jnp.ones`

Make a vector of ones with length `i`

#### Solution: 

Here we use `jnp.arange` and `jnp.where`.  The trick is using an "if condition" that's never met, so all the elements are set to 1.

```python
def ones(a : int):
    jnp.where(jnp.arange(i) < 0, 0, 1)
```

## Puzzle 2:  Implement `jnp.sum`

#### Solution:

Here we're doing an inner product between the array and a vector of ones.  
The inner product is a loss of information, as it collapses two vectors into a scalar.

```python
def sum(a):
    return a @ jnp.ones(a.shape[0])
```

Summing all numbers 0 through 5

```python
sum(jnp.arange(6))
```

yields `Array(15, dtype=int32)`.  This is the right value, but it's in an array with empty shape.
Regular numpy would yield a scalar `int` or `float`.  This is a difference between JAX and numpy.


## Puzzle 3:  Implement `jnp.outer`

#### Solution:

This uses array broadcasting to convert two vectors a and b into a row and a column vector.  The `[:,None]` bit adds an extra dimension.
Just like the [broadcasting rules in numpy](), broadcasting in JAX works because at least one of the two arrays has a trailing dimension of 1.  

```python
def outer(a, b):
    return a[:,None] * b[None,:]
```

If a is length 5 and b is length 7, then `a[:,None]` has shape `(5,1)` and `b[None,:]` has shape `(1,7)`, so the multiplication will have shape `(5,7)`, which is what we wanted.

Note:  Using `@` instead of `*` on my machine (Mac M2 with Jax Metal) throws an XLA error if a and b aren't floats, like the ints produced in `jnp.arange()`.

Using something like `jax.random.normal(key, (5,))` will work with `@` however.

## Puzzle 4:  Implement `jnp.diag`  

#### Solution:

This solution is a nice example of how you can use tensors as indices.  As indices, they work like `a[(0,1,2,3,4),(0,1,2,3,4)]`.  Not quite like base Python's `zip`, but similar.

```python
def diag(a):
    return a[jnp.arange(a.shape[0]),jnp.arange(a.shape[0])]
```

## Puzzle 5: Implement `jnp.eye`

#### Solution:

I found this one very tricky.  The key is to broadcast *with* a boolean comparison.  It seems like the `True` / `False` diagonal matrix is created out of thin air.

```python
def eye(j):
    return jnp.where(jnp.arange(j)[:,None] == jnp.arange(j)[None,:], 1, 0)
```


