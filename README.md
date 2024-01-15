# JAX Tensor Puzzles

Solving Sasha Rush's [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) with JAX NumPy.

[JAX](https://jax.readthedocs.io/en/latest/index.html) is Google's Python array library for GPU computing and autodifferentiation.  JAX NumPy (`jax.numpy`) is a drop-in replacement for NumPy.  The major reason to use `jnp` is that it runs matrix computations on GPUs or TPUs automatically, *and* supports the GPU on Apple Silicon with [JAX Metal](https://developer.apple.com/metal/jax/).  A nice bonus is that it can compute gradients on almost everything, should you need to do this for machine learning or physics projects.

Can Sasha's PyTorch Tensor Puzzles be solved in JAX NumPy?  Are there any major differences?  I'm guessing not, but let's see if that's true.

## Puzzle Intro

My variant of 'Tensor Puzzles' was to implement standard `jax.numpy` (`jnp`) functions only using broadcasting, math, and simpler `jnp` functions.
So unlike Sasha's puzzles, my solutions use JAX NumPy arrays instead of PyTorch tensors.

Implementing the function completes the puzzle, and to start, you're only allowed to use the `jnp.arange()` constructor, multiplication, `@`,  `.shape`, and broadcasting and indexing. After solving a puzzle, you can use its implementation in the standard `jnp` lib for all the later puzzles.

For these challenges I'm using JAX Metal on my M2 MacBook.

Note:  The original Tensor Puzzles require using TorchTyping for all of the solutions.  
These are not just type hints; the solution checker uses the expected types returned to certify that only tensor operations are being used.  
There is a `jax.typing` module but I'll be ignoring it and focusing on the array solutions.

## Puzzle 0:  Implement `jnp.where`

This function replaces an if statement.   An if statement can't be used in `vmap` or `@jit` blocks (see [here](https://github.com/google/jax/discussions/4951)), so this function (or better, its library implementation `jnp.where`) is of fundamental value in JAX.

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
Just like the [broadcasting rules in NumPy](), broadcasting in JAX works because at least one of the two arrays has a trailing dimension of 1.  

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

## Puzzle 6:  Implement `jnp.triu`

Compute the upper triangular matrix for a square matrix by providing the `i` dimension.  
This is not exactly the way `jnp.triu` works, which takes a matrix as input.

#### Solution:

`jnp.eye`, the previous puzzle, was a red herring.  This one is easy using the trick for `diag` above.

```python
def triu(j):
    return jnp.where(jnp.arange(j)[:,None] <=  jnp.arange(j)[None,:],1, 0)
```
## Puzzle 7:  Implement `jnp.cumsum`

#### Solution:

The cumulative sum is the dot between the column vector and an upper triangular matrix.  
I'm using the `jnp.ones` from above, which unlike the implementation above, can actually take a tuple as input to specify the dimensions.

```python
def cumsum(a):
    return a @ jnp.triu(jnp.ones((a.shape[0],a.shape[0]) ))
```

## Puzzle 8:  Implement `jnp.diff`

This returns a vector with the first difference.  The real jnp.diff returns a vector missing the first element, this one the initial value should be returned unaltered.

#### Solution:

We need to use a `where` function to fix the first element. Because otherwise `a[jnp.arange(a.shape[0]) - 1]` on its own returns a circular permutation of `a`.

```python
def jnp_diff(a):
    return a - jnp.where(jnp.arange(a.shape[0]) != 0,  a[jnp.arange(a.shape[0]) - 1], 0)
```

## Puzzle 9:  Implement `jnp.vstack`

#### Solution:

The idea is you can broadcast `ones` and `arange(2)` and check equivalence to get a binary mask for `vstack`.  You may find out the order in the mask needs to be reversed.  There are few different options: swapping `a` and `b`, turning `==` into `!=`, or the solution I opted for, indexing the `arange(2)` in reverse.

```python
def vstack(a,b):
    jnp.where(jnp.ones(a.shape[0]) == jnp.arange(2)[::-1,None], a, b)
```

## Puzzle 10:  Implement `jnp.roll`
Circularly permute a vector (to the right) by an integer amount `i`.

#### Solution:

This one is much easier after doing `jnp.diff`.

```python
def roll(a, i):
    return a[arange(a.shape[0]) - i]
```

## Puzzle 11: Implement `jnp.flip`

Return a vector in reverse order.  

#### Solution:

```python
def flip(a):
    a[arange(a.shape[0])[::-1]]
```

## Puzzle 12:  Implement `jnp.compress`


