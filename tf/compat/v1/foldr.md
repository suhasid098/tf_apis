description: foldr on the list of tensors unpacked from elems on dimension 0.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.foldr" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.foldr

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/functional_ops.py">View source</a>



foldr on the list of tensors unpacked from `elems` on dimension 0.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.foldr(
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This foldr operator repeatedly applies the callable `fn` to a sequence
of elements from last to first. The elements are made of the tensors
unpacked from `elems`. The callable fn takes two tensors as arguments.
The first argument is the accumulated value computed from the preceding
invocation of fn, and the second is the value at the current position of
`elems`. If `initializer` is None, `elems` must contain at least one element,
and its first element is used as the initializer.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
of the result tensor is `fn(initializer, values[0]).shape`.

This method also allows multi-arity `elems` and output of `fn`.  If `elems`
is a (possibly nested) list or tuple of tensors, then each of these tensors
must have a matching first (unpack) dimension.  The signature of `fn` may
match the structure of `elems`.  That is, if `elems` is
`(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
`fn = lambda (t1, [t2, t3, [t4, t5]]):`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fn`
</td>
<td>
The callable to be performed.
</td>
</tr><tr>
<td>
`elems`
</td>
<td>
A tensor or (possibly nested) sequence of tensors, each of which will
be unpacked along their first dimension.  The nested sequence of the
resulting slices will be the first argument to `fn`.
</td>
</tr><tr>
<td>
`initializer`
</td>
<td>
(optional) A tensor or (possibly nested) sequence of tensors,
as the initial value for the accumulator.
</td>
</tr><tr>
<td>
`parallel_iterations`
</td>
<td>
(optional) The number of iterations allowed to run in
parallel.
</td>
</tr><tr>
<td>
`back_prop`
</td>
<td>
(optional) True enables support for back propagation.
</td>
</tr><tr>
<td>
`swap_memory`
</td>
<td>
(optional) True enables GPU-CPU memory swapping.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
(optional) Name prefix for the returned tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor or (possibly nested) sequence of tensors, resulting from applying
`fn` consecutively to the list of tensors unpacked from `elems`, from last
to first.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
if `fn` is not callable.
</td>
</tr>
</table>



#### Example:

```python
elems = [1, 2, 3, 4, 5, 6]
sum = foldr(lambda a, x: a + x, elems)
# sum == 21
```
