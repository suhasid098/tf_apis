description: Computes log(sum(exp(elements across dimensions of a tensor))).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.math.reduce_logsumexp" />
<meta itemprop="path" content="Stable" />
</div>

# tf.math.reduce_logsumexp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/math_ops.py">View source</a>



Computes log(sum(exp(elements across dimensions of a tensor))).

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.reduce_logsumexp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.math.reduce_logsumexp(
    input_tensor, axis=None, keepdims=False, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
of the entries in `axis`, which must be unique. If `keepdims` is true, the
reduced dimensions are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

This function is more numerically stable than log(sum(exp(input))). It avoids
overflows caused by taking the exp of large inputs and underflows caused by
taking the log of small inputs.

#### For example:



```python
x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
tf.reduce_logsumexp(x)  # log(6)
tf.reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
tf.reduce_logsumexp(x, 1)  # [log(3), log(3)]
tf.reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
tf.reduce_logsumexp(x, [0, 1])  # log(6)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_tensor`
</td>
<td>
The tensor to reduce. Should have numeric type.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
The dimensions to reduce. If `None` (the default), reduces all
dimensions. Must be in the range `[-rank(input_tensor),
rank(input_tensor))`.
</td>
</tr><tr>
<td>
`keepdims`
</td>
<td>
If true, retains reduced dimensions with length 1.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The reduced tensor.
</td>
</tr>

</table>

