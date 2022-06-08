description: Performs the max pooling on the input.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.max_pool" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.nn.max_pool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Performs the max pooling on the input.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format=&#x27;NHWC&#x27;,
    name=None,
    input=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
A 4-D `Tensor` of the format specified by `data_format`.
</td>
</tr><tr>
<td>
`ksize`
</td>
<td>
An int or list of `ints` that has length `1`, `2` or `4`.
The size of the window for each dimension of the input tensor.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
An int or list of `ints` that has length `1`, `2` or `4`.
The stride of the sliding window for each dimension of the input tensor.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
Either the `string` `"SAME"` or `"VALID"` indicating the type of
padding algorithm to use, or a list indicating the explicit paddings at
the start and end of each dimension. When explicit padding is used and
data_format is `"NHWC"`, this should be in the form `[[0, 0], [pad_top,
pad_bottom], [pad_left, pad_right], [0, 0]]`. When explicit padding used
and data_format is `"NCHW"`, this should be in the form `[[0, 0], [0, 0],
[pad_top, pad_bottom], [pad_left, pad_right]]`. When using explicit
padding, the size of the paddings cannot be greater than the sliding
window size.
</td>
</tr><tr>
<td>
`data_format`
</td>
<td>
A string. 'NHWC', 'NCHW' and 'NCHW_VECT_C' are supported.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the operation.
</td>
</tr><tr>
<td>
`input`
</td>
<td>
Alias for value.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of format specified by `data_format`.
The max pooled output tensor.
</td>
</tr>

</table>

