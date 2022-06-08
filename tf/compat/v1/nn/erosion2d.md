description: Computes the grayscale erosion of 4-D value and 3-D kernel tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.erosion2d" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.nn.erosion2d

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/nn_ops.py">View source</a>



Computes the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.erosion2d(
    value, kernel, strides, rates, padding, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The `value` tensor has shape `[batch, in_height, in_width, depth]` and the
`kernel` tensor has shape `[kernel_height, kernel_width, depth]`, i.e.,
each input channel is processed independently of the others with its own
structuring function. The `output` tensor has shape
`[batch, out_height, out_width, depth]`. The spatial dimensions of the
output tensor depend on the `padding` algorithm. We currently only support the
default "NHWC" `data_format`.

In detail, the grayscale morphological 2-D erosion is given by:

    output[b, y, x, c] =
       min_{dy, dx} value[b,
                          strides[1] * y - rates[1] * dy,
                          strides[2] * x - rates[2] * dx,
                          c] -
                    kernel[dy, dx, c]

Duality: The erosion of `value` by the `kernel` is equal to the negation of
the dilation of `-value` by the reflected `kernel`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
</td>
</tr><tr>
<td>
`kernel`
</td>
<td>
A `Tensor`. Must have the same type as `value`.
3-D with shape `[kernel_height, kernel_width, depth]`.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
A list of `ints` that has length `>= 4`.
1-D of length 4. The stride of the sliding window for each dimension of
the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
</td>
</tr><tr>
<td>
`rates`
</td>
<td>
A list of `ints` that has length `>= 4`.
1-D of length 4. The input stride for atrous morphological dilation.
Must be: `[1, rate_height, rate_width, 1]`.
</td>
</tr><tr>
<td>
`padding`
</td>
<td>
A `string` from: `"SAME", "VALID"`.
The type of padding algorithm to use.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional). If not specified "erosion2d"
is used.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor`. Has the same type as `value`.
4-D with shape `[batch, out_height, out_width, depth]`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the `value` depth does not match `kernel`' shape, or if
padding is other than `'VALID'` or `'SAME'`.
</td>
</tr>
</table>

