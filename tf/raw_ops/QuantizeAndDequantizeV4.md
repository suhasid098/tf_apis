description: Quantizes then dequantizes a tensor.
robots: noindex

# tf.raw_ops.QuantizeAndDequantizeV4

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Quantizes then dequantizes a tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.QuantizeAndDequantizeV4`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.QuantizeAndDequantizeV4(
    input,
    input_min,
    input_max,
    signed_input=True,
    num_bits=8,
    range_given=False,
    round_mode=&#x27;HALF_TO_EVEN&#x27;,
    narrow_range=False,
    axis=-1,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is almost identical to QuantizeAndDequantizeV2, except that it returns a
gradient of 1 for inputs that are within the quantization range, or 0 otherwise.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
Tensor to quantize and then dequantize.
</td>
</tr><tr>
<td>
`input_min`
</td>
<td>
A `Tensor`. Must have the same type as `input`.
If `range_given == True`, this specifies the minimum input value that needs to
be represented, otherwise it is determined from the min value of the `input`
tensor.
</td>
</tr><tr>
<td>
`input_max`
</td>
<td>
A `Tensor`. Must have the same type as `input`.
If `range_given == True`, this specifies the maximum input value that needs to
be represented, otherwise it is determined from the max value of the `input`
tensor.
</td>
</tr><tr>
<td>
`signed_input`
</td>
<td>
An optional `bool`. Defaults to `True`.
Whether the quantization is signed or unsigned. (actually this parameter should
have been called <b>`signed_output`</b>)
</td>
</tr><tr>
<td>
`num_bits`
</td>
<td>
An optional `int`. Defaults to `8`.
The bitwidth of the quantization.
</td>
</tr><tr>
<td>
`range_given`
</td>
<td>
An optional `bool`. Defaults to `False`.
Whether the range is given or should be determined from the `input` tensor.
</td>
</tr><tr>
<td>
`round_mode`
</td>
<td>
An optional `string` from: `"HALF_TO_EVEN", "HALF_UP"`. Defaults to `"HALF_TO_EVEN"`.
The 'round_mode' attribute controls which rounding tie-breaking algorithm is
used when rounding float values to their quantized equivalents. The following
rounding modes are currently supported:

*   HALF_TO_EVEN: this is the default round_mode.
*   HALF_UP: round towards positive. In this mode 7.5 rounds up to 8 and -7.5
    rounds up to -7.
</td>
</tr><tr>
<td>
`narrow_range`
</td>
<td>
An optional `bool`. Defaults to `False`.
If True, then the absolute value of the quantized minimum value is the same as
the quantized maximum value, instead of 1 greater.
i.e. for 8 bit quantization, the minimum value is -127 instead of -128.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
An optional `int`. Defaults to `-1`.
If specified, this axis is treated as a channel or slice axis, and a separate
quantization range is used for each channel or slice along this axis.
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
A `Tensor`. Has the same type as `input`.
</td>
</tr>

</table>

