description: Update '*var' according to the Adam algorithm.
robots: noindex

# tf.raw_ops.ApplyAdam

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Update '*var' according to the Adam algorithm.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.ApplyAdam`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.ApplyAdam(
    var,
    m,
    v,
    beta1_power,
    beta2_power,
    lr,
    beta1,
    beta2,
    epsilon,
    grad,
    use_locking=False,
    use_nesterov=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

$$\text{lr}_t := \mathrm{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$$
$$m_t := \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g$$
$$v_t := \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g^2$$
$$\text{var} := \begin{cases} \text{var} - (m_t \beta_1 + g \cdot (1 - \beta_1))\cdot\text{lr}_t/(\sqrt{v_t} + \epsilon), &\text{if use_nesterov}\\\\  \text{var} - m_t \cdot \text{lr}_t /(\sqrt{v_t} + \epsilon), &\text{otherwise} \end{cases}$$

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`var`
</td>
<td>
A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
Should be from a Variable().
</td>
</tr><tr>
<td>
`m`
</td>
<td>
A mutable `Tensor`. Must have the same type as `var`.
Should be from a Variable().
</td>
</tr><tr>
<td>
`v`
</td>
<td>
A mutable `Tensor`. Must have the same type as `var`.
Should be from a Variable().
</td>
</tr><tr>
<td>
`beta1_power`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Must be a scalar.
</td>
</tr><tr>
<td>
`beta2_power`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Must be a scalar.
</td>
</tr><tr>
<td>
`lr`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Scaling factor. Must be a scalar.
</td>
</tr><tr>
<td>
`beta1`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Momentum factor. Must be a scalar.
</td>
</tr><tr>
<td>
`beta2`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Momentum factor. Must be a scalar.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
A `Tensor`. Must have the same type as `var`.
Ridge term. Must be a scalar.
</td>
</tr><tr>
<td>
`grad`
</td>
<td>
A `Tensor`. Must have the same type as `var`. The gradient.
</td>
</tr><tr>
<td>
`use_locking`
</td>
<td>
An optional `bool`. Defaults to `False`.
If `True`, updating of the var, m, and v tensors will be protected
by a lock; otherwise the behavior is undefined, but may exhibit less
contention.
</td>
</tr><tr>
<td>
`use_nesterov`
</td>
<td>
An optional `bool`. Defaults to `False`.
If `True`, uses the nesterov update.
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
A mutable `Tensor`. Has the same type as `var`.
</td>
</tr>

</table>

