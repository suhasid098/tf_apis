description: Adds a value to the current value of a variable.
robots: noindex

# tf.raw_ops.AssignAddVariableOp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Adds a value to the current value of a variable.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.AssignAddVariableOp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.AssignAddVariableOp(
    resource, value, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Any ReadVariableOp with a control dependency on this op is guaranteed to
see the incremented value or a subsequent newer one.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`resource`
</td>
<td>
A `Tensor` of type `resource`.
handle to the resource in which to store the variable.
</td>
</tr><tr>
<td>
`value`
</td>
<td>
A `Tensor`. the value by which the variable will be incremented.
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
The created Operation.
</td>
</tr>

</table>

