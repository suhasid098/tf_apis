description: Returns a list which has the passed-in Tensor as last element and the other elements of the given list in input_handle.
robots: noindex

# tf.raw_ops.TensorListPushBack

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorListPushBack`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorListPushBack(
    input_handle, tensor, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

tensor: The tensor to put on the list.
input_handle: The old list.
output_handle: A list with the elements of the old list followed by tensor.
element_dtype: the type of elements in the list.
element_shape: a shape compatible with that of elements in the list.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_handle`
</td>
<td>
A `Tensor` of type `variant`.
</td>
</tr><tr>
<td>
`tensor`
</td>
<td>
A `Tensor`.
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
A `Tensor` of type `variant`.
</td>
</tr>

</table>

