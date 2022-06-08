description: Initializes the multi device iterator with the given dataset.
robots: noindex

# tf.raw_ops.MultiDeviceIteratorInit

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Initializes the multi device iterator with the given dataset.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.MultiDeviceIteratorInit`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.MultiDeviceIteratorInit(
    dataset, multi_device_iterator, max_buffer_size, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset`
</td>
<td>
A `Tensor` of type `variant`. Dataset to be iterated upon.
</td>
</tr><tr>
<td>
`multi_device_iterator`
</td>
<td>
A `Tensor` of type `resource`.
A MultiDeviceIteratorResource.
</td>
</tr><tr>
<td>
`max_buffer_size`
</td>
<td>
A `Tensor` of type `int64`.
The maximum size of the host side per device buffer to keep.
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
A `Tensor` of type `int64`.
</td>
</tr>

</table>

