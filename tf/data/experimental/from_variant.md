description: Constructs a dataset from the given variant and (nested) structure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.from_variant" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.from_variant

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/ops/dataset_ops.py">View source</a>



Constructs a dataset from the given variant and (nested) structure.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.from_variant`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.from_variant(
    variant, structure
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`variant`
</td>
<td>
A scalar <a href="../../../tf.md#variant"><code>tf.variant</code></a> tensor representing a dataset.
</td>
</tr><tr>
<td>
`structure`
</td>
<td>
A (nested) structure of <a href="../../../tf/TypeSpec.md"><code>tf.TypeSpec</code></a> objects representing the
structure of each element in the dataset.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> instance.
</td>
</tr>

</table>

