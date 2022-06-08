description: Wrapper for Graph.get_collection_ref() using the default graph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.get_collection_ref" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.get_collection_ref

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/ops.py">View source</a>



Wrapper for `Graph.get_collection_ref()` using the default graph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.get_collection_ref(
    key
)
</code></pre>



<!-- Placeholder for "Used in" -->

See <a href="../../../tf/Graph.md#get_collection_ref"><code>tf.Graph.get_collection_ref</code></a>
for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`key`
</td>
<td>
The key for the collection. For example, the `GraphKeys` class contains
many standard names for collections.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The list of values in the collection with the given `name`, or an empty
list if no value has been added to that collection.  Note that this returns
the collection list itself, which can be modified in place to change the
collection.
</td>
</tr>

</table>




 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

Collections are not supported when eager execution is enabled.


 </devsite-expandable></section>

