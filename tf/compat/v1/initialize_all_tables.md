description: Returns an Op that initializes all tables of the default graph. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.initialize_all_tables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.initialize_all_tables

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/lookup_ops.py">View source</a>



Returns an Op that initializes all tables of the default graph. (deprecated)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.initialize_all_tables(
    name=&#x27;init_all_tables&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.tables_initializer` instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
Optional name for the initialization op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An Op that initializes all tables.  Note that if there are
not tables the returned Op is a NoOp.
</td>
</tr>

</table>

