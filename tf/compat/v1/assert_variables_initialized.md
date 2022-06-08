description: Returns an Op to check if variables are initialized.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.assert_variables_initialized" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.assert_variables_initialized

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/variables.py">View source</a>



Returns an Op to check if variables are initialized.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.assert_variables_initialized(
    var_list=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

NOTE: This function is obsolete and will be removed in 6 months.  Please
change your implementation to use `report_uninitialized_variables()`.

When run, the returned Op will raise the exception `FailedPreconditionError`
if any of the variables has not yet been initialized.

Note: This function is implemented by trying to fetch the values of the
variables. If one of the variables is not initialized a message may be
logged by the C++ runtime. This is expected.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`var_list`
</td>
<td>
List of `Variable` objects to check. Defaults to the value of
`global_variables().`
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An Op, or None if there are no variables.
</td>
</tr>

</table>


Note: The output of this function should be used. If it is not, a warning will be logged or an error may be raised. To mark the output as used, call its .mark_used() method.