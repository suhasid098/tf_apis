description: Adds ops to list the names of uninitialized variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.report_uninitialized_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.report_uninitialized_variables

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/variables.py">View source</a>



Adds ops to list the names of uninitialized variables.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.report_uninitialized_variables(
    var_list=None, name=&#x27;report_uninitialized_variables&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

When run, it returns a 1-D tensor containing the names of uninitialized
variables if there are any, or an empty array if there are none.

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
`global_variables() + local_variables()`
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name of the `Operation`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 1-D tensor containing names of the uninitialized variables, or an empty
1-D tensor if there are no variables or no uninitialized variables.
</td>
</tr>

</table>


Note: The output of this function should be used. If it is not, a warning will be logged or an error may be raised. To mark the output as used, call its .mark_used() method.