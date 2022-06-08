description: Assert the condition x != y holds for all elements.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.assert_none_equal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.assert_none_equal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/check_ops.py">View source</a>



Assert the condition `x != y` holds for all elements.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.assert_none_equal(
    x, y, summarize=None, message=None, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This Op checks that `x[i] != y[i]` holds for every pair of (possibly
broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
trivially satisfied.

If any elements of `x` and `y` are equal, `message`, as well as the first
`summarize` entries of `x` and `y` are printed, and `InvalidArgumentError`
is raised.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`
</td>
<td>
 Numeric `Tensor`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
 Numeric `Tensor`, same dtype as and broadcastable to `x`.
</td>
</tr><tr>
<td>
`summarize`
</td>
<td>
Print this many entries of each tensor.
</td>
</tr><tr>
<td>
`message`
</td>
<td>
A string to prefix to the default message.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for this operation (optional).  Defaults to
"assert_none_equal".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Op that raises `InvalidArgumentError` if `x != y` is ever False. This can
be used with <a href="../../tf/control_dependencies.md"><code>tf.control_dependencies</code></a> inside of <a href="../../tf/function.md"><code>tf.function</code></a>s to block
followup computation until the check has executed.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
if the check can be performed immediately and
`x != y` is False for any pair of elements in `x` and `y`. The check can
be performed immediately during eager execution or if `x` and `y` are
statically known.
</td>
</tr>
</table>



 <section><devsite-expandable expanded>
 <h2 class="showalways">eager compatibility</h2>

returns None


 </devsite-expandable></section>

