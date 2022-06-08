description: Defines an alias flag for an existing one.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_alias" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_alias

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Defines an alias flag for an existing one.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_alias(
    name, original_name, flag_values=_flagvalues.FLAGS, module_name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
str, the flag name.
</td>
</tr><tr>
<td>
`original_name`
</td>
<td>
str, the original flag name.
</td>
</tr><tr>
<td>
`flag_values`
</td>
<td>
FlagValues, the FlagValues instance with which the flag will be
registered. This should almost never need to be overridden.
</td>
</tr><tr>
<td>
`module_name`
</td>
<td>
A string, the name of the module that defines this flag.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a handle to defined flag.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`flags.FlagError`
</td>
<td>
  UnrecognizedFlagError: if the referenced flag doesn't exist.
DuplicateFlagError: if the alias name has been used by some existing flag.
</td>
</tr>
</table>

