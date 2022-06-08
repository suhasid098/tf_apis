description: Registers a flag whose value must be a float.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_float" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_float

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a flag whose value must be a float.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_float(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args
)
</code></pre>



<!-- Placeholder for "Used in" -->

If lower_bound or upper_bound are set, then this flag must be
within the given range.

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
`default`
</td>
<td>
float|str|None, the default value of the flag.
</td>
</tr><tr>
<td>
`help`
</td>
<td>
str, the help message.
</td>
</tr><tr>
<td>
`lower_bound`
</td>
<td>
float, min value of the flag.
</td>
</tr><tr>
<td>
`upper_bound`
</td>
<td>
float, max value of the flag.
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
`required`
</td>
<td>
bool, is this a required flag. This must be used as a keyword
argument.
</td>
</tr><tr>
<td>
`**args`
</td>
<td>
dict, the extra keyword args that are passed to DEFINE.
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

