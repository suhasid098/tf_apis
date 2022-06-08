description: Registers a flag whose value can be a list of any strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_multi_string" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_multi_string

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a flag whose value can be a list of any strings.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_multi_string(
    name, default, help, flag_values=_flagvalues.FLAGS, required=False, **args
)
</code></pre>



<!-- Placeholder for "Used in" -->

Use the flag on the command line multiple times to place multiple
string values into the list.  The 'default' may be a single string
(which will be converted into a single-element list) or a list of
strings.


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
Union[Iterable[Text], Text, None], the default value of the flag;
see `DEFINE_multi`.
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
Dictionary with extra keyword args that are passed to the Flag
__init__.
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

