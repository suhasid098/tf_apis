description: Registers a generic Flag object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a generic Flag object.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE(
    parser,
    name,
    default,
    help,
    flag_values=_flagvalues.FLAGS,
    serializer=None,
    module_name=None,
    required=False,
    **args
)
</code></pre>



<!-- Placeholder for "Used in" -->

NOTE: in the docstrings of all DEFINE* functions, "registers" is short
for "creates a new flag and registers it".

Auxiliary function: clients should use the specialized DEFINE_<type>
function instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`parser`
</td>
<td>
ArgumentParser, used to parse the flag arguments.
</td>
</tr><tr>
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
The default value of the flag.
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
`serializer`
</td>
<td>
ArgumentSerializer, the flag serializer instance.
</td>
</tr><tr>
<td>
`module_name`
</td>
<td>
str, the name of the Python module declaring this flag. If not
provided, it will be computed using the stack trace of this call.
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
dict, the extra keyword args that are passed to Flag __init__.
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

