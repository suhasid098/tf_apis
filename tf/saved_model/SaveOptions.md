description: Options for saving to SavedModel.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.SaveOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.saved_model.SaveOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/save_options.py">View source</a>



Options for saving to SavedModel.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.saved_model.SaveOptions`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.SaveOptions(
    namespace_whitelist=None,
    save_debug_info=False,
    function_aliases=None,
    experimental_io_device=None,
    experimental_variable_policy=None,
    experimental_custom_gradients=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function may be used in the `options` argument in functions that
save a SavedModel (<a href="../../tf/saved_model/save.md"><code>tf.saved_model.save</code></a>, <a href="../../tf/keras/models/save_model.md"><code>tf.keras.models.save_model</code></a>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`namespace_whitelist`
</td>
<td>
List of strings containing op namespaces to whitelist
when saving a model. Saving an object that uses namespaced ops must
explicitly add all namespaces to the whitelist. The namespaced ops must
be registered into the framework when loading the SavedModel. If no
whitelist is provided, all namespaced ops will be allowed.
</td>
</tr><tr>
<td>
`save_debug_info`
</td>
<td>
Boolean indicating whether debug information is saved. If
True, then a debug/saved_model_debug_info.pb file will be written with
the contents of a GraphDebugInfo binary protocol buffer containing stack
trace information for all ops and functions that are saved.
</td>
</tr><tr>
<td>
`function_aliases`
</td>
<td>
Python dict. Mapping from string to object returned by
@tf.function. A single tf.function can generate many ConcreteFunctions.
If a downstream tool wants to refer to all concrete functions generated
by a single tf.function you can use the `function_aliases` argument to
store a map from the alias name to all concrete function names.
E.g.

```
>>> class Adder(tf.Module):
...   @tf.function
...   def double(self, x):
...     return x + x
```

```
>>> model = Adder()
>>> model.double.get_concrete_function(
...   tf.TensorSpec(shape=[], dtype=tf.float32, name="float_input"))
>>> model.double.get_concrete_function(
...   tf.TensorSpec(shape=[], dtype=tf.string, name="string_input"))
```

```
>>> options = tf.saved_model.SaveOptions(
...   function_aliases={'double': model.double})
>>> tf.saved_model.save(model, '/tmp/adder', options=options)
```
</td>
</tr><tr>
<td>
`experimental_io_device`
</td>
<td>
string. Applies in a distributed setting.
Tensorflow device to use to access the filesystem. If `None` (default)
then for each variable the filesystem is accessed from the CPU:0 device
of the host where that variable is assigned. If specified, the
filesystem is instead accessed from that device for all variables.

This is for example useful if you want to save to a local directory,
such as "/tmp" when running in a distributed setting. In that case pass
a device for the host where the "/tmp" directory is accessible.
</td>
</tr><tr>
<td>
`experimental_variable_policy`
</td>
<td>
The policy to apply to variables when
saving. This is either a <a href="../../tf/saved_model/experimental/VariablePolicy.md"><code>saved_model.experimental.VariablePolicy</code></a> enum
instance or one of its value strings (case is not important). See that
enum documentation for details. A value of `None` corresponds to the
default policy.
</td>
</tr><tr>
<td>
`experimental_custom_gradients`
</td>
<td>
Boolean. When True, will save traced
gradient functions for the functions decorated by <a href="../../tf/custom_gradient.md"><code>tf.custom_gradient</code></a>.
Defaults to `True`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`experimental_custom_gradients`
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_io_device`
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_variable_policy`
</td>
<td>

</td>
</tr><tr>
<td>
`function_aliases`
</td>
<td>

</td>
</tr><tr>
<td>
`namespace_whitelist`
</td>
<td>

</td>
</tr><tr>
<td>
`save_debug_info`
</td>
<td>

</td>
</tr>
</table>



