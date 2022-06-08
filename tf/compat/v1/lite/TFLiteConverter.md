description: Convert a TensorFlow model into output_format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.lite.TFLiteConverter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert"/>
<meta itemprop="property" content="from_frozen_graph"/>
<meta itemprop="property" content="from_keras_model_file"/>
<meta itemprop="property" content="from_saved_model"/>
<meta itemprop="property" content="from_session"/>
<meta itemprop="property" content="get_input_arrays"/>
</div>

# tf.compat.v1.lite.TFLiteConverter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>



Convert a TensorFlow model into `output_format`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.lite.TFLiteConverter(
    graph_def,
    input_tensors,
    output_tensors,
    input_arrays_with_shape=None,
    output_arrays=None,
    experimental_debug_info_func=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
model into either a TFLite FlatBuffer or graph visualization.

#### Example usage:


```python
# Converting a GraphDef from session.
converter = tf.compat.v1.lite.TFLiteConverter.from_session(
  sess, in_tensors, out_tensors)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# Converting a GraphDef from file.
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# Converting a SavedModel.
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(
    saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

# Converting a tf.keras model.
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(
    keras_model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`graph_def`
</td>
<td>
Frozen TensorFlow GraphDef.
</td>
</tr><tr>
<td>
`input_tensors`
</td>
<td>
List of input tensors. Type and shape are computed using
`foo.shape` and `foo.dtype`.
</td>
</tr><tr>
<td>
`output_tensors`
</td>
<td>
List of output tensors (only .name is used from this).
</td>
</tr><tr>
<td>
`input_arrays_with_shape`
</td>
<td>
Tuple of strings representing input tensor names
and list of integers representing input shapes
(e.g., [("foo" : [1, 16, 16, 3])]). Use only when graph cannot be loaded
  into TensorFlow and when `input_tensors` and `output_tensors` are
  None. (default None)
</td>
</tr><tr>
<td>
`output_arrays`
</td>
<td>
List of output tensors to freeze graph with. Use only when
graph cannot be loaded into TensorFlow and when `input_tensors` and
`output_tensors` are None. (default None)
</td>
</tr><tr>
<td>
`experimental_debug_info_func`
</td>
<td>
An experimental function to retrieve the
graph debug info for a set of nodes from the `graph_def`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
Invalid arguments.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`optimizations`
</td>
<td>
Experimental flag, subject to change. Set of optimizations to
apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
set of values of type <a href="../../../../tf/lite/Optimize.md"><code>tf.lite.Optimize</code></a>)
</td>
</tr><tr>
<td>
`representative_dataset`
</td>
<td>
A generator function used for integer quantization
where each generated sample has the same order, type and shape as the
inputs to the model. Usually, this is a small subset of a few hundred
samples randomly chosen, in no particular order, from the training or
evaluation dataset. This is an optional attribute, but required for full
integer quantization, i.e, if <a href="../../../../tf.md#int8"><code>tf.int8</code></a> is the only supported type in
`target_spec.supported_types`. Refer to <a href="../../../../tf/lite/RepresentativeDataset.md"><code>tf.lite.RepresentativeDataset</code></a>.
(default None)
</td>
</tr><tr>
<td>
`target_spec`
</td>
<td>
Experimental flag, subject to change. Specifications of target
device, including supported ops set, supported types and a set of user's
defined TensorFlow operators required in the TensorFlow Lite runtime.
Refer to <a href="../../../../tf/lite/TargetSpec.md"><code>tf.lite.TargetSpec</code></a>.
</td>
</tr><tr>
<td>
`inference_type`
</td>
<td>
Data type of numeric arrays, excluding the input layer.
(default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
</td>
</tr><tr>
<td>
`inference_input_type`
</td>
<td>
Data type of the numeric arrays in the input layer. If
`inference_input_type` is in {tf.int8, tf.uint8}, then
`quantized_input_stats` must be provided. (default is the value assigned
to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
</td>
</tr><tr>
<td>
`inference_output_type`
</td>
<td>
Data type of the numeric arrays in the output layer.
(default is the value assigned to `inference_type`, must be in
{tf.float32, tf.int8, tf.uint8})
</td>
</tr><tr>
<td>
`quantized_input_stats`
</td>
<td>
Map of input tensor names to a tuple of floats
representing the mean and standard deviation of the training data.
(e.g., {"foo" : (0., 1.)}). Required if `inference_input_type` is tf.int8
  or tf.uint8. (default None)
</td>
</tr><tr>
<td>
`default_ranges_stats`
</td>
<td>
Tuple of integers (min, max) representing range values
for all numeric arrays without a specified range. Intended for
experimenting with quantization via "dummy quantization". (default None)
</td>
</tr><tr>
<td>
`allow_custom_ops`
</td>
<td>
Boolean indicating whether to allow custom operations.
When False any unknown operation is an error. When True, custom ops are
created for any op that is unknown. The developer will need to provide
these to the TensorFlow Lite runtime with a custom resolver. (default
False)
</td>
</tr><tr>
<td>
`drop_control_dependency`
</td>
<td>
Boolean indicating whether to drop control
dependencies silently. This is due to TFLite not supporting control
dependencies. (default True)
</td>
</tr><tr>
<td>
`reorder_across_fake_quant`
</td>
<td>
Boolean indicating whether to reorder FakeQuant
nodes in unexpected locations. Used when the location of the FakeQuant
nodes is preventing graph transformations necessary to convert the graph.
Results in a graph that differs from the quantized training graph,
potentially causing differing arithmetic behavior. (default False)
</td>
</tr><tr>
<td>
`change_concat_input_ranges`
</td>
<td>
Boolean to change behavior of min/max ranges for
inputs and outputs of the concat operator for quantized models. Changes
the ranges of concat operator overlap when true. (default False)
</td>
</tr><tr>
<td>
`output_format`
</td>
<td>
Output file format. (default
tf.compat.v1.lite.constants.TFLITE, must be in
{tf.compat.v1.lite.constants.TFLITE,
tf.compat.v1.lite.constants.GRAPHVIZ_DOT})
</td>
</tr><tr>
<td>
`dump_graphviz_dir`
</td>
<td>
Full filepath of folder to dump the graphs at various
stages of processing GraphViz .dot files. Preferred over
`output_format=tf.compat.v1.lite.constants.GRAPHVIZ_DOT` in order to keep
the requirements of the output file. (default None)
</td>
</tr><tr>
<td>
`dump_graphviz_video`
</td>
<td>
Boolean indicating whether to dump the GraphViz .dot
files after every graph transformation. Requires the `dump_graphviz_dir`
flag to be specified. (default False)
</td>
</tr><tr>
<td>
`conversion_summary_dir`
</td>
<td>
Full path of the directory to store conversion logs.
(default None)
</td>
</tr><tr>
<td>
`exclude_conversion_metadata`
</td>
<td>
Whether not to embed the conversion metadata
into the converted model. (default False)
</td>
</tr><tr>
<td>
`target_ops`
</td>
<td>
Deprecated. Please use `target_spec.supported_ops` instead.
</td>
</tr><tr>
<td>
`post_training_quantize`
</td>
<td>
Deprecated. Please use `optimizations` instead and
set it to `{tf.lite.Optimize.DEFAULT}`. (default False)
</td>
</tr><tr>
<td>
`experimental_new_converter`
</td>
<td>
Experimental flag, subject to change. Enables
MLIR-based conversion. (default True)
</td>
</tr><tr>
<td>
`experimental_new_quantizer`
</td>
<td>
Experimental flag, subject to change. Enables
MLIR-based quantization conversion instead of Flatbuffer-based conversion.
(default True)
</td>
</tr>
</table>



## Methods

<h3 id="convert"><code>convert</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert()
</code></pre>

Converts a TensorFlow GraphDef based on instance variables.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The converted data in serialized format. Either a TFLite Flatbuffer or a
Graphviz graph depending on value in `output_format`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
  Input shape is not specified.
None value for dimension in input_tensor.
</td>
</tr>
</table>



<h3 id="from_frozen_graph"><code>from_frozen_graph</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_frozen_graph(
    graph_def_file, input_arrays, output_arrays, input_shapes=None
)
</code></pre>

Creates a TFLiteConverter class from a file containing a frozen GraphDef.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`graph_def_file`
</td>
<td>
Full filepath of file containing frozen GraphDef.
</td>
</tr><tr>
<td>
`input_arrays`
</td>
<td>
List of input tensors to freeze graph with.
</td>
</tr><tr>
<td>
`output_arrays`
</td>
<td>
List of output tensors to freeze graph with.
</td>
</tr><tr>
<td>
`input_shapes`
</td>
<td>
Dict of strings representing input tensor names to list of
integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
Automatically determined when input shapes is None (e.g., {"foo" :
  None}). (default None)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TFLiteConverter class.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`IOError`
</td>
<td>
  File not found.
Unable to parse input file.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
  The graph is not frozen.
input_arrays or output_arrays contains an invalid tensor name.
input_shapes is not correctly defined when required
</td>
</tr>
</table>



<h3 id="from_keras_model_file"><code>from_keras_model_file</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_keras_model_file(
    model_file,
    input_arrays=None,
    input_shapes=None,
    output_arrays=None,
    custom_objects=None
)
</code></pre>

Creates a TFLiteConverter class from a tf.keras model file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_file`
</td>
<td>
Full filepath of HDF5 file containing the tf.keras model.
</td>
</tr><tr>
<td>
`input_arrays`
</td>
<td>
List of input tensors to freeze graph with. Uses input
arrays from SignatureDef when none are provided. (default None)
</td>
</tr><tr>
<td>
`input_shapes`
</td>
<td>
Dict of strings representing input tensor names to list of
integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
Automatically determined when input shapes is None (e.g., {"foo" :
  None}). (default None)
</td>
</tr><tr>
<td>
`output_arrays`
</td>
<td>
List of output tensors to freeze graph with. Uses output
arrays from SignatureDef when none are provided. (default None)
</td>
</tr><tr>
<td>
`custom_objects`
</td>
<td>
Dict mapping names (strings) to custom classes or
functions to be considered during model deserialization. (default None)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TFLiteConverter class.
</td>
</tr>

</table>



<h3 id="from_saved_model"><code>from_saved_model</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_saved_model(
    saved_model_dir,
    input_arrays=None,
    input_shapes=None,
    output_arrays=None,
    tag_set=None,
    signature_key=None
)
</code></pre>

Creates a TFLiteConverter class from a SavedModel.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`saved_model_dir`
</td>
<td>
SavedModel directory to convert.
</td>
</tr><tr>
<td>
`input_arrays`
</td>
<td>
List of input tensors to freeze graph with. Uses input
arrays from SignatureDef when none are provided. (default None)
</td>
</tr><tr>
<td>
`input_shapes`
</td>
<td>
Dict of strings representing input tensor names to list of
integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
Automatically determined when input shapes is None (e.g., {"foo" :
  None}). (default None)
</td>
</tr><tr>
<td>
`output_arrays`
</td>
<td>
List of output tensors to freeze graph with. Uses output
arrays from SignatureDef when none are provided. (default None)
</td>
</tr><tr>
<td>
`tag_set`
</td>
<td>
Set of tags identifying the MetaGraphDef within the SavedModel to
analyze. All tags in the tag set must be present. (default
{tf.saved_model.SERVING})
</td>
</tr><tr>
<td>
`signature_key`
</td>
<td>
Key identifying SignatureDef containing inputs and outputs.
(default tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TFLiteConverter class.
</td>
</tr>

</table>



<h3 id="from_session"><code>from_session</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_session(
    sess, input_tensors, output_tensors
)
</code></pre>

Creates a TFLiteConverter class from a TensorFlow Session.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`sess`
</td>
<td>
TensorFlow Session.
</td>
</tr><tr>
<td>
`input_tensors`
</td>
<td>
List of input tensors. Type and shape are computed using
`foo.shape` and `foo.dtype`.
</td>
</tr><tr>
<td>
`output_tensors`
</td>
<td>
List of output tensors (only .name is used from this).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
TFLiteConverter class.
</td>
</tr>

</table>



<h3 id="get_input_arrays"><code>get_input_arrays</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/lite/python/lite.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_input_arrays()
</code></pre>

Returns a list of the names of the input tensors.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of strings.
</td>
</tr>

</table>





