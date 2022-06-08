description: Outputs a Summary protocol buffer containing a single scalar value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.summary.scalar" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.summary.scalar

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/summary/summary.py">View source</a>



Outputs a `Summary` protocol buffer containing a single scalar value.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.summary.scalar(
    name, tensor, collections=None, family=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

For compatibility purposes, when invoked in TF2 where the outermost context is
eager mode, this API will check if there is a suitable TF2 summary writer
context available, and if so will forward this call to that writer instead. A
"suitable" writer context means that the writer is set as the default writer,
and there is an associated non-empty value for `step` (see
<a href="../../../../tf/summary/SummaryWriter.md#as_default"><code>tf.summary.SummaryWriter.as_default</code></a>, `tf.summary.experimental.set_step` or
alternatively <a href="../../../../tf/compat/v1/train/create_global_step.md"><code>tf.compat.v1.train.create_global_step</code></a>). For the forwarded
call, the arguments here will be passed to the TF2 implementation of
<a href="../../../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a>, and the return value will be an empty bytestring tensor,
to avoid duplicate summary writing. This forwarding is best-effort and not all
arguments will be preserved.

To migrate to TF2, please use <a href="../../../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a> instead. Please check
[Migrating tf.summary usage to
TF 2.0](https://www.tensorflow.org/tensorboard/migrate#in_tf_1x) for concrete
steps for migration. <a href="../../../../tf/summary/scalar.md"><code>tf.summary.scalar</code></a> can also log training metrics in
Keras, you can check [Logging training metrics in
Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras) for detials.

#### How to Map Arguments

| TF1 Arg Name  | TF2 Arg Name    | Note                                   |
| :------------ | :-------------- | :------------------------------------- |
| `name`        | `name`          | -                                      |
| `tensor`      | `data`          | -                                      |
| -             | `step`          | Explicit int64-castable monotonic step |
:               :                 : value. If omitted, this defaults to    :
:               :                 : `tf.summary.experimental.get_step()`.  :
| `collections` | Not Supported   | -                                      |
| `family`      | Removed         | Please use <a href="../../../../tf/name_scope.md"><code>tf.name_scope</code></a> instead to  |
:               :                 : manage summary name prefix.            :
| -             | `description`   | Optional long-form `str` description   |
:               :                 : for the summary. Markdown is supported.:
:               :                 : Defaults to empty.                     :



 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

The generated Summary has a Tensor.proto containing the input Tensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A name for the generated node. Will also serve as the series name in
TensorBoard.
</td>
</tr><tr>
<td>
`tensor`
</td>
<td>
A real numeric Tensor containing a single value.
</td>
</tr><tr>
<td>
`collections`
</td>
<td>
Optional list of graph collections keys. The new summary op is
added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
</td>
</tr><tr>
<td>
`family`
</td>
<td>
Optional; if provided, used as the prefix of the summary tag name,
which controls the tab name used for display on Tensorboard.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
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
If tensor has the wrong shape or type.
</td>
</tr>
</table>


