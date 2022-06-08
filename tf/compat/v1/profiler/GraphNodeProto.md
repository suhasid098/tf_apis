description: A ProtocolMessage

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.profiler.GraphNodeProto" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="InputShapesEntry"/>
</div>

# tf.compat.v1.profiler.GraphNodeProto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/profiler/tfprof_output.proto">View source</a>



A ProtocolMessage

<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`accelerator_exec_micros`
</td>
<td>
`int64 accelerator_exec_micros`
</td>
</tr><tr>
<td>
`children`
</td>
<td>
`repeated GraphNodeProto children`
</td>
</tr><tr>
<td>
`cpu_exec_micros`
</td>
<td>
`int64 cpu_exec_micros`
</td>
</tr><tr>
<td>
`devices`
</td>
<td>
`repeated string devices`
</td>
</tr><tr>
<td>
`exec_micros`
</td>
<td>
`int64 exec_micros`
</td>
</tr><tr>
<td>
`float_ops`
</td>
<td>
`int64 float_ops`
</td>
</tr><tr>
<td>
`input_shapes`
</td>
<td>
`repeated InputShapesEntry input_shapes`
</td>
</tr><tr>
<td>
`name`
</td>
<td>
`string name`
</td>
</tr><tr>
<td>
`output_bytes`
</td>
<td>
`int64 output_bytes`
</td>
</tr><tr>
<td>
`parameters`
</td>
<td>
`int64 parameters`
</td>
</tr><tr>
<td>
`peak_bytes`
</td>
<td>
`int64 peak_bytes`
</td>
</tr><tr>
<td>
`requested_bytes`
</td>
<td>
`int64 requested_bytes`
</td>
</tr><tr>
<td>
`residual_bytes`
</td>
<td>
`int64 residual_bytes`
</td>
</tr><tr>
<td>
`run_count`
</td>
<td>
`int64 run_count`
</td>
</tr><tr>
<td>
`shapes`
</td>
<td>
`repeated TensorShapeProto shapes`
</td>
</tr><tr>
<td>
`tensor_value`
</td>
<td>
`TFProfTensorProto tensor_value`
</td>
</tr><tr>
<td>
`total_accelerator_exec_micros`
</td>
<td>
`int64 total_accelerator_exec_micros`
</td>
</tr><tr>
<td>
`total_cpu_exec_micros`
</td>
<td>
`int64 total_cpu_exec_micros`
</td>
</tr><tr>
<td>
`total_definition_count`
</td>
<td>
`int64 total_definition_count`
</td>
</tr><tr>
<td>
`total_exec_micros`
</td>
<td>
`int64 total_exec_micros`
</td>
</tr><tr>
<td>
`total_float_ops`
</td>
<td>
`int64 total_float_ops`
</td>
</tr><tr>
<td>
`total_output_bytes`
</td>
<td>
`int64 total_output_bytes`
</td>
</tr><tr>
<td>
`total_parameters`
</td>
<td>
`int64 total_parameters`
</td>
</tr><tr>
<td>
`total_peak_bytes`
</td>
<td>
`int64 total_peak_bytes`
</td>
</tr><tr>
<td>
`total_requested_bytes`
</td>
<td>
`int64 total_requested_bytes`
</td>
</tr><tr>
<td>
`total_residual_bytes`
</td>
<td>
`int64 total_residual_bytes`
</td>
</tr><tr>
<td>
`total_run_count`
</td>
<td>
`int64 total_run_count`
</td>
</tr>
</table>



## Child Classes
[`class InputShapesEntry`](../../../../tf/compat/v1/profiler/GraphNodeProto/InputShapesEntry.md)

