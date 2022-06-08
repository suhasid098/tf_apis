description: Options for finer control over the profiler.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.profiler.experimental.ProfilerOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.profiler.experimental.ProfilerOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>



Options for finer control over the profiler.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.profiler.experimental.ProfilerOptions(
    host_tracer_level=2,
    python_tracer_level=0,
    device_tracer_level=1,
    delay_ms=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Use <a href="../../../tf/profiler/experimental/ProfilerOptions.md"><code>tf.profiler.experimental.ProfilerOptions</code></a> to control <a href="../../../tf/profiler.md"><code>tf.profiler</code></a>
behavior.

#### Fields:


* <b>`host_tracer_level`</b>: Adjust CPU tracing level. Values are: 1 - critical info
only, 2 - info, 3 - verbose. [default value is 2]
* <b>`python_tracer_level`</b>: Toggle tracing of Python function calls. Values are: 1
- enabled, 0 - disabled [default value is 0]
* <b>`device_tracer_level`</b>: Adjust device (TPU/GPU) tracing level. Values are: 1 -
enabled, 0 - disabled [default value is 1]
* <b>`delay_ms`</b>: Requests for all hosts to start profiling at a timestamp that is
  `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
  zero, each host will start profiling immediately upon receiving the
  request. Default value is None, allowing the profiler guess the best
  value.




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`host_tracer_level`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`python_tracer_level`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`device_tracer_level`
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr><tr>
<td>
`delay_ms`
</td>
<td>
A `namedtuple` alias for field number 3
</td>
</tr>
</table>



