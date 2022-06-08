description: Context-manager profile API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.profiler.experimental.Profile" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf.profiler.experimental.Profile

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>



Context-manager profile API.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.profiler.experimental.Profile(
    logdir, options=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Profiling will start when entering the scope, and stop and save the results to
the logdir when exits the scope. Open TensorBoard profile tab to view results.

#### Example usage:


```python
with tf.profiler.experimental.Profile("/path/to/logdir"):
  # do some work
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`logdir`
</td>
<td>
profile data will save to this directory.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
An optional <a href="../../../tf/profiler/experimental/ProfilerOptions.md"><code>tf.profiler.experimental.ProfilerOptions</code></a> can be
provided to fine tune the profiler's behavior.
</td>
</tr>
</table>



## Methods

<h3 id="__enter__"><code>__enter__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__enter__()
</code></pre>




<h3 id="__exit__"><code>__exit__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/profiler/profiler_v2.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__exit__(
    typ, value, tb
)
</code></pre>






