description: An in-process tf.data service worker server.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.WorkerServer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="join"/>
<meta itemprop="property" content="start"/>
</div>

# tf.data.experimental.service.WorkerServer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/service/server_lib.py">View source</a>



An in-process tf.data service worker server.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.service.WorkerServer(
    config, start=True
)
</code></pre>



<!-- Placeholder for "Used in" -->

A <a href="../../../../tf/data/experimental/service/WorkerServer.md"><code>tf.data.experimental.service.WorkerServer</code></a> performs <a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>
processing for user-defined datasets, and provides the resulting elements over
RPC. A worker is associated with a single
<a href="../../../../tf/data/experimental/service/DispatchServer.md"><code>tf.data.experimental.service.DispatchServer</code></a>.

```
>>> dispatcher = tf.data.experimental.service.DispatchServer()
>>> dispatcher_address = dispatcher.target.split("://")[1]
>>> worker = tf.data.experimental.service.WorkerServer(
...     tf.data.experimental.service.WorkerConfig(
...         dispatcher_address=dispatcher_address))
>>> dataset = tf.data.Dataset.range(10)
>>> dataset = dataset.apply(tf.data.experimental.service.distribute(
...     processing_mode="parallel_epochs", service=dispatcher.target))
>>> print(list(dataset.as_numpy_iterator()))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

When starting a dedicated tf.data worker process, use join() to block
indefinitely after starting up the server.

```
worker = tf.data.experimental.service.WorkerServer(
    port=5051, dispatcher_address="localhost:5050")
worker.join()
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`
</td>
<td>
A <a href="../../../../tf/data/experimental/service/WorkerConfig.md"><code>tf.data.experimental.service.WorkerConfig</code></a> configration.
</td>
</tr><tr>
<td>
`start`
</td>
<td>
(Optional.) Boolean, indicating whether to start the server after
creating it. Defaults to True.
</td>
</tr>
</table>



## Methods

<h3 id="join"><code>join</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/service/server_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>join()
</code></pre>

Blocks until the server has shut down.

This is useful when starting a dedicated worker process.

```
worker_server = tf.data.experimental.service.WorkerServer(
    port=5051, dispatcher_address="localhost:5050")
worker_server.join()
```

This method currently blocks forever.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses if an error occurs while
joining the server.
</td>
</tr>
</table>



<h3 id="start"><code>start</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/service/server_lib.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start()
</code></pre>

Starts this server.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`tf.errors.OpError`
</td>
<td>
Or one of its subclasses if an error occurs while
starting the server.
</td>
</tr>
</table>





