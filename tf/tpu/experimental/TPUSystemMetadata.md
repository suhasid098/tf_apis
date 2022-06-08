description: Describes some metadata about the TPU system.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.TPUSystemMetadata" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.tpu.experimental.TPUSystemMetadata

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_system_metadata.py">View source</a>



Describes some metadata about the TPU system.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.TPUSystemMetadata`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.TPUSystemMetadata(
    num_cores, num_hosts, num_of_cores_per_host, topology, devices
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`num_cores`
</td>
<td>
interger. Total number of TPU cores in the TPU system.
</td>
</tr><tr>
<td>
`num_hosts`
</td>
<td>
interger. Total number of hosts (TPU workers) in the TPU system.
</td>
</tr><tr>
<td>
`num_of_cores_per_host`
</td>
<td>
interger. Number of TPU cores per host (TPU worker).
</td>
</tr><tr>
<td>
`topology`
</td>
<td>
an instance of <a href="../../../tf/tpu/experimental/Topology.md"><code>tf.tpu.experimental.Topology</code></a>, which describes the
physical topology of TPU system.
</td>
</tr><tr>
<td>
`devices`
</td>
<td>
a tuple of strings, which describes all the TPU devices in the
system.
</td>
</tr>
</table>



