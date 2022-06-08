description: Abstraction for a locally visible physical device.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.PhysicalDevice" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.config.PhysicalDevice

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Abstraction for a locally visible physical device.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.PhysicalDevice`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.PhysicalDevice(
    name, device_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

TensorFlow can utilize various devices such as the CPU or multiple GPUs
for computation. Before initializing a local device for use, the user can
customize certain properties of the device such as it's visibility or memory
configuration.

Once a visible <a href="../../tf/config/PhysicalDevice.md"><code>tf.config.PhysicalDevice</code></a> is initialized one or more
<a href="../../tf/config/LogicalDevice.md"><code>tf.config.LogicalDevice</code></a> objects are created. Use
<a href="../../tf/config/set_visible_devices.md"><code>tf.config.set_visible_devices</code></a> to configure the visibility of a physical
device and <a href="../../tf/config/set_logical_device_configuration.md"><code>tf.config.set_logical_device_configuration</code></a> to configure multiple
<a href="../../tf/config/LogicalDevice.md"><code>tf.config.LogicalDevice</code></a> objects for a <a href="../../tf/config/PhysicalDevice.md"><code>tf.config.PhysicalDevice</code></a>. This is
useful when separation between models is needed or to simulate a multi-device
environment.

#### Fields:


* <b>`name`</b>: Unique identifier for device.
* <b>`device_type`</b>: String declaring the type of device such as "CPU" or "GPU".




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`device_type`
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr>
</table>



