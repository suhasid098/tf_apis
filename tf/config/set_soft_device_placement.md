description: Enable or disable soft device placement.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.set_soft_device_placement" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.set_soft_device_placement

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Enable or disable soft device placement.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.set_soft_device_placement`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.set_soft_device_placement(
    enabled
)
</code></pre>



<!-- Placeholder for "Used in" -->

If enabled, an op will be placed on CPU if any of the following are true
  1. there's no GPU implementation for the OP
  2. no GPU devices are known or registered
  3. need to co-locate with reftype input(s) which are from CPU

Note: by default soft device placement is enabled when running in eager mode
(for convenience) and disabled in graph mode (for performance).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enabled`
</td>
<td>
A boolean indicating whether to enable soft placement.
</td>
</tr>
</table>

