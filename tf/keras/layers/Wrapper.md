description: Abstract wrapper base class.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Wrapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.Wrapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/rnn/base_wrapper.py#L29-L71">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Abstract wrapper base class.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.Wrapper`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Wrapper(
    layer, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Wrappers take another layer and augment it in various ways.
Do not use this class as a layer, it is only an abstract base class.
Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer`
</td>
<td>
The layer to be wrapped.
</td>
</tr>
</table>



