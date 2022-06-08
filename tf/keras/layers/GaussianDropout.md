description: Apply multiplicative 1-centered Gaussian noise.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.GaussianDropout" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.GaussianDropout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/regularization/gaussian_dropout.py#L28-L81">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Apply multiplicative 1-centered Gaussian noise.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.GaussianDropout`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.GaussianDropout(
    rate, seed=None, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

As it is a regularization layer, it is only active at training time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`rate`
</td>
<td>
Float, drop probability (as with `Dropout`).
The multiplicative noise will have
standard deviation `sqrt(rate / (1 - rate))`.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
Integer, optional random seed to enable deterministic behavior.
</td>
</tr>
</table>



#### Call arguments:


* <b>`inputs`</b>: Input tensor (of any rank).
* <b>`training`</b>: Python boolean indicating whether the layer should behave in
  training mode (adding dropout) or in inference mode (doing nothing).


#### Input shape:

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.



#### Output shape:

Same shape as input.


