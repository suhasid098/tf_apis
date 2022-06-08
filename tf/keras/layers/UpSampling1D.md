description: Upsampling layer for 1D inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.UpSampling1D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.UpSampling1D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/reshaping/up_sampling1d.py#L26-L80">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Upsampling layer for 1D inputs.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.UpSampling1D`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.UpSampling1D(
    size=2, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Repeats each temporal step `size` times along the time axis.

#### Examples:



```
>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> print(x)
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = tf.keras.layers.UpSampling1D(size=2)(x)
>>> print(y)
tf.Tensor(
  [[[ 0  1  2]
    [ 0  1  2]
    [ 3  4  5]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 6  7  8]
    [ 9 10 11]
    [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`size`
</td>
<td>
Integer. Upsampling factor.
</td>
</tr>
</table>



#### Input shape:

3D tensor with shape: `(batch_size, steps, features)`.



#### Output shape:

3D tensor with shape: `(batch_size, upsampled_steps, features)`.


