description: A preprocessing layer which crosses features using the "hashing trick".

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.experimental.preprocessing.HashedCrossing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# tf.keras.layers.experimental.preprocessing.HashedCrossing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/layers/preprocessing/hashed_crossing.py#L32-L198">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which crosses features using the "hashing trick".

Inherits From: [`Layer`](../../../../../tf/keras/layers/Layer.md), [`Module`](../../../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.layers.experimental.preprocessing.HashedCrossing`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.experimental.preprocessing.HashedCrossing(
    num_bins, output_mode=&#x27;int&#x27;, sparse=False, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer performs crosses of categorical features using the "hasing trick".
Conceptually, the transformation can be thought of as:
hash(concatenation of features) % `num_bins`.

This layer currently only performs crosses of scalar inputs and batches of
scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size,)` and
`()`.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_bins`
</td>
<td>
Number of hash bins.
</td>
</tr><tr>
<td>
`output_mode`
</td>
<td>
Specification for the output of the layer. Defaults to `"int"`.
Values can be `"int"`, or `"one_hot"` configuring the layer as follows:
  - `"int"`: Return the integer bin indices directly.
  - `"one_hot"`: Encodes each individual element in the input into an
    array the same size as `num_bins`, containing a 1 at the input's bin
    index.
</td>
</tr><tr>
<td>
`sparse`
</td>
<td>
Boolean. Only applicable to `"one_hot"` mode. If True, returns a
`SparseTensor` instead of a dense `Tensor`. Defaults to False.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to construct a layer.
</td>
</tr>
</table>



#### Examples:



**Crossing two scalar features.**

```
>>> layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
...     num_bins=5)
>>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
>>> feat2 = tf.constant([101, 101, 101, 102, 102])
>>> layer((feat1, feat2))
<tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 4, 1, 1, 3])>
```

**Crossing and one-hotting two scalar features.**

```
>>> layer = tf.keras.layers.experimental.preprocessing.HashedCrossing(
...     num_bins=5, output_mode='one_hot')
>>> feat1 = tf.constant(['A', 'B', 'A', 'B', 'A'])
>>> feat2 = tf.constant([101, 101, 101, 102, 102])
>>> layer((feat1, feat2))
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
  array([[0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 1.],
         [0., 1., 0., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0.]], dtype=float32)>
```

