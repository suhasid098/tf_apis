description: Initializer that generates tensors with a normal distribution.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.initializers.RandomNormal" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.initializers.RandomNormal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/initializers/initializers_v2.py#L319-L386">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Initializer that generates tensors with a normal distribution.

Inherits From: [`Initializer`](../../../tf/keras/initializers/Initializer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.initializers.random_normal`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.05, seed=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Also available via the shortcut function
<a href="../../../tf/keras/initializers/RandomNormal.md"><code>tf.keras.initializers.random_normal</code></a>.

#### Examples:



```
>>> # Standalone usage:
>>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
>>> values = initializer(shape=(2, 2))
```

```
>>> # Usage in a Keras layer:
>>> initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
>>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mean`
</td>
<td>
a python scalar or a scalar tensor. Mean of the random values to
generate.
</td>
</tr><tr>
<td>
`stddev`
</td>
<td>
a python scalar or a scalar tensor. Standard deviation of the random
values to generate.
</td>
</tr><tr>
<td>
`seed`
</td>
<td>
A Python integer. Used to make the behavior of the initializer
deterministic. Note that a seeded
initializer will not produce the same random values across multiple calls,
but multiple initializers will produce the same sequence when constructed
with the same seed value.
</td>
</tr>
</table>



## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/initializers/initializers_v2.py#L93-L112">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates an initializer from a configuration dictionary.


#### Example:



```python
initializer = RandomUniform(-1, 1)
config = initializer.get_config()
initializer = RandomUniform.from_config(config)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, the output of `get_config`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/keras/initializers/Initializer.md"><code>tf.keras.initializers.Initializer</code></a> instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/initializers/initializers_v2.py#L381-L386">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the configuration of the initializer as a JSON-serializable dict.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable Python dict.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/initializers/initializers_v2.py#L356-L379">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    shape, dtype=None, **kwargs
)
</code></pre>

Returns a tensor object initialized to random normal values.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`shape`
</td>
<td>
Shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Optional dtype of the tensor. Only floating point types are
supported. If not specified, <a href="../../../tf/keras/backend/floatx.md"><code>tf.keras.backend.floatx()</code></a> is used, which
default to `float32` unless you configured it otherwise (via
<a href="../../../tf/keras/backend/set_floatx.md"><code>tf.keras.backend.set_floatx(float_dtype)</code></a>)
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments.
</td>
</tr>
</table>





