description: Optimizer that implements the NAdam algorithm.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.legacy.Nadam" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.optimizers.legacy.Nadam

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/optimizers/legacy/nadam.py#L22-L24">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Optimizer that implements the NAdam algorithm.

Inherits From: [`Nadam`](../../../../tf/keras/optimizers/Nadam.md), [`Optimizer`](../../../../tf/keras/optimizers/Optimizer.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.optimizers.legacy.Nadam`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.legacy.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name=&#x27;Nadam&#x27;,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->
Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
Nesterov momentum.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`learning_rate`
</td>
<td>
A Tensor or a floating point value.  The learning rate.
</td>
</tr><tr>
<td>
`beta_1`
</td>
<td>
A float value or a constant float tensor. The exponential decay
rate for the 1st moment estimates.
</td>
</tr><tr>
<td>
`beta_2`
</td>
<td>
A float value or a constant float tensor. The exponential decay
rate for the exponentially weighted infinity norm.
</td>
</tr><tr>
<td>
`epsilon`
</td>
<td>
A small constant for numerical stability.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the operations created when applying gradients.
Defaults to `"Nadam"`.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
keyword arguments. Allowed arguments are `clipvalue`,
`clipnorm`, `global_clipnorm`.
If `clipvalue` (float) is set, the gradient of each weight
is clipped to be no higher than this value.
If `clipnorm` (float) is set, the gradient of each weight
is individually clipped so that its norm is no higher than this value.
If `global_clipnorm` (float) is set the gradient of all weights is
clipped so that their global norm is no higher than this value.
</td>
</tr>
</table>



#### Usage Example:

```
>>> opt = tf.keras.optimizers.Nadam(learning_rate=0.2)
>>> var1 = tf.Variable(10.0)
>>> loss = lambda: (var1 ** 2) / 2.0
>>> step_count = opt.minimize(loss, [var1]).numpy()
>>> "{:.1f}".format(var1.numpy())
9.8
```



#### Reference:

- [Dozat, 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
in case of any invalid argument.
</td>
</tr>
</table>



