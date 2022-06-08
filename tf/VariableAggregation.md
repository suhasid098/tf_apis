description: Indicates how a distributed variable will be aggregated.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.VariableAggregation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MEAN"/>
<meta itemprop="property" content="NONE"/>
<meta itemprop="property" content="ONLY_FIRST_REPLICA"/>
<meta itemprop="property" content="SUM"/>
</div>

# tf.VariableAggregation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/variables.py">View source</a>



Indicates how a distributed variable will be aggregated.

<!-- Placeholder for "Used in" -->

<a href="../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a> distributes a model by making multiple copies
(called "replicas") acting data-parallel on different elements of the input
batch. When performing some variable-update operation, say
`var.assign_add(x)`, in a model, we need to resolve how to combine the
different values for `x` computed in the different replicas.

* `NONE`: This is the default, giving an error if you use a
  variable-update operation with multiple replicas.
* `SUM`: Add the updates across replicas.
* `MEAN`: Take the arithmetic mean ("average") of the updates across replicas.
* `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
  update, but we only want to perform the update once. Used, e.g., for the
  global step counter.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
MEAN<a id="MEAN"></a>
</td>
<td>
`<VariableAggregationV2.MEAN: 2>`
</td>
</tr><tr>
<td>
NONE<a id="NONE"></a>
</td>
<td>
`<VariableAggregationV2.NONE: 0>`
</td>
</tr><tr>
<td>
ONLY_FIRST_REPLICA<a id="ONLY_FIRST_REPLICA"></a>
</td>
<td>
`<VariableAggregationV2.ONLY_FIRST_REPLICA: 3>`
</td>
</tr><tr>
<td>
SUM<a id="SUM"></a>
</td>
<td>
`<VariableAggregationV2.SUM: 1>`
</td>
</tr>
</table>

