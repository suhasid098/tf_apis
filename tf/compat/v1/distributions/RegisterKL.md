description: Decorator to register a KL divergence implementation function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.distributions.RegisterKL" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tf.compat.v1.distributions.RegisterKL

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/distributions/kullback_leibler.py">View source</a>



Decorator to register a KL divergence implementation function.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.distributions.RegisterKL(
    dist_cls_a, dist_cls_b
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Usage:



@distributions.RegisterKL(distributions.Normal, distributions.Normal)
def _kl_normal_mvn(norm_a, norm_b):
  # Return KL(norm_a || norm_b)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dist_cls_a`
</td>
<td>
the class of the first argument of the KL divergence.
</td>
</tr><tr>
<td>
`dist_cls_b`
</td>
<td>
the class of the second argument of the KL divergence.
</td>
</tr>
</table>



## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/distributions/kullback_leibler.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    kl_fn
)
</code></pre>

Perform the KL registration.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`kl_fn`
</td>
<td>
The function to use for the KL divergence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
kl_fn
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
if kl_fn is not a callable.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
if a KL divergence function has already been registered for
the given argument classes.
</td>
</tr>
</table>





