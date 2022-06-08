description: Enables experimental MLIR-Based TensorFlow Compiler Optimizations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.enable_mlir_graph_optimization" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.enable_mlir_graph_optimization

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Enables experimental MLIR-Based TensorFlow Compiler Optimizations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.experimental.enable_mlir_graph_optimization`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.enable_mlir_graph_optimization()
</code></pre>



<!-- Placeholder for "Used in" -->

DO NOT USE, DEV AND TESTING ONLY AT THE MOMENT.

NOTE: MLIR-Based TensorFlow Compiler is under active development and has
missing features, please refrain from using. This API exists for development
and testing only.

TensorFlow Compiler Optimizations are responsible general graph level
optimizations that in the current stack mostly done by Grappler graph
optimizers.