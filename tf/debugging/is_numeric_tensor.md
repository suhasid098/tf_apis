description: Returns True if the elements of tensor are numbers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.is_numeric_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.is_numeric_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/check_ops.py">View source</a>



Returns `True` if the elements of `tensor` are numbers.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.debugging.is_numeric_tensor`, `tf.compat.v1.is_numeric_tensor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.debugging.is_numeric_tensor(
    tensor
)
</code></pre>



<!-- Placeholder for "Used in" -->

Specifically, returns `True` if the dtype of `tensor` is one of the following:

* <a href="../../tf.md#float16"><code>tf.float16</code></a>
* <a href="../../tf.md#float32"><code>tf.float32</code></a>
* <a href="../../tf.md#float64"><code>tf.float64</code></a>
* <a href="../../tf.md#int8"><code>tf.int8</code></a>
* <a href="../../tf.md#int16"><code>tf.int16</code></a>
* <a href="../../tf.md#int32"><code>tf.int32</code></a>
* <a href="../../tf.md#int64"><code>tf.int64</code></a>
* <a href="../../tf.md#uint8"><code>tf.uint8</code></a>
* <a href="../../tf.md#uint16"><code>tf.uint16</code></a>
* <a href="../../tf.md#uint32"><code>tf.uint32</code></a>
* <a href="../../tf.md#uint64"><code>tf.uint64</code></a>
* <a href="../../tf.md#qint8"><code>tf.qint8</code></a>
* <a href="../../tf.md#qint16"><code>tf.qint16</code></a>
* <a href="../../tf.md#qint32"><code>tf.qint32</code></a>
* <a href="../../tf.md#quint8"><code>tf.quint8</code></a>
* <a href="../../tf.md#quint16"><code>tf.quint16</code></a>
* <a href="../../tf.md#complex64"><code>tf.complex64</code></a>
* <a href="../../tf.md#complex128"><code>tf.complex128</code></a>
* <a href="../../tf.md#bfloat16"><code>tf.bfloat16</code></a>

Returns `False` if `tensor` is of a non-numeric type or if `tensor` is not
a <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> object.