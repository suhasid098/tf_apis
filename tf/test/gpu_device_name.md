description: Returns the name of a GPU device if available or a empty string.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.test.gpu_device_name" />
<meta itemprop="path" content="Stable" />
</div>

# tf.test.gpu_device_name

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/test_util.py">View source</a>



Returns the name of a GPU device if available or a empty string.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.test.gpu_device_name`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.test.gpu_device_name()
</code></pre>



<!-- Placeholder for "Used in" -->

This method should only be used in tests written with <a href="../../tf/test/TestCase.md"><code>tf.test.TestCase</code></a>.

```
>>> class MyTest(tf.test.TestCase):
...
...   def test_add_on_gpu(self):
...     if not tf.test.is_built_with_gpu_support():
...       self.skipTest("test is only applicable on GPU")
...
...     with tf.device(tf.test.gpu_device_name()):
...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)
```