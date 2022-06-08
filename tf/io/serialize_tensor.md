description: Transforms a Tensor into a serialized TensorProto proto.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.serialize_tensor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.serialize_tensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/io_ops.py">View source</a>



Transforms a Tensor into a serialized TensorProto proto.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.serialize_tensor`, `tf.compat.v1.serialize_tensor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.serialize_tensor(
    tensor, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This operation transforms data in a <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> into a <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> of type
<a href="../../tf.md#string"><code>tf.string</code></a> containing the data in a binary string format. This operation can
transform scalar data and linear arrays, but it is most useful in converting
multidimensional arrays into a format accepted by binary storage formats such
as a `TFRecord` or <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a>.

#### See also:


- <a href="../../tf/io/parse_tensor.md"><code>tf.io.parse_tensor</code></a>: inverse operation of <a href="../../tf/io/serialize_tensor.md"><code>tf.io.serialize_tensor</code></a> that
transforms a scalar string containing a serialized Tensor into a Tensor of a
specified type.
- <a href="../../tf/ensure_shape.md"><code>tf.ensure_shape</code></a>: `parse_tensor` cannot statically determine the shape of
the parsed tensor. Use <a href="../../tf/ensure_shape.md"><code>tf.ensure_shape</code></a> to set the static shape when running
under a <a href="../../tf/function.md"><code>tf.function</code></a>
- `.SerializeToString`, serializes a proto to a binary-string

Example of serializing scalar data:

```
>>> t = tf.constant(1)
>>> tf.io.serialize_tensor(t)
<tf.Tensor: shape=(), dtype=string, numpy=b'\x08...\x00'>
```

Example of storing non-scalar data into a <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a>:

```
>>> t1 = [[1, 2]]
>>> t2 = [[7, 8]]
>>> nonscalar = tf.concat([t1, t2], 0)
>>> nonscalar
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
       [7, 8]], dtype=int32)>
```

Serialize the data using <a href="../../tf/io/serialize_tensor.md"><code>tf.io.serialize_tensor</code></a>.

```
>>> serialized_nonscalar = tf.io.serialize_tensor(nonscalar)
>>> serialized_nonscalar
<tf.Tensor: shape=(), dtype=string, numpy=b'\x08...\x00'>
```

Store the data in a <a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a>.

```
>>> feature_of_bytes = tf.train.Feature(
...   bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))
>>> feature_of_bytes
bytes_list {
  value: "\010...\000"
}
```

Put the <a href="../../tf/train/Feature.md"><code>tf.train.Feature</code></a> message into a <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a>.

```
>>> features_for_example = {
...   'feature0': feature_of_bytes
... }
>>> example_proto = tf.train.Example(
...   features=tf.train.Features(feature=features_for_example))
>>> example_proto
features {
  feature {
    key: "feature0"
    value {
      bytes_list {
        value: "\010...\000"
      }
    }
  }
}
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`
</td>
<td>
A <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a>.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
string.  Optional name for the op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor of dtype string.
</td>
</tr>

</table>

