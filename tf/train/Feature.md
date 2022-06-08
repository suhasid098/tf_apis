description: Used in <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Contains a list of values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.Feature" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.Feature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/core/example/feature.proto">View source</a>



Used in <a href="../../tf/train/Example.md"><code>tf.train.Example</code></a> protos. Contains a list of values.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.Feature`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

An `Example` proto is a representation of the following python type:

```
Dict[str,
     Union[List[bytes],
           List[int64],
           List[float]]]
```

This proto implements the `Union`.

The contained list can be one of three types:

  - <a href="../../tf/train/BytesList.md"><code>tf.train.BytesList</code></a>
  - <a href="../../tf/train/FloatList.md"><code>tf.train.FloatList</code></a>
  - <a href="../../tf/train/Int64List.md"><code>tf.train.Int64List</code></a>

```
>>> int_feature = tf.train.Feature(
...     int64_list=tf.train.Int64List(value=[1, 2, 3, 4]))
>>> float_feature = tf.train.Feature(
...     float_list=tf.train.FloatList(value=[1., 2., 3., 4.]))
>>> bytes_feature = tf.train.Feature(
...     bytes_list=tf.train.BytesList(value=[b"abc", b"1234"]))
>>>
>>> example = tf.train.Example(
...     features=tf.train.Features(feature={
...         'my_ints': int_feature,
...         'my_floats': float_feature,
...         'my_bytes': bytes_feature,
...     }))
```

Use <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example</code></a> to extract tensors from a serialized `Example` proto:

```
>>> tf.io.parse_example(
...     example.SerializeToString(),
...     features = {
...         'my_ints': tf.io.RaggedFeature(dtype=tf.int64),
...         'my_floats': tf.io.RaggedFeature(dtype=tf.float32),
...         'my_bytes': tf.io.RaggedFeature(dtype=tf.string)})
{'my_bytes': <tf.Tensor: shape=(2,), dtype=string,
                         numpy=array([b'abc', b'1234'], dtype=object)>,
 'my_floats': <tf.Tensor: shape=(4,), dtype=float32,
                          numpy=array([1., 2., 3., 4.], dtype=float32)>,
 'my_ints': <tf.Tensor: shape=(4,), dtype=int64,
                        numpy=array([1, 2, 3, 4])>}
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`bytes_list`
</td>
<td>
`BytesList bytes_list`
</td>
</tr><tr>
<td>
`float_list`
</td>
<td>
`FloatList float_list`
</td>
</tr><tr>
<td>
`int64_list`
</td>
<td>
`Int64List int64_list`
</td>
</tr>
</table>



