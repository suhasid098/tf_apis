description: The op extracts fields from a serialized protocol buffers message into tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.decode_proto" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.decode_proto

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



The op extracts fields from a serialized protocol buffers message into tensors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.decode_proto`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.decode_proto(
    bytes,
    message_type,
    field_names,
    output_types,
    descriptor_source=&#x27;local://&#x27;,
    message_format=&#x27;binary&#x27;,
    sanitize=False,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: This API is designed for orthogonality rather than human-friendliness. It
can be used to parse input protos by hand, but it is intended for use in
generated code.

The `decode_proto` op extracts fields from a serialized protocol buffers
message into tensors.  The fields in `field_names` are decoded and converted
to the corresponding `output_types` if possible.

A `message_type` name must be provided to give context for the field names.
The actual message descriptor can be looked up either in the linked-in
descriptor pool or a filename provided by the caller using the
`descriptor_source` attribute.

Each output tensor is a dense tensor. This means that it is padded to hold
the largest number of repeated elements seen in the input minibatch. (The
shape is also padded by one to prevent zero-sized dimensions). The actual
repeat counts for each example in the minibatch can be found in the `sizes`
output. In many cases the output of `decode_proto` is fed immediately into
tf.squeeze if missing values are not a concern. When using tf.squeeze, always
pass the squeeze dimension explicitly to avoid surprises.

For the most part, the mapping between Proto field types and TensorFlow dtypes
is straightforward. However, there are a few special cases:

- A proto field that contains a submessage or group can only be converted
to `DT_STRING` (the serialized submessage). This is to reduce the complexity
of the API. The resulting string can be used as input to another instance of
the decode_proto op.

- TensorFlow lacks support for unsigned integers. The ops represent uint64
types as a `DT_INT64` with the same twos-complement bit pattern (the obvious
way). Unsigned int32 values can be represented exactly by specifying type
`DT_INT64`, or using twos-complement if the caller specifies `DT_INT32` in
the `output_types` attribute.

- `map` fields are not directly decoded. They are treated as `repeated` fields,
of the appropriate entry type. The proto-compiler defines entry types for each
map field. The type-name is the field name, converted to "CamelCase" with
"Entry" appended. The <a href="../../tf/train/Features/FeatureEntry.md"><code>tf.train.Features.FeatureEntry</code></a> message is an example of
one of these implicit `Entry` types.

- `enum` fields should be read as int32.

Both binary and text proto serializations are supported, and can be
chosen using the `format` attribute.

The `descriptor_source` attribute selects the source of protocol
descriptors to consult when looking up `message_type`. This may be:

- An empty string  or "local://", in which case protocol descriptors are
created for C++ (not Python) proto definitions linked to the binary.

- A file, in which case protocol descriptors are created from the file,
which is expected to contain a `FileDescriptorSet` serialized as a string.
NOTE: You can build a `descriptor_source` file using the `--descriptor_set_out`
and `--include_imports` options to the protocol compiler `protoc`.

- A "bytes://<bytes>", in which protocol descriptors are created from `<bytes>`,
which is expected to be a `FileDescriptorSet` serialized as a string.

#### Here is an example:



The, internal, `Summary.Value` proto contains a
`oneof {float simple_value; Image image; ...}`

```
>>> from google.protobuf import text_format
>>>
>>> # A Summary.Value contains: oneof {float simple_value; Image image}
>>> values = [
...    "simple_value: 2.2",
...    "simple_value: 1.2",
...    "image { height: 128 width: 512 }",
...    "image { height: 256 width: 256 }",]
>>> values = [
...    text_format.Parse(v, tf.compat.v1.Summary.Value()).SerializeToString()
...    for v in values]
```

The following can decode both fields from the serialized strings:

```
>>> sizes, [simple_value, image]  = tf.io.decode_proto(
...  values,
...  tf.compat.v1.Summary.Value.DESCRIPTOR.full_name,
...  field_names=['simple_value', 'image'],
...  output_types=[tf.float32, tf.string])
```

The `sizes` has the same shape as the input, with an additional axis across the
fields that were decoded. Here the first column of `sizes` is the size of the
decoded `simple_value` field:

```
>>> print(sizes)
tf.Tensor(
[[1 0]
 [1 0]
 [0 1]
 [0 1]], shape=(4, 2), dtype=int32)
```

The result tensors each have one more index than the input byte-strings.
The valid elements of each result tensor are indicated by
the appropriate column of `sizes`. The invalid elements are padded with a
default value:

```
>>> print(simple_value)
tf.Tensor(
[[2.2]
 [1.2]
 [0. ]
 [0. ]], shape=(4, 1), dtype=float32)
```

Nested protos are extracted as string tensors:

```
>>> print(image.dtype)
<dtype: 'string'>
>>> print(image.shape.as_list())
[4, 1]
```

To convert to a <a href="../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> representation use:

```
>>> tf.RaggedTensor.from_tensor(simple_value, lengths=sizes[:, 0]).to_list()
[[2.2], [1.2], [], []]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`bytes`
</td>
<td>
A `Tensor` of type `string`.
Tensor of serialized protos with shape `batch_shape`.
</td>
</tr><tr>
<td>
`message_type`
</td>
<td>
A `string`. Name of the proto message type to decode.
</td>
</tr><tr>
<td>
`field_names`
</td>
<td>
A list of `strings`.
List of strings containing proto field names. An extension field can be decoded
by using its full name, e.g. EXT_PACKAGE.EXT_FIELD_NAME.
</td>
</tr><tr>
<td>
`output_types`
</td>
<td>
A list of `tf.DTypes`.
List of TF types to use for the respective field in field_names.
</td>
</tr><tr>
<td>
`descriptor_source`
</td>
<td>
An optional `string`. Defaults to `"local://"`.
Either the special value `local://` or a path to a file containing
a serialized `FileDescriptorSet`.
</td>
</tr><tr>
<td>
`message_format`
</td>
<td>
An optional `string`. Defaults to `"binary"`.
Either `binary` or `text`.
</td>
</tr><tr>
<td>
`sanitize`
</td>
<td>
An optional `bool`. Defaults to `False`.
Whether to sanitize the result or not.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `Tensor` objects (sizes, values).
</td>
</tr>
<tr>
<td>
`sizes`
</td>
<td>
A `Tensor` of type `int32`.
</td>
</tr><tr>
<td>
`values`
</td>
<td>
A list of `Tensor` objects of type `output_types`.
</td>
</tr>
</table>

