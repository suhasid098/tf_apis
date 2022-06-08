description: Describes a tf.Tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.TensorSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="from_spec"/>
<meta itemprop="property" content="from_tensor"/>
<meta itemprop="property" content="is_compatible_with"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
<meta itemprop="property" content="most_specific_compatible_type"/>
</div>

# tf.TensorSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>



Describes a tf.Tensor.

Inherits From: [`TypeSpec`](../tf/TypeSpec.md), [`TraceType`](../tf/types/experimental/TraceType.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.TensorSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.TensorSpec(
    shape,
    dtype=<a href="../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Metadata for describing the <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> objects accepted or returned
by some TensorFlow APIs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`
</td>
<td>
Value convertible to <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a>. The shape of the tensor.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Value convertible to <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a>. The type of the tensor values.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Optional name for the Tensor.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
If shape is not convertible to a <a href="../tf/TensorShape.md"><code>tf.TensorShape</code></a>, or dtype is
not convertible to a <a href="../tf/dtypes/DType.md"><code>tf.DType</code></a>.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dtype`
</td>
<td>
Returns the `dtype` of elements in the tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
Returns the (optionally provided) name of the described tensor.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
Returns the `TensorShape` that represents the shape of the tensor.
</td>
</tr><tr>
<td>
`value_type`
</td>
<td>
The Python type for values that are compatible with this TypeSpec.
</td>
</tr>
</table>



## Methods

<h3 id="from_spec"><code>from_spec</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_spec(
    spec, name=None
)
</code></pre>

Returns a `TensorSpec` with the same shape and dtype as `spec`.

```
>>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="OriginalName")
>>> tf.TensorSpec.from_spec(spec, "NewName")
TensorSpec(shape=(8, 3), dtype=tf.int32, name='NewName')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec`
</td>
<td>
The `TypeSpec` used to create the new `TensorSpec`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
The name for the new `TensorSpec`.  Defaults to `spec.name`.
</td>
</tr>
</table>



<h3 id="from_tensor"><code>from_tensor</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_tensor(
    tensor, name=None
)
</code></pre>

Returns a `TensorSpec` that describes `tensor`.

```
>>> tf.TensorSpec.from_tensor(tf.constant([1, 2, 3]))
TensorSpec(shape=(3,), dtype=tf.int32, name=None)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensor`
</td>
<td>
The <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> that should be described.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the `TensorSpec`.  Defaults to `tensor.op.name`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `TensorSpec` that describes `tensor`.
</td>
</tr>

</table>



<h3 id="is_compatible_with"><code>is_compatible_with</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_compatible_with(
    spec_or_tensor
)
</code></pre>

Returns True if spec_or_tensor is compatible with this TensorSpec.

Two tensors are considered compatible if they have the same dtype
and their shapes are compatible (see <a href="../tf/TensorShape.md#is_compatible_with"><code>tf.TensorShape.is_compatible_with</code></a>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`spec_or_tensor`
</td>
<td>
A tf.TensorSpec or a tf.Tensor
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if spec_or_tensor is compatible with self.
</td>
</tr>

</table>



<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_subtype_of(
    other: <a href="../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>
) -> bool
</code></pre>

Returns True if `self` is a subtype of `other`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A TraceType object.
</td>
</tr>
</table>



<h3 id="most_specific_common_supertype"><code>most_specific_common_supertype</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_common_supertype(
    others: Sequence[<a href="../tf/types/experimental/TraceType.md"><code>tf.types.experimental.TraceType</code></a>]
) -> Optional['TypeSpec']
</code></pre>

Returns the most specific supertype TypeSpec  of `self` and `others`.

Implements the tf.types.experimental.func.TraceType interface.

If not overridden by a subclass, the default behavior is to assume the
TypeSpec is covariant upon attributes that implement TraceType and
invariant upon rest of the attributes as well as the structure and type
of the TypeSpec.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`others`
</td>
<td>
A sequence of TraceTypes.
</td>
</tr>
</table>



<h3 id="most_specific_compatible_type"><code>most_specific_compatible_type</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/type_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>most_specific_compatible_type(
    other: 'TypeSpec'
) -> 'TypeSpec'
</code></pre>

Returns the most specific TypeSpec compatible with `self` and `other`. (deprecated)

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use most_specific_common_supertype instead.

Deprecated. Please use `most_specific_common_supertype` instead.
Do not override this function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A `TypeSpec`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there is no TypeSpec that is compatible with both `self`
and `other`.
</td>
</tr>
</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/tensor_spec.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




