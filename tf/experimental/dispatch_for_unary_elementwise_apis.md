description: Decorator to override default implementation for unary elementwise APIs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_unary_elementwise_apis" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_unary_elementwise_apis

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator to override default implementation for unary elementwise APIs.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_unary_elementwise_apis`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_unary_elementwise_apis(
    x_type
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "elementwise api handler") overrides
the default implementation for any unary elementwise API whenever the value
for the first argument (typically named `x`) matches the type annotation
`x_type`. The elementwise api handler is called with two arguments:

  `elementwise_api_handler(api_func, x)`

Where `api_func` is a function that takes a single parameter and performs the
elementwise operation (e.g., <a href="../../tf/math/abs.md"><code>tf.abs</code></a>), and `x` is the first argument to the
elementwise api.

The following example shows how this decorator can be used to update all
unary elementwise operations to handle a `MaskedTensor` type:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
>>> @dispatch_for_unary_elementwise_apis(MaskedTensor)
... def unary_elementwise_api_handler(api_func, x):
...   return MaskedTensor(api_func(x.values), x.mask)
>>> mt = MaskedTensor([1, -2, -3], [True, False, True])
>>> abs_mt = tf.abs(mt)
>>> print(f"values={abs_mt.values.numpy()}, mask={abs_mt.mask.numpy()}")
values=[1 2 3], mask=[ True False True]
```

For unary elementwise operations that take extra arguments beyond `x`, those
arguments are *not* passed to the elementwise api handler, but are
automatically added when `api_func` is called.  E.g., in the following
example, the `dtype` parameter is not passed to
`unary_elementwise_api_handler`, but is added by `api_func`.

```
>>> ones_mt = tf.ones_like(mt, dtype=tf.float32)
>>> print(f"values={ones_mt.values.numpy()}, mask={ones_mt.mask.numpy()}")
values=[1.0 1.0 1.0], mask=[ True False True]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x_type`
</td>
<td>
A type annotation indicating when the api handler should be called.
See `dispatch_for_api` for a list of supported annotation types.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator.
</td>
</tr>

</table>


#### Registered APIs

The unary elementwise APIs are:

* <a href="../../tf/bitwise/invert.md"><code>tf.bitwise.invert(x, name)</code></a>
* <a href="../../tf/cast.md"><code>tf.cast(x, dtype, name)</code></a>
* <a href="../../tf/clip_by_value.md"><code>tf.clip_by_value(t, clip_value_min, clip_value_max, name)</code></a>
* <a href="../../tf/compat/v1/math/log_softmax.md"><code>tf.compat.v1.nn.log_softmax(logits, axis, name, dim)</code></a>
* <a href="../../tf/compat/v1/ones_like.md"><code>tf.compat.v1.ones_like(tensor, dtype, name, optimize)</code></a>
* <a href="../../tf/compat/v1/strings/length.md"><code>tf.compat.v1.strings.length(input, name, unit)</code></a>
* <a href="../../tf/compat/v1/strings/substr.md"><code>tf.compat.v1.strings.substr(input, pos, len, name, unit)</code></a>
* <a href="../../tf/compat/v1/string_to_hash_bucket.md"><code>tf.compat.v1.strings.to_hash_bucket(string_tensor, num_buckets, name, input)</code></a>
* <a href="../../tf/compat/v1/substr.md"><code>tf.compat.v1.substr(input, pos, len, name, unit)</code></a>
* <a href="../../tf/compat/v1/to_bfloat16.md"><code>tf.compat.v1.to_bfloat16(x, name)</code></a>
* <a href="../../tf/compat/v1/to_complex128.md"><code>tf.compat.v1.to_complex128(x, name)</code></a>
* <a href="../../tf/compat/v1/to_complex64.md"><code>tf.compat.v1.to_complex64(x, name)</code></a>
* <a href="../../tf/compat/v1/to_double.md"><code>tf.compat.v1.to_double(x, name)</code></a>
* <a href="../../tf/compat/v1/to_float.md"><code>tf.compat.v1.to_float(x, name)</code></a>
* <a href="../../tf/compat/v1/to_int32.md"><code>tf.compat.v1.to_int32(x, name)</code></a>
* <a href="../../tf/compat/v1/to_int64.md"><code>tf.compat.v1.to_int64(x, name)</code></a>
* <a href="../../tf/compat/v1/zeros_like.md"><code>tf.compat.v1.zeros_like(tensor, dtype, name, optimize)</code></a>
* <a href="../../tf/debugging/check_numerics.md"><code>tf.debugging.check_numerics(tensor, message, name)</code></a>
* <a href="../../tf/dtypes/saturate_cast.md"><code>tf.dtypes.saturate_cast(value, dtype, name)</code></a>
* <a href="../../tf/image/adjust_brightness.md"><code>tf.image.adjust_brightness(image, delta)</code></a>
* <a href="../../tf/image/adjust_gamma.md"><code>tf.image.adjust_gamma(image, gamma, gain)</code></a>
* <a href="../../tf/image/convert_image_dtype.md"><code>tf.image.convert_image_dtype(image, dtype, saturate, name)</code></a>
* <a href="../../tf/image/random_brightness.md"><code>tf.image.random_brightness(image, max_delta, seed)</code></a>
* <a href="../../tf/image/stateless_random_brightness.md"><code>tf.image.stateless_random_brightness(image, max_delta, seed)</code></a>
* <a href="../../tf/io/decode_base64.md"><code>tf.io.decode_base64(input, name)</code></a>
* <a href="../../tf/io/decode_compressed.md"><code>tf.io.decode_compressed(bytes, compression_type, name)</code></a>
* <a href="../../tf/io/encode_base64.md"><code>tf.io.encode_base64(input, pad, name)</code></a>
* <a href="../../tf/math/abs.md"><code>tf.math.abs(x, name)</code></a>
* <a href="../../tf/math/acos.md"><code>tf.math.acos(x, name)</code></a>
* <a href="../../tf/math/acosh.md"><code>tf.math.acosh(x, name)</code></a>
* <a href="../../tf/math/angle.md"><code>tf.math.angle(input, name)</code></a>
* <a href="../../tf/math/asin.md"><code>tf.math.asin(x, name)</code></a>
* <a href="../../tf/math/asinh.md"><code>tf.math.asinh(x, name)</code></a>
* <a href="../../tf/math/atan.md"><code>tf.math.atan(x, name)</code></a>
* <a href="../../tf/math/atanh.md"><code>tf.math.atanh(x, name)</code></a>
* <a href="../../tf/math/bessel_i0.md"><code>tf.math.bessel_i0(x, name)</code></a>
* <a href="../../tf/math/bessel_i0e.md"><code>tf.math.bessel_i0e(x, name)</code></a>
* <a href="../../tf/math/bessel_i1.md"><code>tf.math.bessel_i1(x, name)</code></a>
* <a href="../../tf/math/bessel_i1e.md"><code>tf.math.bessel_i1e(x, name)</code></a>
* <a href="../../tf/math/ceil.md"><code>tf.math.ceil(x, name)</code></a>
* <a href="../../tf/math/conj.md"><code>tf.math.conj(x, name)</code></a>
* <a href="../../tf/math/cos.md"><code>tf.math.cos(x, name)</code></a>
* <a href="../../tf/math/cosh.md"><code>tf.math.cosh(x, name)</code></a>
* <a href="../../tf/math/digamma.md"><code>tf.math.digamma(x, name)</code></a>
* <a href="../../tf/math/erf.md"><code>tf.math.erf(x, name)</code></a>
* <a href="../../tf/math/erfc.md"><code>tf.math.erfc(x, name)</code></a>
* <a href="../../tf/math/erfcinv.md"><code>tf.math.erfcinv(x, name)</code></a>
* <a href="../../tf/math/erfinv.md"><code>tf.math.erfinv(x, name)</code></a>
* <a href="../../tf/math/exp.md"><code>tf.math.exp(x, name)</code></a>
* <a href="../../tf/math/expm1.md"><code>tf.math.expm1(x, name)</code></a>
* <a href="../../tf/math/floor.md"><code>tf.math.floor(x, name)</code></a>
* <a href="../../tf/math/imag.md"><code>tf.math.imag(input, name)</code></a>
* <a href="../../tf/math/is_finite.md"><code>tf.math.is_finite(x, name)</code></a>
* <a href="../../tf/math/is_inf.md"><code>tf.math.is_inf(x, name)</code></a>
* <a href="../../tf/math/is_nan.md"><code>tf.math.is_nan(x, name)</code></a>
* <a href="../../tf/math/lgamma.md"><code>tf.math.lgamma(x, name)</code></a>
* <a href="../../tf/math/log.md"><code>tf.math.log(x, name)</code></a>
* <a href="../../tf/math/log1p.md"><code>tf.math.log1p(x, name)</code></a>
* <a href="../../tf/math/log_sigmoid.md"><code>tf.math.log_sigmoid(x, name)</code></a>
* <a href="../../tf/math/logical_not.md"><code>tf.math.logical_not(x, name)</code></a>
* <a href="../../tf/math/ndtri.md"><code>tf.math.ndtri(x, name)</code></a>
* <a href="../../tf/math/negative.md"><code>tf.math.negative(x, name)</code></a>
* <a href="../../tf/math/nextafter.md"><code>tf.math.nextafter(x1, x2, name)</code></a>
* <a href="../../tf/math/real.md"><code>tf.math.real(input, name)</code></a>
* <a href="../../tf/math/reciprocal.md"><code>tf.math.reciprocal(x, name)</code></a>
* <a href="../../tf/math/reciprocal_no_nan.md"><code>tf.math.reciprocal_no_nan(x, name)</code></a>
* <a href="../../tf/math/rint.md"><code>tf.math.rint(x, name)</code></a>
* <a href="../../tf/math/round.md"><code>tf.math.round(x, name)</code></a>
* <a href="../../tf/math/rsqrt.md"><code>tf.math.rsqrt(x, name)</code></a>
* <a href="../../tf/math/sigmoid.md"><code>tf.math.sigmoid(x, name)</code></a>
* <a href="../../tf/math/sign.md"><code>tf.math.sign(x, name)</code></a>
* <a href="../../tf/math/sin.md"><code>tf.math.sin(x, name)</code></a>
* <a href="../../tf/math/sinh.md"><code>tf.math.sinh(x, name)</code></a>
* <a href="../../tf/math/softplus.md"><code>tf.math.softplus(features, name)</code></a>
* <a href="../../tf/math/special/bessel_j0.md"><code>tf.math.special.bessel_j0(x, name)</code></a>
* <a href="../../tf/math/special/bessel_j1.md"><code>tf.math.special.bessel_j1(x, name)</code></a>
* <a href="../../tf/math/special/bessel_k0.md"><code>tf.math.special.bessel_k0(x, name)</code></a>
* <a href="../../tf/math/special/bessel_k0e.md"><code>tf.math.special.bessel_k0e(x, name)</code></a>
* <a href="../../tf/math/special/bessel_k1.md"><code>tf.math.special.bessel_k1(x, name)</code></a>
* <a href="../../tf/math/special/bessel_k1e.md"><code>tf.math.special.bessel_k1e(x, name)</code></a>
* <a href="../../tf/math/special/bessel_y0.md"><code>tf.math.special.bessel_y0(x, name)</code></a>
* <a href="../../tf/math/special/bessel_y1.md"><code>tf.math.special.bessel_y1(x, name)</code></a>
* <a href="../../tf/math/special/dawsn.md"><code>tf.math.special.dawsn(x, name)</code></a>
* <a href="../../tf/math/special/expint.md"><code>tf.math.special.expint(x, name)</code></a>
* <a href="../../tf/math/special/fresnel_cos.md"><code>tf.math.special.fresnel_cos(x, name)</code></a>
* <a href="../../tf/math/special/fresnel_sin.md"><code>tf.math.special.fresnel_sin(x, name)</code></a>
* <a href="../../tf/math/special/spence.md"><code>tf.math.special.spence(x, name)</code></a>
* <a href="../../tf/math/sqrt.md"><code>tf.math.sqrt(x, name)</code></a>
* <a href="../../tf/math/square.md"><code>tf.math.square(x, name)</code></a>
* <a href="../../tf/math/tan.md"><code>tf.math.tan(x, name)</code></a>
* <a href="../../tf/math/tanh.md"><code>tf.math.tanh(x, name)</code></a>
* <a href="../../tf/nn/elu.md"><code>tf.nn.elu(features, name)</code></a>
* <a href="../../tf/nn/gelu.md"><code>tf.nn.gelu(features, approximate, name)</code></a>
* <a href="../../tf/nn/leaky_relu.md"><code>tf.nn.leaky_relu(features, alpha, name)</code></a>
* <a href="../../tf/nn/relu.md"><code>tf.nn.relu(features, name)</code></a>
* <a href="../../tf/nn/relu6.md"><code>tf.nn.relu6(features, name)</code></a>
* <a href="../../tf/nn/selu.md"><code>tf.nn.selu(features, name)</code></a>
* <a href="../../tf/nn/silu.md"><code>tf.nn.silu(features, beta)</code></a>
* <a href="../../tf/nn/softsign.md"><code>tf.nn.softsign(features, name)</code></a>
* <a href="../../tf/ones_like.md"><code>tf.ones_like(input, dtype, name)</code></a>
* <a href="../../tf/strings/as_string.md"><code>tf.strings.as_string(input, precision, scientific, shortest, width, fill, name)</code></a>
* <a href="../../tf/strings/length.md"><code>tf.strings.length(input, unit, name)</code></a>
* <a href="../../tf/strings/lower.md"><code>tf.strings.lower(input, encoding, name)</code></a>
* <a href="../../tf/strings/regex_full_match.md"><code>tf.strings.regex_full_match(input, pattern, name)</code></a>
* <a href="../../tf/strings/regex_replace.md"><code>tf.strings.regex_replace(input, pattern, rewrite, replace_global, name)</code></a>
* <a href="../../tf/strings/strip.md"><code>tf.strings.strip(input, name)</code></a>
* <a href="../../tf/strings/substr.md"><code>tf.strings.substr(input, pos, len, unit, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket.md"><code>tf.strings.to_hash_bucket(input, num_buckets, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket_fast.md"><code>tf.strings.to_hash_bucket_fast(input, num_buckets, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket_strong.md"><code>tf.strings.to_hash_bucket_strong(input, num_buckets, key, name)</code></a>
* <a href="../../tf/strings/to_number.md"><code>tf.strings.to_number(input, out_type, name)</code></a>
* <a href="../../tf/strings/unicode_script.md"><code>tf.strings.unicode_script(input, name)</code></a>
* <a href="../../tf/strings/unicode_transcode.md"><code>tf.strings.unicode_transcode(input, input_encoding, output_encoding, errors, replacement_char, replace_control_characters, name)</code></a>
* <a href="../../tf/strings/upper.md"><code>tf.strings.upper(input, encoding, name)</code></a>
* <a href="../../tf/zeros_like.md"><code>tf.zeros_like(input, dtype, name)</code></a>