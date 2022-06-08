description: Ragged Tensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ragged" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.ragged

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Ragged Tensors.


This package defines ops for manipulating ragged tensors (<a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>),
which are tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> class documentation
and the [Ragged Tensor Guide](/guide/ragged_tensor).


### Additional ops that support `RaggedTensor`

Arguments that accept `RaggedTensor`s are marked in **bold**.

* `tf.__operators__.eq`(**self**, **other**)
* `tf.__operators__.ne`(**self**, **other**)
* <a href="../../../tf/bitcast.md"><code>tf.bitcast</code></a>(**input**, type, name=`None`)
* <a href="../../../tf/bitwise/bitwise_and.md"><code>tf.bitwise.bitwise_and</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/bitwise/bitwise_or.md"><code>tf.bitwise.bitwise_or</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/bitwise/bitwise_xor.md"><code>tf.bitwise.bitwise_xor</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/bitwise/invert.md"><code>tf.bitwise.invert</code></a>(**x**, name=`None`)
* <a href="../../../tf/bitwise/left_shift.md"><code>tf.bitwise.left_shift</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/bitwise/right_shift.md"><code>tf.bitwise.right_shift</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/broadcast_to.md"><code>tf.broadcast_to</code></a>(**input**, **shape**, name=`None`)
* <a href="../../../tf/cast.md"><code>tf.cast</code></a>(**x**, dtype, name=`None`)
* <a href="../../../tf/clip_by_value.md"><code>tf.clip_by_value</code></a>(**t**, clip_value_min, clip_value_max, name=`None`)
* <a href="../../../tf/concat.md"><code>tf.concat</code></a>(**values**, axis, name=`'concat'`)
* <a href="../../../tf/debugging/check_numerics.md"><code>tf.debugging.check_numerics</code></a>(**tensor**, message, name=`None`)
* <a href="../../../tf/dtypes/complex.md"><code>tf.dtypes.complex</code></a>(**real**, **imag**, name=`None`)
* <a href="../../../tf/dtypes/saturate_cast.md"><code>tf.dtypes.saturate_cast</code></a>(**value**, dtype, name=`None`)
* <a href="../../../tf/dynamic_partition.md"><code>tf.dynamic_partition</code></a>(**data**, **partitions**, num_partitions, name=`None`)
* <a href="../../../tf/expand_dims.md"><code>tf.expand_dims</code></a>(**input**, axis, name=`None`)
* <a href="../../../tf/gather_nd.md"><code>tf.gather_nd</code></a>(**params**, **indices**, batch_dims=`0`, name=`None`)
* <a href="../../../tf/gather.md"><code>tf.gather</code></a>(**params**, **indices**, validate_indices=`None`, axis=`None`, batch_dims=`0`, name=`None`)
* <a href="../../../tf/image/adjust_brightness.md"><code>tf.image.adjust_brightness</code></a>(**image**, delta)
* <a href="../../../tf/image/adjust_gamma.md"><code>tf.image.adjust_gamma</code></a>(**image**, gamma=`1`, gain=`1`)
* <a href="../../../tf/image/convert_image_dtype.md"><code>tf.image.convert_image_dtype</code></a>(**image**, dtype, saturate=`False`, name=`None`)
* <a href="../../../tf/image/random_brightness.md"><code>tf.image.random_brightness</code></a>(**image**, max_delta, seed=`None`)
* <a href="../../../tf/image/resize.md"><code>tf.image.resize</code></a>(**images**, size, method=`'bilinear'`, preserve_aspect_ratio=`False`, antialias=`False`, name=`None`)
* <a href="../../../tf/image/stateless_random_brightness.md"><code>tf.image.stateless_random_brightness</code></a>(**image**, max_delta, seed)
* <a href="../../../tf/io/decode_base64.md"><code>tf.io.decode_base64</code></a>(**input**, name=`None`)
* <a href="../../../tf/io/decode_compressed.md"><code>tf.io.decode_compressed</code></a>(**bytes**, compression_type=`''`, name=`None`)
* <a href="../../../tf/io/encode_base64.md"><code>tf.io.encode_base64</code></a>(**input**, pad=`False`, name=`None`)
* <a href="../../../tf/linalg/matmul.md"><code>tf.linalg.matmul</code></a>(**a**, **b**, transpose_a=`False`, transpose_b=`False`, adjoint_a=`False`, adjoint_b=`False`, a_is_sparse=`False`, b_is_sparse=`False`, output_type=`None`, name=`None`)
* <a href="../../../tf/math/abs.md"><code>tf.math.abs</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/acos.md"><code>tf.math.acos</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/acosh.md"><code>tf.math.acosh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/add_n.md"><code>tf.math.add_n</code></a>(**inputs**, name=`None`)
* <a href="../../../tf/math/add.md"><code>tf.math.add</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/angle.md"><code>tf.math.angle</code></a>(**input**, name=`None`)
* <a href="../../../tf/math/asin.md"><code>tf.math.asin</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/asinh.md"><code>tf.math.asinh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/atan2.md"><code>tf.math.atan2</code></a>(**y**, **x**, name=`None`)
* <a href="../../../tf/math/atan.md"><code>tf.math.atan</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/atanh.md"><code>tf.math.atanh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/bessel_i0.md"><code>tf.math.bessel_i0</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/bessel_i0e.md"><code>tf.math.bessel_i0e</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/bessel_i1.md"><code>tf.math.bessel_i1</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/bessel_i1e.md"><code>tf.math.bessel_i1e</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/ceil.md"><code>tf.math.ceil</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/conj.md"><code>tf.math.conj</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/cos.md"><code>tf.math.cos</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/cosh.md"><code>tf.math.cosh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/digamma.md"><code>tf.math.digamma</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/divide_no_nan.md"><code>tf.math.divide_no_nan</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/divide.md"><code>tf.math.divide</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/equal.md"><code>tf.math.equal</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/erf.md"><code>tf.math.erf</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/erfc.md"><code>tf.math.erfc</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/erfcinv.md"><code>tf.math.erfcinv</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/erfinv.md"><code>tf.math.erfinv</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/exp.md"><code>tf.math.exp</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/expm1.md"><code>tf.math.expm1</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/floor.md"><code>tf.math.floor</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/floordiv.md"><code>tf.math.floordiv</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/floormod.md"><code>tf.math.floormod</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/greater_equal.md"><code>tf.math.greater_equal</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/greater.md"><code>tf.math.greater</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/imag.md"><code>tf.math.imag</code></a>(**input**, name=`None`)
* <a href="../../../tf/math/is_finite.md"><code>tf.math.is_finite</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/is_inf.md"><code>tf.math.is_inf</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/is_nan.md"><code>tf.math.is_nan</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/less_equal.md"><code>tf.math.less_equal</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/less.md"><code>tf.math.less</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/lgamma.md"><code>tf.math.lgamma</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/log1p.md"><code>tf.math.log1p</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/log_sigmoid.md"><code>tf.math.log_sigmoid</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/log.md"><code>tf.math.log</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/logical_and.md"><code>tf.math.logical_and</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/logical_not.md"><code>tf.math.logical_not</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/logical_or.md"><code>tf.math.logical_or</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/logical_xor.md"><code>tf.math.logical_xor</code></a>(**x**, **y**, name=`'LogicalXor'`)
* <a href="../../../tf/math/maximum.md"><code>tf.math.maximum</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/minimum.md"><code>tf.math.minimum</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/multiply_no_nan.md"><code>tf.math.multiply_no_nan</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/multiply.md"><code>tf.math.multiply</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/ndtri.md"><code>tf.math.ndtri</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/negative.md"><code>tf.math.negative</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/nextafter.md"><code>tf.math.nextafter</code></a>(**x1**, x2, name=`None`)
* <a href="../../../tf/math/not_equal.md"><code>tf.math.not_equal</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/pow.md"><code>tf.math.pow</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/real.md"><code>tf.math.real</code></a>(**input**, name=`None`)
* <a href="../../../tf/math/reciprocal_no_nan.md"><code>tf.math.reciprocal_no_nan</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/reciprocal.md"><code>tf.math.reciprocal</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/reduce_all.md"><code>tf.math.reduce_all</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_any.md"><code>tf.math.reduce_any</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_max.md"><code>tf.math.reduce_max</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_mean.md"><code>tf.math.reduce_mean</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_min.md"><code>tf.math.reduce_min</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_prod.md"><code>tf.math.reduce_prod</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_std.md"><code>tf.math.reduce_std</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_sum.md"><code>tf.math.reduce_sum</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/reduce_variance.md"><code>tf.math.reduce_variance</code></a>(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* <a href="../../../tf/math/rint.md"><code>tf.math.rint</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/round.md"><code>tf.math.round</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/rsqrt.md"><code>tf.math.rsqrt</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/scalar_mul.md"><code>tf.math.scalar_mul</code></a>(**scalar**, **x**, name=`None`)
* <a href="../../../tf/math/sigmoid.md"><code>tf.math.sigmoid</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/sign.md"><code>tf.math.sign</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/sin.md"><code>tf.math.sin</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/sinh.md"><code>tf.math.sinh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/softplus.md"><code>tf.math.softplus</code></a>(**features**, name=`None`)
* <a href="../../../tf/math/special/bessel_j0.md"><code>tf.math.special.bessel_j0</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_j1.md"><code>tf.math.special.bessel_j1</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_k0.md"><code>tf.math.special.bessel_k0</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_k0e.md"><code>tf.math.special.bessel_k0e</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_k1.md"><code>tf.math.special.bessel_k1</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_k1e.md"><code>tf.math.special.bessel_k1e</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_y0.md"><code>tf.math.special.bessel_y0</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/bessel_y1.md"><code>tf.math.special.bessel_y1</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/dawsn.md"><code>tf.math.special.dawsn</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/expint.md"><code>tf.math.special.expint</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/fresnel_cos.md"><code>tf.math.special.fresnel_cos</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/fresnel_sin.md"><code>tf.math.special.fresnel_sin</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/special/spence.md"><code>tf.math.special.spence</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/sqrt.md"><code>tf.math.sqrt</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/square.md"><code>tf.math.square</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/squared_difference.md"><code>tf.math.squared_difference</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/subtract.md"><code>tf.math.subtract</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/tan.md"><code>tf.math.tan</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/tanh.md"><code>tf.math.tanh</code></a>(**x**, name=`None`)
* <a href="../../../tf/math/truediv.md"><code>tf.math.truediv</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/unsorted_segment_max.md"><code>tf.math.unsorted_segment_max</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/unsorted_segment_mean.md"><code>tf.math.unsorted_segment_mean</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/unsorted_segment_min.md"><code>tf.math.unsorted_segment_min</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/unsorted_segment_prod.md"><code>tf.math.unsorted_segment_prod</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/unsorted_segment_sqrt_n.md"><code>tf.math.unsorted_segment_sqrt_n</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/unsorted_segment_sum.md"><code>tf.math.unsorted_segment_sum</code></a>(**data**, **segment_ids**, num_segments, name=`None`)
* <a href="../../../tf/math/xdivy.md"><code>tf.math.xdivy</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/xlog1py.md"><code>tf.math.xlog1py</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/xlogy.md"><code>tf.math.xlogy</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/math/zeta.md"><code>tf.math.zeta</code></a>(**x**, **q**, name=`None`)
* <a href="../../../tf/nn/dropout.md"><code>tf.nn.dropout</code></a>(**x**, rate, noise_shape=`None`, seed=`None`, name=`None`)
* <a href="../../../tf/nn/elu.md"><code>tf.nn.elu</code></a>(**features**, name=`None`)
* <a href="../../../tf/nn/gelu.md"><code>tf.nn.gelu</code></a>(**features**, approximate=`False`, name=`None`)
* <a href="../../../tf/nn/leaky_relu.md"><code>tf.nn.leaky_relu</code></a>(**features**, alpha=`0.2`, name=`None`)
* <a href="../../../tf/nn/relu6.md"><code>tf.nn.relu6</code></a>(**features**, name=`None`)
* <a href="../../../tf/nn/relu.md"><code>tf.nn.relu</code></a>(**features**, name=`None`)
* <a href="../../../tf/nn/selu.md"><code>tf.nn.selu</code></a>(**features**, name=`None`)
* <a href="../../../tf/nn/sigmoid_cross_entropy_with_logits.md"><code>tf.nn.sigmoid_cross_entropy_with_logits</code></a>(**labels**=`None`, **logits**=`None`, name=`None`)
* <a href="../../../tf/nn/silu.md"><code>tf.nn.silu</code></a>(**features**, beta=`1.0`)
* <a href="../../../tf/nn/softmax.md"><code>tf.nn.softmax</code></a>(**logits**, axis=`None`, name=`None`)
* <a href="../../../tf/nn/softsign.md"><code>tf.nn.softsign</code></a>(**features**, name=`None`)
* <a href="../../../tf/one_hot.md"><code>tf.one_hot</code></a>(**indices**, depth, on_value=`None`, off_value=`None`, axis=`None`, dtype=`None`, name=`None`)
* <a href="../../../tf/ones_like.md"><code>tf.ones_like</code></a>(**input**, dtype=`None`, name=`None`)
* <a href="../../../tf/print.md"><code>tf.print</code></a>(***inputs**, **kwargs)
* <a href="../../../tf/rank.md"><code>tf.rank</code></a>(**input**, name=`None`)
* <a href="../../../tf/realdiv.md"><code>tf.realdiv</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/reshape.md"><code>tf.reshape</code></a>(**tensor**, **shape**, name=`None`)
* <a href="../../../tf/reverse.md"><code>tf.reverse</code></a>(**tensor**, axis, name=`None`)
* <a href="../../../tf/size.md"><code>tf.size</code></a>(**input**, out_type=<a href="../../../tf.md#int32"><code>tf.int32</code></a>, name=`None`)
* <a href="../../../tf/split.md"><code>tf.split</code></a>(**value**, num_or_size_splits, axis=`0`, num=`None`, name=`'split'`)
* <a href="../../../tf/squeeze.md"><code>tf.squeeze</code></a>(**input**, axis=`None`, name=`None`)
* <a href="../../../tf/stack.md"><code>tf.stack</code></a>(**values**, axis=`0`, name=`'stack'`)
* <a href="../../../tf/strings/as_string.md"><code>tf.strings.as_string</code></a>(**input**, precision=`-1`, scientific=`False`, shortest=`False`, width=`-1`, fill=`''`, name=`None`)
* <a href="../../../tf/strings/format.md"><code>tf.strings.format</code></a>(**template**, **inputs**, placeholder=`'{}'`, summarize=`3`, name=`None`)
* <a href="../../../tf/strings/join.md"><code>tf.strings.join</code></a>(**inputs**, separator=`''`, name=`None`)
* <a href="../../../tf/strings/length.md"><code>tf.strings.length</code></a>(**input**, unit=`'BYTE'`, name=`None`)
* <a href="../../../tf/strings/lower.md"><code>tf.strings.lower</code></a>(**input**, encoding=`''`, name=`None`)
* <a href="../../../tf/strings/reduce_join.md"><code>tf.strings.reduce_join</code></a>(**inputs**, axis=`None`, keepdims=`False`, separator=`''`, name=`None`)
* <a href="../../../tf/strings/regex_full_match.md"><code>tf.strings.regex_full_match</code></a>(**input**, pattern, name=`None`)
* <a href="../../../tf/strings/regex_replace.md"><code>tf.strings.regex_replace</code></a>(**input**, pattern, rewrite, replace_global=`True`, name=`None`)
* <a href="../../../tf/strings/strip.md"><code>tf.strings.strip</code></a>(**input**, name=`None`)
* <a href="../../../tf/strings/substr.md"><code>tf.strings.substr</code></a>(**input**, pos, len, unit=`'BYTE'`, name=`None`)
* <a href="../../../tf/strings/to_hash_bucket_fast.md"><code>tf.strings.to_hash_bucket_fast</code></a>(**input**, num_buckets, name=`None`)
* <a href="../../../tf/strings/to_hash_bucket_strong.md"><code>tf.strings.to_hash_bucket_strong</code></a>(**input**, num_buckets, key, name=`None`)
* <a href="../../../tf/strings/to_hash_bucket.md"><code>tf.strings.to_hash_bucket</code></a>(**input**, num_buckets, name=`None`)
* <a href="../../../tf/strings/to_number.md"><code>tf.strings.to_number</code></a>(**input**, out_type=<a href="../../../tf.md#float32"><code>tf.float32</code></a>, name=`None`)
* <a href="../../../tf/strings/unicode_script.md"><code>tf.strings.unicode_script</code></a>(**input**, name=`None`)
* <a href="../../../tf/strings/unicode_transcode.md"><code>tf.strings.unicode_transcode</code></a>(**input**, input_encoding, output_encoding, errors=`'replace'`, replacement_char=`65533`, replace_control_characters=`False`, name=`None`)
* <a href="../../../tf/strings/upper.md"><code>tf.strings.upper</code></a>(**input**, encoding=`''`, name=`None`)
* <a href="../../../tf/tile.md"><code>tf.tile</code></a>(**input**, multiples, name=`None`)
* <a href="../../../tf/truncatediv.md"><code>tf.truncatediv</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/truncatemod.md"><code>tf.truncatemod</code></a>(**x**, **y**, name=`None`)
* <a href="../../../tf/where.md"><code>tf.where</code></a>(**condition**, **x**=`None`, **y**=`None`, name=`None`)
* <a href="../../../tf/zeros_like.md"><code>tf.zeros_like</code></a>(**input**, dtype=`None`, name=`None`)n

## Classes

[`class RaggedTensorValue`](../../../tf/compat/v1/ragged/RaggedTensorValue.md): Represents the value of a `RaggedTensor`.

## Functions

[`boolean_mask(...)`](../../../tf/ragged/boolean_mask.md): Applies a boolean mask to `data` without flattening the mask dimensions.

[`constant(...)`](../../../tf/ragged/constant.md): Constructs a constant RaggedTensor from a nested Python list.

[`constant_value(...)`](../../../tf/compat/v1/ragged/constant_value.md): Constructs a RaggedTensorValue from a nested Python list.

[`cross(...)`](../../../tf/ragged/cross.md): Generates feature cross from a list of tensors.

[`cross_hashed(...)`](../../../tf/ragged/cross_hashed.md): Generates hashed feature cross from a list of tensors.

[`map_flat_values(...)`](../../../tf/ragged/map_flat_values.md): Applies `op` to the `flat_values` of one or more RaggedTensors.

[`placeholder(...)`](../../../tf/compat/v1/ragged/placeholder.md): Creates a placeholder for a <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> that will always be fed.

[`range(...)`](../../../tf/ragged/range.md): Returns a `RaggedTensor` containing the specified sequences of numbers.

[`row_splits_to_segment_ids(...)`](../../../tf/ragged/row_splits_to_segment_ids.md): Generates the segmentation corresponding to a RaggedTensor `row_splits`.

[`segment_ids_to_row_splits(...)`](../../../tf/ragged/segment_ids_to_row_splits.md): Generates the RaggedTensor `row_splits` corresponding to a segmentation.

[`stack(...)`](../../../tf/ragged/stack.md): Stacks a list of rank-`R` tensors into one rank-`(R+1)` `RaggedTensor`.

[`stack_dynamic_partitions(...)`](../../../tf/ragged/stack_dynamic_partitions.md): Stacks dynamic partitions of a Tensor or RaggedTensor.

