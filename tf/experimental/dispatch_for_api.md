description: Decorator that overrides the default implementation for a TensorFlow API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dispatch_for_api" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dispatch_for_api

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/dispatch.py">View source</a>



Decorator that overrides the default implementation for a TensorFlow API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.dispatch_for_api`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dispatch_for_api(
    api, *signatures
)
</code></pre>



<!-- Placeholder for "Used in" -->

The decorated function (known as the "dispatch target") will override the
default implementation for the API when the API is called with parameters that
match a specified type signature.  Signatures are specified using dictionaries
that map parameter names to type annotations.  E.g., in the following example,
`masked_add` will be called for <a href="../../tf/math/add.md"><code>tf.add</code></a> if both `x` and `y` are
`MaskedTensor`s:

```
>>> class MaskedTensor(tf.experimental.ExtensionType):
...   values: tf.Tensor
...   mask: tf.Tensor
```

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor, 'y': MaskedTensor})
... def masked_add(x, y, name=None):
...   return MaskedTensor(x.values + y.values, x.mask & y.mask)
```

```
>>> mt = tf.add(MaskedTensor([1, 2], [True, False]), MaskedTensor(10, True))
>>> print(f"values={mt.values.numpy()}, mask={mt.mask.numpy()}")
values=[11 12], mask=[ True False]
```

If multiple type signatures are specified, then the dispatch target will be
called if any of the signatures match.  For example, the following code
registers `masked_add` to be called if `x` is a `MaskedTensor` *or* `y` is
a `MaskedTensor`.

```
>>> @dispatch_for_api(tf.math.add, {'x': MaskedTensor}, {'y':MaskedTensor})
... def masked_add(x, y):
...   x_values = x.values if isinstance(x, MaskedTensor) else x
...   x_mask = x.mask if isinstance(x, MaskedTensor) else True
...   y_values = y.values if isinstance(y, MaskedTensor) else y
...   y_mask = y.mask if isinstance(y, MaskedTensor) else True
...   return MaskedTensor(x_values + y_values, x_mask & y_mask)
```

The type annotations in type signatures may be type objects (e.g.,
`MaskedTensor`), `typing.List` values, or `typing.Union` values.   For
example, the following will register `masked_concat` to be called if `values`
is a list of `MaskedTensor` values:

```
>>> @dispatch_for_api(tf.concat, {'values': typing.List[MaskedTensor]})
... def masked_concat(values, axis):
...   return MaskedTensor(tf.concat([v.values for v in values], axis),
...                       tf.concat([v.mask for v in values], axis))
```

Each type signature must contain at least one subclass of `tf.CompositeTensor`
(which includes subclasses of `tf.ExtensionType`), and dispatch will only be
triggered if at least one type-annotated parameter contains a
`CompositeTensor` value.  This rule avoids invoking dispatch in degenerate
cases, such as the following examples:

* `@dispatch_for_api(tf.concat, {'values': List[MaskedTensor]})`: Will not
  dispatch to the decorated dispatch target when the user calls
  `tf.concat([])`.

* `@dispatch_for_api(tf.add, {'x': Union[MaskedTensor, Tensor], 'y':
  Union[MaskedTensor, Tensor]})`: Will not dispatch to the decorated dispatch
  target when the user calls `tf.add(tf.constant(1), tf.constant(2))`.

The dispatch target's signature must match the signature of the API that is
being overridden.  In particular, parameters must have the same names, and
must occur in the same order.  The dispatch target may optionally elide the
"name" parameter, in which case it will be wrapped with a call to
<a href="../../tf/name_scope.md"><code>tf.name_scope</code></a> when appropraite.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`api`
</td>
<td>
The TensorFlow API to override.
</td>
</tr><tr>
<td>
`*signatures`
</td>
<td>
Dictionaries mapping parameter names or indices to type
annotations, specifying when the dispatch target should be called.  In
particular, the dispatch target will be called if any signature matches;
and a signature matches if all of the specified parameters have types that
match with the indicated type annotations.  If no signatures are
specified, then a signature will be read from the dispatch target
function's type annotations.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A decorator that overrides the default implementation for `api`.
</td>
</tr>

</table>


#### Registered APIs

The TensorFlow APIs that may be overridden by `@dispatch_for_api` are:

* `tf.__operators__.add(x, y, name)`
* `tf.__operators__.eq(self, other)`
* `tf.__operators__.getitem(tensor, slice_spec, var)`
* `tf.__operators__.ne(self, other)`
* `tf.__operators__.ragged_getitem(rt_input, key)`
* <a href="../../tf/argsort.md"><code>tf.argsort(values, axis, direction, stable, name)</code></a>
* <a href="../../tf/audio/decode_wav.md"><code>tf.audio.decode_wav(contents, desired_channels, desired_samples, name)</code></a>
* <a href="../../tf/audio/encode_wav.md"><code>tf.audio.encode_wav(audio, sample_rate, name)</code></a>
* <a href="../../tf/batch_to_space.md"><code>tf.batch_to_space(input, block_shape, crops, name)</code></a>
* <a href="../../tf/bitcast.md"><code>tf.bitcast(input, type, name)</code></a>
* <a href="../../tf/bitwise/bitwise_and.md"><code>tf.bitwise.bitwise_and(x, y, name)</code></a>
* <a href="../../tf/bitwise/bitwise_or.md"><code>tf.bitwise.bitwise_or(x, y, name)</code></a>
* <a href="../../tf/bitwise/bitwise_xor.md"><code>tf.bitwise.bitwise_xor(x, y, name)</code></a>
* <a href="../../tf/bitwise/invert.md"><code>tf.bitwise.invert(x, name)</code></a>
* <a href="../../tf/bitwise/left_shift.md"><code>tf.bitwise.left_shift(x, y, name)</code></a>
* <a href="../../tf/bitwise/right_shift.md"><code>tf.bitwise.right_shift(x, y, name)</code></a>
* <a href="../../tf/boolean_mask.md"><code>tf.boolean_mask(tensor, mask, axis, name)</code></a>
* <a href="../../tf/broadcast_dynamic_shape.md"><code>tf.broadcast_dynamic_shape(shape_x, shape_y)</code></a>
* <a href="../../tf/broadcast_static_shape.md"><code>tf.broadcast_static_shape(shape_x, shape_y)</code></a>
* <a href="../../tf/broadcast_to.md"><code>tf.broadcast_to(input, shape, name)</code></a>
* <a href="../../tf/case.md"><code>tf.case(pred_fn_pairs, default, exclusive, strict, name)</code></a>
* <a href="../../tf/cast.md"><code>tf.cast(x, dtype, name)</code></a>
* <a href="../../tf/clip_by_global_norm.md"><code>tf.clip_by_global_norm(t_list, clip_norm, use_norm, name)</code></a>
* <a href="../../tf/clip_by_norm.md"><code>tf.clip_by_norm(t, clip_norm, axes, name)</code></a>
* <a href="../../tf/clip_by_value.md"><code>tf.clip_by_value(t, clip_value_min, clip_value_max, name)</code></a>
* <a href="../../tf/compat/v1/Print.md"><code>tf.compat.v1.Print(input_, data, message, first_n, summarize, name)</code></a>
* <a href="../../tf/compat/v1/arg_max.md"><code>tf.compat.v1.arg_max(input, dimension, output_type, name)</code></a>
* <a href="../../tf/compat/v1/arg_min.md"><code>tf.compat.v1.arg_min(input, dimension, output_type, name)</code></a>
* <a href="../../tf/compat/v1/batch_gather.md"><code>tf.compat.v1.batch_gather(params, indices, name)</code></a>
* <a href="../../tf/compat/v1/batch_to_space.md"><code>tf.compat.v1.batch_to_space(input, crops, block_size, name, block_shape)</code></a>
* <a href="../../tf/compat/v1/batch_to_space_nd.md"><code>tf.compat.v1.batch_to_space_nd(input, block_shape, crops, name)</code></a>
* <a href="../../tf/compat/v1/boolean_mask.md"><code>tf.compat.v1.boolean_mask(tensor, mask, name, axis)</code></a>
* <a href="../../tf/compat/v1/case.md"><code>tf.compat.v1.case(pred_fn_pairs, default, exclusive, strict, name)</code></a>
* <a href="../../tf/compat/v1/clip_by_average_norm.md"><code>tf.compat.v1.clip_by_average_norm(t, clip_norm, name)</code></a>
* <a href="../../tf/compat/v1/cond.md"><code>tf.compat.v1.cond(pred, true_fn, false_fn, strict, name, fn1, fn2)</code></a>
* <a href="../../tf/compat/v1/convert_to_tensor.md"><code>tf.compat.v1.convert_to_tensor(value, dtype, name, preferred_dtype, dtype_hint)</code></a>
* <a href="../../tf/compat/v1/verify_tensor_all_finite.md"><code>tf.compat.v1.debugging.assert_all_finite(t, msg, name, x, message)</code></a>
* <a href="../../tf/compat/v1/assert_equal.md"><code>tf.compat.v1.debugging.assert_equal(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_greater.md"><code>tf.compat.v1.debugging.assert_greater(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_greater_equal.md"><code>tf.compat.v1.debugging.assert_greater_equal(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_integer.md"><code>tf.compat.v1.debugging.assert_integer(x, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_less.md"><code>tf.compat.v1.debugging.assert_less(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_less_equal.md"><code>tf.compat.v1.debugging.assert_less_equal(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_near.md"><code>tf.compat.v1.debugging.assert_near(x, y, rtol, atol, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_negative.md"><code>tf.compat.v1.debugging.assert_negative(x, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_non_negative.md"><code>tf.compat.v1.debugging.assert_non_negative(x, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_non_positive.md"><code>tf.compat.v1.debugging.assert_non_positive(x, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_none_equal.md"><code>tf.compat.v1.debugging.assert_none_equal(x, y, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_positive.md"><code>tf.compat.v1.debugging.assert_positive(x, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_rank.md"><code>tf.compat.v1.debugging.assert_rank(x, rank, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_rank_at_least.md"><code>tf.compat.v1.debugging.assert_rank_at_least(x, rank, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_rank_in.md"><code>tf.compat.v1.debugging.assert_rank_in(x, ranks, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_scalar.md"><code>tf.compat.v1.debugging.assert_scalar(tensor, name, message)</code></a>
* <a href="../../tf/compat/v1/debugging/assert_shapes.md"><code>tf.compat.v1.debugging.assert_shapes(shapes, data, summarize, message, name)</code></a>
* <a href="../../tf/compat/v1/assert_type.md"><code>tf.compat.v1.debugging.assert_type(tensor, tf_type, message, name)</code></a>
* <a href="../../tf/compat/v1/decode_raw.md"><code>tf.compat.v1.decode_raw(input_bytes, out_type, little_endian, name, bytes)</code></a>
* <a href="../../tf/compat/v1/div.md"><code>tf.compat.v1.div(x, y, name)</code></a>
* <a href="../../tf/compat/v1/expand_dims.md"><code>tf.compat.v1.expand_dims(input, axis, name, dim)</code></a>
* <a href="../../tf/compat/v1/floor_div.md"><code>tf.compat.v1.floor_div(x, y, name)</code></a>
* <a href="../../tf/compat/v1/foldl.md"><code>tf.compat.v1.foldl(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, name)</code></a>
* <a href="../../tf/compat/v1/foldr.md"><code>tf.compat.v1.foldr(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, name)</code></a>
* <a href="../../tf/compat/v1/gather.md"><code>tf.compat.v1.gather(params, indices, validate_indices, name, axis, batch_dims)</code></a>
* <a href="../../tf/compat/v1/gather_nd.md"><code>tf.compat.v1.gather_nd(params, indices, name, batch_dims)</code></a>
* <a href="../../tf/compat/v1/image/crop_and_resize.md"><code>tf.compat.v1.image.crop_and_resize(image, boxes, box_ind, crop_size, method, extrapolation_value, name, box_indices)</code></a>
* <a href="../../tf/compat/v1/image/draw_bounding_boxes.md"><code>tf.compat.v1.image.draw_bounding_boxes(images, boxes, name, colors)</code></a>
* <a href="../../tf/compat/v1/image/extract_glimpse.md"><code>tf.compat.v1.image.extract_glimpse(input, size, offsets, centered, normalized, uniform_noise, name)</code></a>
* <a href="../../tf/compat/v1/extract_image_patches.md"><code>tf.compat.v1.image.extract_image_patches(images, ksizes, strides, rates, padding, name, sizes)</code></a>
* <a href="../../tf/compat/v1/image/resize_area.md"><code>tf.compat.v1.image.resize_area(images, size, align_corners, name)</code></a>
* <a href="../../tf/compat/v1/image/resize_bicubic.md"><code>tf.compat.v1.image.resize_bicubic(images, size, align_corners, name, half_pixel_centers)</code></a>
* <a href="../../tf/compat/v1/image/resize_bilinear.md"><code>tf.compat.v1.image.resize_bilinear(images, size, align_corners, name, half_pixel_centers)</code></a>
* <a href="../../tf/compat/v1/image/resize_image_with_pad.md"><code>tf.compat.v1.image.resize_image_with_pad(image, target_height, target_width, method, align_corners)</code></a>
* <a href="../../tf/compat/v1/image/resize.md"><code>tf.compat.v1.image.resize_images(images, size, method, align_corners, preserve_aspect_ratio, name)</code></a>
* <a href="../../tf/compat/v1/image/resize_nearest_neighbor.md"><code>tf.compat.v1.image.resize_nearest_neighbor(images, size, align_corners, name, half_pixel_centers)</code></a>
* <a href="../../tf/compat/v1/image/sample_distorted_bounding_box.md"><code>tf.compat.v1.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed, seed2, min_object_covered, aspect_ratio_range, area_range, max_attempts, use_image_if_no_bounding_boxes, name)</code></a>
* <a href="../../tf/compat/v1/decode_csv.md"><code>tf.compat.v1.io.decode_csv(records, record_defaults, field_delim, use_quote_delim, name, na_value, select_cols)</code></a>
* <a href="../../tf/compat/v1/parse_example.md"><code>tf.compat.v1.io.parse_example(serialized, features, name, example_names)</code></a>
* <a href="../../tf/compat/v1/parse_single_example.md"><code>tf.compat.v1.io.parse_single_example(serialized, features, name, example_names)</code></a>
* <a href="../../tf/compat/v1/serialize_many_sparse.md"><code>tf.compat.v1.io.serialize_many_sparse(sp_input, name, out_type)</code></a>
* <a href="../../tf/compat/v1/serialize_sparse.md"><code>tf.compat.v1.io.serialize_sparse(sp_input, name, out_type)</code></a>
* <a href="../../tf/compat/v1/losses/absolute_difference.md"><code>tf.compat.v1.losses.absolute_difference(labels, predictions, weights, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/compute_weighted_loss.md"><code>tf.compat.v1.losses.compute_weighted_loss(losses, weights, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/cosine_distance.md"><code>tf.compat.v1.losses.cosine_distance(labels, predictions, axis, weights, scope, loss_collection, reduction, dim)</code></a>
* <a href="../../tf/compat/v1/losses/hinge_loss.md"><code>tf.compat.v1.losses.hinge_loss(labels, logits, weights, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/huber_loss.md"><code>tf.compat.v1.losses.huber_loss(labels, predictions, weights, delta, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/log_loss.md"><code>tf.compat.v1.losses.log_loss(labels, predictions, weights, epsilon, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/mean_pairwise_squared_error.md"><code>tf.compat.v1.losses.mean_pairwise_squared_error(labels, predictions, weights, scope, loss_collection)</code></a>
* <a href="../../tf/compat/v1/losses/mean_squared_error.md"><code>tf.compat.v1.losses.mean_squared_error(labels, predictions, weights, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/sigmoid_cross_entropy.md"><code>tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels, logits, weights, label_smoothing, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/softmax_cross_entropy.md"><code>tf.compat.v1.losses.softmax_cross_entropy(onehot_labels, logits, weights, label_smoothing, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/losses/sparse_softmax_cross_entropy.md"><code>tf.compat.v1.losses.sparse_softmax_cross_entropy(labels, logits, weights, scope, loss_collection, reduction)</code></a>
* <a href="../../tf/compat/v1/argmax.md"><code>tf.compat.v1.math.argmax(input, axis, name, dimension, output_type)</code></a>
* <a href="../../tf/compat/v1/argmin.md"><code>tf.compat.v1.math.argmin(input, axis, name, dimension, output_type)</code></a>
* <a href="../../tf/compat/v1/confusion_matrix.md"><code>tf.compat.v1.math.confusion_matrix(labels, predictions, num_classes, dtype, name, weights)</code></a>
* <a href="../../tf/compat/v1/count_nonzero.md"><code>tf.compat.v1.math.count_nonzero(input_tensor, axis, keepdims, dtype, name, reduction_indices, keep_dims, input)</code></a>
* <a href="../../tf/compat/v1/math/in_top_k.md"><code>tf.compat.v1.math.in_top_k(predictions, targets, k, name)</code></a>
* <a href="../../tf/compat/v1/reduce_all.md"><code>tf.compat.v1.math.reduce_all(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_any.md"><code>tf.compat.v1.math.reduce_any(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_logsumexp.md"><code>tf.compat.v1.math.reduce_logsumexp(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_max.md"><code>tf.compat.v1.math.reduce_max(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_mean.md"><code>tf.compat.v1.math.reduce_mean(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_min.md"><code>tf.compat.v1.math.reduce_min(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_prod.md"><code>tf.compat.v1.math.reduce_prod(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/reduce_sum.md"><code>tf.compat.v1.math.reduce_sum(input_tensor, axis, keepdims, name, reduction_indices, keep_dims)</code></a>
* <a href="../../tf/compat/v1/scalar_mul.md"><code>tf.compat.v1.math.scalar_mul(scalar, x, name)</code></a>
* <a href="../../tf/compat/v1/nn/avg_pool.md"><code>tf.compat.v1.nn.avg_pool(value, ksize, strides, padding, data_format, name, input)</code></a>
* <a href="../../tf/compat/v1/nn/batch_norm_with_global_normalization.md"><code>tf.compat.v1.nn.batch_norm_with_global_normalization(t, m, v, beta, gamma, variance_epsilon, scale_after_normalization, name, input, mean, variance)</code></a>
* <a href="../../tf/compat/v1/nn/bidirectional_dynamic_rnn.md"><code>tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length, initial_state_fw, initial_state_bw, dtype, parallel_iterations, swap_memory, time_major, scope)</code></a>
* <a href="../../tf/compat/v1/nn/conv1d.md"><code>tf.compat.v1.nn.conv1d(value, filters, stride, padding, use_cudnn_on_gpu, data_format, name, input, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/conv2d.md"><code>tf.compat.v1.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, data_format, dilations, name, filters)</code></a>
* <a href="../../tf/compat/v1/nn/conv2d_backprop_filter.md"><code>tf.compat.v1.nn.conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, use_cudnn_on_gpu, data_format, dilations, name)</code></a>
* <a href="../../tf/compat/v1/nn/conv2d_backprop_input.md"><code>tf.compat.v1.nn.conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, use_cudnn_on_gpu, data_format, dilations, name, filters)</code></a>
* <a href="../../tf/compat/v1/nn/conv2d_transpose.md"><code>tf.compat.v1.nn.conv2d_transpose(value, filter, output_shape, strides, padding, data_format, name, input, filters, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/conv3d.md"><code>tf.compat.v1.nn.conv3d(input, filter, strides, padding, data_format, dilations, name, filters)</code></a>
* <a href="../../tf/compat/v1/nn/conv3d_backprop_filter.md"><code>tf.compat.v1.nn.conv3d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/compat/v1/nn/conv3d_transpose.md"><code>tf.compat.v1.nn.conv3d_transpose(value, filter, output_shape, strides, padding, data_format, name, input, filters, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/convolution.md"><code>tf.compat.v1.nn.convolution(input, filter, padding, strides, dilation_rate, name, data_format, filters, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/crelu.md"><code>tf.compat.v1.nn.crelu(features, name, axis)</code></a>
* <a href="../../tf/compat/v1/nn/ctc_beam_search_decoder.md"><code>tf.compat.v1.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width, top_paths, merge_repeated)</code></a>
* <a href="../../tf/compat/v1/nn/ctc_loss.md"><code>tf.compat.v1.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated, ctc_merge_repeated, ignore_longer_outputs_than_inputs, time_major, logits)</code></a>
* <a href="../../tf/compat/v1/nn/ctc_loss_v2.md"><code>tf.compat.v1.nn.ctc_loss_v2(labels, logits, label_length, logit_length, logits_time_major, unique, blank_index, name)</code></a>
* <a href="../../tf/compat/v1/depth_to_space.md"><code>tf.compat.v1.nn.depth_to_space(input, block_size, name, data_format)</code></a>
* <a href="../../tf/compat/v1/nn/depthwise_conv2d.md"><code>tf.compat.v1.nn.depthwise_conv2d(input, filter, strides, padding, rate, name, data_format, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/depthwise_conv2d_native.md"><code>tf.compat.v1.nn.depthwise_conv2d_native(input, filter, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/compat/v1/nn/dilation2d.md"><code>tf.compat.v1.nn.dilation2d(input, filter, strides, rates, padding, name, filters, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/dropout.md"><code>tf.compat.v1.nn.dropout(x, keep_prob, noise_shape, seed, name, rate)</code></a>
* <a href="../../tf/compat/v1/nn/dynamic_rnn.md"><code>tf.compat.v1.nn.dynamic_rnn(cell, inputs, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)</code></a>
* <a href="../../tf/compat/v1/nn/embedding_lookup.md"><code>tf.compat.v1.nn.embedding_lookup(params, ids, partition_strategy, name, validate_indices, max_norm)</code></a>
* <a href="../../tf/compat/v1/nn/embedding_lookup_sparse.md"><code>tf.compat.v1.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, partition_strategy, name, combiner, max_norm)</code></a>
* <a href="../../tf/compat/v1/nn/erosion2d.md"><code>tf.compat.v1.nn.erosion2d(value, kernel, strides, rates, padding, name)</code></a>
* <a href="../../tf/compat/v1/nn/fractional_avg_pool.md"><code>tf.compat.v1.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random, overlapping, deterministic, seed, seed2, name)</code></a>
* <a href="../../tf/compat/v1/nn/fractional_max_pool.md"><code>tf.compat.v1.nn.fractional_max_pool(value, pooling_ratio, pseudo_random, overlapping, deterministic, seed, seed2, name)</code></a>
* <a href="../../tf/compat/v1/nn/fused_batch_norm.md"><code>tf.compat.v1.nn.fused_batch_norm(x, scale, offset, mean, variance, epsilon, data_format, is_training, name, exponential_avg_factor)</code></a>
* <a href="../../tf/compat/v1/math/log_softmax.md"><code>tf.compat.v1.nn.log_softmax(logits, axis, name, dim)</code></a>
* <a href="../../tf/compat/v1/nn/max_pool.md"><code>tf.compat.v1.nn.max_pool(value, ksize, strides, padding, data_format, name, input)</code></a>
* <a href="../../tf/compat/v1/nn/max_pool_with_argmax.md"><code>tf.compat.v1.nn.max_pool_with_argmax(input, ksize, strides, padding, data_format, Targmax, name, output_dtype, include_batch_in_index)</code></a>
* <a href="../../tf/compat/v1/nn/moments.md"><code>tf.compat.v1.nn.moments(x, axes, shift, name, keep_dims, keepdims)</code></a>
* <a href="../../tf/compat/v1/nn/nce_loss.md"><code>tf.compat.v1.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values, remove_accidental_hits, partition_strategy, name)</code></a>
* <a href="../../tf/compat/v1/nn/pool.md"><code>tf.compat.v1.nn.pool(input, window_shape, pooling_type, padding, dilation_rate, strides, name, data_format, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_avg_pool.md"><code>tf.compat.v1.nn.quantized_avg_pool(input, min_input, max_input, ksize, strides, padding, name)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_conv2d.md"><code>tf.compat.v1.nn.quantized_conv2d(input, filter, min_input, max_input, min_filter, max_filter, strides, padding, out_type, dilations, name)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_max_pool.md"><code>tf.compat.v1.nn.quantized_max_pool(input, min_input, max_input, ksize, strides, padding, name)</code></a>
* <a href="../../tf/compat/v1/nn/quantized_relu_x.md"><code>tf.compat.v1.nn.quantized_relu_x(features, max_value, min_features, max_features, out_type, name)</code></a>
* <a href="../../tf/compat/v1/nn/raw_rnn.md"><code>tf.compat.v1.nn.raw_rnn(cell, loop_fn, parallel_iterations, swap_memory, scope)</code></a>
* <a href="../../tf/compat/v1/nn/relu_layer.md"><code>tf.compat.v1.nn.relu_layer(x, weights, biases, name)</code></a>
* <a href="../../tf/compat/v1/nn/safe_embedding_lookup_sparse.md"><code>tf.compat.v1.nn.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights, combiner, default_id, name, partition_strategy, max_norm)</code></a>
* <a href="../../tf/compat/v1/nn/sampled_softmax_loss.md"><code>tf.compat.v1.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values, remove_accidental_hits, partition_strategy, name, seed)</code></a>
* <a href="../../tf/compat/v1/nn/separable_conv2d.md"><code>tf.compat.v1.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate, name, data_format, dilations)</code></a>
* <a href="../../tf/compat/v1/nn/sigmoid_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(_sentinel, labels, logits, name)</code></a>
* <a href="../../tf/compat/v1/math/softmax.md"><code>tf.compat.v1.nn.softmax(logits, axis, name, dim)</code></a>
* <a href="../../tf/compat/v1/nn/softmax_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.softmax_cross_entropy_with_logits(_sentinel, labels, logits, dim, name, axis)</code></a>
* <a href="../../tf/compat/v1/nn/softmax_cross_entropy_with_logits_v2.md"><code>tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels, logits, axis, name, dim)</code></a>
* <a href="../../tf/compat/v1/space_to_batch.md"><code>tf.compat.v1.nn.space_to_batch(input, paddings, block_size, name, block_shape)</code></a>
* <a href="../../tf/compat/v1/space_to_depth.md"><code>tf.compat.v1.nn.space_to_depth(input, block_size, name, data_format)</code></a>
* <a href="../../tf/compat/v1/nn/sparse_softmax_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(_sentinel, labels, logits, name)</code></a>
* <a href="../../tf/compat/v1/nn/static_bidirectional_rnn.md"><code>tf.compat.v1.nn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw, initial_state_bw, dtype, sequence_length, scope)</code></a>
* <a href="../../tf/compat/v1/nn/static_rnn.md"><code>tf.compat.v1.nn.static_rnn(cell, inputs, initial_state, dtype, sequence_length, scope)</code></a>
* <a href="../../tf/compat/v1/nn/static_state_saving_rnn.md"><code>tf.compat.v1.nn.static_state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length, scope)</code></a>
* <a href="../../tf/compat/v1/nn/sufficient_statistics.md"><code>tf.compat.v1.nn.sufficient_statistics(x, axes, shift, keep_dims, name, keepdims)</code></a>
* <a href="../../tf/compat/v1/nn/weighted_cross_entropy_with_logits.md"><code>tf.compat.v1.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight, name, targets)</code></a>
* <a href="../../tf/compat/v1/nn/weighted_moments.md"><code>tf.compat.v1.nn.weighted_moments(x, axes, frequency_weights, name, keep_dims, keepdims)</code></a>
* <a href="../../tf/compat/v1/nn/xw_plus_b.md"><code>tf.compat.v1.nn.xw_plus_b(x, weights, biases, name)</code></a>
* <a href="../../tf/compat/v1/norm.md"><code>tf.compat.v1.norm(tensor, ord, axis, keepdims, name, keep_dims)</code></a>
* <a href="../../tf/compat/v1/ones_like.md"><code>tf.compat.v1.ones_like(tensor, dtype, name, optimize)</code></a>
* <a href="../../tf/compat/v1/pad.md"><code>tf.compat.v1.pad(tensor, paddings, mode, name, constant_values)</code></a>
* <a href="../../tf/compat/v1/py_func.md"><code>tf.compat.v1.py_func(func, inp, Tout, stateful, name)</code></a>
* <a href="../../tf/compat/v1/quantize_v2.md"><code>tf.compat.v1.quantize_v2(input, min_range, max_range, T, mode, name, round_mode, narrow_range, axis, ensure_minimum_range)</code></a>
* <a href="../../tf/compat/v1/ragged/constant_value.md"><code>tf.compat.v1.ragged.constant_value(pylist, dtype, ragged_rank, inner_shape, row_splits_dtype)</code></a>
* <a href="../../tf/compat/v1/ragged/placeholder.md"><code>tf.compat.v1.ragged.placeholder(dtype, ragged_rank, value_shape, name)</code></a>
* <a href="../../tf/compat/v1/multinomial.md"><code>tf.compat.v1.random.multinomial(logits, num_samples, seed, name, output_dtype)</code></a>
* <a href="../../tf/compat/v1/random_poisson.md"><code>tf.compat.v1.random.poisson(lam, shape, dtype, seed, name)</code></a>
* <a href="../../tf/compat/v1/random/stateless_multinomial.md"><code>tf.compat.v1.random.stateless_multinomial(logits, num_samples, seed, output_dtype, name)</code></a>
* <a href="../../tf/compat/v1/scan.md"><code>tf.compat.v1.scan(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, infer_shape, reverse, name)</code></a>
* <a href="../../tf/compat/v1/setdiff1d.md"><code>tf.compat.v1.setdiff1d(x, y, index_dtype, name)</code></a>
* <a href="../../tf/compat/v1/shape.md"><code>tf.compat.v1.shape(input, name, out_type)</code></a>
* <a href="../../tf/compat/v1/size.md"><code>tf.compat.v1.size(input, name, out_type)</code></a>
* <a href="../../tf/compat/v1/sparse_to_dense.md"><code>tf.compat.v1.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, validate_indices, name)</code></a>
* <a href="../../tf/compat/v1/squeeze.md"><code>tf.compat.v1.squeeze(input, axis, name, squeeze_dims)</code></a>
* <a href="../../tf/compat/v1/string_split.md"><code>tf.compat.v1.string_split(source, sep, skip_empty, delimiter, result_type, name)</code></a>
* <a href="../../tf/compat/v1/strings/length.md"><code>tf.compat.v1.strings.length(input, name, unit)</code></a>
* <a href="../../tf/compat/v1/reduce_join.md"><code>tf.compat.v1.strings.reduce_join(inputs, axis, keep_dims, separator, name, reduction_indices, keepdims)</code></a>
* <a href="../../tf/compat/v1/strings/split.md"><code>tf.compat.v1.strings.split(input, sep, maxsplit, result_type, source, name)</code></a>
* <a href="../../tf/compat/v1/strings/substr.md"><code>tf.compat.v1.strings.substr(input, pos, len, name, unit)</code></a>
* <a href="../../tf/compat/v1/string_to_hash_bucket.md"><code>tf.compat.v1.strings.to_hash_bucket(string_tensor, num_buckets, name, input)</code></a>
* <a href="../../tf/compat/v1/string_to_number.md"><code>tf.compat.v1.strings.to_number(string_tensor, out_type, name, input)</code></a>
* <a href="../../tf/compat/v1/substr.md"><code>tf.compat.v1.substr(input, pos, len, name, unit)</code></a>
* <a href="../../tf/compat/v1/to_bfloat16.md"><code>tf.compat.v1.to_bfloat16(x, name)</code></a>
* <a href="../../tf/compat/v1/to_complex128.md"><code>tf.compat.v1.to_complex128(x, name)</code></a>
* <a href="../../tf/compat/v1/to_complex64.md"><code>tf.compat.v1.to_complex64(x, name)</code></a>
* <a href="../../tf/compat/v1/to_double.md"><code>tf.compat.v1.to_double(x, name)</code></a>
* <a href="../../tf/compat/v1/to_float.md"><code>tf.compat.v1.to_float(x, name)</code></a>
* <a href="../../tf/compat/v1/to_int32.md"><code>tf.compat.v1.to_int32(x, name)</code></a>
* <a href="../../tf/compat/v1/to_int64.md"><code>tf.compat.v1.to_int64(x, name)</code></a>
* <a href="../../tf/compat/v1/train/sdca_fprint.md"><code>tf.compat.v1.train.sdca_fprint(input, name)</code></a>
* <a href="../../tf/compat/v1/train/sdca_optimizer.md"><code>tf.compat.v1.train.sdca_optimizer(sparse_example_indices, sparse_feature_indices, sparse_feature_values, dense_features, example_weights, example_labels, sparse_indices, sparse_weights, dense_weights, example_state_data, loss_type, l1, l2, num_loss_partitions, num_inner_iterations, adaptative, name)</code></a>
* <a href="../../tf/compat/v1/train/sdca_shrink_l1.md"><code>tf.compat.v1.train.sdca_shrink_l1(weights, l1, l2, name)</code></a>
* <a href="../../tf/compat/v1/transpose.md"><code>tf.compat.v1.transpose(a, perm, name, conjugate)</code></a>
* <a href="../../tf/compat/v1/tuple.md"><code>tf.compat.v1.tuple(tensors, name, control_inputs)</code></a>
* <a href="../../tf/compat/v1/where.md"><code>tf.compat.v1.where(condition, x, y, name)</code></a>
* <a href="../../tf/compat/v1/zeros_like.md"><code>tf.compat.v1.zeros_like(tensor, dtype, name, optimize)</code></a>
* <a href="../../tf/concat.md"><code>tf.concat(values, axis, name)</code></a>
* <a href="../../tf/cond.md"><code>tf.cond(pred, true_fn, false_fn, name)</code></a>
* <a href="../../tf/convert_to_tensor.md"><code>tf.convert_to_tensor(value, dtype, dtype_hint, name)</code></a>
* <a href="../../tf/debugging/Assert.md"><code>tf.debugging.Assert(condition, data, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_all_finite.md"><code>tf.debugging.assert_all_finite(x, message, name)</code></a>
* <a href="../../tf/debugging/assert_equal.md"><code>tf.debugging.assert_equal(x, y, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_greater.md"><code>tf.debugging.assert_greater(x, y, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_greater_equal.md"><code>tf.debugging.assert_greater_equal(x, y, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_integer.md"><code>tf.debugging.assert_integer(x, message, name)</code></a>
* <a href="../../tf/debugging/assert_less.md"><code>tf.debugging.assert_less(x, y, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_less_equal.md"><code>tf.debugging.assert_less_equal(x, y, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_near.md"><code>tf.debugging.assert_near(x, y, rtol, atol, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_negative.md"><code>tf.debugging.assert_negative(x, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_non_negative.md"><code>tf.debugging.assert_non_negative(x, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_non_positive.md"><code>tf.debugging.assert_non_positive(x, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_none_equal.md"><code>tf.debugging.assert_none_equal(x, y, summarize, message, name)</code></a>
* <a href="../../tf/debugging/assert_positive.md"><code>tf.debugging.assert_positive(x, message, summarize, name)</code></a>
* <a href="../../tf/debugging/assert_proper_iterable.md"><code>tf.debugging.assert_proper_iterable(values)</code></a>
* <a href="../../tf/debugging/assert_rank.md"><code>tf.debugging.assert_rank(x, rank, message, name)</code></a>
* <a href="../../tf/debugging/assert_rank_at_least.md"><code>tf.debugging.assert_rank_at_least(x, rank, message, name)</code></a>
* <a href="../../tf/debugging/assert_rank_in.md"><code>tf.debugging.assert_rank_in(x, ranks, message, name)</code></a>
* <a href="../../tf/debugging/assert_same_float_dtype.md"><code>tf.debugging.assert_same_float_dtype(tensors, dtype)</code></a>
* <a href="../../tf/debugging/assert_scalar.md"><code>tf.debugging.assert_scalar(tensor, message, name)</code></a>
* <a href="../../tf/debugging/assert_shapes.md"><code>tf.debugging.assert_shapes(shapes, data, summarize, message, name)</code></a>
* <a href="../../tf/debugging/assert_type.md"><code>tf.debugging.assert_type(tensor, tf_type, message, name)</code></a>
* <a href="../../tf/debugging/check_numerics.md"><code>tf.debugging.check_numerics(tensor, message, name)</code></a>
* <a href="../../tf/dtypes/complex.md"><code>tf.dtypes.complex(real, imag, name)</code></a>
* <a href="../../tf/dtypes/saturate_cast.md"><code>tf.dtypes.saturate_cast(value, dtype, name)</code></a>
* <a href="../../tf/dynamic_partition.md"><code>tf.dynamic_partition(data, partitions, num_partitions, name)</code></a>
* <a href="../../tf/dynamic_stitch.md"><code>tf.dynamic_stitch(indices, data, name)</code></a>
* <a href="../../tf/edit_distance.md"><code>tf.edit_distance(hypothesis, truth, normalize, name)</code></a>
* <a href="../../tf/ensure_shape.md"><code>tf.ensure_shape(x, shape, name)</code></a>
* <a href="../../tf/expand_dims.md"><code>tf.expand_dims(input, axis, name)</code></a>
* <a href="../../tf/extract_volume_patches.md"><code>tf.extract_volume_patches(input, ksizes, strides, padding, name)</code></a>
* <a href="../../tf/eye.md"><code>tf.eye(num_rows, num_columns, batch_shape, dtype, name)</code></a>
* <a href="../../tf/fill.md"><code>tf.fill(dims, value, name)</code></a>
* <a href="../../tf/fingerprint.md"><code>tf.fingerprint(data, method, name)</code></a>
* <a href="../../tf/foldl.md"><code>tf.foldl(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, name)</code></a>
* <a href="../../tf/foldr.md"><code>tf.foldr(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, name)</code></a>
* <a href="../../tf/gather.md"><code>tf.gather(params, indices, validate_indices, axis, batch_dims, name)</code></a>
* <a href="../../tf/gather_nd.md"><code>tf.gather_nd(params, indices, batch_dims, name)</code></a>
* <a href="../../tf/histogram_fixed_width.md"><code>tf.histogram_fixed_width(values, value_range, nbins, dtype, name)</code></a>
* <a href="../../tf/histogram_fixed_width_bins.md"><code>tf.histogram_fixed_width_bins(values, value_range, nbins, dtype, name)</code></a>
* <a href="../../tf/identity.md"><code>tf.identity(input, name)</code></a>
* <a href="../../tf/identity_n.md"><code>tf.identity_n(input, name)</code></a>
* <a href="../../tf/image/adjust_brightness.md"><code>tf.image.adjust_brightness(image, delta)</code></a>
* <a href="../../tf/image/adjust_contrast.md"><code>tf.image.adjust_contrast(images, contrast_factor)</code></a>
* <a href="../../tf/image/adjust_gamma.md"><code>tf.image.adjust_gamma(image, gamma, gain)</code></a>
* <a href="../../tf/image/adjust_hue.md"><code>tf.image.adjust_hue(image, delta, name)</code></a>
* <a href="../../tf/image/adjust_jpeg_quality.md"><code>tf.image.adjust_jpeg_quality(image, jpeg_quality, name)</code></a>
* <a href="../../tf/image/adjust_saturation.md"><code>tf.image.adjust_saturation(image, saturation_factor, name)</code></a>
* <a href="../../tf/image/central_crop.md"><code>tf.image.central_crop(image, central_fraction)</code></a>
* <a href="../../tf/image/combined_non_max_suppression.md"><code>tf.image.combined_non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, pad_per_class, clip_boxes, name)</code></a>
* <a href="../../tf/image/convert_image_dtype.md"><code>tf.image.convert_image_dtype(image, dtype, saturate, name)</code></a>
* <a href="../../tf/image/crop_and_resize.md"><code>tf.image.crop_and_resize(image, boxes, box_indices, crop_size, method, extrapolation_value, name)</code></a>
* <a href="../../tf/image/crop_to_bounding_box.md"><code>tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)</code></a>
* <a href="../../tf/image/draw_bounding_boxes.md"><code>tf.image.draw_bounding_boxes(images, boxes, colors, name)</code></a>
* <a href="../../tf/image/extract_glimpse.md"><code>tf.image.extract_glimpse(input, size, offsets, centered, normalized, noise, name)</code></a>
* <a href="../../tf/image/extract_patches.md"><code>tf.image.extract_patches(images, sizes, strides, rates, padding, name)</code></a>
* <a href="../../tf/image/flip_left_right.md"><code>tf.image.flip_left_right(image)</code></a>
* <a href="../../tf/image/flip_up_down.md"><code>tf.image.flip_up_down(image)</code></a>
* <a href="../../tf/image/generate_bounding_box_proposals.md"><code>tf.image.generate_bounding_box_proposals(scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size, post_nms_topn, name)</code></a>
* <a href="../../tf/image/grayscale_to_rgb.md"><code>tf.image.grayscale_to_rgb(images, name)</code></a>
* <a href="../../tf/image/hsv_to_rgb.md"><code>tf.image.hsv_to_rgb(images, name)</code></a>
* <a href="../../tf/image/image_gradients.md"><code>tf.image.image_gradients(image)</code></a>
* <a href="../../tf/image/non_max_suppression.md"><code>tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold, score_threshold, name)</code></a>
* <a href="../../tf/image/non_max_suppression_overlaps.md"><code>tf.image.non_max_suppression_overlaps(overlaps, scores, max_output_size, overlap_threshold, score_threshold, name)</code></a>
* <a href="../../tf/image/non_max_suppression_padded.md"><code>tf.image.non_max_suppression_padded(boxes, scores, max_output_size, iou_threshold, score_threshold, pad_to_max_output_size, name, sorted_input, canonicalized_coordinates, tile_size)</code></a>
* <a href="../../tf/image/non_max_suppression_with_scores.md"><code>tf.image.non_max_suppression_with_scores(boxes, scores, max_output_size, iou_threshold, score_threshold, soft_nms_sigma, name)</code></a>
* <a href="../../tf/image/pad_to_bounding_box.md"><code>tf.image.pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width)</code></a>
* <a href="../../tf/image/per_image_standardization.md"><code>tf.image.per_image_standardization(image)</code></a>
* <a href="../../tf/image/psnr.md"><code>tf.image.psnr(a, b, max_val, name)</code></a>
* <a href="../../tf/image/random_brightness.md"><code>tf.image.random_brightness(image, max_delta, seed)</code></a>
* <a href="../../tf/image/random_contrast.md"><code>tf.image.random_contrast(image, lower, upper, seed)</code></a>
* <a href="../../tf/image/random_crop.md"><code>tf.image.random_crop(value, size, seed, name)</code></a>
* <a href="../../tf/image/random_flip_left_right.md"><code>tf.image.random_flip_left_right(image, seed)</code></a>
* <a href="../../tf/image/random_flip_up_down.md"><code>tf.image.random_flip_up_down(image, seed)</code></a>
* <a href="../../tf/image/random_hue.md"><code>tf.image.random_hue(image, max_delta, seed)</code></a>
* <a href="../../tf/image/random_jpeg_quality.md"><code>tf.image.random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed)</code></a>
* <a href="../../tf/image/random_saturation.md"><code>tf.image.random_saturation(image, lower, upper, seed)</code></a>
* <a href="../../tf/image/resize.md"><code>tf.image.resize(images, size, method, preserve_aspect_ratio, antialias, name)</code></a>
* <a href="../../tf/image/resize_with_crop_or_pad.md"><code>tf.image.resize_with_crop_or_pad(image, target_height, target_width)</code></a>
* <a href="../../tf/image/resize_with_pad.md"><code>tf.image.resize_with_pad(image, target_height, target_width, method, antialias)</code></a>
* <a href="../../tf/image/rgb_to_grayscale.md"><code>tf.image.rgb_to_grayscale(images, name)</code></a>
* <a href="../../tf/image/rgb_to_hsv.md"><code>tf.image.rgb_to_hsv(images, name)</code></a>
* <a href="../../tf/image/rgb_to_yiq.md"><code>tf.image.rgb_to_yiq(images)</code></a>
* <a href="../../tf/image/rgb_to_yuv.md"><code>tf.image.rgb_to_yuv(images)</code></a>
* <a href="../../tf/image/rot90.md"><code>tf.image.rot90(image, k, name)</code></a>
* <a href="../../tf/image/sample_distorted_bounding_box.md"><code>tf.image.sample_distorted_bounding_box(image_size, bounding_boxes, seed, min_object_covered, aspect_ratio_range, area_range, max_attempts, use_image_if_no_bounding_boxes, name)</code></a>
* <a href="../../tf/image/sobel_edges.md"><code>tf.image.sobel_edges(image)</code></a>
* <a href="../../tf/image/ssim.md"><code>tf.image.ssim(img1, img2, max_val, filter_size, filter_sigma, k1, k2)</code></a>
* <a href="../../tf/image/ssim_multiscale.md"><code>tf.image.ssim_multiscale(img1, img2, max_val, power_factors, filter_size, filter_sigma, k1, k2)</code></a>
* <a href="../../tf/image/stateless_random_brightness.md"><code>tf.image.stateless_random_brightness(image, max_delta, seed)</code></a>
* <a href="../../tf/image/stateless_random_contrast.md"><code>tf.image.stateless_random_contrast(image, lower, upper, seed)</code></a>
* <a href="../../tf/image/stateless_random_crop.md"><code>tf.image.stateless_random_crop(value, size, seed, name)</code></a>
* <a href="../../tf/image/stateless_random_flip_left_right.md"><code>tf.image.stateless_random_flip_left_right(image, seed)</code></a>
* <a href="../../tf/image/stateless_random_flip_up_down.md"><code>tf.image.stateless_random_flip_up_down(image, seed)</code></a>
* <a href="../../tf/image/stateless_random_hue.md"><code>tf.image.stateless_random_hue(image, max_delta, seed)</code></a>
* <a href="../../tf/image/stateless_random_jpeg_quality.md"><code>tf.image.stateless_random_jpeg_quality(image, min_jpeg_quality, max_jpeg_quality, seed)</code></a>
* <a href="../../tf/image/stateless_random_saturation.md"><code>tf.image.stateless_random_saturation(image, lower, upper, seed)</code></a>
* <a href="../../tf/image/stateless_sample_distorted_bounding_box.md"><code>tf.image.stateless_sample_distorted_bounding_box(image_size, bounding_boxes, seed, min_object_covered, aspect_ratio_range, area_range, max_attempts, use_image_if_no_bounding_boxes, name)</code></a>
* <a href="../../tf/image/total_variation.md"><code>tf.image.total_variation(images, name)</code></a>
* <a href="../../tf/image/transpose.md"><code>tf.image.transpose(image, name)</code></a>
* <a href="../../tf/image/yiq_to_rgb.md"><code>tf.image.yiq_to_rgb(images)</code></a>
* <a href="../../tf/image/yuv_to_rgb.md"><code>tf.image.yuv_to_rgb(images)</code></a>
* <a href="../../tf/io/decode_and_crop_jpeg.md"><code>tf.io.decode_and_crop_jpeg(contents, crop_window, channels, ratio, fancy_upscaling, try_recover_truncated, acceptable_fraction, dct_method, name)</code></a>
* <a href="../../tf/io/decode_base64.md"><code>tf.io.decode_base64(input, name)</code></a>
* <a href="../../tf/io/decode_bmp.md"><code>tf.io.decode_bmp(contents, channels, name)</code></a>
* <a href="../../tf/io/decode_compressed.md"><code>tf.io.decode_compressed(bytes, compression_type, name)</code></a>
* <a href="../../tf/io/decode_csv.md"><code>tf.io.decode_csv(records, record_defaults, field_delim, use_quote_delim, na_value, select_cols, name)</code></a>
* <a href="../../tf/io/decode_gif.md"><code>tf.io.decode_gif(contents, name)</code></a>
* <a href="../../tf/io/decode_image.md"><code>tf.io.decode_image(contents, channels, dtype, name, expand_animations)</code></a>
* <a href="../../tf/io/decode_jpeg.md"><code>tf.io.decode_jpeg(contents, channels, ratio, fancy_upscaling, try_recover_truncated, acceptable_fraction, dct_method, name)</code></a>
* <a href="../../tf/io/decode_png.md"><code>tf.io.decode_png(contents, channels, dtype, name)</code></a>
* <a href="../../tf/io/decode_proto.md"><code>tf.io.decode_proto(bytes, message_type, field_names, output_types, descriptor_source, message_format, sanitize, name)</code></a>
* <a href="../../tf/io/decode_raw.md"><code>tf.io.decode_raw(input_bytes, out_type, little_endian, fixed_length, name)</code></a>
* <a href="../../tf/io/deserialize_many_sparse.md"><code>tf.io.deserialize_many_sparse(serialized_sparse, dtype, rank, name)</code></a>
* <a href="../../tf/io/encode_base64.md"><code>tf.io.encode_base64(input, pad, name)</code></a>
* <a href="../../tf/io/encode_jpeg.md"><code>tf.io.encode_jpeg(image, format, quality, progressive, optimize_size, chroma_downsampling, density_unit, x_density, y_density, xmp_metadata, name)</code></a>
* <a href="../../tf/io/encode_png.md"><code>tf.io.encode_png(image, compression, name)</code></a>
* <a href="../../tf/io/encode_proto.md"><code>tf.io.encode_proto(sizes, values, field_names, message_type, descriptor_source, name)</code></a>
* <a href="../../tf/io/extract_jpeg_shape.md"><code>tf.io.extract_jpeg_shape(contents, output_type, name)</code></a>
* <a href="../../tf/io/matching_files.md"><code>tf.io.matching_files(pattern, name)</code></a>
* <a href="../../tf/io/parse_example.md"><code>tf.io.parse_example(serialized, features, example_names, name)</code></a>
* <a href="../../tf/io/parse_sequence_example.md"><code>tf.io.parse_sequence_example(serialized, context_features, sequence_features, example_names, name)</code></a>
* <a href="../../tf/io/parse_single_example.md"><code>tf.io.parse_single_example(serialized, features, example_names, name)</code></a>
* <a href="../../tf/io/parse_single_sequence_example.md"><code>tf.io.parse_single_sequence_example(serialized, context_features, sequence_features, example_name, name)</code></a>
* <a href="../../tf/io/parse_tensor.md"><code>tf.io.parse_tensor(serialized, out_type, name)</code></a>
* <a href="../../tf/io/serialize_many_sparse.md"><code>tf.io.serialize_many_sparse(sp_input, out_type, name)</code></a>
* <a href="../../tf/io/serialize_sparse.md"><code>tf.io.serialize_sparse(sp_input, out_type, name)</code></a>
* <a href="../../tf/io/write_file.md"><code>tf.io.write_file(filename, contents, name)</code></a>
* <a href="../../tf/linalg/adjoint.md"><code>tf.linalg.adjoint(matrix, name)</code></a>
* <a href="../../tf/linalg/band_part.md"><code>tf.linalg.band_part(input, num_lower, num_upper, name)</code></a>
* <a href="../../tf/linalg/cholesky.md"><code>tf.linalg.cholesky(input, name)</code></a>
* <a href="../../tf/linalg/cholesky_solve.md"><code>tf.linalg.cholesky_solve(chol, rhs, name)</code></a>
* <a href="../../tf/linalg/cross.md"><code>tf.linalg.cross(a, b, name)</code></a>
* <a href="../../tf/linalg/det.md"><code>tf.linalg.det(input, name)</code></a>
* <a href="../../tf/linalg/diag.md"><code>tf.linalg.diag(diagonal, name, k, num_rows, num_cols, padding_value, align)</code></a>
* <a href="../../tf/linalg/diag_part.md"><code>tf.linalg.diag_part(input, name, k, padding_value, align)</code></a>
* <a href="../../tf/linalg/eig.md"><code>tf.linalg.eig(tensor, name)</code></a>
* <a href="../../tf/linalg/eigh.md"><code>tf.linalg.eigh(tensor, name)</code></a>
* <a href="../../tf/linalg/eigh_tridiagonal.md"><code>tf.linalg.eigh_tridiagonal(alpha, beta, eigvals_only, select, select_range, tol, name)</code></a>
* <a href="../../tf/linalg/eigvals.md"><code>tf.linalg.eigvals(tensor, name)</code></a>
* <a href="../../tf/linalg/eigvalsh.md"><code>tf.linalg.eigvalsh(tensor, name)</code></a>
* <a href="../../tf/linalg/experimental/conjugate_gradient.md"><code>tf.linalg.experimental.conjugate_gradient(operator, rhs, preconditioner, x, tol, max_iter, name)</code></a>
* <a href="../../tf/linalg/expm.md"><code>tf.linalg.expm(input, name)</code></a>
* <a href="../../tf/linalg/global_norm.md"><code>tf.linalg.global_norm(t_list, name)</code></a>
* <a href="../../tf/linalg/inv.md"><code>tf.linalg.inv(input, adjoint, name)</code></a>
* <a href="../../tf/linalg/logdet.md"><code>tf.linalg.logdet(matrix, name)</code></a>
* <a href="../../tf/linalg/logm.md"><code>tf.linalg.logm(input, name)</code></a>
* <a href="../../tf/linalg/lstsq.md"><code>tf.linalg.lstsq(matrix, rhs, l2_regularizer, fast, name)</code></a>
* <a href="../../tf/linalg/lu.md"><code>tf.linalg.lu(input, output_idx_type, name)</code></a>
* <a href="../../tf/linalg/lu_matrix_inverse.md"><code>tf.linalg.lu_matrix_inverse(lower_upper, perm, validate_args, name)</code></a>
* <a href="../../tf/linalg/lu_reconstruct.md"><code>tf.linalg.lu_reconstruct(lower_upper, perm, validate_args, name)</code></a>
* <a href="../../tf/linalg/lu_solve.md"><code>tf.linalg.lu_solve(lower_upper, perm, rhs, validate_args, name)</code></a>
* <a href="../../tf/linalg/matmul.md"><code>tf.linalg.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b, a_is_sparse, b_is_sparse, output_type, name)</code></a>
* <a href="../../tf/linalg/matrix_rank.md"><code>tf.linalg.matrix_rank(a, tol, validate_args, name)</code></a>
* <a href="../../tf/linalg/matrix_transpose.md"><code>tf.linalg.matrix_transpose(a, name, conjugate)</code></a>
* <a href="../../tf/linalg/matvec.md"><code>tf.linalg.matvec(a, b, transpose_a, adjoint_a, a_is_sparse, b_is_sparse, name)</code></a>
* <a href="../../tf/linalg/normalize.md"><code>tf.linalg.normalize(tensor, ord, axis, name)</code></a>
* <a href="../../tf/linalg/pinv.md"><code>tf.linalg.pinv(a, rcond, validate_args, name)</code></a>
* <a href="../../tf/linalg/qr.md"><code>tf.linalg.qr(input, full_matrices, name)</code></a>
* <a href="../../tf/linalg/set_diag.md"><code>tf.linalg.set_diag(input, diagonal, name, k, align)</code></a>
* <a href="../../tf/linalg/slogdet.md"><code>tf.linalg.slogdet(input, name)</code></a>
* <a href="../../tf/linalg/solve.md"><code>tf.linalg.solve(matrix, rhs, adjoint, name)</code></a>
* <a href="../../tf/linalg/sqrtm.md"><code>tf.linalg.sqrtm(input, name)</code></a>
* <a href="../../tf/linalg/svd.md"><code>tf.linalg.svd(tensor, full_matrices, compute_uv, name)</code></a>
* <a href="../../tf/linalg/tensor_diag.md"><code>tf.linalg.tensor_diag(diagonal, name)</code></a>
* <a href="../../tf/linalg/tensor_diag_part.md"><code>tf.linalg.tensor_diag_part(input, name)</code></a>
* <a href="../../tf/linalg/trace.md"><code>tf.linalg.trace(x, name)</code></a>
* <a href="../../tf/linalg/triangular_solve.md"><code>tf.linalg.triangular_solve(matrix, rhs, lower, adjoint, name)</code></a>
* <a href="../../tf/linalg/tridiagonal_matmul.md"><code>tf.linalg.tridiagonal_matmul(diagonals, rhs, diagonals_format, name)</code></a>
* <a href="../../tf/linalg/tridiagonal_solve.md"><code>tf.linalg.tridiagonal_solve(diagonals, rhs, diagonals_format, transpose_rhs, conjugate_rhs, name, partial_pivoting, perturb_singular)</code></a>
* <a href="../../tf/linspace.md"><code>tf.linspace(start, stop, num, name, axis)</code></a>
* <a href="../../tf/math/abs.md"><code>tf.math.abs(x, name)</code></a>
* <a href="../../tf/math/accumulate_n.md"><code>tf.math.accumulate_n(inputs, shape, tensor_dtype, name)</code></a>
* <a href="../../tf/math/acos.md"><code>tf.math.acos(x, name)</code></a>
* <a href="../../tf/math/acosh.md"><code>tf.math.acosh(x, name)</code></a>
* <a href="../../tf/math/add.md"><code>tf.math.add(x, y, name)</code></a>
* <a href="../../tf/math/add_n.md"><code>tf.math.add_n(inputs, name)</code></a>
* <a href="../../tf/math/angle.md"><code>tf.math.angle(input, name)</code></a>
* <a href="../../tf/math/argmax.md"><code>tf.math.argmax(input, axis, output_type, name)</code></a>
* <a href="../../tf/math/argmin.md"><code>tf.math.argmin(input, axis, output_type, name)</code></a>
* <a href="../../tf/math/asin.md"><code>tf.math.asin(x, name)</code></a>
* <a href="../../tf/math/asinh.md"><code>tf.math.asinh(x, name)</code></a>
* <a href="../../tf/math/atan.md"><code>tf.math.atan(x, name)</code></a>
* <a href="../../tf/math/atan2.md"><code>tf.math.atan2(y, x, name)</code></a>
* <a href="../../tf/math/atanh.md"><code>tf.math.atanh(x, name)</code></a>
* <a href="../../tf/math/bessel_i0.md"><code>tf.math.bessel_i0(x, name)</code></a>
* <a href="../../tf/math/bessel_i0e.md"><code>tf.math.bessel_i0e(x, name)</code></a>
* <a href="../../tf/math/bessel_i1.md"><code>tf.math.bessel_i1(x, name)</code></a>
* <a href="../../tf/math/bessel_i1e.md"><code>tf.math.bessel_i1e(x, name)</code></a>
* <a href="../../tf/math/betainc.md"><code>tf.math.betainc(a, b, x, name)</code></a>
* <a href="../../tf/math/ceil.md"><code>tf.math.ceil(x, name)</code></a>
* <a href="../../tf/math/confusion_matrix.md"><code>tf.math.confusion_matrix(labels, predictions, num_classes, weights, dtype, name)</code></a>
* <a href="../../tf/math/conj.md"><code>tf.math.conj(x, name)</code></a>
* <a href="../../tf/math/cos.md"><code>tf.math.cos(x, name)</code></a>
* <a href="../../tf/math/cosh.md"><code>tf.math.cosh(x, name)</code></a>
* <a href="../../tf/math/count_nonzero.md"><code>tf.math.count_nonzero(input, axis, keepdims, dtype, name)</code></a>
* <a href="../../tf/math/cumprod.md"><code>tf.math.cumprod(x, axis, exclusive, reverse, name)</code></a>
* <a href="../../tf/math/cumsum.md"><code>tf.math.cumsum(x, axis, exclusive, reverse, name)</code></a>
* <a href="../../tf/math/cumulative_logsumexp.md"><code>tf.math.cumulative_logsumexp(x, axis, exclusive, reverse, name)</code></a>
* <a href="../../tf/math/digamma.md"><code>tf.math.digamma(x, name)</code></a>
* <a href="../../tf/math/divide.md"><code>tf.math.divide(x, y, name)</code></a>
* <a href="../../tf/math/divide_no_nan.md"><code>tf.math.divide_no_nan(x, y, name)</code></a>
* <a href="../../tf/math/equal.md"><code>tf.math.equal(x, y, name)</code></a>
* <a href="../../tf/math/erf.md"><code>tf.math.erf(x, name)</code></a>
* <a href="../../tf/math/erfc.md"><code>tf.math.erfc(x, name)</code></a>
* <a href="../../tf/math/erfcinv.md"><code>tf.math.erfcinv(x, name)</code></a>
* <a href="../../tf/math/erfinv.md"><code>tf.math.erfinv(x, name)</code></a>
* <a href="../../tf/math/exp.md"><code>tf.math.exp(x, name)</code></a>
* <a href="../../tf/math/expm1.md"><code>tf.math.expm1(x, name)</code></a>
* <a href="../../tf/math/floor.md"><code>tf.math.floor(x, name)</code></a>
* <a href="../../tf/math/floordiv.md"><code>tf.math.floordiv(x, y, name)</code></a>
* <a href="../../tf/math/floormod.md"><code>tf.math.floormod(x, y, name)</code></a>
* <a href="../../tf/math/greater.md"><code>tf.math.greater(x, y, name)</code></a>
* <a href="../../tf/math/greater_equal.md"><code>tf.math.greater_equal(x, y, name)</code></a>
* <a href="../../tf/math/igamma.md"><code>tf.math.igamma(a, x, name)</code></a>
* <a href="../../tf/math/igammac.md"><code>tf.math.igammac(a, x, name)</code></a>
* <a href="../../tf/math/imag.md"><code>tf.math.imag(input, name)</code></a>
* <a href="../../tf/math/in_top_k.md"><code>tf.math.in_top_k(targets, predictions, k, name)</code></a>
* <a href="../../tf/math/invert_permutation.md"><code>tf.math.invert_permutation(x, name)</code></a>
* <a href="../../tf/math/is_finite.md"><code>tf.math.is_finite(x, name)</code></a>
* <a href="../../tf/math/is_inf.md"><code>tf.math.is_inf(x, name)</code></a>
* <a href="../../tf/math/is_nan.md"><code>tf.math.is_nan(x, name)</code></a>
* <a href="../../tf/math/is_non_decreasing.md"><code>tf.math.is_non_decreasing(x, name)</code></a>
* <a href="../../tf/math/is_strictly_increasing.md"><code>tf.math.is_strictly_increasing(x, name)</code></a>
* <a href="../../tf/math/l2_normalize.md"><code>tf.math.l2_normalize(x, axis, epsilon, name, dim)</code></a>
* <a href="../../tf/math/lbeta.md"><code>tf.math.lbeta(x, name)</code></a>
* <a href="../../tf/math/less.md"><code>tf.math.less(x, y, name)</code></a>
* <a href="../../tf/math/less_equal.md"><code>tf.math.less_equal(x, y, name)</code></a>
* <a href="../../tf/math/lgamma.md"><code>tf.math.lgamma(x, name)</code></a>
* <a href="../../tf/math/log.md"><code>tf.math.log(x, name)</code></a>
* <a href="../../tf/math/log1p.md"><code>tf.math.log1p(x, name)</code></a>
* <a href="../../tf/math/log_sigmoid.md"><code>tf.math.log_sigmoid(x, name)</code></a>
* <a href="../../tf/math/logical_and.md"><code>tf.math.logical_and(x, y, name)</code></a>
* <a href="../../tf/math/logical_not.md"><code>tf.math.logical_not(x, name)</code></a>
* <a href="../../tf/math/logical_or.md"><code>tf.math.logical_or(x, y, name)</code></a>
* <a href="../../tf/math/logical_xor.md"><code>tf.math.logical_xor(x, y, name)</code></a>
* <a href="../../tf/math/maximum.md"><code>tf.math.maximum(x, y, name)</code></a>
* <a href="../../tf/math/minimum.md"><code>tf.math.minimum(x, y, name)</code></a>
* <a href="../../tf/math/multiply.md"><code>tf.math.multiply(x, y, name)</code></a>
* <a href="../../tf/math/multiply_no_nan.md"><code>tf.math.multiply_no_nan(x, y, name)</code></a>
* <a href="../../tf/math/ndtri.md"><code>tf.math.ndtri(x, name)</code></a>
* <a href="../../tf/math/negative.md"><code>tf.math.negative(x, name)</code></a>
* <a href="../../tf/math/nextafter.md"><code>tf.math.nextafter(x1, x2, name)</code></a>
* <a href="../../tf/math/not_equal.md"><code>tf.math.not_equal(x, y, name)</code></a>
* <a href="../../tf/math/polygamma.md"><code>tf.math.polygamma(a, x, name)</code></a>
* <a href="../../tf/math/polyval.md"><code>tf.math.polyval(coeffs, x, name)</code></a>
* <a href="../../tf/math/pow.md"><code>tf.math.pow(x, y, name)</code></a>
* <a href="../../tf/math/real.md"><code>tf.math.real(input, name)</code></a>
* <a href="../../tf/math/reciprocal.md"><code>tf.math.reciprocal(x, name)</code></a>
* <a href="../../tf/math/reciprocal_no_nan.md"><code>tf.math.reciprocal_no_nan(x, name)</code></a>
* <a href="../../tf/math/reduce_all.md"><code>tf.math.reduce_all(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_any.md"><code>tf.math.reduce_any(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_euclidean_norm.md"><code>tf.math.reduce_euclidean_norm(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_logsumexp.md"><code>tf.math.reduce_logsumexp(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_max.md"><code>tf.math.reduce_max(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_mean.md"><code>tf.math.reduce_mean(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_min.md"><code>tf.math.reduce_min(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_prod.md"><code>tf.math.reduce_prod(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_std.md"><code>tf.math.reduce_std(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_sum.md"><code>tf.math.reduce_sum(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/reduce_variance.md"><code>tf.math.reduce_variance(input_tensor, axis, keepdims, name)</code></a>
* <a href="../../tf/math/rint.md"><code>tf.math.rint(x, name)</code></a>
* <a href="../../tf/math/round.md"><code>tf.math.round(x, name)</code></a>
* <a href="../../tf/math/rsqrt.md"><code>tf.math.rsqrt(x, name)</code></a>
* <a href="../../tf/math/scalar_mul.md"><code>tf.math.scalar_mul(scalar, x, name)</code></a>
* <a href="../../tf/math/segment_max.md"><code>tf.math.segment_max(data, segment_ids, name)</code></a>
* <a href="../../tf/math/segment_mean.md"><code>tf.math.segment_mean(data, segment_ids, name)</code></a>
* <a href="../../tf/math/segment_min.md"><code>tf.math.segment_min(data, segment_ids, name)</code></a>
* <a href="../../tf/math/segment_prod.md"><code>tf.math.segment_prod(data, segment_ids, name)</code></a>
* <a href="../../tf/math/segment_sum.md"><code>tf.math.segment_sum(data, segment_ids, name)</code></a>
* <a href="../../tf/math/sigmoid.md"><code>tf.math.sigmoid(x, name)</code></a>
* <a href="../../tf/math/sign.md"><code>tf.math.sign(x, name)</code></a>
* <a href="../../tf/math/sin.md"><code>tf.math.sin(x, name)</code></a>
* <a href="../../tf/math/sinh.md"><code>tf.math.sinh(x, name)</code></a>
* <a href="../../tf/math/sobol_sample.md"><code>tf.math.sobol_sample(dim, num_results, skip, dtype, name)</code></a>
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
* <a href="../../tf/math/squared_difference.md"><code>tf.math.squared_difference(x, y, name)</code></a>
* <a href="../../tf/math/subtract.md"><code>tf.math.subtract(x, y, name)</code></a>
* <a href="../../tf/math/tan.md"><code>tf.math.tan(x, name)</code></a>
* <a href="../../tf/math/tanh.md"><code>tf.math.tanh(x, name)</code></a>
* <a href="../../tf/math/top_k.md"><code>tf.math.top_k(input, k, sorted, name)</code></a>
* <a href="../../tf/math/truediv.md"><code>tf.math.truediv(x, y, name)</code></a>
* <a href="../../tf/math/unsorted_segment_max.md"><code>tf.math.unsorted_segment_max(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/unsorted_segment_mean.md"><code>tf.math.unsorted_segment_mean(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/unsorted_segment_min.md"><code>tf.math.unsorted_segment_min(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/unsorted_segment_prod.md"><code>tf.math.unsorted_segment_prod(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/unsorted_segment_sqrt_n.md"><code>tf.math.unsorted_segment_sqrt_n(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/unsorted_segment_sum.md"><code>tf.math.unsorted_segment_sum(data, segment_ids, num_segments, name)</code></a>
* <a href="../../tf/math/xdivy.md"><code>tf.math.xdivy(x, y, name)</code></a>
* <a href="../../tf/math/xlog1py.md"><code>tf.math.xlog1py(x, y, name)</code></a>
* <a href="../../tf/math/xlogy.md"><code>tf.math.xlogy(x, y, name)</code></a>
* <a href="../../tf/math/zero_fraction.md"><code>tf.math.zero_fraction(value, name)</code></a>
* <a href="../../tf/math/zeta.md"><code>tf.math.zeta(x, q, name)</code></a>
* <a href="../../tf/nn/atrous_conv2d.md"><code>tf.nn.atrous_conv2d(value, filters, rate, padding, name)</code></a>
* <a href="../../tf/nn/atrous_conv2d_transpose.md"><code>tf.nn.atrous_conv2d_transpose(value, filters, output_shape, rate, padding, name)</code></a>
* <a href="../../tf/nn/avg_pool.md"><code>tf.nn.avg_pool(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/avg_pool1d.md"><code>tf.nn.avg_pool1d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/avg_pool2d.md"><code>tf.nn.avg_pool2d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/avg_pool3d.md"><code>tf.nn.avg_pool3d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/batch_norm_with_global_normalization.md"><code>tf.nn.batch_norm_with_global_normalization(input, mean, variance, beta, gamma, variance_epsilon, scale_after_normalization, name)</code></a>
* <a href="../../tf/nn/batch_normalization.md"><code>tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name)</code></a>
* <a href="../../tf/nn/bias_add.md"><code>tf.nn.bias_add(value, bias, data_format, name)</code></a>
* <a href="../../tf/nn/collapse_repeated.md"><code>tf.nn.collapse_repeated(labels, seq_length, name)</code></a>
* <a href="../../tf/nn/compute_accidental_hits.md"><code>tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed, name)</code></a>
* <a href="../../tf/nn/compute_average_loss.md"><code>tf.nn.compute_average_loss(per_example_loss, sample_weight, global_batch_size)</code></a>
* <a href="../../tf/nn/conv1d.md"><code>tf.nn.conv1d(input, filters, stride, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv1d_transpose.md"><code>tf.nn.conv1d_transpose(input, filters, output_shape, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv2d.md"><code>tf.nn.conv2d(input, filters, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv2d_transpose.md"><code>tf.nn.conv2d_transpose(input, filters, output_shape, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv3d.md"><code>tf.nn.conv3d(input, filters, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv3d_transpose.md"><code>tf.nn.conv3d_transpose(input, filters, output_shape, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/conv_transpose.md"><code>tf.nn.conv_transpose(input, filters, output_shape, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/convolution.md"><code>tf.nn.convolution(input, filters, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/crelu.md"><code>tf.nn.crelu(features, axis, name)</code></a>
* <a href="../../tf/nn/ctc_beam_search_decoder.md"><code>tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width, top_paths)</code></a>
* <a href="../../tf/nn/ctc_greedy_decoder.md"><code>tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated, blank_index)</code></a>
* <a href="../../tf/nn/ctc_loss.md"><code>tf.nn.ctc_loss(labels, logits, label_length, logit_length, logits_time_major, unique, blank_index, name)</code></a>
* <a href="../../tf/nn/ctc_unique_labels.md"><code>tf.nn.ctc_unique_labels(labels, name)</code></a>
* <a href="../../tf/nn/depth_to_space.md"><code>tf.nn.depth_to_space(input, block_size, data_format, name)</code></a>
* <a href="../../tf/nn/depthwise_conv2d.md"><code>tf.nn.depthwise_conv2d(input, filter, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/depthwise_conv2d_backprop_filter.md"><code>tf.nn.depthwise_conv2d_backprop_filter(input, filter_sizes, out_backprop, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/depthwise_conv2d_backprop_input.md"><code>tf.nn.depthwise_conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/dilation2d.md"><code>tf.nn.dilation2d(input, filters, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/dropout.md"><code>tf.nn.dropout(x, rate, noise_shape, seed, name)</code></a>
* <a href="../../tf/nn/elu.md"><code>tf.nn.elu(features, name)</code></a>
* <a href="../../tf/nn/embedding_lookup.md"><code>tf.nn.embedding_lookup(params, ids, max_norm, name)</code></a>
* <a href="../../tf/nn/embedding_lookup_sparse.md"><code>tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, combiner, max_norm, name)</code></a>
* <a href="../../tf/nn/erosion2d.md"><code>tf.nn.erosion2d(value, filters, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/experimental/stateless_dropout.md"><code>tf.nn.experimental.stateless_dropout(x, rate, seed, rng_alg, noise_shape, name)</code></a>
* <a href="../../tf/nn/fractional_avg_pool.md"><code>tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random, overlapping, seed, name)</code></a>
* <a href="../../tf/nn/fractional_max_pool.md"><code>tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random, overlapping, seed, name)</code></a>
* <a href="../../tf/nn/gelu.md"><code>tf.nn.gelu(features, approximate, name)</code></a>
* <a href="../../tf/nn/isotonic_regression.md"><code>tf.nn.isotonic_regression(inputs, decreasing, axis)</code></a>
* <a href="../../tf/nn/l2_loss.md"><code>tf.nn.l2_loss(t, name)</code></a>
* <a href="../../tf/nn/leaky_relu.md"><code>tf.nn.leaky_relu(features, alpha, name)</code></a>
* <a href="../../tf/nn/local_response_normalization.md"><code>tf.nn.local_response_normalization(input, depth_radius, bias, alpha, beta, name)</code></a>
* <a href="../../tf/nn/log_poisson_loss.md"><code>tf.nn.log_poisson_loss(targets, log_input, compute_full_loss, name)</code></a>
* <a href="../../tf/nn/log_softmax.md"><code>tf.nn.log_softmax(logits, axis, name)</code></a>
* <a href="../../tf/nn/max_pool.md"><code>tf.nn.max_pool(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/max_pool1d.md"><code>tf.nn.max_pool1d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/max_pool2d.md"><code>tf.nn.max_pool2d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/max_pool3d.md"><code>tf.nn.max_pool3d(input, ksize, strides, padding, data_format, name)</code></a>
* <a href="../../tf/nn/max_pool_with_argmax.md"><code>tf.nn.max_pool_with_argmax(input, ksize, strides, padding, data_format, output_dtype, include_batch_in_index, name)</code></a>
* <a href="../../tf/nn/moments.md"><code>tf.nn.moments(x, axes, shift, keepdims, name)</code></a>
* <a href="../../tf/nn/nce_loss.md"><code>tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values, remove_accidental_hits, name)</code></a>
* <a href="../../tf/nn/normalize_moments.md"><code>tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift, name)</code></a>
* <a href="../../tf/nn/pool.md"><code>tf.nn.pool(input, window_shape, pooling_type, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/relu.md"><code>tf.nn.relu(features, name)</code></a>
* <a href="../../tf/nn/relu6.md"><code>tf.nn.relu6(features, name)</code></a>
* <a href="../../tf/nn/safe_embedding_lookup_sparse.md"><code>tf.nn.safe_embedding_lookup_sparse(embedding_weights, sparse_ids, sparse_weights, combiner, default_id, max_norm, name)</code></a>
* <a href="../../tf/nn/sampled_softmax_loss.md"><code>tf.nn.sampled_softmax_loss(weights, biases, labels, inputs, num_sampled, num_classes, num_true, sampled_values, remove_accidental_hits, seed, name)</code></a>
* <a href="../../tf/nn/scale_regularization_loss.md"><code>tf.nn.scale_regularization_loss(regularization_loss)</code></a>
* <a href="../../tf/nn/selu.md"><code>tf.nn.selu(features, name)</code></a>
* <a href="../../tf/nn/separable_conv2d.md"><code>tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, data_format, dilations, name)</code></a>
* <a href="../../tf/nn/sigmoid_cross_entropy_with_logits.md"><code>tf.nn.sigmoid_cross_entropy_with_logits(labels, logits, name)</code></a>
* <a href="../../tf/nn/silu.md"><code>tf.nn.silu(features, beta)</code></a>
* <a href="../../tf/nn/softmax.md"><code>tf.nn.softmax(logits, axis, name)</code></a>
* <a href="../../tf/nn/softmax_cross_entropy_with_logits.md"><code>tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis, name)</code></a>
* <a href="../../tf/nn/softsign.md"><code>tf.nn.softsign(features, name)</code></a>
* <a href="../../tf/nn/space_to_depth.md"><code>tf.nn.space_to_depth(input, block_size, data_format, name)</code></a>
* <a href="../../tf/nn/sparse_softmax_cross_entropy_with_logits.md"><code>tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name)</code></a>
* <a href="../../tf/nn/sufficient_statistics.md"><code>tf.nn.sufficient_statistics(x, axes, shift, keepdims, name)</code></a>
* <a href="../../tf/nn/weighted_cross_entropy_with_logits.md"><code>tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight, name)</code></a>
* <a href="../../tf/nn/weighted_moments.md"><code>tf.nn.weighted_moments(x, axes, frequency_weights, keepdims, name)</code></a>
* <a href="../../tf/nn/with_space_to_batch.md"><code>tf.nn.with_space_to_batch(input, dilation_rate, padding, op, filter_shape, spatial_dims, data_format)</code></a>
* <a href="../../tf/no_op.md"><code>tf.no_op(name)</code></a>
* <a href="../../tf/norm.md"><code>tf.norm(tensor, ord, axis, keepdims, name)</code></a>
* <a href="../../tf/numpy_function.md"><code>tf.numpy_function(func, inp, Tout, stateful, name)</code></a>
* <a href="../../tf/one_hot.md"><code>tf.one_hot(indices, depth, on_value, off_value, axis, dtype, name)</code></a>
* <a href="../../tf/ones.md"><code>tf.ones(shape, dtype, name)</code></a>
* <a href="../../tf/ones_like.md"><code>tf.ones_like(input, dtype, name)</code></a>
* <a href="../../tf/pad.md"><code>tf.pad(tensor, paddings, mode, constant_values, name)</code></a>
* <a href="../../tf/parallel_stack.md"><code>tf.parallel_stack(values, name)</code></a>
* <a href="../../tf/py_function.md"><code>tf.py_function(func, inp, Tout, name)</code></a>
* <a href="../../tf/quantization/dequantize.md"><code>tf.quantization.dequantize(input, min_range, max_range, mode, name, axis, narrow_range, dtype)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_args.md"><code>tf.quantization.fake_quant_with_min_max_args(inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_args_gradient.md"><code>tf.quantization.fake_quant_with_min_max_args_gradient(gradients, inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars.md"><code>tf.quantization.fake_quant_with_min_max_vars(inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_gradient.md"><code>tf.quantization.fake_quant_with_min_max_vars_gradient(gradients, inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_per_channel.md"><code>tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/fake_quant_with_min_max_vars_per_channel_gradient.md"><code>tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient(gradients, inputs, min, max, num_bits, narrow_range, name)</code></a>
* <a href="../../tf/quantization/quantize.md"><code>tf.quantization.quantize(input, min_range, max_range, T, mode, round_mode, name, narrow_range, axis, ensure_minimum_range)</code></a>
* <a href="../../tf/quantization/quantize_and_dequantize.md"><code>tf.quantization.quantize_and_dequantize(input, input_min, input_max, signed_input, num_bits, range_given, round_mode, name, narrow_range, axis)</code></a>
* <a href="../../tf/quantization/quantize_and_dequantize_v2.md"><code>tf.quantization.quantize_and_dequantize_v2(input, input_min, input_max, signed_input, num_bits, range_given, round_mode, name, narrow_range, axis)</code></a>
* <a href="../../tf/quantization/quantized_concat.md"><code>tf.quantization.quantized_concat(concat_dim, values, input_mins, input_maxes, name)</code></a>
* <a href="../../tf/ragged/boolean_mask.md"><code>tf.ragged.boolean_mask(data, mask, name)</code></a>
* <a href="../../tf/ragged/constant.md"><code>tf.ragged.constant(pylist, dtype, ragged_rank, inner_shape, name, row_splits_dtype)</code></a>
* <a href="../../tf/ragged/cross.md"><code>tf.ragged.cross(inputs, name)</code></a>
* <a href="../../tf/ragged/cross_hashed.md"><code>tf.ragged.cross_hashed(inputs, num_buckets, hash_key, name)</code></a>
* <a href="../../tf/ragged/range.md"><code>tf.ragged.range(starts, limits, deltas, dtype, name, row_splits_dtype)</code></a>
* <a href="../../tf/ragged/row_splits_to_segment_ids.md"><code>tf.ragged.row_splits_to_segment_ids(splits, name, out_type)</code></a>
* <a href="../../tf/ragged/segment_ids_to_row_splits.md"><code>tf.ragged.segment_ids_to_row_splits(segment_ids, num_segments, out_type, name)</code></a>
* <a href="../../tf/ragged/stack.md"><code>tf.ragged.stack(values, axis, name)</code></a>
* <a href="../../tf/ragged/stack_dynamic_partitions.md"><code>tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions, name)</code></a>
* <a href="../../tf/random/categorical.md"><code>tf.random.categorical(logits, num_samples, dtype, seed, name)</code></a>
* <a href="../../tf/random/experimental/index_shuffle.md"><code>tf.random.experimental.index_shuffle(index, seed, max_index)</code></a>
* <a href="../../tf/random/experimental/stateless_fold_in.md"><code>tf.random.experimental.stateless_fold_in(seed, data, alg)</code></a>
* <a href="../../tf/random/experimental/stateless_split.md"><code>tf.random.experimental.stateless_split(seed, num, alg)</code></a>
* <a href="../../tf/random/fixed_unigram_candidate_sampler.md"><code>tf.random.fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, vocab_file, distortion, num_reserved_ids, num_shards, shard, unigrams, seed, name)</code></a>
* <a href="../../tf/random/gamma.md"><code>tf.random.gamma(shape, alpha, beta, dtype, seed, name)</code></a>
* <a href="../../tf/random/learned_unigram_candidate_sampler.md"><code>tf.random.learned_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed, name)</code></a>
* <a href="../../tf/random/log_uniform_candidate_sampler.md"><code>tf.random.log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed, name)</code></a>
* <a href="../../tf/random/normal.md"><code>tf.random.normal(shape, mean, stddev, dtype, seed, name)</code></a>
* <a href="../../tf/random/poisson.md"><code>tf.random.poisson(shape, lam, dtype, seed, name)</code></a>
* <a href="../../tf/random/shuffle.md"><code>tf.random.shuffle(value, seed, name)</code></a>
* <a href="../../tf/random/stateless_binomial.md"><code>tf.random.stateless_binomial(shape, seed, counts, probs, output_dtype, name)</code></a>
* <a href="../../tf/random/stateless_categorical.md"><code>tf.random.stateless_categorical(logits, num_samples, seed, dtype, name)</code></a>
* <a href="../../tf/random/stateless_gamma.md"><code>tf.random.stateless_gamma(shape, seed, alpha, beta, dtype, name)</code></a>
* <a href="../../tf/random/stateless_normal.md"><code>tf.random.stateless_normal(shape, seed, mean, stddev, dtype, name, alg)</code></a>
* <a href="../../tf/random/stateless_parameterized_truncated_normal.md"><code>tf.random.stateless_parameterized_truncated_normal(shape, seed, means, stddevs, minvals, maxvals, name)</code></a>
* <a href="../../tf/random/stateless_poisson.md"><code>tf.random.stateless_poisson(shape, seed, lam, dtype, name)</code></a>
* <a href="../../tf/random/stateless_truncated_normal.md"><code>tf.random.stateless_truncated_normal(shape, seed, mean, stddev, dtype, name, alg)</code></a>
* <a href="../../tf/random/stateless_uniform.md"><code>tf.random.stateless_uniform(shape, seed, minval, maxval, dtype, name, alg)</code></a>
* <a href="../../tf/random/truncated_normal.md"><code>tf.random.truncated_normal(shape, mean, stddev, dtype, seed, name)</code></a>
* <a href="../../tf/random/uniform.md"><code>tf.random.uniform(shape, minval, maxval, dtype, seed, name)</code></a>
* <a href="../../tf/random/uniform_candidate_sampler.md"><code>tf.random.uniform_candidate_sampler(true_classes, num_true, num_sampled, unique, range_max, seed, name)</code></a>
* <a href="../../tf/random_index_shuffle.md"><code>tf.random_index_shuffle(index, seed, max_index, name)</code></a>
* <a href="../../tf/range.md"><code>tf.range(start, limit, delta, dtype, name)</code></a>
* <a href="../../tf/rank.md"><code>tf.rank(input, name)</code></a>
* <a href="../../tf/realdiv.md"><code>tf.realdiv(x, y, name)</code></a>
* <a href="../../tf/repeat.md"><code>tf.repeat(input, repeats, axis, name)</code></a>
* <a href="../../tf/reshape.md"><code>tf.reshape(tensor, shape, name)</code></a>
* <a href="../../tf/reverse.md"><code>tf.reverse(tensor, axis, name)</code></a>
* <a href="../../tf/reverse_sequence.md"><code>tf.reverse_sequence(input, seq_lengths, seq_axis, batch_axis, name)</code></a>
* <a href="../../tf/roll.md"><code>tf.roll(input, shift, axis, name)</code></a>
* <a href="../../tf/scan.md"><code>tf.scan(fn, elems, initializer, parallel_iterations, back_prop, swap_memory, infer_shape, reverse, name)</code></a>
* <a href="../../tf/scatter_nd.md"><code>tf.scatter_nd(indices, updates, shape, name)</code></a>
* <a href="../../tf/searchsorted.md"><code>tf.searchsorted(sorted_sequence, values, side, out_type, name)</code></a>
* <a href="../../tf/sequence_mask.md"><code>tf.sequence_mask(lengths, maxlen, dtype, name)</code></a>
* <a href="../../tf/sets/difference.md"><code>tf.sets.difference(a, b, aminusb, validate_indices)</code></a>
* <a href="../../tf/sets/intersection.md"><code>tf.sets.intersection(a, b, validate_indices)</code></a>
* <a href="../../tf/sets/size.md"><code>tf.sets.size(a, validate_indices)</code></a>
* <a href="../../tf/sets/union.md"><code>tf.sets.union(a, b, validate_indices)</code></a>
* <a href="../../tf/shape.md"><code>tf.shape(input, out_type, name)</code></a>
* <a href="../../tf/shape_n.md"><code>tf.shape_n(input, out_type, name)</code></a>
* <a href="../../tf/signal/dct.md"><code>tf.signal.dct(input, type, n, axis, norm, name)</code></a>
* <a href="../../tf/signal/fft.md"><code>tf.signal.fft(input, name)</code></a>
* <a href="../../tf/signal/fft2d.md"><code>tf.signal.fft2d(input, name)</code></a>
* <a href="../../tf/signal/fft3d.md"><code>tf.signal.fft3d(input, name)</code></a>
* <a href="../../tf/signal/fftshift.md"><code>tf.signal.fftshift(x, axes, name)</code></a>
* <a href="../../tf/signal/frame.md"><code>tf.signal.frame(signal, frame_length, frame_step, pad_end, pad_value, axis, name)</code></a>
* <a href="../../tf/signal/hamming_window.md"><code>tf.signal.hamming_window(window_length, periodic, dtype, name)</code></a>
* <a href="../../tf/signal/hann_window.md"><code>tf.signal.hann_window(window_length, periodic, dtype, name)</code></a>
* <a href="../../tf/signal/idct.md"><code>tf.signal.idct(input, type, n, axis, norm, name)</code></a>
* <a href="../../tf/signal/ifft.md"><code>tf.signal.ifft(input, name)</code></a>
* <a href="../../tf/signal/ifft2d.md"><code>tf.signal.ifft2d(input, name)</code></a>
* <a href="../../tf/signal/ifft3d.md"><code>tf.signal.ifft3d(input, name)</code></a>
* <a href="../../tf/signal/ifftshift.md"><code>tf.signal.ifftshift(x, axes, name)</code></a>
* <a href="../../tf/signal/inverse_mdct.md"><code>tf.signal.inverse_mdct(mdcts, window_fn, norm, name)</code></a>
* <a href="../../tf/signal/inverse_stft.md"><code>tf.signal.inverse_stft(stfts, frame_length, frame_step, fft_length, window_fn, name)</code></a>
* <a href="../../tf/signal/inverse_stft_window_fn.md"><code>tf.signal.inverse_stft_window_fn(frame_step, forward_window_fn, name)</code></a>
* <a href="../../tf/signal/irfft.md"><code>tf.signal.irfft(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/irfft2d.md"><code>tf.signal.irfft2d(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/irfft3d.md"><code>tf.signal.irfft3d(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/kaiser_bessel_derived_window.md"><code>tf.signal.kaiser_bessel_derived_window(window_length, beta, dtype, name)</code></a>
* <a href="../../tf/signal/kaiser_window.md"><code>tf.signal.kaiser_window(window_length, beta, dtype, name)</code></a>
* <a href="../../tf/signal/linear_to_mel_weight_matrix.md"><code>tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz, dtype, name)</code></a>
* <a href="../../tf/signal/mdct.md"><code>tf.signal.mdct(signals, frame_length, window_fn, pad_end, norm, name)</code></a>
* <a href="../../tf/signal/mfccs_from_log_mel_spectrograms.md"><code>tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms, name)</code></a>
* <a href="../../tf/signal/overlap_and_add.md"><code>tf.signal.overlap_and_add(signal, frame_step, name)</code></a>
* <a href="../../tf/signal/rfft.md"><code>tf.signal.rfft(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/rfft2d.md"><code>tf.signal.rfft2d(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/rfft3d.md"><code>tf.signal.rfft3d(input_tensor, fft_length, name)</code></a>
* <a href="../../tf/signal/stft.md"><code>tf.signal.stft(signals, frame_length, frame_step, fft_length, window_fn, pad_end, name)</code></a>
* <a href="../../tf/signal/vorbis_window.md"><code>tf.signal.vorbis_window(window_length, dtype, name)</code></a>
* <a href="../../tf/size.md"><code>tf.size(input, out_type, name)</code></a>
* <a href="../../tf/slice.md"><code>tf.slice(input_, begin, size, name)</code></a>
* <a href="../../tf/sort.md"><code>tf.sort(values, axis, direction, name)</code></a>
* <a href="../../tf/space_to_batch.md"><code>tf.space_to_batch(input, block_shape, paddings, name)</code></a>
* <a href="../../tf/space_to_batch_nd.md"><code>tf.space_to_batch_nd(input, block_shape, paddings, name)</code></a>
* <a href="../../tf/split.md"><code>tf.split(value, num_or_size_splits, axis, num, name)</code></a>
* <a href="../../tf/squeeze.md"><code>tf.squeeze(input, axis, name)</code></a>
* <a href="../../tf/stack.md"><code>tf.stack(values, axis, name)</code></a>
* <a href="../../tf/stop_gradient.md"><code>tf.stop_gradient(input, name)</code></a>
* <a href="../../tf/strided_slice.md"><code>tf.strided_slice(input_, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, var, name)</code></a>
* <a href="../../tf/strings/as_string.md"><code>tf.strings.as_string(input, precision, scientific, shortest, width, fill, name)</code></a>
* <a href="../../tf/strings/bytes_split.md"><code>tf.strings.bytes_split(input, name)</code></a>
* <a href="../../tf/strings/format.md"><code>tf.strings.format(template, inputs, placeholder, summarize, name)</code></a>
* <a href="../../tf/strings/join.md"><code>tf.strings.join(inputs, separator, name)</code></a>
* <a href="../../tf/strings/length.md"><code>tf.strings.length(input, unit, name)</code></a>
* <a href="../../tf/strings/lower.md"><code>tf.strings.lower(input, encoding, name)</code></a>
* <a href="../../tf/strings/ngrams.md"><code>tf.strings.ngrams(data, ngram_width, separator, pad_values, padding_width, preserve_short_sequences, name)</code></a>
* <a href="../../tf/strings/reduce_join.md"><code>tf.strings.reduce_join(inputs, axis, keepdims, separator, name)</code></a>
* <a href="../../tf/strings/regex_full_match.md"><code>tf.strings.regex_full_match(input, pattern, name)</code></a>
* <a href="../../tf/strings/regex_replace.md"><code>tf.strings.regex_replace(input, pattern, rewrite, replace_global, name)</code></a>
* <a href="../../tf/strings/split.md"><code>tf.strings.split(input, sep, maxsplit, name)</code></a>
* <a href="../../tf/strings/strip.md"><code>tf.strings.strip(input, name)</code></a>
* <a href="../../tf/strings/substr.md"><code>tf.strings.substr(input, pos, len, unit, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket.md"><code>tf.strings.to_hash_bucket(input, num_buckets, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket_fast.md"><code>tf.strings.to_hash_bucket_fast(input, num_buckets, name)</code></a>
* <a href="../../tf/strings/to_hash_bucket_strong.md"><code>tf.strings.to_hash_bucket_strong(input, num_buckets, key, name)</code></a>
* <a href="../../tf/strings/to_number.md"><code>tf.strings.to_number(input, out_type, name)</code></a>
* <a href="../../tf/strings/unicode_decode.md"><code>tf.strings.unicode_decode(input, input_encoding, errors, replacement_char, replace_control_characters, name)</code></a>
* <a href="../../tf/strings/unicode_decode_with_offsets.md"><code>tf.strings.unicode_decode_with_offsets(input, input_encoding, errors, replacement_char, replace_control_characters, name)</code></a>
* <a href="../../tf/strings/unicode_encode.md"><code>tf.strings.unicode_encode(input, output_encoding, errors, replacement_char, name)</code></a>
* <a href="../../tf/strings/unicode_script.md"><code>tf.strings.unicode_script(input, name)</code></a>
* <a href="../../tf/strings/unicode_split.md"><code>tf.strings.unicode_split(input, input_encoding, errors, replacement_char, name)</code></a>
* <a href="../../tf/strings/unicode_split_with_offsets.md"><code>tf.strings.unicode_split_with_offsets(input, input_encoding, errors, replacement_char, name)</code></a>
* <a href="../../tf/strings/unicode_transcode.md"><code>tf.strings.unicode_transcode(input, input_encoding, output_encoding, errors, replacement_char, replace_control_characters, name)</code></a>
* <a href="../../tf/strings/unsorted_segment_join.md"><code>tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments, separator, name)</code></a>
* <a href="../../tf/strings/upper.md"><code>tf.strings.upper(input, encoding, name)</code></a>
* <a href="../../tf/tensor_scatter_nd_add.md"><code>tf.tensor_scatter_nd_add(tensor, indices, updates, name)</code></a>
* <a href="../../tf/tensor_scatter_nd_max.md"><code>tf.tensor_scatter_nd_max(tensor, indices, updates, name)</code></a>
* <a href="../../tf/tensor_scatter_nd_min.md"><code>tf.tensor_scatter_nd_min(tensor, indices, updates, name)</code></a>
* <a href="../../tf/tensor_scatter_nd_sub.md"><code>tf.tensor_scatter_nd_sub(tensor, indices, updates, name)</code></a>
* <a href="../../tf/tensor_scatter_nd_update.md"><code>tf.tensor_scatter_nd_update(tensor, indices, updates, name)</code></a>
* <a href="../../tf/tensordot.md"><code>tf.tensordot(a, b, axes, name)</code></a>
* <a href="../../tf/tile.md"><code>tf.tile(input, multiples, name)</code></a>
* <a href="../../tf/timestamp.md"><code>tf.timestamp(name)</code></a>
* <a href="../../tf/transpose.md"><code>tf.transpose(a, perm, conjugate, name)</code></a>
* <a href="../../tf/truncatediv.md"><code>tf.truncatediv(x, y, name)</code></a>
* <a href="../../tf/truncatemod.md"><code>tf.truncatemod(x, y, name)</code></a>
* <a href="../../tf/tuple.md"><code>tf.tuple(tensors, control_inputs, name)</code></a>
* <a href="../../tf/unique.md"><code>tf.unique(x, out_idx, name)</code></a>
* <a href="../../tf/unique_with_counts.md"><code>tf.unique_with_counts(x, out_idx, name)</code></a>
* <a href="../../tf/unravel_index.md"><code>tf.unravel_index(indices, dims, name)</code></a>
* <a href="../../tf/unstack.md"><code>tf.unstack(value, num, axis, name)</code></a>
* <a href="../../tf/where.md"><code>tf.where(condition, x, y, name)</code></a>
* `tf.xla_all_reduce(input, group_assignment, reduce_op, mode, name)`
* `tf.xla_broadcast_helper(lhs, rhs, broadcast_dims, name)`
* `tf.xla_cluster_output(input, name)`
* `tf.xla_conv(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, name)`
* `tf.xla_conv_v2(lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation, feature_group_count, dimension_numbers, precision_config, preferred_element_type, batch_group_count, name)`
* `tf.xla_custom_call(args, target_name, backend_config, dtype, shape, name)`
* `tf.xla_dequantize(input, min_range, max_range, mode, transpose_output, name)`
* `tf.xla_dot(lhs, rhs, dimension_numbers, precision_config, name)`
* `tf.xla_dot_v2(lhs, rhs, dimension_numbers, precision_config, preferred_element_type, name)`
* `tf.xla_dynamic_slice(input, start_indices, size_indices, name)`
* `tf.xla_dynamic_update_slice(input, update, indices, name)`
* `tf.xla_einsum(a, b, equation, name)`
* `tf.xla_gather(operand, start_indices, slice_sizes, dimension_numbers, indices_are_sorted, name)`
* `tf.xla_if(cond, inputs, then_branch, else_branch, Tout, name)`
* `tf.xla_key_value_sort(keys, values, name)`
* `tf.xla_launch(constants, args, resources, Tresults, function, name)`
* `tf.xla_optimization_barrier(input, name)`
* `tf.xla_pad(input, padding_value, padding_low, padding_high, padding_interior, name)`
* `tf.xla_recv(dtype, tensor_name, shape, name)`
* `tf.xla_reduce(input, init_value, dimensions_to_reduce, reducer, name)`
* `tf.xla_reduce_scatter(input, group_assignment, scatter_dimension, reduce_op, name)`
* `tf.xla_reduce_window(input, init_value, window_dimensions, window_strides, base_dilations, window_dilations, padding, computation, name)`
* `tf.xla_remove_dynamic_dimension_size(input, dim_index, name)`
* `tf.xla_replica_id(name)`
* `tf.xla_rng_bit_generator(algorithm, initial_state, shape, dtype, name)`
* `tf.xla_scatter(operand, scatter_indices, updates, update_computation, dimension_numbers, indices_are_sorted, name)`
* `tf.xla_select_and_scatter(operand, window_dimensions, window_strides, padding, source, init_value, select, scatter, name)`
* `tf.xla_self_adjoint_eig(a, lower, max_iter, epsilon, name)`
* `tf.xla_send(tensor, tensor_name, name)`
* `tf.xla_set_bound(input, bound, name)`
* `tf.xla_set_dynamic_dimension_size(input, dim_index, size, name)`
* `tf.xla_sharding(input, sharding, unspecified_dims, name)`
* `tf.xla_sort(input, name)`
* `tf.xla_spmd_full_to_shard_shape(input, manual_sharding, dim, unspecified_dims, name)`
* `tf.xla_spmd_shard_to_full_shape(input, manual_sharding, full_shape, dim, unspecified_dims, name)`
* `tf.xla_svd(a, max_iter, epsilon, precision_config, name)`
* `tf.xla_variadic_reduce(input, init_value, dimensions_to_reduce, reducer, name)`
* `tf.xla_variadic_reduce_v2(inputs, init_values, dimensions_to_reduce, reducer, name)`
* `tf.xla_variadic_sort(inputs, dimension, comparator, is_stable, name)`
* `tf.xla_while(input, cond, body, name)`
* <a href="../../tf/zeros.md"><code>tf.zeros(shape, dtype, name)</code></a>
* <a href="../../tf/zeros_like.md"><code>tf.zeros_like(input, dtype, name)</code></a>