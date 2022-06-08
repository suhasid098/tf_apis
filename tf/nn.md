description: Primitive Neural Net (NN) Operations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.nn

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Primitive Neural Net (NN) Operations.


## Notes on padding

Several neural network operations, such as <a href="../tf/nn/conv2d.md"><code>tf.nn.conv2d</code></a> and
<a href="../tf/nn/max_pool2d.md"><code>tf.nn.max_pool2d</code></a>, take a `padding` parameter, which controls how the input is
padded before running the operation. The input is padded by inserting values
(typically zeros) before and after the tensor in each spatial dimension. The
`padding` parameter can either be the string `'VALID'`, which means use no
padding, or `'SAME'` which adds padding according to a formula which is
described below. Certain ops also allow the amount of padding per dimension to
be explicitly specified by passing a list to `padding`.

In the case of convolutions, the input is padded with zeros. In case of pools,
the padded input values are ignored. For example, in a max pool, the sliding
window ignores padded values, which is equivalent to the padded values being
`-infinity`.

### `'VALID'` padding

Passing `padding='VALID'` to an op causes no padding to be used. This causes the
output size to typically be smaller than the input size, even when the stride is
one. In the 2D case, the output size is computed as:

```python
out_height = ceil((in_height - filter_height + 1) / stride_height)
out_width  = ceil((in_width - filter_width + 1) / stride_width)
```

The 1D and 3D cases are similar. Note `filter_height` and `filter_width` refer
to the filter size after dilations (if any) for convolutions, and refer to the
window size for pools.

### `'SAME'` padding

With `'SAME'` padding, padding is applied to each spatial dimension. When the
strides are 1, the input is padded such that the output size is the same as the
input size. In the 2D case, the output size is computed as:

```python
out_height = ceil(in_height / stride_height)
out_width  = ceil(in_width / stride_width)
```

The amount of padding used is the smallest amount that results in the output
size. The formula for the total amount of padding per dimension is:

```python
if (in_height % strides[1] == 0):
  pad_along_height = max(filter_height - stride_height, 0)
else:
  pad_along_height = max(filter_height - (in_height % stride_height), 0)
if (in_width % strides[2] == 0):
  pad_along_width = max(filter_width - stride_width, 0)
else:
  pad_along_width = max(filter_width - (in_width % stride_width), 0)
```

Finally, the padding on the top, bottom, left and right are:

```python
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left
```

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when pad_along_height is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as PyTorch and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

Here is an example of `'SAME'` padding:

```
>>> in_height = 5
>>> filter_height = 3
>>> stride_height = 2
>>>
>>> in_width = 2
>>> filter_width = 2
>>> stride_width = 1
>>>
>>> inp = tf.ones((2, in_height, in_width, 2))
>>> filter = tf.ones((filter_height, filter_width, 2, 2))
>>> strides = [stride_height, stride_width]
>>> output = tf.nn.conv2d(inp, filter, strides, padding='SAME')
>>> output.shape[1]  # output_height: ceil(5 / 2)
3
>>> output.shape[2] # output_width: ceil(2 / 1)
2
```

### Explicit padding

Certain ops, like <a href="../tf/nn/conv2d.md"><code>tf.nn.conv2d</code></a>, also allow a list of explicit padding amounts
to be passed to the `padding` parameter. This list is in the same format as what
is passed to <a href="../tf/pad.md"><code>tf.pad</code></a>, except the padding must be a nested list, not a tensor.
For example, in the 2D case, the list is in the format `[[0, 0], [pad_top,
pad_bottom], [pad_left, pad_right], [0, 0]]` when `data_format` is its default
value of `'NHWC'`. The two `[0, 0]` pairs  indicate the batch and channel
dimensions have no padding, which is required, as only spatial dimensions can
have padding.

#### For example:



```
>>> inp = tf.ones((1, 3, 3, 1))
>>> filter = tf.ones((2, 2, 1, 1))
>>> strides = [1, 1]
>>> padding = [[0, 0], [1, 2], [0, 1], [0, 0]]
>>> output = tf.nn.conv2d(inp, filter, strides, padding=padding)
>>> tuple(output.shape)
(1, 5, 3, 1)
>>> # Equivalently, tf.pad can be used, since convolutions pad with zeros.
>>> inp = tf.pad(inp, padding)
>>> # 'VALID' means to use no padding in conv2d (we already padded inp)
>>> output2 = tf.nn.conv2d(inp, filter, strides, padding='VALID')
>>> tf.debugging.assert_equal(output, output2)
```

## Modules

[`experimental`](../tf/nn/experimental.md) module: Public API for tf.nn.experimental namespace.

## Classes

[`class RNNCellDeviceWrapper`](../tf/nn/RNNCellDeviceWrapper.md): Operator that ensures an RNNCell runs on a particular device.

[`class RNNCellDropoutWrapper`](../tf/nn/RNNCellDropoutWrapper.md): Operator adding dropout to inputs and outputs of the given cell.

[`class RNNCellResidualWrapper`](../tf/nn/RNNCellResidualWrapper.md): RNNCell wrapper that ensures cell inputs are added to the outputs.

## Functions

[`all_candidate_sampler(...)`](../tf/random/all_candidate_sampler.md): Generate the set of all classes.

[`atrous_conv2d(...)`](../tf/nn/atrous_conv2d.md): Atrous convolution (a.k.a. convolution with holes or dilated convolution).

[`atrous_conv2d_transpose(...)`](../tf/nn/atrous_conv2d_transpose.md): The transpose of `atrous_conv2d`.

[`avg_pool(...)`](../tf/nn/avg_pool.md): Performs the avg pooling on the input.

[`avg_pool1d(...)`](../tf/nn/avg_pool1d.md): Performs the average pooling on the input.

[`avg_pool2d(...)`](../tf/nn/avg_pool2d.md): Performs the average pooling on the input.

[`avg_pool3d(...)`](../tf/nn/avg_pool3d.md): Performs the average pooling on the input.

[`batch_norm_with_global_normalization(...)`](../tf/nn/batch_norm_with_global_normalization.md): Batch normalization.

[`batch_normalization(...)`](../tf/nn/batch_normalization.md): Batch normalization.

[`bias_add(...)`](../tf/nn/bias_add.md): Adds `bias` to `value`.

[`collapse_repeated(...)`](../tf/nn/collapse_repeated.md): Merge repeated labels into single labels.

[`compute_accidental_hits(...)`](../tf/nn/compute_accidental_hits.md): Compute the position ids in `sampled_candidates` matching `true_classes`.

[`compute_average_loss(...)`](../tf/nn/compute_average_loss.md): Scales per-example losses with sample_weights and computes their average.

[`conv1d(...)`](../tf/nn/conv1d.md): Computes a 1-D convolution given 3-D input and filter tensors.

[`conv1d_transpose(...)`](../tf/nn/conv1d_transpose.md): The transpose of `conv1d`.

[`conv2d(...)`](../tf/nn/conv2d.md): Computes a 2-D convolution given `input` and 4-D `filters` tensors.

[`conv2d_transpose(...)`](../tf/nn/conv2d_transpose.md): The transpose of `conv2d`.

[`conv3d(...)`](../tf/nn/conv3d.md): Computes a 3-D convolution given 5-D `input` and `filters` tensors.

[`conv3d_transpose(...)`](../tf/nn/conv3d_transpose.md): The transpose of `conv3d`.

[`conv_transpose(...)`](../tf/nn/conv_transpose.md): The transpose of `convolution`.

[`convolution(...)`](../tf/nn/convolution.md): Computes sums of N-D convolutions (actually cross-correlation).

[`crelu(...)`](../tf/nn/crelu.md): Computes Concatenated ReLU.

[`ctc_beam_search_decoder(...)`](../tf/nn/ctc_beam_search_decoder.md): Performs beam search decoding on the logits given in input.

[`ctc_greedy_decoder(...)`](../tf/nn/ctc_greedy_decoder.md): Performs greedy decoding on the logits given in input (best path).

[`ctc_loss(...)`](../tf/nn/ctc_loss.md): Computes CTC (Connectionist Temporal Classification) loss.

[`ctc_unique_labels(...)`](../tf/nn/ctc_unique_labels.md): Get unique labels and indices for batched labels for <a href="../tf/nn/ctc_loss.md"><code>tf.nn.ctc_loss</code></a>.

[`depth_to_space(...)`](../tf/nn/depth_to_space.md): DepthToSpace for tensors of type T.

[`depthwise_conv2d(...)`](../tf/nn/depthwise_conv2d.md): Depthwise 2-D convolution.

[`depthwise_conv2d_backprop_filter(...)`](../tf/nn/depthwise_conv2d_backprop_filter.md): Computes the gradients of depthwise convolution with respect to the filter.

[`depthwise_conv2d_backprop_input(...)`](../tf/nn/depthwise_conv2d_backprop_input.md): Computes the gradients of depthwise convolution with respect to the input.

[`dilation2d(...)`](../tf/nn/dilation2d.md): Computes the grayscale dilation of 4-D `input` and 3-D `filters` tensors.

[`dropout(...)`](../tf/nn/dropout.md): Computes dropout: randomly sets elements to zero to prevent overfitting.

[`elu(...)`](../tf/nn/elu.md): Computes the exponential linear function.

[`embedding_lookup(...)`](../tf/nn/embedding_lookup.md): Looks up embeddings for the given `ids` from a list of tensors.

[`embedding_lookup_sparse(...)`](../tf/nn/embedding_lookup_sparse.md): Looks up embeddings for the given ids and weights from a list of tensors.

[`erosion2d(...)`](../tf/nn/erosion2d.md): Computes the grayscale erosion of 4-D `value` and 3-D `filters` tensors.

[`fixed_unigram_candidate_sampler(...)`](../tf/random/fixed_unigram_candidate_sampler.md): Samples a set of classes using the provided (fixed) base distribution.

[`fractional_avg_pool(...)`](../tf/nn/fractional_avg_pool.md): Performs fractional average pooling on the input.

[`fractional_max_pool(...)`](../tf/nn/fractional_max_pool.md): Performs fractional max pooling on the input.

[`gelu(...)`](../tf/nn/gelu.md): Compute the Gaussian Error Linear Unit (GELU) activation function.

[`in_top_k(...)`](../tf/math/in_top_k.md): Says whether the targets are in the top `K` predictions.

[`isotonic_regression(...)`](../tf/nn/isotonic_regression.md): Solves isotonic regression problems along the given axis.

[`l2_loss(...)`](../tf/nn/l2_loss.md): L2 Loss.

[`l2_normalize(...)`](../tf/math/l2_normalize.md): Normalizes along dimension `axis` using an L2 norm. (deprecated arguments)

[`leaky_relu(...)`](../tf/nn/leaky_relu.md): Compute the Leaky ReLU activation function.

[`learned_unigram_candidate_sampler(...)`](../tf/random/learned_unigram_candidate_sampler.md): Samples a set of classes from a distribution learned during training.

[`local_response_normalization(...)`](../tf/nn/local_response_normalization.md): Local Response Normalization.

[`log_poisson_loss(...)`](../tf/nn/log_poisson_loss.md): Computes log Poisson loss given `log_input`.

[`log_softmax(...)`](../tf/nn/log_softmax.md): Computes log softmax activations.

[`lrn(...)`](../tf/nn/local_response_normalization.md): Local Response Normalization.

[`max_pool(...)`](../tf/nn/max_pool.md): Performs max pooling on the input.

[`max_pool1d(...)`](../tf/nn/max_pool1d.md): Performs the max pooling on the input.

[`max_pool2d(...)`](../tf/nn/max_pool2d.md): Performs max pooling on 2D spatial data such as images.

[`max_pool3d(...)`](../tf/nn/max_pool3d.md): Performs the max pooling on the input.

[`max_pool_with_argmax(...)`](../tf/nn/max_pool_with_argmax.md): Performs max pooling on the input and outputs both max values and indices.

[`moments(...)`](../tf/nn/moments.md): Calculates the mean and variance of `x`.

[`nce_loss(...)`](../tf/nn/nce_loss.md): Computes and returns the noise-contrastive estimation training loss.

[`normalize_moments(...)`](../tf/nn/normalize_moments.md): Calculate the mean and variance of based on the sufficient statistics.

[`pool(...)`](../tf/nn/pool.md): Performs an N-D pooling operation.

[`relu(...)`](../tf/nn/relu.md): Computes rectified linear: `max(features, 0)`.

[`relu6(...)`](../tf/nn/relu6.md): Computes Rectified Linear 6: `min(max(features, 0), 6)`.

[`safe_embedding_lookup_sparse(...)`](../tf/nn/safe_embedding_lookup_sparse.md): Lookup embedding results, accounting for invalid IDs and empty features.

[`sampled_softmax_loss(...)`](../tf/nn/sampled_softmax_loss.md): Computes and returns the sampled softmax training loss.

[`scale_regularization_loss(...)`](../tf/nn/scale_regularization_loss.md): Scales the sum of the given regularization losses by number of replicas.

[`selu(...)`](../tf/nn/selu.md): Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`

[`separable_conv2d(...)`](../tf/nn/separable_conv2d.md): 2-D convolution with separable filters.

[`sigmoid(...)`](../tf/math/sigmoid.md): Computes sigmoid of `x` element-wise.

[`sigmoid_cross_entropy_with_logits(...)`](../tf/nn/sigmoid_cross_entropy_with_logits.md): Computes sigmoid cross entropy given `logits`.

[`silu(...)`](../tf/nn/silu.md): Computes the SiLU or Swish activation function: `x * sigmoid(beta * x)`.

[`softmax(...)`](../tf/nn/softmax.md): Computes softmax activations.

[`softmax_cross_entropy_with_logits(...)`](../tf/nn/softmax_cross_entropy_with_logits.md): Computes softmax cross entropy between `logits` and `labels`.

[`softplus(...)`](../tf/math/softplus.md): Computes elementwise softplus: `softplus(x) = log(exp(x) + 1)`.

[`softsign(...)`](../tf/nn/softsign.md): Computes softsign: `features / (abs(features) + 1)`.

[`space_to_batch(...)`](../tf/space_to_batch.md): SpaceToBatch for N-D tensors of type T.

[`space_to_depth(...)`](../tf/nn/space_to_depth.md): SpaceToDepth for tensors of type T.

[`sparse_softmax_cross_entropy_with_logits(...)`](../tf/nn/sparse_softmax_cross_entropy_with_logits.md): Computes sparse softmax cross entropy between `logits` and `labels`.

[`sufficient_statistics(...)`](../tf/nn/sufficient_statistics.md): Calculate the sufficient statistics for the mean and variance of `x`.

[`swish(...)`](../tf/nn/silu.md): Computes the SiLU or Swish activation function: `x * sigmoid(beta * x)`.

[`tanh(...)`](../tf/math/tanh.md): Computes hyperbolic tangent of `x` element-wise.

[`top_k(...)`](../tf/math/top_k.md): Finds values and indices of the `k` largest entries for the last dimension.

[`weighted_cross_entropy_with_logits(...)`](../tf/nn/weighted_cross_entropy_with_logits.md): Computes a weighted cross entropy.

[`weighted_moments(...)`](../tf/nn/weighted_moments.md): Returns the frequency-weighted mean and variance of `x`.

[`with_space_to_batch(...)`](../tf/nn/with_space_to_batch.md): Performs `op` on the space-to-batch representation of `input`.

[`zero_fraction(...)`](../tf/math/zero_fraction.md): Returns the fraction of zeros in `value`.

