description: Utilities to preprocess data before training.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.keras.preprocessing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Utilities to preprocess data before training.


Deprecated: <a href="../../tf/keras/preprocessing.md"><code>tf.keras.preprocessing</code></a> APIs do not operate on tensors and are
not recommended for new code. Prefer loading data with either
<a href="../../tf/keras/utils/text_dataset_from_directory.md"><code>tf.keras.utils.text_dataset_from_directory</code></a> or
<a href="../../tf/keras/utils/image_dataset_from_directory.md"><code>tf.keras.utils.image_dataset_from_directory</code></a>, and then transforming the output
<a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> with preprocessing layers. These approaches will offer
better performance and intergration with the broader Tensorflow ecosystem. For
more information, see the tutorials for [loading text](
https://www.tensorflow.org/tutorials/load_data/text), [loading images](
https://www.tensorflow.org/tutorials/load_data/images), and [augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as the
[preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).

## Modules

[`image`](../../tf/keras/preprocessing/image.md) module: Utilies for image preprocessing and augmentation.

[`sequence`](../../tf/keras/preprocessing/sequence.md) module: Utilities for preprocessing sequence data.

[`text`](../../tf/keras/preprocessing/text.md) module: Utilities for text input preprocessing.

## Functions

[`image_dataset_from_directory(...)`](../../tf/keras/utils/image_dataset_from_directory.md): Generates a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from image files in a directory.

[`text_dataset_from_directory(...)`](../../tf/keras/utils/text_dataset_from_directory.md): Generates a <a href="../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from text files in a directory.

[`timeseries_dataset_from_array(...)`](../../tf/keras/utils/timeseries_dataset_from_array.md): Creates a dataset of sliding windows over a timeseries provided as array.

