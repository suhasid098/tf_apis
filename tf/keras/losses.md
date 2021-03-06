description: Built-in loss functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.keras.losses

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Built-in loss functions.



## Classes

[`class BinaryCrossentropy`](../../tf/keras/losses/BinaryCrossentropy.md): Computes the cross-entropy loss between true labels and predicted labels.

[`class BinaryFocalCrossentropy`](../../tf/keras/losses/BinaryFocalCrossentropy.md): Computes the focal cross-entropy loss between true labels and predictions.

[`class CategoricalCrossentropy`](../../tf/keras/losses/CategoricalCrossentropy.md): Computes the crossentropy loss between the labels and predictions.

[`class CategoricalHinge`](../../tf/keras/losses/CategoricalHinge.md): Computes the categorical hinge loss between `y_true` and `y_pred`.

[`class CosineSimilarity`](../../tf/keras/losses/CosineSimilarity.md): Computes the cosine similarity between labels and predictions.

[`class Hinge`](../../tf/keras/losses/Hinge.md): Computes the hinge loss between `y_true` and `y_pred`.

[`class Huber`](../../tf/keras/losses/Huber.md): Computes the Huber loss between `y_true` and `y_pred`.

[`class KLDivergence`](../../tf/keras/losses/KLDivergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`class LogCosh`](../../tf/keras/losses/LogCosh.md): Computes the logarithm of the hyperbolic cosine of the prediction error.

[`class Loss`](../../tf/keras/losses/Loss.md): Loss base class.

[`class MeanAbsoluteError`](../../tf/keras/losses/MeanAbsoluteError.md): Computes the mean of absolute difference between labels and predictions.

[`class MeanAbsolutePercentageError`](../../tf/keras/losses/MeanAbsolutePercentageError.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`class MeanSquaredError`](../../tf/keras/losses/MeanSquaredError.md): Computes the mean of squares of errors between labels and predictions.

[`class MeanSquaredLogarithmicError`](../../tf/keras/losses/MeanSquaredLogarithmicError.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`class Poisson`](../../tf/keras/losses/Poisson.md): Computes the Poisson loss between `y_true` and `y_pred`.

[`class Reduction`](../../tf/keras/losses/Reduction.md): Types of loss reduction.

[`class SparseCategoricalCrossentropy`](../../tf/keras/losses/SparseCategoricalCrossentropy.md): Computes the crossentropy loss between the labels and predictions.

[`class SquaredHinge`](../../tf/keras/losses/SquaredHinge.md): Computes the squared hinge loss between `y_true` and `y_pred`.

## Functions

[`KLD(...)`](../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`MAE(...)`](../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`MAPE(...)`](../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`MSE(...)`](../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`MSLE(...)`](../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`binary_crossentropy(...)`](../../tf/keras/metrics/binary_crossentropy.md): Computes the binary crossentropy loss.

[`binary_focal_crossentropy(...)`](../../tf/keras/metrics/binary_focal_crossentropy.md): Computes the binary focal crossentropy loss.

[`categorical_crossentropy(...)`](../../tf/keras/metrics/categorical_crossentropy.md): Computes the categorical crossentropy loss.

[`categorical_hinge(...)`](../../tf/keras/losses/categorical_hinge.md): Computes the categorical hinge loss between `y_true` and `y_pred`.

[`cosine_similarity(...)`](../../tf/keras/losses/cosine_similarity.md): Computes the cosine similarity between labels and predictions.

[`deserialize(...)`](../../tf/keras/losses/deserialize.md): Deserializes a serialized loss class/function instance.

[`get(...)`](../../tf/keras/losses/get.md): Retrieves a Keras loss as a `function`/`Loss` class instance.

[`hinge(...)`](../../tf/keras/metrics/hinge.md): Computes the hinge loss between `y_true` and `y_pred`.

[`huber(...)`](../../tf/keras/losses/huber.md): Computes Huber loss value.

[`kl_divergence(...)`](../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`kld(...)`](../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`kullback_leibler_divergence(...)`](../../tf/keras/metrics/kl_divergence.md): Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

[`log_cosh(...)`](../../tf/keras/losses/log_cosh.md): Logarithm of the hyperbolic cosine of the prediction error.

[`logcosh(...)`](../../tf/keras/losses/log_cosh.md): Logarithm of the hyperbolic cosine of the prediction error.

[`mae(...)`](../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`mape(...)`](../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`mean_absolute_error(...)`](../../tf/keras/metrics/mean_absolute_error.md): Computes the mean absolute error between labels and predictions.

[`mean_absolute_percentage_error(...)`](../../tf/keras/metrics/mean_absolute_percentage_error.md): Computes the mean absolute percentage error between `y_true` and `y_pred`.

[`mean_squared_error(...)`](../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`mean_squared_logarithmic_error(...)`](../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`mse(...)`](../../tf/keras/metrics/mean_squared_error.md): Computes the mean squared error between labels and predictions.

[`msle(...)`](../../tf/keras/metrics/mean_squared_logarithmic_error.md): Computes the mean squared logarithmic error between `y_true` and `y_pred`.

[`poisson(...)`](../../tf/keras/metrics/poisson.md): Computes the Poisson loss between y_true and y_pred.

[`serialize(...)`](../../tf/keras/losses/serialize.md): Serializes loss function or `Loss` instance.

[`sparse_categorical_crossentropy(...)`](../../tf/keras/metrics/sparse_categorical_crossentropy.md): Computes the sparse categorical crossentropy loss.

[`squared_hinge(...)`](../../tf/keras/metrics/squared_hinge.md): Computes the squared hinge loss between `y_true` and `y_pred`.

