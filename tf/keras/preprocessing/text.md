description: Utilities for text input preprocessing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.keras.preprocessing.text

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Utilities for text input preprocessing.


Deprecated: <a href="../../../tf/keras/preprocessing/text.md"><code>tf.keras.preprocessing.text</code></a> APIs are not recommended for new code.
Prefer <a href="../../../tf/keras/utils/text_dataset_from_directory.md"><code>tf.keras.utils.text_dataset_from_directory</code></a> and
<a href="../../../tf/keras/layers/TextVectorization.md"><code>tf.keras.layers.TextVectorization</code></a> which provide a more efficient approach
for preprocessing text input. For an introduction to these APIs, see
the [text loading tutorial]
(https://www.tensorflow.org/tutorials/load_data/text)
and [preprocessing layer guide]
(https://www.tensorflow.org/guide/keras/preprocessing_layers).

## Classes

[`class Tokenizer`](../../../tf/keras/preprocessing/text/Tokenizer.md): Text tokenization utility class.

## Functions

[`hashing_trick(...)`](../../../tf/keras/preprocessing/text/hashing_trick.md): Converts a text to a sequence of indexes in a fixed-size hashing space.

[`one_hot(...)`](../../../tf/keras/preprocessing/text/one_hot.md): One-hot encodes a text into a list of word indexes of size `n`.

[`text_to_word_sequence(...)`](../../../tf/keras/preprocessing/text/text_to_word_sequence.md): Converts a text to a sequence of words (or tokens).

[`tokenizer_from_json(...)`](../../../tf/keras/preprocessing/text/tokenizer_from_json.md): Parses a JSON tokenizer configuration and returns a tokenizer instance.

