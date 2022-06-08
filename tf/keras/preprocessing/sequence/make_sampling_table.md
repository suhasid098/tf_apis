description: Generates a word rank-based probabilistic sampling table.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.sequence.make_sampling_table" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.sequence.make_sampling_table

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/preprocessing/sequence.py#L234-L271">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates a word rank-based probabilistic sampling table.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.preprocessing.sequence.make_sampling_table`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.sequence.make_sampling_table(
    size, sampling_factor=1e-05
)
</code></pre>



<!-- Placeholder for "Used in" -->

Used for generating the `sampling_table` argument for `skipgrams`.
`sampling_table[i]` is the probability of sampling
the word i-th most common word in a dataset
(more common words should be sampled less frequently, for balance).

The sampling probabilities are generated according
to the sampling distribution used in word2vec:

```
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

We assume that the word frequencies follow Zipf's law (s=1) to derive
a numerical approximation of frequency(rank):

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
where `gamma` is the Euler-Mascheroni constant.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`size`
</td>
<td>
Int, number of possible words to sample.
</td>
</tr><tr>
<td>
`sampling_factor`
</td>
<td>
The sampling factor in the word2vec formula.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 1D Numpy array of length `size` where the ith entry
is the probability that a word of rank i should be sampled.
</td>
</tr>

</table>

