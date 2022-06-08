description: Finds the filename of latest saved checkpoint file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.latest_checkpoint" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.latest_checkpoint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/checkpoint_management.py">View source</a>



Finds the filename of latest saved checkpoint file.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.latest_checkpoint`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.latest_checkpoint(
    checkpoint_dir, latest_filename=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Gets the checkpoint state given the provided checkpoint_dir and looks for a
corresponding TensorFlow 2 (preferred) or TensorFlow 1.x checkpoint path.
The latest_filename argument is only applicable if you are saving checkpoint
using <a href="../../tf/compat/v1/train/Saver.md#save"><code>v1.train.Saver.save</code></a>


See the [Training Checkpoints
Guide](https://www.tensorflow.org/guide/checkpoint) for more details and
examples.`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`checkpoint_dir`
</td>
<td>
Directory where the variables were saved.
</td>
</tr><tr>
<td>
`latest_filename`
</td>
<td>
Optional name for the protocol buffer file that
contains the list of most recent checkpoint filenames.
See the corresponding argument to <a href="../../tf/compat/v1/train/Saver.md#save"><code>v1.train.Saver.save</code></a>.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The full path to the latest checkpoint or `None` if no checkpoint was found.
</td>
</tr>

</table>

