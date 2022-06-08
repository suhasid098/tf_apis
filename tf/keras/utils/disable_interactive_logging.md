description: Turn off interactive logging.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.disable_interactive_logging" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.disable_interactive_logging

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/utils/io_utils.py#L43-L51">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Turn off interactive logging.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.disable_interactive_logging`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.disable_interactive_logging()
</code></pre>



<!-- Placeholder for "Used in" -->

When interactive logging is disabled, Keras sends logs to `absl.logging`.
This is the best option when using Keras in a non-interactive
way, such as running a training or inference job on a server.