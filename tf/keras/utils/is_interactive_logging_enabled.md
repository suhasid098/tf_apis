description: Check if interactive logging is enabled.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.is_interactive_logging_enabled" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.is_interactive_logging_enabled

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/utils/io_utils.py#L54-L68">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Check if interactive logging is enabled.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.keras.utils.is_interactive_logging_enabled`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.is_interactive_logging_enabled()
</code></pre>



<!-- Placeholder for "Used in" -->

To switch between writing logs to stdout and `absl.logging`, you may use
<a href="../../../tf/keras/utils/enable_interactive_logging.md"><code>keras.utils.enable_interactive_logging()</code></a> and
`keras.utils.disable_interactie_logging()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Boolean (True if interactive logging is enabled and False otherwise).
</td>
</tr>

</table>

