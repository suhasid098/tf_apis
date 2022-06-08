description: Converts input which is a PathLike object to str type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.path_to_str" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.path_to_str

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/util/compat.py">View source</a>



Converts input which is a `PathLike` object to `str` type.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.compat.path_to_str`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.path_to_str(
    path
)
</code></pre>



<!-- Placeholder for "Used in" -->

Converts from any python constant representation of a `PathLike` object to
a string. If the input is not a `PathLike` object, simply returns the input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`
</td>
<td>
An object that can be converted to path representation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `str` object.
</td>
</tr>

</table>



#### Usage:

In case a simplified `str` version of the path is needed from an
`os.PathLike` object



#### Examples:


```python
$ tf.compat.path_to_str('C:\XYZ\tensorflow\./.././tensorflow')
'C:\XYZ\tensorflow\./.././tensorflow' # Windows OS
$ tf.compat.path_to_str(Path('C:\XYZ\tensorflow\./.././tensorflow'))
'C:\XYZ\tensorflow\..\tensorflow' # Windows OS
$ tf.compat.path_to_str(Path('./corpus'))
'corpus' # Linux OS
$ tf.compat.path_to_str('./.././Corpus')
'./.././Corpus' # Linux OS
$ tf.compat.path_to_str(Path('./.././Corpus'))
'../Corpus' # Linux OS
$ tf.compat.path_to_str(Path('./..////../'))
'../..' # Linux OS

```