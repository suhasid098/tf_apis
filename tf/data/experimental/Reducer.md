description: A reducer is used for reducing a set of elements.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.Reducer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.data.experimental.Reducer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/grouping.py">View source</a>



A reducer is used for reducing a set of elements.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.Reducer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.Reducer(
    init_func, reduce_func, finalize_func
)
</code></pre>



<!-- Placeholder for "Used in" -->

A reducer is represented as a tuple of the three functions:
- init_func - to define initial value: key => initial state
- reducer_func - operation to perform on values with same key: (old state, input) => new state
- finalize_func - value to return in the end: state => result

For example,

```
def init_func(_):
  return (0.0, 0.0)

def reduce_func(state, value):
  return (state[0] + value['features'], state[1] + 1)

def finalize_func(s, n):
  return s / n

reducer = tf.data.experimental.Reducer(init_func, reduce_func, finalize_func)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`finalize_func`
</td>
<td>

</td>
</tr><tr>
<td>
`init_func`
</td>
<td>

</td>
</tr><tr>
<td>
`reduce_func`
</td>
<td>

</td>
</tr>
</table>



