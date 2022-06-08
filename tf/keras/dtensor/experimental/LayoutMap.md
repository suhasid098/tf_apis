description: A dict-like object that maps string to Layout instances.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.dtensor.experimental.LayoutMap" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="get_default_mesh"/>
<meta itemprop="property" content="items"/>
<meta itemprop="property" content="keys"/>
<meta itemprop="property" content="pop"/>
<meta itemprop="property" content="popitem"/>
<meta itemprop="property" content="setdefault"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="values"/>
</div>

# tf.keras.dtensor.experimental.LayoutMap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/layout_map.py#L47-L136">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A dict-like object that maps string to `Layout` instances.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.dtensor.experimental.LayoutMap(
    mesh=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

`LayoutMap` uses a string as key and a `Layout` as value. There is a behavior
difference between a normal Python dict and this class. The string key will be
treated as a regex when retrieving the value. See the docstring of
`get` for more details.

See below for a usage example. You can define the naming schema
of the `Layout`, and then retrieve the corresponding `Layout` instance.

To use the `LayoutMap` with a `Model`, please see the docstring of
<a href="../../../../tf/keras/dtensor/experimental/layout_map_scope.md"><code>tf.keras.dtensor.experimental.layout_map_scope</code></a>.

```python
map = LayoutMap(mesh=None)
map['.*dense.*kernel'] = layout_2d
map['.*dense.*bias'] = layout_1d
map['.*conv2d.*kernel'] = layout_4d
map['.*conv2d.*bias'] = layout_1d

layout_1 = map['dense_1.kernel']    #   layout_1 == layout_2d
layout_2 = map['dense_1.bias']      #   layout_2 == layout_1d
layout_3 = map['dense_2.kernel']    #   layout_3 == layout_2d
layout_4 = map['dense_2.bias']      #   layout_4 == layout_1d
layout_5 = map['my_model/conv2d_123/kernel']    #   layout_5 == layout_4d
layout_6 = map['my_model/conv2d_123/bias']      #   layout_6 == layout_1d
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh`
</td>
<td>
An optional `Mesh` that can be used to create all replicated
layout as default when there isn't a layout found based on the input
string query.
</td>
</tr>
</table>



## Methods

<h3 id="clear"><code>clear</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>

D.clear() -> None.  Remove all items from D.


<h3 id="get"><code>get</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    key, default=None
)
</code></pre>

Retrieve the corresponding layout by the string key.

When there isn't an exact match, all the existing keys in the layout map
will be treated as a regex and map against the input key again. The first
match will be returned, based on the key insertion order. Return None if
there isn't any match found.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
the string key as the query for the layout.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Corresponding layout based on the query.
</td>
</tr>

</table>



<h3 id="get_default_mesh"><code>get_default_mesh</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/layout_map.py#L130-L136">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_default_mesh()
</code></pre>

Return the default `Mesh` set at instance creation.

The `Mesh` can be used to create default replicated `Layout` when there
isn't a match of the input string query.

<h3 id="items"><code>items</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>items()
</code></pre>

D.items() -> a set-like object providing a view on D's items


<h3 id="keys"><code>keys</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>keys()
</code></pre>

D.keys() -> a set-like object providing a view on D's keys


<h3 id="pop"><code>pop</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pop(
    key, default=__marker
)
</code></pre>

D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised.

<h3 id="popitem"><code>popitem</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>popitem()
</code></pre>

D.popitem() -> (k, v), remove and return some (key, value) pair
as a 2-tuple; but raise KeyError if D is empty.

<h3 id="setdefault"><code>setdefault</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>setdefault(
    key, default=None
)
</code></pre>

D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D


<h3 id="update"><code>update</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    other, /, **kwds
)
</code></pre>

D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v

<h3 id="values"><code>values</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>values()
</code></pre>

D.values() -> an object providing a view on D's values


<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key
)
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/layout_map.py#L87-L107">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key
)
</code></pre>

Retrieve the corresponding layout by the string key.

When there isn't an exact match, all the existing keys in the layout map
will be treated as a regex and map against the input key again. The first
match will be returned, based on the key insertion order. Return None if
there isn't any match found.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`key`
</td>
<td>
the string key as the query for the layout.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Corresponding layout based on the query.
</td>
</tr>

</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/layout_map.py#L127-L128">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>




<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/dtensor/layout_map.py#L124-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>






