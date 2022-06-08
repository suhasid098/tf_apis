description: This enumeration represents optional conversion options.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.autograph.experimental.Feature" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ALL"/>
<meta itemprop="property" content="ASSERT_STATEMENTS"/>
<meta itemprop="property" content="AUTO_CONTROL_DEPS"/>
<meta itemprop="property" content="BUILTIN_FUNCTIONS"/>
<meta itemprop="property" content="EQUALITY_OPERATORS"/>
<meta itemprop="property" content="LISTS"/>
<meta itemprop="property" content="NAME_SCOPES"/>
</div>

# tf.autograph.experimental.Feature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/autograph/core/converter.py">View source</a>



This enumeration represents optional conversion options.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.autograph.experimental.Feature`</p>
</p>
</section>

<!-- Placeholder for "Used in" -->

These conversion options are experimental. They are subject to change without
notice and offer no guarantees.

_Example Usage_

```python
optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
@tf.function(experimental_autograph_options=optionals)
def f(i):
  if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
    tf.print('i is zero')
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`ALL`
</td>
<td>
Enable all features.
</td>
</tr><tr>
<td>
`AUTO_CONTROL_DEPS`
</td>
<td>
Insert of control dependencies in the generated code.
</td>
</tr><tr>
<td>
`ASSERT_STATEMENTS`
</td>
<td>
Convert Tensor-dependent assert statements to tf.Assert.
</td>
</tr><tr>
<td>
`BUILTIN_FUNCTIONS`
</td>
<td>
Convert builtin functions applied to Tensors to
their TF counterparts.
</td>
</tr><tr>
<td>
`EQUALITY_OPERATORS`
</td>
<td>
Whether to convert the comparison operators, like
equality. This is soon to be deprecated as support is being added to the
Tensor class.
</td>
</tr><tr>
<td>
`LISTS`
</td>
<td>
Convert list idioms, like initializers, slices, append, etc.
</td>
</tr><tr>
<td>
`NAME_SCOPES`
</td>
<td>
Insert name scopes that name ops according to context, like the
function they were defined in.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
ALL<a id="ALL"></a>
</td>
<td>
`<Feature.ALL: 'ALL'>`
</td>
</tr><tr>
<td>
ASSERT_STATEMENTS<a id="ASSERT_STATEMENTS"></a>
</td>
<td>
`<Feature.ASSERT_STATEMENTS: 'ASSERT_STATEMENTS'>`
</td>
</tr><tr>
<td>
AUTO_CONTROL_DEPS<a id="AUTO_CONTROL_DEPS"></a>
</td>
<td>
`<Feature.AUTO_CONTROL_DEPS: 'AUTO_CONTROL_DEPS'>`
</td>
</tr><tr>
<td>
BUILTIN_FUNCTIONS<a id="BUILTIN_FUNCTIONS"></a>
</td>
<td>
`<Feature.BUILTIN_FUNCTIONS: 'BUILTIN_FUNCTIONS'>`
</td>
</tr><tr>
<td>
EQUALITY_OPERATORS<a id="EQUALITY_OPERATORS"></a>
</td>
<td>
`<Feature.EQUALITY_OPERATORS: 'EQUALITY_OPERATORS'>`
</td>
</tr><tr>
<td>
LISTS<a id="LISTS"></a>
</td>
<td>
`<Feature.LISTS: 'LISTS'>`
</td>
</tr><tr>
<td>
NAME_SCOPES<a id="NAME_SCOPES"></a>
</td>
<td>
`<Feature.NAME_SCOPES: 'NAME_SCOPES'>`
</td>
</tr>
</table>

