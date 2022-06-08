description: Represents the type of object(s) for tf.function tracing purposes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.TraceType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
</div>

# tf.types.experimental.TraceType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>



Represents the type of object(s) for tf.function tracing purposes.

<!-- Placeholder for "Used in" -->

`TraceType` is an abstract class that other classes might inherit from to
provide information regarding associated class(es) for the purposes of
tf.function tracing. The typing logic provided through this mechanism will be
used to make decisions regarding usage of cached concrete functions and
retracing.

For example, if we have the following tf.function and classes:
```python
@tf.function
def get_mixed_flavor(fruit_a, fruit_b):
  return fruit_a.flavor + fruit_b.flavor

class Fruit:
  flavor = tf.constant([0, 0])

class Apple(Fruit):
  flavor = tf.constant([1, 2])

class Mango(Fruit):
  flavor = tf.constant([3, 4])
```

tf.function does not know when to re-use an existing concrete function in
regards to the `Fruit` class so naively it retraces for every new instance.
```python
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function again
```

However, we, as the designers of the `Fruit` class, know that each subclass
has a fixed flavor and we can reuse an existing traced concrete function if
it was the same subclass. Avoiding such unnecessary tracing of concrete
functions can have significant performance benefits.

```python
class FruitTraceType(tf.types.experimental.TraceType):
  def __init__(self, fruit_type):
    self.fruit_type = fruit_type

  def is_subtype_of(self, other):
     return (type(other) is FruitTraceType and
             self.fruit_type is other.fruit_type)

  def most_specific_common_supertype(self, others):
     return self if all(self == other for other in others) else None

class Fruit:

 def __tf_tracing_type__(self, context):
   return FruitTraceType(type(self))
```

Now if we try calling it again:
```python
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
get_mixed_flavor(Apple(), Mango()) # Re-uses the traced concrete function
```

## Methods

<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>is_subtype_of(
    other: 'TraceType'
) -> bool
</code></pre>

Returns True if `self` is a subtype of `other`.

For example, <a href="../../../tf/function.md"><code>tf.function</code></a> uses subtyping for dispatch:
if `a.is_subtype_of(b)` is True, then an argument of `TraceType`
`a` can be used as argument to a `ConcreteFunction` traced with an
a `TraceType` `b`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A TraceType object to be compared against.
</td>
</tr>
</table>



#### Example:



```python
class Dimension(TraceType):
  def __init__(self, value: Optional[int]):
    self.value = value

  def is_subtype_of(self, other):
    # Either the value is the same or other has a generalized value that
    # can represent any specific ones.
    return (self.value == other.value) or (other.value is None)
```

<h3 id="most_specific_common_supertype"><code>most_specific_common_supertype</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>most_specific_common_supertype(
    others: Sequence['TraceType']
) -> Optional['TraceType']
</code></pre>

Returns the most specific supertype of `self` and `others`, if exists.

The returned `TraceType` is a supertype of `self` and `others`, that is,
they are all subtypes (see `is_subtype_of`) of it.
It is also most specific, that is, there it has no subtype that is also
a common supertype of `self` and `others`.

If `self` and `others` have no common supertype, this returns `None`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`others`
</td>
<td>
A sequence of TraceTypes.
</td>
</tr>
</table>



#### Example:


```python
 class Dimension(TraceType):
   def __init__(self, value: Optional[int]):
     self.value = value

   def most_specific_common_supertype(self, other):
      # Either the value is the same or other has a generalized value that
      # can represent any specific ones.
      if self.value == other.value:
        return self.value
      else:
        return Dimension(None)
```

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.




