description: Functions that work with structures.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nest" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.nest

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Functions that work with structures.



#### A structure is either:



* one of the recognized Python collections, holding _nested structures_;
* a value of any other type, typically a TensorFlow data type like Tensor,
  Variable, or of compatible types such as int, float, ndarray, etc. these are
  commonly referred to as _atoms_ of the structure.

A structure of type `T` is a structure whose atomic items are of type `T`.
For example, a structure of <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> only contains <a href="../tf/Tensor.md"><code>tf.Tensor</code></a> as its atoms.

Historically a _nested structure_ was called a _nested sequence_ in TensorFlow.
A nested structure is sometimes called a _nest_ or a _tree_, but the formal
name _nested structure_ is preferred.

Refer to [Nesting Data Structures]
(https://en.wikipedia.org/wiki/Nesting_(computing)#Data_structures).

The following collection types are recognized by <a href="../tf/nest.md"><code>tf.nest</code></a> as nested
structures:

* `collections.abc.Sequence` (except `string` and `bytes`).
  This includes `list`, `tuple`, and `namedtuple`.
* `collections.abc.Mapping` (with sortable keys).
  This includes `dict` and `collections.OrderedDict`.
* `collections.abc.MappingView` (with sortable keys).
* [`attr.s` classes](https://www.attrs.org/).

Any other values are considered **atoms**.  Not all collection types are
considered nested structures.  For example, the following types are
considered atoms:

* `set`; `{"a", "b"}` is an atom, while `["a", "b"]` is a nested structure.
* [`dataclass` classes](https://docs.python.org/library/dataclasses.html)
* <a href="../tf/Tensor.md"><code>tf.Tensor</code></a>
* `numpy.array`

<a href="../tf/nest/is_nested.md"><code>tf.nest.is_nested</code></a> checks whether an object is a nested structure or an atom.
For example:

  ```
  >>> tf.nest.is_nested("1234")
  False
  >>> tf.nest.is_nested([1, 3, [4, 5]])
  True
  >>> tf.nest.is_nested(((7, 8), (5, 6)))
  True
  >>> tf.nest.is_nested([])
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2})
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.keys())
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.values())
  True
  >>> tf.nest.is_nested({"a": 1, "b": 2}.items())
  True
  >>> tf.nest.is_nested(set([1, 2]))
  False
  >>> ones = tf.ones([2, 3])
  >>> tf.nest.is_nested(ones)
  False
  ```

Note: A proper structure shall form a tree. The user shall ensure there is no
cyclic references within the items in the structure,
i.e., no references in the structure of the input of these functions
should be recursive. The behavior is undefined if there is a cycle.

## Functions

[`assert_same_structure(...)`](../tf/nest/assert_same_structure.md): Asserts that two structures are nested in the same way.

[`flatten(...)`](../tf/nest/flatten.md): Returns a flat list from a given structure.

[`is_nested(...)`](../tf/nest/is_nested.md): Returns true if its input is a nested structure.

[`map_structure(...)`](../tf/nest/map_structure.md): Creates a new structure by applying `func` to each atom in `structure`.

[`pack_sequence_as(...)`](../tf/nest/pack_sequence_as.md): Returns a given flattened sequence packed into a given structure.

