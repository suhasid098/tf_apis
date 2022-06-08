description: Half-precision floating-point number type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.numpy.float16" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__abs__"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__and__"/>
<meta itemprop="property" content="__array__"/>
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__floordiv__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__invert__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lshift__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__mod__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__neg__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="__or__"/>
<meta itemprop="property" content="__pos__"/>
<meta itemprop="property" content="__pow__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rand__"/>
<meta itemprop="property" content="__rfloordiv__"/>
<meta itemprop="property" content="__rlshift__"/>
<meta itemprop="property" content="__rmod__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="__ror__"/>
<meta itemprop="property" content="__rpow__"/>
<meta itemprop="property" content="__rrshift__"/>
<meta itemprop="property" content="__rshift__"/>
<meta itemprop="property" content="__rsub__"/>
<meta itemprop="property" content="__rtruediv__"/>
<meta itemprop="property" content="__rxor__"/>
<meta itemprop="property" content="__sub__"/>
<meta itemprop="property" content="__truediv__"/>
<meta itemprop="property" content="__xor__"/>
<meta itemprop="property" content="all"/>
<meta itemprop="property" content="any"/>
<meta itemprop="property" content="argmax"/>
<meta itemprop="property" content="argmin"/>
<meta itemprop="property" content="argsort"/>
<meta itemprop="property" content="as_integer_ratio"/>
<meta itemprop="property" content="astype"/>
<meta itemprop="property" content="byteswap"/>
<meta itemprop="property" content="choose"/>
<meta itemprop="property" content="clip"/>
<meta itemprop="property" content="compress"/>
<meta itemprop="property" content="conj"/>
<meta itemprop="property" content="conjugate"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="cumprod"/>
<meta itemprop="property" content="cumsum"/>
<meta itemprop="property" content="diagonal"/>
<meta itemprop="property" content="dump"/>
<meta itemprop="property" content="dumps"/>
<meta itemprop="property" content="fill"/>
<meta itemprop="property" content="flatten"/>
<meta itemprop="property" content="getfield"/>
<meta itemprop="property" content="is_integer"/>
<meta itemprop="property" content="item"/>
<meta itemprop="property" content="itemset"/>
<meta itemprop="property" content="max"/>
<meta itemprop="property" content="mean"/>
<meta itemprop="property" content="min"/>
<meta itemprop="property" content="newbyteorder"/>
<meta itemprop="property" content="nonzero"/>
<meta itemprop="property" content="prod"/>
<meta itemprop="property" content="ptp"/>
<meta itemprop="property" content="put"/>
<meta itemprop="property" content="ravel"/>
<meta itemprop="property" content="repeat"/>
<meta itemprop="property" content="reshape"/>
<meta itemprop="property" content="resize"/>
<meta itemprop="property" content="round"/>
<meta itemprop="property" content="searchsorted"/>
<meta itemprop="property" content="setfield"/>
<meta itemprop="property" content="setflags"/>
<meta itemprop="property" content="sort"/>
<meta itemprop="property" content="squeeze"/>
<meta itemprop="property" content="std"/>
<meta itemprop="property" content="sum"/>
<meta itemprop="property" content="swapaxes"/>
<meta itemprop="property" content="take"/>
<meta itemprop="property" content="tobytes"/>
<meta itemprop="property" content="tofile"/>
<meta itemprop="property" content="tolist"/>
<meta itemprop="property" content="tostring"/>
<meta itemprop="property" content="trace"/>
<meta itemprop="property" content="transpose"/>
<meta itemprop="property" content="var"/>
<meta itemprop="property" content="view"/>
</div>

# tf.experimental.numpy.float16

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Half-precision floating-point number type.

Inherits From: [`inexact`](../../../tf/experimental/numpy/inexact.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.numpy.float16(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

:Character code: ``'e'``
:Canonical name: `numpy.half`
:Alias on this platform (Windows AMD64): `numpy.float16`: 16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`T`
</td>
<td>
Scalar attribute identical to the corresponding array attribute.

Please see `ndarray.T`.
</td>
</tr><tr>
<td>
`base`
</td>
<td>
Scalar attribute identical to the corresponding array attribute.

Please see `ndarray.base`.
</td>
</tr><tr>
<td>
`data`
</td>
<td>
Pointer to start of data.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
Get array data-descriptor.
</td>
</tr><tr>
<td>
`flags`
</td>
<td>
The integer value of flags.
</td>
</tr><tr>
<td>
`flat`
</td>
<td>
A 1-D view of the scalar.
</td>
</tr><tr>
<td>
`imag`
</td>
<td>
The imaginary part of the scalar.
</td>
</tr><tr>
<td>
`itemsize`
</td>
<td>
The length of one element in bytes.
</td>
</tr><tr>
<td>
`nbytes`
</td>
<td>
The length of the scalar in bytes.
</td>
</tr><tr>
<td>
`ndim`
</td>
<td>
The number of array dimensions.
</td>
</tr><tr>
<td>
`real`
</td>
<td>
The real part of the scalar.
</td>
</tr><tr>
<td>
`shape`
</td>
<td>
Tuple of array dimensions.
</td>
</tr><tr>
<td>
`size`
</td>
<td>
The number of elements in the gentype.
</td>
</tr><tr>
<td>
`strides`
</td>
<td>
Tuple of bytes steps in each dimension.
</td>
</tr>
</table>



## Methods

<h3 id="all"><code>all</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.all`.

<h3 id="any"><code>any</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>any()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.any`.

<h3 id="argmax"><code>argmax</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>argmax()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argmax`.

<h3 id="argmin"><code>argmin</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>argmin()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argmin`.

<h3 id="argsort"><code>argsort</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>argsort()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.argsort`.

<h3 id="as_integer_ratio"><code>as_integer_ratio</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>as_integer_ratio()
</code></pre>

half.as_integer_ratio() -> (int, int)

Return a pair of integers, whose ratio is exactly equal to the original
floating point number, and with a positive denominator.
Raise `OverflowError` on infinities and a `ValueError` on NaNs.

```
>>> np.half(10.0).as_integer_ratio()
(10, 1)
>>> np.half(0.0).as_integer_ratio()
(0, 1)
>>> np.half(-.25).as_integer_ratio()
(-1, 4)
```

<h3 id="astype"><code>astype</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>astype()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.astype`.

<h3 id="byteswap"><code>byteswap</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>byteswap()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.byteswap`.

<h3 id="choose"><code>choose</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>choose()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.choose`.

<h3 id="clip"><code>clip</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clip()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.clip`.

<h3 id="compress"><code>compress</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compress()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.compress`.

<h3 id="conj"><code>conj</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>conj()
</code></pre>




<h3 id="conjugate"><code>conjugate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>conjugate()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.conjugate`.

<h3 id="copy"><code>copy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.copy`.

<h3 id="cumprod"><code>cumprod</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cumprod()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.cumprod`.

<h3 id="cumsum"><code>cumsum</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cumsum()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.cumsum`.

<h3 id="diagonal"><code>diagonal</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>diagonal()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.diagonal`.

<h3 id="dump"><code>dump</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dump()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.dump`.

<h3 id="dumps"><code>dumps</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>dumps()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.dumps`.

<h3 id="fill"><code>fill</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fill()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.fill`.

<h3 id="flatten"><code>flatten</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flatten()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.flatten`.

<h3 id="getfield"><code>getfield</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>getfield()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.getfield`.

<h3 id="is_integer"><code>is_integer</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_integer()
</code></pre>

half.is_integer() -> bool

Return ``True`` if the floating point number is finite with integral
value, and ``False`` otherwise.

.. versionadded:: 1.22

Examples
--------
```
>>> np.half(-2.0).is_integer()
True
>>> np.half(3.2).is_integer()
False
```

<h3 id="item"><code>item</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>item()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.item`.

<h3 id="itemset"><code>itemset</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>itemset()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.itemset`.

<h3 id="max"><code>max</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>max()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.max`.

<h3 id="mean"><code>mean</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mean()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.mean`.

<h3 id="min"><code>min</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>min()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.min`.

<h3 id="newbyteorder"><code>newbyteorder</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>newbyteorder()
</code></pre>

newbyteorder(new_order='S', /)

Return a new `dtype` with a different byte order.

Changes are also made in all fields and sub-arrays of the data type.

The `new_order` code can be any from the following:

* 'S' - swap dtype from current to opposite endian
* {'<', 'little'} - little endian
* {'>', 'big'} - big endian
* {'=', 'native'} - native order
* {'|', 'I'} - ignore (no change to byte order)

Parameters
----------
new_order : str, optional
    Byte order to force; a value from the byte order specifications
    above.  The default value ('S') results in swapping the current
    byte order.


Returns
-------
new_dtype : dtype
    New `dtype` object with the given change to the byte order.

<h3 id="nonzero"><code>nonzero</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>nonzero()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.nonzero`.

<h3 id="prod"><code>prod</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prod()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.prod`.

<h3 id="ptp"><code>ptp</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>ptp()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.ptp`.

<h3 id="put"><code>put</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>put()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.put`.

<h3 id="ravel"><code>ravel</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>ravel()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.ravel`.

<h3 id="repeat"><code>repeat</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>repeat()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.repeat`.

<h3 id="reshape"><code>reshape</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reshape()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.reshape`.

<h3 id="resize"><code>resize</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>resize()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.resize`.

<h3 id="round"><code>round</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>round()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.round`.

<h3 id="searchsorted"><code>searchsorted</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>searchsorted()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.searchsorted`.

<h3 id="setfield"><code>setfield</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>setfield()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.setfield`.

<h3 id="setflags"><code>setflags</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>setflags()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.setflags`.

<h3 id="sort"><code>sort</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sort()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.sort`.

<h3 id="squeeze"><code>squeeze</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>squeeze()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.squeeze`.

<h3 id="std"><code>std</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>std()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.std`.

<h3 id="sum"><code>sum</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sum()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.sum`.

<h3 id="swapaxes"><code>swapaxes</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>swapaxes()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.swapaxes`.

<h3 id="take"><code>take</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>take()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.take`.

<h3 id="tobytes"><code>tobytes</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tobytes()
</code></pre>




<h3 id="tofile"><code>tofile</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tofile()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tofile`.

<h3 id="tolist"><code>tolist</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tolist()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tolist`.

<h3 id="tostring"><code>tostring</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tostring()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.tostring`.

<h3 id="trace"><code>trace</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>trace()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.trace`.

<h3 id="transpose"><code>transpose</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>transpose()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.transpose`.

<h3 id="var"><code>var</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>var()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.var`.

<h3 id="view"><code>view</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>view()
</code></pre>

Scalar method identical to the corresponding array attribute.

Please see `ndarray.view`.

<h3 id="__abs__"><code>__abs__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__abs__()
</code></pre>

abs(self)


<h3 id="__add__"><code>__add__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__add__(
    value, /
)
</code></pre>

Return self+value.


<h3 id="__and__"><code>__and__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__and__(
    value, /
)
</code></pre>

Return self&value.


<h3 id="__array__"><code>__array__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__array__()
</code></pre>

sc.__array__(dtype) return 0-dim array from scalar with specified dtype


<h3 id="__bool__"><code>__bool__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>

self != 0


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    value, /
)
</code></pre>

Return self==value.


<h3 id="__floordiv__"><code>__floordiv__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__floordiv__(
    value, /
)
</code></pre>

Return self//value.


<h3 id="__ge__"><code>__ge__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    value, /
)
</code></pre>

Return self>=value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    key, /
)
</code></pre>

Return self[key].


<h3 id="__gt__"><code>__gt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    value, /
)
</code></pre>

Return self>value.


<h3 id="__invert__"><code>__invert__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__invert__()
</code></pre>

~self


<h3 id="__le__"><code>__le__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    value, /
)
</code></pre>

Return self<=value.


<h3 id="__lshift__"><code>__lshift__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lshift__(
    value, /
)
</code></pre>

Return self<<value.


<h3 id="__lt__"><code>__lt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    value, /
)
</code></pre>

Return self<value.


<h3 id="__mod__"><code>__mod__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mod__(
    value, /
)
</code></pre>

Return self%value.


<h3 id="__mul__"><code>__mul__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__mul__(
    value, /
)
</code></pre>

Return self*value.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    value, /
)
</code></pre>

Return self!=value.


<h3 id="__neg__"><code>__neg__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__neg__()
</code></pre>

-self


<h3 id="__or__"><code>__or__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__or__(
    value, /
)
</code></pre>

Return self|value.


<h3 id="__pos__"><code>__pos__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pos__()
</code></pre>

+self


<h3 id="__pow__"><code>__pow__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__pow__(
    value, mod, /
)
</code></pre>

Return pow(self, value, mod).


<h3 id="__radd__"><code>__radd__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__radd__(
    value, /
)
</code></pre>

Return value+self.


<h3 id="__rand__"><code>__rand__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rand__(
    value, /
)
</code></pre>

Return value&self.


<h3 id="__rfloordiv__"><code>__rfloordiv__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rfloordiv__(
    value, /
)
</code></pre>

Return value//self.


<h3 id="__rlshift__"><code>__rlshift__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rlshift__(
    value, /
)
</code></pre>

Return value<<self.


<h3 id="__rmod__"><code>__rmod__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmod__(
    value, /
)
</code></pre>

Return value%self.


<h3 id="__rmul__"><code>__rmul__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rmul__(
    value, /
)
</code></pre>

Return value*self.


<h3 id="__ror__"><code>__ror__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ror__(
    value, /
)
</code></pre>

Return value|self.


<h3 id="__rpow__"><code>__rpow__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rpow__(
    value, mod, /
)
</code></pre>

Return pow(value, self, mod).


<h3 id="__rrshift__"><code>__rrshift__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rrshift__(
    value, /
)
</code></pre>

Return value>>self.


<h3 id="__rshift__"><code>__rshift__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rshift__(
    value, /
)
</code></pre>

Return self>>value.


<h3 id="__rsub__"><code>__rsub__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rsub__(
    value, /
)
</code></pre>

Return value-self.


<h3 id="__rtruediv__"><code>__rtruediv__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rtruediv__(
    value, /
)
</code></pre>

Return value/self.


<h3 id="__rxor__"><code>__rxor__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__rxor__(
    value, /
)
</code></pre>

Return value^self.


<h3 id="__sub__"><code>__sub__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__sub__(
    value, /
)
</code></pre>

Return self-value.


<h3 id="__truediv__"><code>__truediv__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__truediv__(
    value, /
)
</code></pre>

Return self/value.


<h3 id="__xor__"><code>__xor__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__xor__(
    value, /
)
</code></pre>

Return self^value.




