description: Returns the batched diagonal part of a batched tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.linalg.diag_part" />
<meta itemprop="path" content="Stable" />
</div>

# tf.linalg.diag_part

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/array_ops.py">View source</a>



Returns the batched diagonal part of a batched tensor.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.linalg.diag_part`, `tf.compat.v1.matrix_diag_part`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.linalg.diag_part(
    input,
    name=&#x27;diag_part&#x27;,
    k=0,
    padding_value=0,
    align=&#x27;RIGHT_LEFT&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns a tensor with the `k[0]`-th to `k[1]`-th diagonals of the batched
`input`.

Assume `input` has `r` dimensions `[I, J, ..., L, M, N]`.
Let `max_diag_len` be the maximum length among all diagonals to be extracted,
`max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`
Let `num_diags` be the number of diagonals to extract,
`num_diags = k[1] - k[0] + 1`.

If `num_diags == 1`, the output tensor is of rank `r - 1` with shape
`[I, J, ..., L, max_diag_len]` and values:

```
diagonal[i, j, ..., l, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```
where `y = max(-k[1], 0)`, `x = max(k[1], 0)`.

Otherwise, the output tensor has rank `r` with dimensions
`[I, J, ..., L, num_diags, max_diag_len]` with values:

```
diagonal[i, j, ..., l, m, n]
  = input[i, j, ..., l, n+y, n+x] ; if 0 <= n+y < M and 0 <= n+x < N,
    padding_value                 ; otherwise.
```
where `d = k[1] - m`, `y = max(-d, 0) - offset`, and `x = max(d, 0) - offset`.

`offset` is zero except when the alignment of the diagonal is to the right.
```
offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                           and `d >= 0`) or
                                         (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                           and `d <= 0`)
         0                          ; otherwise
```
where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

The input must be at least a matrix.

#### For example:



```
input = np.array([[[1, 2, 3, 4],  # Input shape: (2, 3, 4)
                   [5, 6, 7, 8],
                   [9, 8, 7, 6]],
                  [[5, 4, 3, 2],
                   [1, 2, 3, 4],
                   [5, 6, 7, 8]]])

# A main diagonal from each batch.
tf.linalg.diag_part(input) ==> [[1, 6, 7],  # Output shape: (2, 3)
                                [5, 2, 7]]

# A superdiagonal from each batch.
tf.linalg.diag_part(input, k = 1)
  ==> [[2, 7, 6],  # Output shape: (2, 3)
       [4, 3, 8]]

# A band from each batch.
tf.linalg.diag_part(input, k = (-1, 2))
  ==> [[[3, 8, 0],  # Output shape: (2, 4, 3)
        [2, 7, 6],
        [1, 6, 7],
        [0, 5, 8]],
       [[3, 4, 0],
        [4, 3, 8],
        [5, 2, 7],
        [0, 1, 6]]]

# RIGHT_LEFT alignment.
tf.linalg.diag_part(input, k = (-1, 2), align="RIGHT_LEFT")
  ==> [[[0, 3, 8],  # Output shape: (2, 4, 3)
        [2, 7, 6],
        [1, 6, 7],
        [5, 8, 0]],
       [[0, 3, 4],
        [4, 3, 8],
        [5, 2, 7],
        [1, 6, 0]]]

# max_diag_len can be shorter than the main diagonal.
tf.linalg.diag_part(input, k = (-2, -1))
  ==> [[[5, 8],
        [0, 9]],
       [[1, 6],
        [0, 5]]]

# padding_value = 9
tf.linalg.diag_part(input, k = (1, 3), padding_value = 9)
  ==> [[[4, 9, 9],  # Output shape: (2, 3, 3)
        [3, 8, 9],
        [2, 7, 6]],
       [[2, 9, 9],
        [3, 4, 9],
        [4, 3, 8]]]

```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`
</td>
<td>
A `Tensor` with `rank k >= 2`.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`k`
</td>
<td>
Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
main diagonal, and negative value means subdiagonals. `k` can be a single
integer (for a single diagonal) or a pair of integers specifying the low
and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
</td>
</tr><tr>
<td>
`padding_value`
</td>
<td>
The value to fill the area outside the specified diagonal
band with. Default is 0.
</td>
</tr><tr>
<td>
`align`
</td>
<td>
Some diagonals are shorter than `max_diag_len` and need to be padded.
`align` is a string specifying how superdiagonals and subdiagonals should
be aligned, respectively. There are four possible alignments: "RIGHT_LEFT"
(default), "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT"
aligns superdiagonals to the right (left-pads the row) and subdiagonals to
the left (right-pads the row). It is the packing format LAPACK uses.
cuSPARSE uses "LEFT_RIGHT", which is the opposite alignment.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor containing diagonals of `input`. Has the same type as `input`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`
</td>
<td>
When `k` is out of bound or when `k[0]>k[1:]`.
</td>
</tr>
</table>

