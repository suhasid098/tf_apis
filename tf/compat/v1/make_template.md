description: Given an arbitrary function, wrap it so that it does variable sharing.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.make_template" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.make_template

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/template.py">View source</a>



Given an arbitrary function, wrap it so that it does variable sharing.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.make_template(
    name_,
    func_,
    create_scope_now_=False,
    unique_name_=None,
    custom_getter_=None,
    **kwargs
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

<a href="../../../tf/compat/v1/make_template.md"><code>tf.compat.v1.make_template</code></a> is a legacy API that is only compatible
with eager execution enabled and <a href="../../../tf/function.md"><code>tf.function</code></a> if you combine it with
<a href="../../../tf/compat/v1/keras/utils/track_tf1_style_variables.md"><code>tf.compat.v1.keras.utils.track_tf1_style_variables</code></a>. See the model mapping
migration guide section on `make_template` for more info:

https://www.tensorflow.org/guide/migrate/model_mapping#using_tfcompatv1make_template_in_the_decorated_method

Even if you use legacy apis for `variable_scope`-based variable reuse,
we recommend using
<a href="../../../tf/compat/v1/keras/utils/track_tf1_style_variables.md"><code>tf.compat.v1.keras.utils.track_tf1_style_variables</code></a> directly and not using
<a href="../../../tf/compat/v1/make_template.md"><code>tf.compat.v1.make_template</code></a>, as it interoperates with eager execution in a
simpler and more predictable fashion than `make_template`.

The TF2 API approach would be tracking your variables using
<a href="../../../tf/Module.md"><code>tf.Module</code></a>s or Keras layers and models rather than relying on
`make_template`.


 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->



This wraps `func_` in a Template and partially evaluates it. Templates are
functions that create variables the first time they are called and reuse them
thereafter. In order for `func_` to be compatible with a `Template` it must
have the following properties:

* The function should create all trainable variables and any variables that
   should be reused by calling <a href="../../../tf/compat/v1/get_variable.md"><code>tf.compat.v1.get_variable</code></a>. If a trainable
   variable is
   created using <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>, then a ValueError will be thrown. Variables
   that are intended to be locals can be created by specifying
   <a href="../../../tf/Variable.md"><code>tf.Variable(..., trainable=false)</code></a>.
* The function may use variable scopes and other templates internally to
    create and reuse variables, but it shouldn't use
    <a href="../../../tf/compat/v1/global_variables.md"><code>tf.compat.v1.global_variables</code></a> to
    capture variables that are defined outside of the scope of the function.
* Internal scopes and variable names should not depend on any arguments that
    are not supplied to `make_template`. In general you will get a ValueError
    telling you that you are trying to reuse a variable that doesn't exist
    if you make a mistake.

In the following example, both `z` and `w` will be scaled by the same `y`. It
is important to note that if we didn't assign `scalar_name` and used a
different name for z and w that a `ValueError` would be thrown because it
couldn't reuse the variable.

```python
def my_op(x, scalar_name):
  var1 = tf.compat.v1.get_variable(scalar_name,
                         shape=[],
                         initializer=tf.compat.v1.constant_initializer(1))
  return x * var1

scale_by_y = tf.compat.v1.make_template('scale_by_y', my_op, scalar_name='y')

z = scale_by_y(input1)
w = scale_by_y(input2)
```

As a safe-guard, the returned function will raise a `ValueError` after the
first call if trainable variables are created by calling <a href="../../../tf/Variable.md"><code>tf.Variable</code></a>.

If all of these are true, then 2 properties are enforced by the template:

1. Calling the same template multiple times will share all non-local
    variables.
2. Two different templates are guaranteed to be unique, unless you reenter the
    same variable scope as the initial definition of a template and redefine
    it. An examples of this exception:

```python
def my_op(x, scalar_name):
  var1 = tf.compat.v1.get_variable(scalar_name,
                         shape=[],
                         initializer=tf.compat.v1.constant_initializer(1))
  return x * var1

with tf.compat.v1.variable_scope('scope') as vs:
  scale_by_y = tf.compat.v1.make_template('scale_by_y', my_op,
  scalar_name='y')
  z = scale_by_y(input1)
  w = scale_by_y(input2)

# Creates a template that reuses the variables above.
with tf.compat.v1.variable_scope(vs, reuse=True):
  scale_by_y2 = tf.compat.v1.make_template('scale_by_y', my_op,
  scalar_name='y')
  z2 = scale_by_y2(input1)
  w2 = scale_by_y2(input2)
```

Depending on the value of `create_scope_now_`, the full variable scope may be
captured either at the time of first call or at the time of construction. If
this option is set to True, then all Tensors created by repeated calls to the
template will have an extra trailing _N+1 to their name, as the first time the
scope is entered in the Template constructor no Tensors are created.

Note: `name_`, `func_` and `create_scope_now_` have a trailing underscore to
reduce the likelihood of collisions with kwargs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name_`
</td>
<td>
A name for the scope created by this template. If necessary, the name
will be made unique by appending `_N` to the name.
</td>
</tr><tr>
<td>
`func_`
</td>
<td>
The function to wrap.
</td>
</tr><tr>
<td>
`create_scope_now_`
</td>
<td>
Boolean controlling whether the scope should be created
when the template is constructed or when the template is called. Default
is False, meaning the scope is created when the template is called.
</td>
</tr><tr>
<td>
`unique_name_`
</td>
<td>
When used, it overrides name_ and is not made unique. If a
template of the same scope/unique_name already exists and reuse is false,
an error is raised. Defaults to None.
</td>
</tr><tr>
<td>
`custom_getter_`
</td>
<td>
Optional custom getter for variables used in `func_`. See
the <a href="../../../tf/compat/v1/get_variable.md"><code>tf.compat.v1.get_variable</code></a> `custom_getter` documentation for more
information.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword arguments to apply to `func_`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A function to encapsulate a set of variables which should be created once
and reused. An enclosing scope will be created either when `make_template`
is called or when the result is called, depending on the value of
`create_scope_now_`. Regardless of the value, the first time the template
is called it will enter the scope with no reuse, and call `func_` to create
variables, which are guaranteed to be unique. All subsequent calls will
re-enter the scope and reuse those variables.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `name_` is None.
</td>
</tr>
</table>

