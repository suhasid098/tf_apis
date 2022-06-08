description: Adds methods that call original methods with eager_op_as_function enabled.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.test.with_eager_op_as_function" />
<meta itemprop="path" content="Stable" />
</div>

# tf.test.with_eager_op_as_function

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/test_util.py">View source</a>



Adds methods that call original methods with eager_op_as_function enabled.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.test.with_eager_op_as_function`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.test.with_eager_op_as_function(
    cls=None, only_as_function=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Example:



@test_util.with_eager_op_as_function
class SessionTest(test.TestCase):

  def testEnabledForEagerOpAsFunction(self):
    ...

  @disable_eager_op_as_function("b/xyzabc")
  def testDisabledForEagerOpAsFunction(self):
    ...

#### Generated class:


class SessionTest(test.TestCase):

  def testEnabledForEagerOpAsFunction(self):
    ...

  def testEnabledForEagerOpAsFunctionWithEagerOpAsFunctionEnabled(self):
    // Enable run_eager_op_as_function
    // Reset context
    testEnabledForEagerOpAsFunction(self)
    // Disable run_eager_op_as_function
    // Reset context

  def testDisabledForEagerOpAsFunction(self):
    ...

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cls`
</td>
<td>
class to decorate.
</td>
</tr><tr>
<td>
`only_as_function`
</td>
<td>
whether to run all the tests in the TestCase in eager mode
and in eager_op_as_function mode. By default it will run all tests in both
modes. When `only_as_function=True` tests will not be run in eager mode.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
cls with new test methods added.
</td>
</tr>

</table>

