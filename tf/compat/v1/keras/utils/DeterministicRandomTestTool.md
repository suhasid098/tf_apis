description: DeterministicRandomTestTool is a testing tool.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.keras.utils.DeterministicRandomTestTool" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="scope"/>
</div>

# tf.compat.v1.keras.utils.DeterministicRandomTestTool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/migration_utils.py#L15-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



DeterministicRandomTestTool is a testing tool.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.keras.utils.DeterministicRandomTestTool(
    seed: int = 42, mode=&#x27;constant&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This tool is used to validate random number generation semantics match between
TF1.x graphs/sessions and eager execution.

This is useful when you are migrating from TF 1.x to TF2 and need to make sure
your computation is still happening correctly along the way. See the
validating correctness migration guide for more info :
https://www.tensorflow.org/guide/migrate/validate_correctness

The following DeterministicRandomTestTool object provides a context manager
scope() that can make stateful random operations use the same seed across both
TF1 graphs/sessions and eager execution,The tool provides two testing modes:
- constant which uses the same seed for every single operation no matter how
many times it has been called and,
- num_random_ops which uses the number of previously-observed stateful random
operations as the operation seed.
The num_random_ops mode serves as a more sensitive validation check than the
constant mode. It ensures that the random numbers initialization does not get
accidentaly reused.(for example if several weights take on the same
initializations), you can use the num_random_ops mode to avoid this. In the
num_random_ops mode, the generated random numbers will depend on the ordering
of random ops in the program.

This applies both to the stateful random operations used for creating and
initializing variables, and to the stateful random operations used in
computation (such as for dropout layers).



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`operation_seed`
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="scope"><code>scope</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.9.0/keras/legacy_tf_layers/migration_utils.py#L65-L103">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>scope()
</code></pre>

set random seed.




