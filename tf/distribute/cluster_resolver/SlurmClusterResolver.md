description: ClusterResolver for system with Slurm workload manager.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.cluster_resolver.SlurmClusterResolver" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="cluster_spec"/>
<meta itemprop="property" content="get_task_info"/>
<meta itemprop="property" content="master"/>
<meta itemprop="property" content="num_accelerators"/>
</div>

# tf.distribute.cluster_resolver.SlurmClusterResolver

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py">View source</a>



ClusterResolver for system with Slurm workload manager.

Inherits From: [`ClusterResolver`](../../../tf/distribute/cluster_resolver/ClusterResolver.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.distribute.cluster_resolver.SlurmClusterResolver`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.distribute.cluster_resolver.SlurmClusterResolver(
    jobs=None,
    port_base=8888,
    gpus_per_node=None,
    gpus_per_task=None,
    tasks_per_node=None,
    auto_set_gpu=True,
    rpc_layer=&#x27;grpc&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is an implementation of ClusterResolver for Slurm clusters. This allows
the specification of jobs and task counts, number of tasks per node, number
of GPUs on each node and number of GPUs for each task. It retrieves system
attributes by Slurm environment variables, resolves allocated computing node
names, constructs a cluster and returns a ClusterResolver object which can be
used for distributed TensorFlow.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`jobs`
</td>
<td>
Dictionary with job names as key and number of tasks in the job as
value. Defaults to as many 'worker's as there are (Slurm) tasks.
</td>
</tr><tr>
<td>
`port_base`
</td>
<td>
The first port number to start with for processes on a node.
</td>
</tr><tr>
<td>
`gpus_per_node`
</td>
<td>
Number of GPUs available on each node. Defaults to the
number of GPUs reported by nvidia-smi
</td>
</tr><tr>
<td>
`gpus_per_task`
</td>
<td>
Number of GPUs to be used for each task. Default is to
evenly distribute the gpus_per_node to tasks_per_node.
</td>
</tr><tr>
<td>
`tasks_per_node`
</td>
<td>
Number of tasks running on each node. Can be an integer if
the number of tasks per node is constant or a dictionary mapping
hostnames to number of tasks on that node. If not set the Slurm
environment is queried for the correct mapping.
</td>
</tr><tr>
<td>
`auto_set_gpu`
</td>
<td>
Set the visible CUDA devices automatically while resolving
the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
Defaults to True.
</td>
</tr><tr>
<td>
`rpc_layer`
</td>
<td>
The protocol TensorFlow used to communicate between nodes.
Defaults to 'grpc'.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If requested more GPUs per node then available or
requested more tasks then assigned tasks or
resolving missing values from the environment failed.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`environment`
</td>
<td>
Returns the current environment which TensorFlow is running in.

There are two possible return values, "google" (when TensorFlow is running
in a Google-internal environment) or an empty string (when TensorFlow is
running elsewhere).

If you are implementing a ClusterResolver that works in both the Google
environment and the open-source world (for instance, a TPU ClusterResolver
or similar), you will have to return the appropriate string depending on the
environment, which you will have to detect.

Otherwise, if you are implementing a ClusterResolver that will only work
in open-source TensorFlow, you do not need to implement this property.
</td>
</tr><tr>
<td>
`task_id`
</td>
<td>
Returns the task id this `ClusterResolver` indicates.

In TensorFlow distributed environment, each job may have an applicable
task id, which is the index of the instance within its task type. This is
useful when user needs to run specific code according to task index. For
example,

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                        task_id=0)

...

if cluster_resolver.task_type == 'worker' and cluster_resolver.task_id == 0:
  # Perform something that's only applicable on 'worker' type, id 0. This
  # block will run on this particular instance since we've specified this
  # task to be a 'worker', id 0 in above cluster resolver.
else:
  # Perform something that's only applicable on other ids. This block will
  # not run on this particular instance.
```

Returns `None` if such information is not available or is not applicable
in the current distributed environment, such as training with
<a href="../../../tf/distribute/cluster_resolver/TPUClusterResolver.md"><code>tf.distribute.cluster_resolver.TPUClusterResolver</code></a>.

For more information, please see
<a href="../../../tf/distribute/cluster_resolver/ClusterResolver.md"><code>tf.distribute.cluster_resolver.ClusterResolver</code></a>'s class docstring.
</td>
</tr><tr>
<td>
`task_type`
</td>
<td>
Returns the task type this `ClusterResolver` indicates.

In TensorFlow distributed environment, each job may have an applicable
task type. Valid task types in TensorFlow include
'chief': a worker that is designated with more responsibility,
'worker': a regular worker for training/evaluation,
'ps': a parameter server, or
'evaluator': an evaluator that evaluates the checkpoints for metrics.

See [Multi-worker configuration](
https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras#multi-worker_configuration)
for more information about 'chief' and 'worker' task type, which are most
commonly used.

Having access to such information is useful when user needs to run specific
code according to task types. For example,

```python
cluster_spec = tf.train.ClusterSpec({
    "ps": ["localhost:2222", "localhost:2223"],
    "worker": ["localhost:2224", "localhost:2225", "localhost:2226"]
})

# SimpleClusterResolver is used here for illustration; other cluster
# resolvers may be used for other source of task type/id.
simple_resolver = SimpleClusterResolver(cluster_spec, task_type="worker",
                                        task_id=1)

...

if cluster_resolver.task_type == 'worker':
  # Perform something that's only applicable on workers. This block
  # will run on this particular instance since we've specified this task to
  # be a worker in above cluster resolver.
elif cluster_resolver.task_type == 'ps':
  # Perform something that's only applicable on parameter servers. This
  # block will not run on this particular instance.
```

Returns `None` if such information is not available or is not applicable
in the current distributed environment, such as training with
<a href="../../../tf/distribute/experimental/TPUStrategy.md"><code>tf.distribute.experimental.TPUStrategy</code></a>.

For more information, please see
<a href="../../../tf/distribute/cluster_resolver/ClusterResolver.md"><code>tf.distribute.cluster_resolver.ClusterResolver</code></a>'s class doc.
</td>
</tr>
</table>



## Methods

<h3 id="cluster_spec"><code>cluster_spec</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cluster_spec()
</code></pre>

Returns a ClusterSpec object based on the latest instance group info.

This returns a ClusterSpec object for use based on information from the
specified initialization parameters and Slurm environment variables. The
cluster specification is resolved each time this function is called. The
resolver extract hostnames of nodes by scontrol and pack tasks in that
order until a node a has number of tasks that is equal to specification.
GPUs on nodes are allocated to tasks by specification through setting
CUDA_VISIBLE_DEVICES environment variable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A ClusterSpec containing host information retrieved from Slurm's
environment variables.
</td>
</tr>

</table>



<h3 id="get_task_info"><code>get_task_info</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_task_info()
</code></pre>

Returns job name and task_id for the process which calls this.

This returns the job name and task index for the process which calls this
function according to its rank and cluster specification. The job name and
task index are set after a cluster is constructed by cluster_spec otherwise
defaults to None.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A string specifying job name the process belongs to and an integer
specifying the task index the process belongs to in that job.
</td>
</tr>

</table>



<h3 id="master"><code>master</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>master(
    task_type=None, task_id=None, rpc_layer=None
)
</code></pre>

Returns the master string for connecting to a TensorFlow master.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`task_type`
</td>
<td>
(Optional) Overrides the default auto-selected task type.
</td>
</tr><tr>
<td>
`task_id`
</td>
<td>
(Optional) Overrides the default auto-selected task index.
</td>
</tr><tr>
<td>
`rpc_layer`
</td>
<td>
(Optional) Overrides the default RPC protocol TensorFlow uses
to communicate across nodes.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A connection string for connecting to a TensorFlow master.
</td>
</tr>

</table>



<h3 id="num_accelerators"><code>num_accelerators</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/cluster_resolver/slurm_cluster_resolver.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>num_accelerators(
    task_type=None, task_id=None, config_proto=None
)
</code></pre>

Returns the number of accelerator cores per worker.

This returns the number of accelerator cores (such as GPUs and TPUs)
available per worker.

Optionally, we allow callers to specify the task_type, and task_id, for
if they want to target a specific TensorFlow task to query
the number of accelerators. This is to support heterogenous environments,
where the number of accelerators cores per host is different.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`task_type`
</td>
<td>
(Optional) The type of the TensorFlow task of the machine we
want to query.
</td>
</tr><tr>
<td>
`task_id`
</td>
<td>
(Optional) The index of the TensorFlow task of the machine we
want to query.
</td>
</tr><tr>
<td>
`config_proto`
</td>
<td>
(Optional) Configuration for starting a new session to
query how many accelerator cores it has.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A map of accelerator types to number of cores.
</td>
</tr>

</table>





