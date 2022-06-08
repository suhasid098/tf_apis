description: Initializes Multi Client DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.initialize_multi_client" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.initialize_multi_client

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/mesh_util.py">View source</a>



Initializes Multi Client DTensor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.initialize_multi_client(
    enable_coordination_service: Optional[bool] = False
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

The following environment variables controls the behavior of this function.
If the variables are unset, DTensor will be configured to run in single-client
mode.

- DTENSOR_CLIENT_ID: integer, between 0 to num_clients - 1, to identify the
    client id of the current process.
- DTENSOR_NUM_CLIENTS: integer, the number of clients.
- DTENSOR_JOB_NAME: string, a hostname like string for the name of the dtensor
    job. The job name is used by TensorFlow in the job name section of
    the DeviceSpec.
- DTENSOR_JOBS: string, a comma separated list. Each item in the list is
    of format `{hostname}:{port}` and the items must be sorted in alphabet
    order. The implication is the RPC port numbers of the clients from
    the same host must be ordered by the client ID.
    Examples of valid DTENSOR_JOBS values:
    - 4 clients on localhost:
      `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
    - 2 clients on host1, 2 clients on host2
      `host1:10000,host1:10001,host2:10000,host2:10003`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enable_coordination_service`
</td>
<td>
If true, enable distributed coordination
service to make sure that workers know the devices on each other, a
prerequisite for data transfer through cross-worker rendezvous.
</td>
</tr>
</table>

