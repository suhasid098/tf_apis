description: Public API for tf.config.experimental namespace.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.config.experimental" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf.compat.v1.config.experimental

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf.config.experimental namespace.



## Classes

[`class ClusterDeviceFilters`](../../../../tf/config/experimental/ClusterDeviceFilters.md): Represent a collection of device filters for the remote workers in cluster.

[`class VirtualDeviceConfiguration`](../../../../tf/config/LogicalDeviceConfiguration.md): Configuration class for a logical devices.

## Functions

[`disable_mlir_bridge(...)`](../../../../tf/config/experimental/disable_mlir_bridge.md): Disables experimental MLIR-Based TensorFlow Compiler Bridge.

[`disable_mlir_graph_optimization(...)`](../../../../tf/config/experimental/disable_mlir_graph_optimization.md): Disables experimental MLIR-Based TensorFlow Compiler Optimizations.

[`enable_mlir_bridge(...)`](../../../../tf/config/experimental/enable_mlir_bridge.md): Enables experimental MLIR-Based TensorFlow Compiler Bridge.

[`enable_mlir_graph_optimization(...)`](../../../../tf/config/experimental/enable_mlir_graph_optimization.md): Enables experimental MLIR-Based TensorFlow Compiler Optimizations.

[`enable_tensor_float_32_execution(...)`](../../../../tf/config/experimental/enable_tensor_float_32_execution.md): Enable or disable the use of TensorFloat-32 on supported hardware.

[`get_device_details(...)`](../../../../tf/config/experimental/get_device_details.md): Returns details about a physical devices.

[`get_device_policy(...)`](../../../../tf/config/experimental/get_device_policy.md): Gets the current device policy.

[`get_memory_growth(...)`](../../../../tf/config/experimental/get_memory_growth.md): Get if memory growth is enabled for a `PhysicalDevice`.

[`get_memory_info(...)`](../../../../tf/config/experimental/get_memory_info.md): Get memory info for the chosen device, as a dict.

[`get_memory_usage(...)`](../../../../tf/config/experimental/get_memory_usage.md): Get the current memory usage, in bytes, for the chosen device. (deprecated)

[`get_synchronous_execution(...)`](../../../../tf/config/experimental/get_synchronous_execution.md): Gets whether operations are executed synchronously or asynchronously.

[`get_virtual_device_configuration(...)`](../../../../tf/config/get_logical_device_configuration.md): Get the virtual device configuration for a <a href="../../../../tf/config/PhysicalDevice.md"><code>tf.config.PhysicalDevice</code></a>.

[`get_visible_devices(...)`](../../../../tf/config/get_visible_devices.md): Get the list of visible physical devices.

[`list_logical_devices(...)`](../../../../tf/config/list_logical_devices.md): Return a list of logical devices created by runtime.

[`list_physical_devices(...)`](../../../../tf/config/list_physical_devices.md): Return a list of physical devices visible to the host runtime.

[`reset_memory_stats(...)`](../../../../tf/config/experimental/reset_memory_stats.md): Resets the tracked memory stats for the chosen device.

[`set_device_policy(...)`](../../../../tf/config/experimental/set_device_policy.md): Sets the current thread device policy.

[`set_memory_growth(...)`](../../../../tf/config/experimental/set_memory_growth.md): Set if memory growth should be enabled for a `PhysicalDevice`.

[`set_synchronous_execution(...)`](../../../../tf/config/experimental/set_synchronous_execution.md): Specifies whether operations are executed synchronously or asynchronously.

[`set_virtual_device_configuration(...)`](../../../../tf/config/set_logical_device_configuration.md): Set the logical device configuration for a <a href="../../../../tf/config/PhysicalDevice.md"><code>tf.config.PhysicalDevice</code></a>.

[`set_visible_devices(...)`](../../../../tf/config/set_visible_devices.md): Set the list of visible devices.

[`tensor_float_32_execution_enabled(...)`](../../../../tf/config/experimental/tensor_float_32_execution_enabled.md): Returns whether TensorFloat-32 is enabled.

