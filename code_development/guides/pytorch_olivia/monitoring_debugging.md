(pytorch-monitoring-debugging)=

# Monitoring GPU Utilization and Debugging Techniques
One convenient way to monitor GPU utilization during your training is by using the `nvidia-smi` command within your job script. By placing the monitoring code both before and after the actual training command (e.g., the srun command), you can track GPU utilization throughout the training process and save the logs to a separate file for later analysis.

`nvidia-smi` is a command-line utility provided by NVIDIA that allows you to monitor and manage GPUs. The `--query-gpu` option lets you specify the GPU metrics you want to track, such as the timestamp, GPU index, GPU name, GPU utilization, memory utilization, total memory, and used memory. The output is saved in CSV format, making it easy to parse and analyze. Additionally, the `-l 5` option logs the data every 5 seconds, providing a real-time view of GPU performance. The & at the end of the command ensures that it runs in the background, allowing the rest of the job script to execute without interruption.

```bash
# Start GPU utilization monitoring in the background
GPU_LOG_FILE="utilization.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

srun .....

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"
```
However, we are not limited to this and there are several other ways to monitor the GPU utilization depending on your needs.
For instance, we can use `watch -n 1 nvidia-smi` by logging into the specific gpu node for manual debugging to display GPU metrics in real time or use NVIDIA profiling tools like NVIDIA Nsight Systems and Nsight Compute for advanced profiling capabilities.
Moreover, when we are using frameworks like PyTorch, we can leverage built-in utilities such as `torch.cuda.utilization()` or integrate monitoring into your training loop with tools like TensorBoard. This allows us to correlate GPU usage with training steps, dataloader performance, and other metrics.

## Ensuring Optimal Hardware Utilization During Training
When running training across multiple GPUs, NCCL (NVIDIA Collective Communications Library) plays a critical role in facilitating communication between GPUs. To achieve maximum performance, it’s essential to ensure that NCCL is utilizing the hardware optimally. Enabling NCCL debug statements can help identify and diagnose potential issues related to communication and performance.

To enable NCCL debug logs, include the following lines in your job script:

```bash
# Debugging: Enable NCCL logs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

These debug logs provide valuable insights, such as whether the appropriate versions of CUDA, NCCL, libfabrics, and CXI (for Slingshot interconnects) are being used. If the logs indicate improper usage or missing components, it may suggest that the hardware is not being utilized optimally, which could lead to subpar performance. By analyzing these logs, you can ensure that your training setup is configured correctly for maximum efficiency.