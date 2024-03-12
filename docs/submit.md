# Submit Batch Jobs

This section details how to request access to a GPU (graphics processing unit) on NeSI for batch jobs.
We will also see how to make sure that the installed deep learning packages properly find and use them.

## Available GPUs

![](imgs/a100.jpg)

GPU are dedicated piece of hardware filled with specialised compute units to handle massively parallel computations[^1].
The massively parallel architecture of deep learning models make them well-suited to run on GPUs, drastically decreasing their training time.

[^1]: Recent cards from NVIDIA have even more dedicated processing units, *[Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)*, that massively accelerate lower-precision matrix-matrix multiplications.

NeSI HPC platform gives access to different types of GPUs.
Here is a little tour of the available capacity as of March 2024:

GPU type | Location | Access type
---------|----------|------------
9 NVIDIA Tesla P100 PCIe 12GB cards (1 node with 1 GPU, 4 nodes with 2 GPUs) | Mahuika | Slurm and Jupyter
5 NVIDIA Tesla P100 PCIe 12GB cards (5 nodes with 1 GPU) | Māui Ancil. | Slurm
7 A100-1g.5gb instances (1 NVIDIA A100 PCIe 40GB card divided into 7 MIG GPU slices with 5GB memory each)| Mahuika | Slurm and Jupyter
7 NVIDIA A100 PCIe 40GB cards (4 nodes with 1 GPU, 2 nodes with 2 GPUs) | Mahuika | Slurm
4 NVIDIA HGX A100 boards (4 GPUs per board with 80GB memory each, 16 A100 GPUs in total)| Mahuika | Slurm

Which one should you use?

- for small experimentations, start with a A100-1g.5gb or a P100,
- if you need to run legacy code (e.g. TensorFlow 1.x) try a P100,
- otherwise use the PCIe or HGX A100,
- and if you need large memory and/or multiple GPUs, use the HGX A100s.

## Slurm job submission

When preparing our Slurm job script, we need to make sure we tell Slurm that we need a GPU, using
the `--gpus-per-node` option.
In a job submission script, the syntax is the following:

```bash
#SBATCH --gpus-per-node=<gpu_type>:<gpu_number>
```

Depending on the GPU type, we *may* also need to specify a partition using `--partition`.

GPU type | Slurm option
---------|-------------
Mahuika P100 | <pre><code>#SBATCH --gpus-per-node=P100:1</code></pre>
Māui Ancil. P100 | <pre><code>#SBATCH --partition=nesi_gpu<br>#SBATCH --gpus-per-node=1</code></pre>
A100-1g.5gb | TODO
PCIe A100 (40GB) | TODO
HGX A100 (80GB) | TODO

For today's exercises, we will use a big one, an HGX A100 GPU.

TODO query gpu with sbatch

TODO nvidia smi check as dummy job

```bash title="gpujob.sl" linenums="1"
#!/usr/bin/env bash
#SBATCH --account=nesi99991
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --partition=hgx
#SBATCH --gpus-per-node=A100:1

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
```

TODO do not specify CUDA_VISIBLE_DEVICES

## Limits on GPU Jobs

- per-project limit of 6 GPUs being used at a time.
- There is also a per-project limit of 360 GPU-hours being allocated to running jobs. This reduces the number of GPUs available for longer jobs, so for example you can use 8 GPUs at a time if your jobs run for a day, but only two GPUs if your jobs run for a week. The intention is to guarantee that all users can get short debugging jobs on to a GPU in a reasonably timely manner.
- Each GPU job can use no more than 64 CPUs. This is to ensure that GPUs are not left idle just because their node has no remaining free CPUs.
There is a limit of one A100-1g.5gb GPU job per user.

## TensorFlow model training example

TODO

## CUDA/cuDNN modules and other subtleties

TODO when to load CUDA/cuDNN, which versions?
- env module vs. conda vs. container
- PyTorch: no cudnn? no cuda?

TODO TF: XLA compiler

TODO check the logs about GPU usage

TODO apptainer --nv

## References

- https://support.nesi.org.nz/hc/en-gb/articles/4963040656783-Available-GPUs-on-NeSI
- https://support.nesi.org.nz/hc/en-gb/articles/360001471955-GPU-use-on-NeSI
- https://support.nesi.org.nz/hc/en-gb/articles/360000204076-Mahuika-Slurm-Partitions
