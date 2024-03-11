# Software Installation

In this section, we will explore multiple ways to install Deep Learning packages on the NeSI platform.
We will focus on Python packages although many advice apply to other languages.

## Environment Modules

The first method to install software is... to not install them but use those preinstalled 😁.

On NeSI's platform, we use *Environment Modules* to provide software in a flexible way, allowing users to load specific versions and their dependencies.

Here we will use TensorFlow as an example.
So let's first assert that this package is not available in the default environment:

```bash
python3 -c "import tensorflow; print(tensorflow.__version__)"
```

??? success "output"

    ```
    Traceback (most recent call last):
    File "<string>", line 1, in <module>
    ModuleNotFoundError: No module named 'tensorflow'
    ```

Next, let's check if there is a TensorFlow envionment module on the platform, using the search command `module spider`:

```bash
module spider TensorFlow
```

??? success "output"

    ```
    -------------------------------------------------------------------------------------------
      TensorFlow:
    -------------------------------------------------------------------------------------------
        Description:
          An open-source software library for Machine Intelligence

         Versions:
            TensorFlow/2.0.1-gimkl-2018b-Python-3.8.1
            TensorFlow/2.2.0-gimkl-2018b-Python-3.8.1
            TensorFlow/2.2.2-gimkl-2018b-Python-3.8.1
            TensorFlow/2.2.3-gimkl-2018b-Python-3.8.1
            TensorFlow/2.3.1-gimkl-2020a-Python-3.8.2
            TensorFlow/2.4.1-gimkl-2020a-Python-3.8.2
            TensorFlow/2.8.2-gimkl-2022a-Python-3.10.5
            TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3

         Other possible modules matches:
            tensorflow

    -------------------------------------------------------------------------------------------
      To find other possible module matches do:
          module -r spider '.*TensorFlow.*'

    -------------------------------------------------------------------------------------------
      For detailed information about a specific "TensorFlow" module (including how to load the modules) use the module's full name.
      For example:

         $ module spider TensorFlow/2.8.2-gimkl-2022a-Python-3.10.5
    -------------------------------------------------------------------------------------------
    ```

Success! There are a couple of versions, let's load version 2.13.0.

```bash
module purge
module load TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3
```

!!! tip "Tips"

    The `module purge` command ensures that we start from a clean state.

    Always specify the version of a module, this will save you -- and people helping you -- a lot of time when debugging installation issues 😉.


We can now check that TensorFlow is available from a Python interpreter:

```bash
python3 -c "import tensorflow; print(tensorflow.__version__)"
```

??? success "output"

    ```
    2024-03-11 12:29:51.145566: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2.13.0
    ```

It is also interesting to list all the modules loaded as dependencies from TensorFlow:

```bash
module list
```

??? success "output"

    ```
    Currently Loaded Modules:
      1) XALT/minimal
      2) slurm
      3) NeSI                                                   (S)
      4) GCCcore/11.3.0
      5) zlib/1.2.11-GCCcore-11.3.0
      6) binutils/2.38-GCCcore-11.3.0
      7) GCC/11.3.0
      8) libpmi/2-slurm
      9) numactl/2.0.14-GCC-11.3.0
     10) UCX/1.12.1-GCC-11.3.0
     11) impi/2021.5.1-GCC-11.3.0
     12) AlwaysIntelMKL/1.0
     13) imkl/2022.0.2
     14) gimpi/2022a
     15) imkl-FFTW/2022.0.2-gimpi-2022a
     16) gimkl/2022a
     17) bzip2/1.0.8-GCCcore-11.3.0
     18) XZ/5.2.5-GCCcore-11.3.0
     19) libpng/1.6.37-GCCcore-11.3.0
     20) freetype/2.11.1-GCCcore-11.3.0
     21) Szip/2.1.1-GCCcore-11.3.0
     22) HDF5/1.12.2-gimpi-2022a
     23) libjpeg-turbo/2.1.3-GCCcore-11.3.0
     24) ncurses/6.2-GCCcore-11.3.0
     25) libreadline/8.1-GCCcore-11.3.0
     26) libxml2/2.9.10-GCCcore-11.3.0
     27) libxslt/1.1.34-GCCcore-11.3.0
     28) cURL/7.83.1-GCCcore-11.3.0
     29) netCDF/4.8.1-gimpi-2022a
     30) SQLite/3.36.0-GCCcore-11.3.0
     31) METIS/5.1.0-GCC-11.3.0
     32) GMP/6.2.1-GCCcore-11.3.0
     33) MPFR/4.1.0-GCC-11.3.0
     34) SuiteSparse/5.13.0-gimkl-2022a
     35) Tcl/8.6.10-GCCcore-11.3.0
     36) Tk/8.6.10-GCCcore-11.3.0
     37) ZeroMQ/4.3.4-GCCcore-11.3.0
     38) OpenSSL/1.1.1k-GCCcore-11.3.0
     39) Python/3.11.3-gimkl-2022a
     40) CUDA/11.8.0
     41) cuDNN/8.6.0.163-CUDA-11.8.0
     42) TensorRT/8.6.1.6-gimkl-2022a-Python-3.11.3-CUDA-11.8.0
     43) NCCL/2.16.5-CUDA-11.8.0
     44) TensorFlow/2.13.0-gimkl-2022a-Python-3.11.3
    ```

You will note that CUDA and cuDNN are loaded, and the right version to make it just works.

Modules are prepared with ❤️ by NeSI's Research Support Team.
If you need a package or a version that is not available, do not hesitate to contact us at <support@nesi.org.nz> to get it installed for you.

!!! info "See also"
    The [TensorFlow support page](https://support.nesi.org.nz/hc/en-gb/articles/360000990436-TensorFlow-on-GPUs#use_nesi_modules) also hightlights how to use Python virtual environments, with and without reusing system packages provided by an environment module.

## Conda Environment

If you want to be fully in control of you software stack, there are handful of solutions.
One of them is to use the [Conda package manager](https://docs.conda.io/projects/conda).
Conda lets you create "conda environments", which allow you to install a set of packages in isolation.

Before using `conda`, we need to load the corresponding environment module:

```bash
module purge
module load Miniconda3/23.10.0-1
conda -V
```

??? success "output"

    ```
    conda 23.10.0
    ```

Because deep learning frameworks tend to be... fairly large 😅, we need to be careful how we install these packages to avoid wasting storage space and locking our home storage (20GB maximum).
We will set a default folder to store the package cache in `nobackup` storage:

```bash
conda config --add pkgs_dirs /nesi/nobackup/nesi99991/conda_pkgs/$USER
```

You only need to do this once.

!!! tip

    Using `nobackup` storage will ensure that downloaded packages will be automatically removed after 120 days.
    This won't affect the conda environments where these packages are installed.

Next, let's create a conda environment from the definition file `environment.yml`, installing TensorFlow in it:

```bash
export PYTHONNOUSERSITE=1
cd /nesi/project/nesi99991/hpcintrodl_20240314/$USER/content
conda env create -f environment.yml -p /nesi/nobackup/nesi99991/hpcintrodl_20240314/$USER/venv
```

??? success "output"

    ```
    Channels:
     - conda-forge
     - defaults
    Platform: linux-64
    Collecting package metadata (repodata.json): done
    Solving environment: done

    Downloading and Extracting Packages:

    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Installing pip dependencies: \ Ran pip subprocess with arguments:
    ['/nesi/nobackup/nesi99991/hpcintrodl_20240314/riom/venv/bin/python', '-m', 'pip', 'install', '-U', '-r', '/scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt', '--exists-action=b']
    Pip subprocess output:
    Collecting tensorflow==2.12.0 (from -r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading tensorflow-2.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.4 kB)
    Collecting tensorflow-datasets (from -r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading tensorflow_datasets-4.9.4-py3-none-any.whl.metadata (9.2 kB)
    Collecting absl-py>=1.0.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting astunparse>=1.6.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
    Collecting flatbuffers>=2.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading flatbuffers-24.3.7-py2.py3-none-any.whl.metadata (849 bytes)
    Collecting gast<=0.4.0,>=0.2.1 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading gast-0.4.0-py3-none-any.whl.metadata (1.1 kB)
    Collecting google-pasta>=0.1.1 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
    Collecting grpcio<2.0,>=1.24.3 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading grpcio-1.62.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
    Collecting h5py>=2.9.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading h5py-3.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
    Collecting jax>=0.3.15 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading jax-0.4.25-py3-none-any.whl.metadata (24 kB)
    Collecting keras<2.13,>=2.12.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading keras-2.12.0-py2.py3-none-any.whl.metadata (1.4 kB)
    Collecting libclang>=13.0.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading libclang-16.0.6-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)
    Collecting numpy<1.24,>=1.22 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading numpy-1.23.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
    Collecting opt-einsum>=2.3.2 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
    Collecting packaging (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading packaging-24.0-py3-none-any.whl.metadata (3.2 kB)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl.metadata (541 bytes)
    Requirement already satisfied: setuptools in /scale_wlg_nobackup/filesets/nobackup/nesi99991/hpcintrodl_20240314/riom/venv/lib/python3.10/site-packages (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1)) (69.1.1)
    Collecting six>=1.12.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
    Collecting tensorboard<2.13,>=2.12 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading tensorboard-2.12.3-py3-none-any.whl.metadata (1.8 kB)
    Collecting tensorflow-estimator<2.13,>=2.12.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading tensorflow_estimator-2.12.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting termcolor>=1.1.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
    Collecting typing-extensions>=3.6.6 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading typing_extensions-4.10.0-py3-none-any.whl.metadata (3.0 kB)
    Collecting wrapt<1.15,>=1.11.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading wrapt-1.14.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)
    Collecting click (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Using cached click-8.1.7-py3-none-any.whl.metadata (3.0 kB)
    Collecting dm-tree (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading dm_tree-0.1.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.9 kB)
    Collecting etils>=0.9.0 (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading etils-1.7.0-py3-none-any.whl.metadata (6.4 kB)
    Collecting promise (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading promise-2.3.tar.gz (19 kB)
      Preparing metadata (setup.py): started
      Preparing metadata (setup.py): finished with status 'done'
    Collecting psutil (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Using cached psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (21 kB)
    Collecting requests>=2.19.0 (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting tensorflow-metadata (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading tensorflow_metadata-1.14.0-py3-none-any.whl.metadata (2.1 kB)
    Collecting toml (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)
    Collecting tqdm (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Using cached tqdm-4.66.2-py3-none-any.whl.metadata (57 kB)
    Collecting array-record>=0.5.0 (from tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading array_record-0.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (503 bytes)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /scale_wlg_nobackup/filesets/nobackup/nesi99991/hpcintrodl_20240314/riom/venv/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1)) (0.42.0)
    Collecting fsspec (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading fsspec-2024.2.0-py3-none-any.whl.metadata (6.8 kB)
    Collecting importlib_resources (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading importlib_resources-6.1.3-py3-none-any.whl.metadata (3.9 kB)
    Collecting zipp (from etils[enp,epath,etree]>=0.9.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading zipp-3.17.0-py3-none-any.whl.metadata (3.7 kB)
    Collecting ml-dtypes>=0.2.0 (from jax>=0.3.15->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading ml_dtypes-0.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
    Collecting scipy>=1.9 (from jax>=0.3.15->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.4/60.4 kB 10.4 MB/s eta 0:00:00
    Collecting charset-normalizer<4,>=2 (from requests>=2.19.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
    Collecting idna<4,>=2.5 (from requests>=2.19.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Using cached idna-3.6-py3-none-any.whl.metadata (9.9 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.19.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
    Collecting certifi>=2017.4.17 (from requests>=2.19.0->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Using cached certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
    Collecting google-auth<3,>=1.6.3 (from tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading google_auth-2.28.2-py2.py3-none-any.whl.metadata (4.7 kB)
    Collecting google-auth-oauthlib<1.1,>=0.5 (from tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl.metadata (2.7 kB)
    Collecting markdown>=2.6.8 (from tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading Markdown-3.5.2-py3-none-any.whl.metadata (7.0 kB)
    Collecting tensorboard-data-server<0.8.0,>=0.7.0 (from tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading tensorboard_data_server-0.7.2-py3-none-any.whl.metadata (1.1 kB)
    Collecting werkzeug>=1.0.1 (from tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading werkzeug-3.0.1-py3-none-any.whl.metadata (4.1 kB)
    Collecting absl-py>=1.0.0 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting googleapis-common-protos<2,>=1.52.0 (from tensorflow-metadata->tensorflow-datasets->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 2))
      Downloading googleapis_common_protos-1.63.0-py2.py3-none-any.whl.metadata (1.5 kB)
    Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading protobuf-3.20.3-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (679 bytes)
    Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
    Collecting pyasn1-modules>=0.2.1 (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl.metadata (3.6 kB)
    Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading requests_oauthlib-1.4.0-py2.py3-none-any.whl.metadata (11 kB)
    Collecting MarkupSafe>=2.1.1 (from werkzeug>=1.0.1->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
    Collecting pyasn1<0.6.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading pyasn1-0.5.1-py2.py3-none-any.whl.metadata (8.6 kB)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard<2.13,>=2.12->tensorflow==2.12.0->-r /scale_wlg_persistent/filesets/project/nesi99999/riom/hpc-intro-dl/content/condaenv.rsqilbad.requirements.txt (line 1))
      Downloading oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
    Downloading tensorflow-2.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (585.9 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 585.9/585.9 MB 2.8 MB/s eta 0:00:00
    Downloading tensorflow_datasets-4.9.4-py3-none-any.whl (5.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 2.8 MB/s eta 0:00:00
    Downloading array_record-0.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.0 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.0/3.0 MB 4.1 MB/s eta 0:00:00
    Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Downloading etils-1.7.0-py3-none-any.whl (152 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 152.4/152.4 kB 1.1 MB/s eta 0:00:00
    Downloading flatbuffers-24.3.7-py2.py3-none-any.whl (26 kB)
    Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.5/57.5 kB 6.0 MB/s eta 0:00:00
    Downloading grpcio-1.62.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.5 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 9.6 MB/s eta 0:00:00
    Downloading h5py-3.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.8 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 6.3 MB/s eta 0:00:00
    Downloading jax-0.4.25-py3-none-any.whl (1.8 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 11.4 MB/s eta 0:00:00
    Downloading keras-2.12.0-py2.py3-none-any.whl (1.7 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 4.9 MB/s eta 0:00:00
    Downloading libclang-16.0.6-py2.py3-none-manylinux2010_x86_64.whl (22.9 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22.9/22.9 MB 18.8 MB/s eta 0:00:00
    Downloading numpy-1.23.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.1/17.1 MB 4.9 MB/s eta 0:00:00
    Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 65.5/65.5 kB 8.4 MB/s eta 0:00:00
    Downloading requests-2.31.0-py3-none-any.whl (62 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.6/62.6 kB 2.3 MB/s eta 0:00:00
    Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
    Downloading tensorboard-2.12.3-py3-none-any.whl (5.6 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.6/5.6 MB 4.9 MB/s eta 0:00:00
    Downloading tensorflow_estimator-2.12.0-py2.py3-none-any.whl (440 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 440.7/440.7 kB 5.1 MB/s eta 0:00:00
    Downloading tensorflow_io_gcs_filesystem-0.36.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 5.4 MB/s eta 0:00:00
    Downloading termcolor-2.4.0-py3-none-any.whl (7.7 kB)
    Downloading typing_extensions-4.10.0-py3-none-any.whl (33 kB)
    Downloading wrapt-1.14.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (77 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 321.6 kB/s eta 0:00:00
    Using cached click-8.1.7-py3-none-any.whl (97 kB)
    Downloading dm_tree-0.1.8-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (152 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 152.8/152.8 kB 26.7 MB/s eta 0:00:00
    Downloading packaging-24.0-py3-none-any.whl (53 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.5/53.5 kB 218.2 kB/s eta 0:00:00
    Using cached psutil-5.9.8-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (288 kB)
    Downloading tensorflow_metadata-1.14.0-py3-none-any.whl (28 kB)
    Downloading absl_py-1.4.0-py3-none-any.whl (126 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 126.5/126.5 kB 738.1 kB/s eta 0:00:00
    Downloading protobuf-3.20.3-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 17.8 MB/s eta 0:00:00
    Downloading toml-0.10.2-py2.py3-none-any.whl (16 kB)
    Using cached tqdm-4.66.2-py3-none-any.whl (78 kB)
    Using cached certifi-2024.2.2-py3-none-any.whl (163 kB)
    Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 21.7 MB/s eta 0:00:00
    Downloading google_auth-2.28.2-py2.py3-none-any.whl (186 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 186.9/186.9 kB 495.4 kB/s eta 0:00:00
    Downloading google_auth_oauthlib-1.0.0-py2.py3-none-any.whl (18 kB)
    Downloading googleapis_common_protos-1.63.0-py2.py3-none-any.whl (229 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.1/229.1 kB 2.8 MB/s eta 0:00:00
    Using cached idna-3.6-py3-none-any.whl (61 kB)
    Downloading Markdown-3.5.2-py3-none-any.whl (103 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.9/103.9 kB 1.2 MB/s eta 0:00:00
    Downloading ml_dtypes-0.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.2 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 3.7 MB/s eta 0:00:00
    Downloading scipy-1.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.4 MB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.4/38.4 MB 39.9 MB/s eta 0:00:00
    Downloading tensorboard_data_server-0.7.2-py3-none-any.whl (2.4 kB)
    Downloading urllib3-2.2.1-py3-none-any.whl (121 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.1/121.1 kB 19.1 MB/s eta 0:00:00
    Downloading werkzeug-3.0.1-py3-none-any.whl (226 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 226.7/226.7 kB 809.5 kB/s eta 0:00:00
    Downloading fsspec-2024.2.0-py3-none-any.whl (170 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 170.9/170.9 kB 17.5 MB/s eta 0:00:00
    Downloading importlib_resources-6.1.3-py3-none-any.whl (34 kB)
    Downloading zipp-3.17.0-py3-none-any.whl (7.4 kB)
    Downloading cachetools-5.3.3-py3-none-any.whl (9.3 kB)
    Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
    Downloading pyasn1_modules-0.3.0-py2.py3-none-any.whl (181 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 181.3/181.3 kB 891.8 kB/s eta 0:00:00
    Downloading requests_oauthlib-1.4.0-py2.py3-none-any.whl (24 kB)
    Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 151.7/151.7 kB 22.3 MB/s eta 0:00:00
    Downloading pyasn1-0.5.1-py2.py3-none-any.whl (84 kB)
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 84.9/84.9 kB 322.5 kB/s eta 0:00:00
    Building wheels for collected packages: promise
      Building wheel for promise (setup.py): started
      Building wheel for promise (setup.py): finished with status 'done'
      Created wheel for promise: filename=promise-2.3-py3-none-any.whl size=21483 sha256=63f989887b83185cbedc8ec08be54f909b2627f3208632cd3aae84af95f3bdc4
      Stored in directory: /scale_wlg_persistent/filesets/home/riom/.cache/pip/wheels/54/4e/28/3ed0e1c8a752867445bab994d2340724928aa3ab059c57c8db
    Successfully built promise
    Installing collected packages: libclang, flatbuffers, dm-tree, zipp, wrapt, urllib3, typing-extensions, tqdm, toml, termcolor, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, six, pyasn1, psutil, protobuf, packaging, oauthlib, numpy, MarkupSafe, markdown, keras, importlib_resources, idna, grpcio, gast, fsspec, etils, click, charset-normalizer, certifi, cachetools, absl-py, werkzeug, scipy, rsa, requests, pyasn1-modules, promise, opt-einsum, ml-dtypes, h5py, googleapis-common-protos, google-pasta, astunparse, tensorflow-metadata, requests-oauthlib, jax, google-auth, google-auth-oauthlib, array-record, tensorboard, tensorflow-datasets, tensorflow
    Successfully installed MarkupSafe-2.1.5 absl-py-1.4.0 array-record-0.5.0 astunparse-1.6.3 cachetools-5.3.3 certifi-2024.2.2 charset-normalizer-3.3.2 click-8.1.7 dm-tree-0.1.8 etils-1.7.0 flatbuffers-24.3.7 fsspec-2024.2.0 gast-0.4.0 google-auth-2.28.2 google-auth-oauthlib-1.0.0 google-pasta-0.2.0 googleapis-common-protos-1.63.0 grpcio-1.62.1 h5py-3.10.0 idna-3.6 importlib_resources-6.1.3 jax-0.4.25 keras-2.12.0 libclang-16.0.6 markdown-3.5.2 ml-dtypes-0.3.2 numpy-1.23.5 oauthlib-3.2.2 opt-einsum-3.3.0 packaging-24.0 promise-2.3 protobuf-3.20.3 psutil-5.9.8 pyasn1-0.5.1 pyasn1-modules-0.3.0 requests-2.31.0 requests-oauthlib-1.4.0 rsa-4.9 scipy-1.12.0 six-1.16.0 tensorboard-2.12.3 tensorboard-data-server-0.7.2 tensorflow-2.12.0 tensorflow-datasets-4.9.4 tensorflow-estimator-2.12.0 tensorflow-io-gcs-filesystem-0.36.0 tensorflow-metadata-1.14.0 termcolor-2.4.0 toml-0.10.2 tqdm-4.66.2 typing-extensions-4.10.0 urllib3-2.2.1 werkzeug-3.0.1 wrapt-1.14.1 zipp-3.17.0

    done
    #
    # To activate this environment, use
    #
    #     $ conda activate /nesi/nobackup/nesi99991/hpcintrodl_20240314/riom/venv
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate
    ```

This will take a bit of time, so it could be a good time for a 🍵 break.

!!! warning

    Setting the variable `PYTHONNOUSERSITE=1` is necessary to ensure that `pip` doesn't look into user's home folder for locally installed Python packages (using `pip install --user`).
    This will ensure that the conda environment is really isolated from the system and more reproducible.

At last, we have our conda environment ready, let's activate it to see if TensorFlow is properly installed:

```bash
conda activate /nesi/nobackup/nesi99991/hpcintrodl_20240314/$USER/venv
```

??? failure "output"

    ```
    usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...
    conda: error: argument COMMAND: invalid choice: 'activate' (choose from 'clean', 'compare', 'config', 'create', 'info', 'init', 'install', 'list', 'notices', 'package', 'remove', 'uninstall', 'rename', 'run', 'search', 'update', 'upgrade', 'content-trust', 'doctor', 'repoquery', 'env')
    ```

Oh no! we cannot activate the environment 😨.

The reason is that we need to one extra step to fully configure conda in addition to loading the environment module:

```bash
source $(conda info --base)/etc/profile.d/conda.sh
```

??? danger "`conda init`, aka our nightmare"

    You might be tempted to use `conda init` here.
    **Do not.**
    Seriously.
    Please 🙏.
    This will insert a small piece of code in your `~/.bashrc` file that is fine on an individual computer but interacts poorly with the environment module system on the HPC platform.

    If you have used this command, you can edit `~/.bashrc` using a terminal based editor like `nano` and remove the offending piece of code:
    ```bash
    nano ~/.bashrc
    ```
    and start a new terminal.

Ok, now we can finally test our installation:

```bash
conda activate /nesi/nobackup/nesi99991/hpcintrodl_20240314/$USER/venv
python3 -c "import tensorflow; print(tensorflow.__version__)"
```

??? success "output"

    ```
    2024-03-11 13:56:36.843990: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-11 13:56:37.871865: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-03-11 13:56:37.872581: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-11 13:56:44.076918: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    2.12.0
    ```

!!! warning

    Using Conda on the HPC has numerous traps, we tried to highlight most of them here.
    In case of doubt, please check our [dedicated support page](https://support.nesi.org.nz/hc/en-gb/articles/360001580415-Miniconda3).

## Apptainer Container

TODO Apptainer, maybe just mention and show an example container pre-pulled?

TODO mention build
