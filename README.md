# AlphaFold
srun --cpus-per-task=18 --time=02:00:00 --mem=80GB --gres=gpu:mi250:1 --pty /bin/bash

srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi100:1 --pty /bin/bash
srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi50:1 --pty /bin/bash
srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi250:1 --pty /bin/bash

cd vast/ml7612/alphafold
wget https://github.com/google/jax/archive/refs/tags/jax-v0.4.24.tar.gz
git clone https://github.com/ROCmSoftwarePlatform/xla.git

/pyenv/versions/3.11.0/bin/python  build/build.py --enable_rocm --rocm_path=/opt/rocm-5.7.0 --bazel_options=--override_repository=xla=/vast/wang/amd-porting/xla


singularity exec –bind ${VAST}/cache:${HOME}/.cache –overlay overlay-10GB-400K.ext3 /scratch/work/public/singularity/rocm5.7.1-ubuntu22.04.3.sif /bin/bash


singularity exec –bind ${VAST}/cache:${HOME}/.cache –overlay overlay-10GB-400K.ext3 /scratch/work/public/singularity/rocm5.7.1-ubuntu22.04.3.sif /bin/bash



srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi100:1 --pty /bin/bash
srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi50:1 --pty /bin/bash
srun --cpus-per-task=10 --mem=60GB --gres=gpu:mi250:1 --pty /bin/bash
# for MI50
singularity exec --overlay XXXX \
--bind $VAST/cache-mi50:$HOME/.cache \
XXX
bash /scratch/work/vip-amd-cuda/alphafold/setup-alphafold.bash
# for MI100
# for MI250
Ready to go production setup
/scratch/work/public/apps/alphafold/2.3.2-hip
NVIDIA GPU version
/scratch/work/public/apps/alphafold/2.3.2/run-alphafold-2.3.2.py
T1050
NVIDIA GPU
AMD GPUs



