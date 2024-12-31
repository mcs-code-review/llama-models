#!/bin/bash
# Partition for the job:
##SBATCH --partition=deeplearn
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="70b-instruct-test"

# The project ID which this job should run under:
#SBATCH --account="punim2247"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# Slurm QoS:
##SBATCH --qos=gpgpudeeplearn
##SBATCH --constraint=dlg5

# Requested memory per node:
#SBATCH --mem=128G

# Use this email address:
#SBATCH --mail-user=mukhammad.karimov@student.unimelb.edu.au

# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-4:0:00

# Standard output and error log
#SBATCH -o logs/70b-instruct-test-%N.%j.out # STDOUT
#SBATCH -e logs/70b-instruct-test-%N.%j.err # STDERR

# Run the job from the directory where it was launched (default)

# The modules to load:
echo "Current modules:"
echo "$(module list)"
echo "Loading modules..."
module load foss/2022a
module load CUDA/12.2.0
module load NCCL/2.19.4-CUDA-12.2.0
module load UCX-CUDA/1.14.1-CUDA-12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load GCCcore/11.3.0
module load Python/3.10.4
echo "Loaded modules:"
echo "$(module list)"

# The job command(s):
source ~/venvs/codellama/bin/activate

### CodeReviewer ###

python code_review_instruction_parallel.py \
    --ckpt_dir ./meta-llama/Llama-3.1-70B-Instruct \
    --tokenizer_path ./meta-llama/Llama-3.1-70B-Instruct \
    --conf_path ../config/zero-shot/llama-31-70B-instruct-cr.json \
    --temperature 0.0 --top_p 0.95 \
    --max_new_tokens 512 \
    --tp_size 4 \
    --debug True

# CHECKPOINT_DIR=$PWD/meta-llama/Llama-3.1-70B-Instruct/original
# NGPUS=2

# PYTHONPATH=$(git rev-parse --show-toplevel) torchrun --nproc_per_node=$NGPUS models/scripts/example_chat_completion.py $CHECKPOINT_DIR --model_parallel_size $NGPUS

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -c -n -s
my-job-stats -a -n -s
nvidia-smi
