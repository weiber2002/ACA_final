#!/bin/bash
#SBATCH --job-name=TP_Long_Sequence_TMP_4GPU      # 工作名稱
#SBATCH --partition=dev                # 使用的 partition (請根據你的系統修改)
#SBATCH --time=00:30:00                # 執行時間上限 (小時:分鐘:秒)
#SBATCH --account=ENT114105
#SBATCH --nodes=1                     # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=4            # Gpus per node
#SBATCH --cpus-per-task=4             # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1         # Maximum number of tasks on each node
#SBATCH -o TP_Long_Sequence_TMP_4GPU_%j.log        # output file (%j expands to jobId)
#SBATCH -e TP_Long_Sequence_TMP_4GPU_%j.err        # output file (%j expands to jobId)


CONDA_PREFIX="/work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1"
CONDA_DEFAULT_ENV="ACA_TP" # 你的 conda 環境名稱

$CONDA_PREFIX/bin/conda run -n "$CONDA_DEFAULT_ENV" torchrun  --nproc_per_node=4 TP.py