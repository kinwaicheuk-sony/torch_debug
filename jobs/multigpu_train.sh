#!/bin/bash

#SBATCH --account=project55
#SBATCH --partition=gc3-h100

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --output=debug_%j.log
#SBATCH --time=10-00:00:00

# #SBATCH --exclude=comp-h100-[15-16]

source /etc/profile.d/modules.sh
module load singularity/3.11.5
module load openmpi/3.1.6

CONTAINER_PATH=/group/project55/yukara/container/mfm_audiocraft_2-2.sif
ROOT_DIR=/group/project55/tda/project/torch_debug

OUTPUT_ROOT_DIR=${ROOT_DIR}
DATASET_DIR=/group/project55/yukara/dataset/KPM/

JOB_ID=$SLURM_JOB_ID

DATASET_CONFIG=${ROOT_DIR}/stable_audio_tools/configs/dataset_yukara/kpm_clap_gaia.json
PRETRANSFORM_CKPT="${ROOT_DIR}/ckpt/vae/sa2_64dim/sa2_vae_v0-0-16_840k-steps.ckpt"
CLAP_CKPT=${ROOT_DIR}/ckpt/clap/music_audioset_epoch_15_esc_90.14.pt

EXTRA_ARGS=""

# JOB_NAME="ggc-fy24-dit"
# MODEL_CONFIG=${ROOT_DIR}/stable_audio_tools/configs/model_yukara/stereo/sa2_melody_clap.json
BS_PER_GPU=6

# JOB_NAME="ggc-fy24-dit-large"
# MODEL_CONFIG=${ROOT_DIR}/stable_audio_tools/configs/model_yukara/stereo/sa2_melody_clap_large.json
# BS_PER_GPU=3

## 3-min model
# JOB_NAME="ggc-fy24-dit-large-3min"
# MODEL_CONFIG=${ROOT_DIR}/stable_audio_tools/configs/model_yukara/stereo/sa2_melody_clap_large_3min.json
# BS_PER_GPU=3
# ## RESUME training
# CKPT_PATH="${ROOT_DIR}/runs/ggc-fy24-dit-large/1251/ggc-fy24-dit-large/01r0s58l/checkpoints/last.ckpt"
# EXTRA_ARGS="--ckpt-path ${CKPT_PATH}"

## 4-min model
JOB_NAME="debug"

MASTER_PORT=25038
NUM_WORKERS=8
SEED=1234

MASTER_ADDR=${HOSTNAME}
NUM_NODES=$SLURM_JOB_NUM_NODES
GPUS_PER_NODE=2
NUM_GPUS=$((${NUM_NODES}*${GPUS_PER_NODE}))
BATCH_SIZE=$((${NUM_GPUS}*${BS_PER_GPU}))

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "Number of Nodes : ${NUM_NODES}"
echo "Number of GPUs : ${NUM_GPUS}"
echo "Batch size : ${BATCH_SIZE}"


# DEBUG
# DEBUG=0

# if [ $DEBUG -eq 0 ]; then
#     OUTPUT_DIR=${OUTPUT_ROOT_DIR}/runs/${JOB_NAME}/${JOB_ID}
# else
#     OUTPUT_DIR=${OUTPUT_ROOT_DIR}/runs/debug/${JOB_NAME}/${JOB_ID}
# fi

# mkdir -p ${OUTPUT_DIR}


time mpirun -n ${NUM_GPUS} -x NCCL_SOCKET_IFNAME=ib -x PATH \
    -x MASTER_ADDR=${MASTER_ADDR} -x MASTER_PORT=${MASTER_PORT}  \
    singularity exec --nv --pwd $ROOT_DIR -B $ROOT_DIR  \
    --env MASTER_PORT=${MASTER_PORT} --env MASTER_ADDR=${MASTER_ADDR} --env HYDRA_FULL_ERROR=1 \
    ${CONTAINER_PATH} \
    ${ROOT_DIR}/python_mpi.bash ${ROOT_DIR}/train.py 