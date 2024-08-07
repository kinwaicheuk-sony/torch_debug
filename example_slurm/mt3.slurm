#!/bin/bash
#SBATCH --partition=sharedp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=MRVAN_2GPU_B4_16p_Lagg128_hop3
#SBATCH --output=slurm-logs/ablation/output_%x.%N.%j.log
#SBATCH --error=slurm-logs/ablation/error_%x.%N.%j.log
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --account=mt
#SBATCH --signal=SIGUSR1@30
#SBATCH --requeue

#
# Export the job name as an environment variable
export JOB_NAME=$SLURM_JOB_NAME

# Load necessary modules (if needed)
# module load python/3.10

# Activate your virtual environment (if needed)
# source /home/kinwai.cheuk/mambaforge/envs/mini_v2/bin/activate

source /home/kinwai.cheuk/mambaforge/bin/activate
conda activate mt3

# cp dataset from beegfs to /data/user if it is not there
SOURCE_DIR="/mnt/beegfs/group/mt/dataset/slakh2100_flac_redux"
DEST_DIR="/data/kinwai.cheuk"
DEST_PATH="${DEST_DIR}/slakh2100_flac_redux"
# Check if the destination directory already exists
if [ -d "$DEST_PATH" ]; then
    echo "The dataset $DEST_PATH already exists. No need to copy."
else
    # Check if the base destination directory exists
    if [ ! -d "$DEST_DIR" ]; then
        echo "The base directory $DEST_DIR does not exist. Creating it."
        mkdir -p "$DEST_DIR"
    fi
    # Copy the source directory to the destination
    echo "Copying $SOURCE_DIR to $DEST_PATH."
    cp -r "$SOURCE_DIR" "$DEST_PATH"
    echo "Copy completed."
fi

# Run the Python script
srun python3 /mnt/beegfs/group/mfm/data/kinwai/MR-MT3/train.py \
--config-path="config" \
--config-name="config_slakh_segmem_mrvan" \
devices=2 \
hydra/job_logging=disabled \
model="VanillaTransformerNetSegMemV2WithPrev" \
dataset="SlakhPrevAugment" \
dataset_use_tf_spectral_ops=False \
dataset_is_randomize_tokens=True \
split_frame_length=2000 \
model_segmem_length=128 \
dataset_prev_augment_frames=3 \
trainer.check_val_every_n_epoch=20 \
eval.eval_after_num_epoch=400 \
eval.eval_first_n_examples=3 \
eval.eval_per_epoch=10 \
eval.contiguous_inference=True \
num_epochs=800 \
trainer.precision=16 \
dataloader.train.batch_size=4 \
dataloader.train.num_workers=32 \
dataloader.val.batch_size=2 \
dataloader.val.num_workers=2 \
optim.num_steps_per_epoch=1289 \
optim.warmup_steps=64500 \
+tag=$JOB_NAME

