#!/bin/bash
#SBATCH --job-name=prosody_array
#SBATCH --output=/home1/nmehlman/arts/vpc/logs/slurm/output_%A_%a.log
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-4  # Replace <TOTAL_JOBS-1> with the total number of jobs minus one

# Load necessary modules
module purge
module load gcc/13.3.0 
module load cudnn/8.9.7.29-12-cuda

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate whisperx

# Define higher-level directory and save directory
DATA_ROOT_MAIN="/project2/shrikann_35/nmehlman/data/psid_data/Vox1/wav"
SAVE_ROOT_MAIN="/project2/shrikann_35/nmehlman/data/psid_data/Vox1/rhythm-feats"

# Collect all subdirectories
SUBDIRS=()
for SUBDIR in "$DATA_ROOT_MAIN"/*; do
    if [ -d "$SUBDIR" ]; then
        SUBDIRS+=("$SUBDIR")
    fi
done

# Calculate total directories and directories per task
TOTAL_DIRS=${#SUBDIRS[@]}
TOTAL_JOBS=5  # Replace <TOTAL_JOBS> with the desired number of SLURM array jobs
DIRS_PER_TASK=$(( (TOTAL_DIRS + TOTAL_JOBS - 1) / TOTAL_JOBS ))  # Compute directories per task dynamically

# Get the batch index from the SLURM array task ID
BATCH_INDEX=$SLURM_ARRAY_TASK_ID

DATA_DIRS=()
SAVE_DIRS=()

# Collect directories for this batch
for ((j=BATCH_INDEX*DIRS_PER_TASK; j<(BATCH_INDEX+1)*DIRS_PER_TASK && j<TOTAL_DIRS; j++)); do
    NAME=$(basename "${SUBDIRS[j]}")
    DATA_DIRS+=("${SUBDIRS[j]}")
    SAVE_DIRS+=("$SAVE_ROOT_MAIN/$NAME")
    mkdir -p "$SAVE_ROOT_MAIN/$NAME"  # Create save directory if it doesn't exist
done

# Run the Python script for this batch
python ../whisperx/prosody_features/extract_prosody_features.py \
    --data-dirs ${DATA_DIRS[@]} \
    --save-dirs ${SAVE_DIRS[@]} \
    --device cuda \
    --compute-type float32 \
    --file-type wav \
    --skip-existing
