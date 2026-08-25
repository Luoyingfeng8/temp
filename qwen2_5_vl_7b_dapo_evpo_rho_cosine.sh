#!/bin/bash

set -x
set -e
set -o pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate wjx

# Set this to the EAAR project root before running.
PROJECT_ROOT="/mnt/543780/workflow_62140109/workspace/EAAR"
cd "${PROJECT_ROOT}" || exit 1

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TENSORBOARD_DIR="${PROJECT_ROOT}/log"

# Enable rho(s) = 0.5 * [1 + cos(pi * s / total_training_steps)].
export USE_EVPO_RHO_COSINE_ANNEAL=true

CUDA_IDS=0,1,2,3,4,5,6,7
N_GPU=8

TOTAL_EPOCHES=2
MAX_STEPS=202
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=128
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096
ORI_ENTROPY_LOSS_COEF=0.03

# Reference-policy KL loss coefficient.
KL_REF_COEF=0
# Keep augmented-image log-probs for EVPO weights without adding the PAPO KL loss.
KL_PRCP_COEF=0.0

MODEL_PATH=${PROJECT_ROOT}/data/Qwen2.5-VL-7B-Instruct
TRAIN_FILE=${PROJECT_ROOT}/data/PAPO_ViRL39K_train/data
VAL_FILE=${PROJECT_ROOT}/data/math12k/data/test-00000-of-00001.parquet

CONFIG_FILE=${PROJECT_ROOT}/examples/configs/config_dapo_evpo.yaml
FORMAT_PROMPT=${PROJECT_ROOT}/examples/format_prompt/math_perception.jinja
REWARD_FUNCTION=${PROJECT_ROOT}/examples/reward_function/math.py:compute_score

EXP_NAME="qwen2_5_vl_7b__dapo__evpo__rho_cosine_0.1_beta1.2_clip_steps${MAX_STEPS}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_mini${MINI_ROLLOUT_BATCH_SIZE}"
LOG_DIR="${TENSORBOARD_DIR}/${EXP_NAME}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"

# Keep asset preparation separate from the training command.
# bash "${PROJECT_ROOT}/examples/evpo_dapo/prepare_qwen2_5_vl_assets.sh"

# Download/check locations used by this script:
#   model:      /data/Qwen2.5-VL-7B-Instruct
#   train data: /data/PAPO_ViRL39K_train/data
#   val data:   /data/math12k/data/test-00000-of-00001.parquet


# Verify local project files before launching Ray. Assets are verified by the preparation script.
# for REQUIRED_PATH in "${CONFIG_FILE}" "${TRAIN_FILE}" "${VAL_FILE}" "${FORMAT_PROMPT}" "${MODEL_PATH}/config.json"; do
#     if [ ! -e "${REQUIRED_PATH}" ]; then
#         echo "Required path still does not exist after the download check: ${REQUIRED_PATH}" >&2
#         exit 1
#     fi
# done

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONFIG_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.filter_overlong_prompts_workers=1 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_coef=${KL_REF_COEF} \
    algorithm.online_filtering=true \
    algorithm.filter_key=accuracy \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=0.99 \
    algorithm.use_kl_prcp=true \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    algorithm.use_aug_entropy_loss=false \
    algorithm.use_ori_entropy_loss=true \
    algorithm.ori_entropy_loss_coef=${ORI_ENTROPY_LOSS_COEF} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.max_steps=${MAX_STEPS} \
    2>&1 | tee "${LOG_FILE}"
