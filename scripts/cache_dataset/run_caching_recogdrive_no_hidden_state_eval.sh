TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_test

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"

# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and run dataset caching
# cache_hidden_state:
# In IL/RL training, we save the VLM’s last_hidden_state together with historical trajectories, instructions, and ego status
# to accelerate training by avoiding repeated VLM forward passes. You can refer to navsim\agents\recogdrive\recogdrive_features.py.
# We are also exploring joint training of VLM and DiT, as VLM forward is the main bottleneck that slows training.
# During evaluation, we provide two modes: with or without caching the hidden state.
# For GPUs like 3090/4090, set cache_hidden_state=False to disable caching,
# allowing VLM and DiT to perform joint inference during evaluation.


torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=False \
    agent.cache_mode=True \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/path/to/ReCogDrive_2b" \ 
    cache_path=$CACHE_PATH  > caching_dataset_test.txt 2>&1