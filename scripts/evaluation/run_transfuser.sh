TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/high_perf_store3/world-model/yongkangli/data/NAVSIM/exp/transfuser_seed_0.ckpt
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/high_perf_store3/world-model/yongkangli/data/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/high_perf_store3/world-model/yongkangli/data/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/high_perf_store3/world-model/yongkangli/data/NAVSIM/dataset"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval 
