EAGLE_DIR="/scratch_1/tmp/hf_models/EAGLE-LLaMA3-Instruct-70B"
BASE_MODEL="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct"
MERGED_MODEL_DIR="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct_eagle_modelopt"

EAGLE_NUM_LAYERS=1

python /scratch/trtllm-slurm-benchmark/eagle/load_eagle_ckpt.py \
    --eagle_dir $EAGLE_DIR \
    --base_model $BASE_MODEL \
    --eagle_num_layers $EAGLE_NUM_LAYERS \
    --output_dir $MERGED_MODEL_DIR