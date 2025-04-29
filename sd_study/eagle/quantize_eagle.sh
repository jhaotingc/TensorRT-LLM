MERGED_MODEL_DIR="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct_eagle_modelopt"
MODELOPT_OUTPUT_DIR="/scratch_1/tmp/modelopt/saved_models_Meta-Llama-3-70B-Instruct_eagle_modelopt_fp8_hf"

MODELOPT_ROOT="/scratch/TensorRT-Model-Optimizer"

python $MODELOPT_ROOT/examples/llm_ptq/hf_ptq.py \
    --pyt_ckpt_path=$MERGED_MODEL_DIR \
    --export_path=$MODELOPT_OUTPUT_DIR \
    --sparsity_fmt=dense \
    --qformat=fp8 \
    --calib_size=512 \
    --batch_size=1 \
    --inference_tensor_parallel=1 \
    --inference_pipeline_parallel=1 \
    --export_fmt=hf