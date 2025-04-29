WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=8
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

MAX_DRAFT_LEN=63
MAX_NON_LEAVES_PER_LAYER=10

MODELOPT_OUTPUT_DIR="/scratch_1/tmp/modelopt/saved_models_Meta-Llama-3-70B-Instruct_eagle_modelopt_fp8_hf"
CKPT_DIR="/scratch_1/tmp/trt_models/Meta-Llama-3-70B-Instruct_eagle_modelopt/tp${TP_SIZE}_pp${PP_SIZE}_max_draft_len${MAX_DRAFT_LEN}_max_non_leaves${MAX_NON_LEAVES_PER_LAYER}"

python $WORK_DIR/examples/eagle/convert_checkpoint.py --model_dir $MODELOPT_OUTPUT_DIR \
                            --output_dir $CKPT_DIR \
                            --dtype float16 \
                            --max_draft_len $MAX_DRAFT_LEN \
                            --max_non_leaves_per_layer $MAX_NON_LEAVES_PER_LAYER \
                            --tp_size $TP_SIZE \
                            --pp_size $PP_SIZE \
                            --workers $WORLD_SIZE