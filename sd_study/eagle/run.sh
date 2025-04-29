WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

MAX_DRAFT_LEN=63
MAX_NON_LEAVES_PER_LAYER=10

BASE_MODEL="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct"
ENGINE_DIR="/scratch_1/tmp/trt_engines/Meta-Llama-3-70B-Instruct_eagle_modelopt/tp${TP_SIZE}_pp${PP_SIZE}_max_draft_len${MAX_DRAFT_LEN}_max_non_leaves${MAX_NON_LEAVES_PER_LAYER}"

mpirun -n $WORLD_SIZE --allow-run-as-root python $WORK_DIR/examples/run.py \
    --engine_dir $ENGINE_DIR \
    --tokenizer_dir $BASE_MODEL \
    --max_output_len=100 \
    --eagle_choices="[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]" \
    --input_text "Once upon"