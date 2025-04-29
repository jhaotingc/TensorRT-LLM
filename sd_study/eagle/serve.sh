WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

MAX_DRAFT_LEN=63
MAX_NON_LEAVES_PER_LAYER=10

BASE_MODEL="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct"
ENGINE_DIR="/scratch_1/tmp/trt_engines/Meta-Llama-3-70B-Instruct_eagle_modelopt/tp${TP_SIZE}_pp${PP_SIZE}_max_draft_len${MAX_DRAFT_LEN}_max_non_leaves${MAX_NON_LEAVES_PER_LAYER}"

MAX_BATCH_SIZE=64
ISL=1024
OSL=1024
MAX_NUM_TOKENS=$((ISL+OSL+MAX_DRAFT_LEN*(MAX_BATCH_SIZE+1)))
MAX_SEQ_LEN=$((ISL+OSL))

HOST=localhost
PORT=8000

trtllm-serve $ENGINE_DIR \
        --tokenizer $BASE_MODEL \
        --host $HOST \
        --port $PORT \
        --max_seq_len $MAX_SEQ_LEN \
        --max_batch_size $MAX_BATCH_SIZE \
        --tp_size $TP_SIZE \
        --kv_cache_free_gpu_memory_fraction 0.90 \
        --extra_llm_api_options eagle/extra_llm_api_options.yaml