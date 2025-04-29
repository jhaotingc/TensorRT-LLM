WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

MAX_DRAFT_LEN=63
MAX_NON_LEAVES_PER_LAYER=10

CKPT_DIR="/scratch_1/tmp/trt_models/Meta-Llama-3-70B-Instruct_eagle_modelopt/tp${TP_SIZE}_pp${PP_SIZE}_max_draft_len${MAX_DRAFT_LEN}_max_non_leaves${MAX_NON_LEAVES_PER_LAYER}"
ENGINE_DIR="/scratch_1/tmp/trt_engines/Meta-Llama-3-70B-Instruct_eagle_modelopt/tp${TP_SIZE}_pp${PP_SIZE}_max_draft_len${MAX_DRAFT_LEN}_max_non_leaves${MAX_NON_LEAVES_PER_LAYER}"

MAX_BATCH_SIZE=64
ISL=1024
OSL=1024
MAX_NUM_TOKENS=$((ISL+OSL+MAX_DRAFT_LEN*(MAX_BATCH_SIZE+1)))
MAX_SEQ_LEN=$((ISL+OSL))

# echo all args
echo "TP_SIZE: $TP_SIZE"
echo "PP_SIZE: $PP_SIZE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "MAX_DRAFT_LEN: $MAX_DRAFT_LEN"
echo "MAX_NON_LEAVES_PER_LAYER: $MAX_NON_LEAVES_PER_LAYER"
echo "MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
echo "ISL: $ISL"
echo "OSL: $OSL"
echo "MAX_NUM_TOKENS: $MAX_NUM_TOKENS"
echo "MAX_SEQ_LEN: $MAX_SEQ_LEN"

trtllm-build --checkpoint_dir $CKPT_DIR \
             --output_dir $ENGINE_DIR \
             --speculative_decoding_mode eagle \
             --gemm_plugin auto \
             --workers $WORLD_SIZE \
             --max_batch_size $MAX_BATCH_SIZE \
             --low_latency_gemm_plugin fp8 \
             --max_input_len $ISL \
             --max_seq_len $MAX_SEQ_LEN \
             --max_num_tokens $MAX_NUM_TOKENS
             