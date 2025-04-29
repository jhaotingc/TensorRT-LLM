WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

CKPT_DIR="/scratch_1/tmp/trt_models/Meta-Llama-3-70B-Instruct/tp${TP_SIZE}_pp${PP_SIZE}"
ENGINE_DIR="/scratch_1/tmp/trt_engines/Meta-Llama-3-70B-Instruct/tp${TP_SIZE}_pp${PP_SIZE}"

trtllm-build --checkpoint_dir $CKPT_DIR \
             --output_dir $ENGINE_DIR \
             --gemm_plugin auto \
             --workers $WORLD_SIZE \
             --low_latency_gemm_plugin fp8