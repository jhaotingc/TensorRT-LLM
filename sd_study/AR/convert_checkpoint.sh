WORK_DIR="/scratch/TensorRT-LLM-SD"

TP_SIZE=4
PP_SIZE=1
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

BASE_MODEL="/scratch_1/tmp/hf_models/Meta-Llama-3-70B-Instruct"
CKPT_DIR="/scratch_1/tmp/trt_models/Meta-Llama-3-70B-Instruct/tp${TP_SIZE}_pp${PP_SIZE}"

python $WORK_DIR/examples/quantization/quantize.py --model_dir $BASE_MODEL \
                                   --dtype float16 \
                                   --qformat fp8 \
                                   --kv_cache_dtype fp8 \
                                   --output_dir $CKPT_DIR \
                                   --calib_size 512 \
                                   --tp_size $TP_SIZE