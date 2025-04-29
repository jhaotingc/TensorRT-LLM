# Llama + Eagle end-to-end script 
(tested on 41a6c98)

1. Git clone [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) and [yuhuili/EAGLE-LLaMA3-Instruct-70B](https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-70B)

2. Merge LLama and Eagle checkpoint for ModelOpt to do FP8 PTQ together with [eagle/merge_eagle.sh](eagle/merge_eagle.sh)  

3. Clone [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and build from source with `pip install -e ".[all]" --extra-index-url https://pypi.nvidia.com`

4. Quantize BF16 checkpoint with ModelOpt using [eagle/quantize_eagle.sh](eagle/quantize_eagle.sh)  
    4.1 Feel free to change calib_size / batch_size / calibration dataset.  
    4.2 Keep `--inference_tensor_parallel=1`, `--inference_pipeline_parallel=1` and `--export_fmt=hf`. TRTLLM will handle TP/PP later  

5. Convert ModelOpt/HF checkpoint to TRTLLM checkpoint with [eagle/convert_checkpoint.sh](eagle/convert_checkpoint.sh)

6. Build TRTLLM FP8 engine with [eagle/build_model.sh](eagle/build_model.sh)  
    6.1 Feel free to change MAX_BATCH_SIZE / MAX_INPUT_LEN / MAX_SEQ_LEN / MAX_NUM_TOKENS.  
    6.2 Keep `--speculative_decoding_mode eagle`  

7. (Optional) Test run the engine with [eagle/run.sh](eagle/run.sh)

8. Launch TRTLLM server with [eagle/serve.sh](eagle/serve.sh)  
    8.1 run `eagle/serve.sh &` for server to launch in background  
    8.2 configure `PORT`, `HOST` if needed.  
    8.3 Set eagle config in [eagle/extra_llm_api_options.yaml](eagle/extra_llm_api_options.yaml)  

8.3.1 Eagle 1 example:  
```
speculative_config:
    eagle_choices: [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    decoding_type: Eagle
    greedy_sampling: false
    num_eagle_layers: 4
    max_non_leaves_per_layer: 10
    max_draft_len: 63
    pytorch_eagle_weights_path: null
```

        8.3.2 Eagle 2 example:  
            Note: Currently Eagle 2 trtllm-serve is having a bug where `eagle_choices` expect a value. Please wait for the quick fix. 
            Should be solvable very quick zzz.


```yaml
speculative_config:
    eagle_choices: null
    decoding_type: Eagle
    greedy_sampling: false
    num_eagle_layers: 4
    max_non_leaves_per_layer: 10
    max_draft_len: 63
    pytorch_eagle_weights_path: null
    use_dynamic_tree: true
    dynamic_tree_max_top_k: 10
```

9. Send curl.
    ```bash
    curl localhost:8000/v1/completions     -H "Content-Type: application/json"     -d '{
            "model": "Llama3-70b-eagle",
            "prompt": "NVIDIA is a great company because",
            "max_tokens": 1024,
            "temperature": 0
        }' -w "\n" --write-out "Time_Total: %{time_total}\n"
    ```

        See the response
        
    ```bash
    INFO:     ::1:52492 - "POST /v1/completions HTTP/1.1" 200 OK
    {"id":"cmpl-f200d26e65da4d0c98323bb3aaa42bdd","object":"text_completion","created":1745902407,"model":"tp4_pp1_max_draft_len63_max_non_leaves10","choices":[{"index":0,"text":" of its innovative products and its commitment to making a positive impact on society. NVIDIA is a leader in the field of artificial intelligence, and its products are used in a wide range of applications, from gaming to healthcare to autonomous vehicles. The company is also committed to using its technology to make a positive impact on society, whether it's through its work in AI for good, its support for STEM education, or its efforts to reduce its environmental footprint.\n\nOne of the things that I admire most about NVIDIA is its commitment to innovation. The company is constantly pushing the boundaries of what is possible with technology, and its products are always at the cutting edge","logprobs":null,"context_logits":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null}],"usage":{"prompt_tokens":8,"total_tokens":136,"completion_tokens":128}}
    ```
