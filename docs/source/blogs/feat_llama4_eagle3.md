# How to run Llama4 Maverick + Eagle3 with feat/llama4

1. Git clone, checkout to branch `feat/llama4`
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout feat/llama4
git submodule update --init --recursive
git lfs pull
```

2. Build the release docker image ([installation guide](https://nvidia.github.io/TensorRT-LLM/installation/build-from-source-linux.html#option-1-build-tensorrt-llm-in-one-stepp))
```

make -C docker release_build
```

3. (Optional) tag and push the docker image to your own registry
```
docker tag tensorrt_llm/release:latest docker.io/<username>/tensorrt_llm:llama4
docker push docker.io/<username>/tensorrt_llm:llama4
```

4. Start the server with the following command
4.1 be sure to mount Maverick and Eagle3 checkpoints to `/config/models/maverick` and `/config/models/eagle` respectively.
4.2 the `-d` flag runs the container in detached mode (run the server in the background)
4.3 the `-p 8000:8000` flag maps port 8000 on the host to port 8000 on the container
```
docker run -d --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8000:8000 --gpus=all -e "TRTLLM_ENABLE_PDL=1" -v /path/to/maverick:/config/models/maverick -v /path/to/eagle:/config/models/eagle docker.io/<username>/tensorrt_llm:llama4 sh -c "echo -e 'enable_attention_dp: false\npytorch_backend_config:\n  enable_overlap_scheduler: true\n  autotuner_enabled: false\n  use_cuda_graph: true\n  cuda_graph_max_batch_size: 8\nspeculative_config:\n  decoding_type: Eagle\n  max_draft_len: 3\n  pytorch_eagle_weights_path: /config/models/eagle\nkv_cache_config:\n  enable_block_reuse: false' > c.yaml && trtllm-serve /config/models/maverick --host 0.0.0.0 --port 8000 --backend pytorch --max_batch_size 8 --max_num_tokens 8192 --max_seq_len 8192 --tp_size 8 --ep_size 1 --trust_remote_code --extra_llm_api_options c.yaml --kv_cache_free_gpu_memory_fraction 0.75"
```

5. Send requests to the server. Here's an example curl request.
```
curl localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
        "model": "Llama4-eagle",
        "prompt": "NVIDIA is a great company because",
        "max_tokens": 1024
    }' -w "\n"

# response: {"id":"cmpl-ceb4791612564e25bf94274ee6179811","object":"text_completion","created":1747263257,"model":"fp8-128e-ckpt","choices":[{"index":0,"text":" it has a strong brand, a diverse product portfolio, and a dominant position in the graphics processing unit (GPU) market. The company has a long history of innovation and has been at the forefront of GPU technology for many years. NVIDIA's GPUs are used in a wide range of applications, including gaming, professional visualization, data center, and automotive. The company's GPUs are known for their high performance, power efficiency, and advanced features such as ray tracing and artificial intelligence (AI) acceleration.\nIn addition to its GPU business, NVIDIA has a growing presence in other areas such as datacenter, automotive, and professional visualization. The","logprobs":null,"context_logits":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null}],"usage":{"prompt_tokens":7,"total_tokens":132,"completion_tokens":125}}
```

6. (Optional) retrieve the logs of an active container (trtllm-serve server)
```
docker ps # get the container id
docker logs -f <container_id>
```

7. (Optional) kill the server
```
docker ps # get the container id
docker kill <container_id>
```
