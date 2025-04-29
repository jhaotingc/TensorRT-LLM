# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import transformers
from transformers import AutoTokenizer

import modelopt.torch.speculative as mtsp
import argparse
import os
import json

#init main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eagle_dir", type=str, default="/trt_llm/data/llm-models/EAGLE-Vicuna-7B-v1.3")
    parser.add_argument("--base_model", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--eagle_num_layers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    # Now "model" is a EAGLE model in ModelOpt format

    # Load in the EAGLE module weight from your eagle ckpt
    # https://huggingface.co/yuhuili/EAGLE-Vicuna-7B-v1.3/tree/main
    EAGLE_DIR = args.eagle_dir
    eagle_weghts = torch.load(os.path.join(EAGLE_DIR, "pytorch_model.bin"), map_location="cpu")

    #print all eagle weights
    print(f"EAGLE_DIR: {EAGLE_DIR}")
    for key, value in eagle_weghts.items():
        print(f"\t{key}:\t{value.shape}")

    # Adjust the parameters depending on your eagle module weight
    eagle_config = {
        "eagle_num_layers": args.eagle_num_layers,
    }


    # load in your base model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
    )

    mtsp.convert(model, [("eagle", eagle_config)])

    for i in range(args.eagle_num_layers):
        # Replace the eagle weight in modelopt eagle model
        model.eagle_module.layers[i].self_attn.q_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.self_attn.q_proj.weight"])
        model.eagle_module.layers[i].self_attn.k_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.self_attn.k_proj.weight"])
        model.eagle_module.layers[i].self_attn.v_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.self_attn.v_proj.weight"])
        model.eagle_module.layers[i].self_attn.o_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.self_attn.o_proj.weight"])
        model.eagle_module.layers[i].mlp.gate_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.mlp.gate_proj.weight"])
        model.eagle_module.layers[i].mlp.up_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.mlp.up_proj.weight"])
        model.eagle_module.layers[i].mlp.down_proj.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.mlp.down_proj.weight"])
        model.eagle_module.layers[i].post_attention_layernorm.weight = torch.nn.Parameter(eagle_weghts[f"layers.{i}.post_attention_layernorm.weight"])

    model.eagle_module.fc.weight = torch.nn.Parameter(eagle_weghts[f"fc.weight"])
    if "fc.bias" in eagle_weghts:
        model.eagle_module.fc.bias = torch.nn.Parameter(eagle_weghts[f"fc.bias"])
    else:
        # print(model.eagle_module.fc)
        # model.eagle_module.fc = torch.nn.Linear(eagle_weghts[f"fc.weight"].shape[0], eagle_weghts[f"fc.weight"].shape[1], bias=False)
        model.eagle_module.fc.bias = torch.nn.Parameter(torch.zeros(eagle_weghts[f"fc.weight"].shape[0], dtype=eagle_weghts[f"fc.weight"].dtype))

    # create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model
    )
    tokenizer.save_pretrained(args.output_dir)
