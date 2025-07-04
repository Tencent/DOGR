from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import json
import tqdm

import os
import sys
import warnings

test_file_path = "/group/40079/yinanzhou/LLaVA-NeXT/rcg_inference_output/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-64npu-qwen_1_5/mplug_text_recognition_llava_result.json"

with open(test_file_path, "r") as f:
    test_data = json.load(f)

template_bbox2t ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "bbox2t_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
# template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

output = {}
rst_list = []
for item in tqdm.tqdm(test_data):
    

    template_bbox2t_new = copy.deepcopy(template_bbox2t)
    template_bbox2t_new["image"].append(item)

    template_bbox2t_new["model_answer"]=test_data[item]["pred"]
    template_bbox2t_new["gt_answer"]=test_data[item]["gt"]
    rst_list.append(template_bbox2t_new)




    # import pdb; pdb.set_trace()
    



with open(f"test_result_rcg.jsonl", "w") as f:
    for line in rst_list:
        json.dump(line, f)
        f.write('\n')