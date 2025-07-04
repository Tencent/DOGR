from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import torch_npu
import json
import tqdm

import os
import sys
import warnings


import argparse

parser = argparse.ArgumentParser(description='benchmark evaluation')
parser.add_argument('--model_path', type=str, help='the directory path of model')
parser.add_argument('--npu', type=str, help='device')
parser.add_argument('--dataset', type=str, choices=['mplug','crello',"chartqa",'ccmain'])
# 高分辨率resize+padding的ckpt:
# pretrained = "/group/40079/uasonchen/projects/LLaVA-NeXT/checkpoints/high/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-64npu-qwen_1_5"
args = parser.parse_args()
# 高分辨率resize的ckpt:
pretrained = args.model_path
warnings.filterwarnings("ignore")
# pretrained = "checkpoints/high/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-64npu-qwen_1_5"
# pretrained = "checkpoints/high/llavanext-InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-stage3-sogou-math-solution-20k-mathv360k-qwen_1_5/checkpoint-2800"
# pretrained = "/group/40079/uasonchen/projects/LLaVA-NeXT/checkpoints/high/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-64npu-qwen_1_5"
model_name = "llava_qwen"
device = "npu:"+args.npu
device_map = "npu:"+args.npu
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
model.tokenizer = tokenizer
model.eval()

# image_dir = "playground/data"
# test_file_path = "playground/data/sogou_math34k/sogou_math34k_answer_test_llava_format.jsonl"
# with open(test_file_path, "r") as f:
#     test_data = [json.loads(line) for line in f]

# test_file_path = "/group/40034/uasonchen/projects/formula-ocr/refine_exam_data/output/refine_math_34k_test_llava.json"
# with open(test_file_path, "r") as f:
#     test_data = json.load(f)
    
# image_dir = "/group/40034/uasonchen/projects/LLaVA-NeXT/playground/data/sogou_math34k/images"

image_dirs = ["[path to]DocStruct4M/DocLocal4K",'[path to]DocStruct4M/DocDownstream-1.0']

test_file_path = {"mplug":"[path to]/mplug_text_grounding_llava.jsonl",
"ccmain":'[path to]ccmain_grounding_test_1round_type_annoted_llava.jsonl',
"chartqa":'[path to]test_t2bbox_with_random_mask_pad_1round_type_annoted_llava.jsonl',
'crello':'[path to]grd_test_1round_type_annoted_llava.jsonl'}




with open(test_file_path[args.dataset], "r") as f:
    test_data = [json.loads(line) for line in f]

# template_bbox2t ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "bbox2t_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

output = {}
rst_list = []
for item in tqdm.tqdm(test_data):
    qid = item["image"].split("/")[-1].split(".")[0]

    # qid = item["qid"]
    if args.dataset=='mplug':
        image_dir = image_dirs[0]

    else:
        image_dir = image_dirs[1]

    img_path = os.path.join(image_dir, item["image"])
    # img_path = os.path.join(image_dir, item["image"].replace("DocStruct4M/",""))
    if os.path.exists(img_path):
        print(img_path)
        None
    elif os.path.exists( os.path.join(image_dir, item["image"].replace("DocStruct4M","DocDownstream-1.0"))):
        
        print(img_path)
    else:
        print('not exists',img_path)
        continue
    # img_path = os.path.join(image_dir, qid + ".png")
    image = Image.open(img_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    prompt =  item["conversations"][0]["value"].replace("<image>",' ')
    gt_answer = item["conversations"][1]["value"]

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    # question = DEFAULT_IMAGE_TOKEN + "\nGive the bounding box of the text <ocr> △ABC </ocr>"
    # question = DEFAULT_IMAGE_TOKEN + "\nplease answer the question in the image."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=1024,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    output[qid] = {"pred": text_outputs[0], "gt": gt_answer, "question": prompt}
    print(f"qid: {qid}\npred: {text_outputs[0]}\ngt: {gt_answer}")

    template_t2bbox_new = copy.deepcopy(template_t2bbox)
    template_t2bbox_new["image"].append(item["image"])
    template_t2bbox_new["task_name"] = item["task_name"]
    template_t2bbox_new["messages"][0]["content"]+=prompt
    template_t2bbox_new["messages"][1]["content"]+=gt_answer
    template_t2bbox_new["model_answer"]=text_outputs[0]
    template_t2bbox_new["gt_answer"]=gt_answer
    rst_list.append(template_t2bbox_new)



    # import pdb; pdb.set_trace()
    

model_name = pretrained.split("/")[-1]
data_name = test_file_path[args.dataset].split("/")[-1].split(".")[0]
output_dir = f"[path to]/grd_inference_output/{args.dataset}/{model_name}"
os.makedirs(output_dir, exist_ok=True)

with open(f"[path to]/inference_with_mplug/grd_inference_output/{args.dataset}/{model_name}/{data_name}_result.jsonl", "w") as f:
    for line in rst_list:
        json.dump(line, f)
        f.write('\n')