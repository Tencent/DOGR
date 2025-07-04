from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
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
parser.add_argument('--dataset', type=str, choices=['DocVQA', 'InfographicsVQA', 'WikiTableQuestions', 'DeepForm', 'KleisterCharity', 'TabFact',
                                                    'ChartQA','ChartQA_PCG','ChartQAE', 'TextVQA', 'TextCaps', 'VisualMRC'])
args = parser.parse_args()

pretrained = args.model_path
dataset = args.dataset
image_dir = "/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0"
test_file_path = os.path.join(image_dir, 'test', dataset+'_llavanext_test.jsonl')
# "0,1,2,3"
model_name = "llava_qwen"
device = "npu:"+args.npu
device_map = "npu:"+args.npu
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
model.tokenizer = tokenizer
model.eval()


with open(test_file_path, "r") as f:
    test_data = [json.loads(line) for line in f]
print("test_length:",len(test_data))
template ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
# template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

output = {}
rst_list = []
for item in tqdm.tqdm(test_data):
    qid = item["image"].split("/")[-1].split(".")[0]

    # qid = item["qid"]
    img_path = os.path.join(image_dir, item["image"])
    image = Image.open(img_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    prompt =  item["conversations"][0]["value"].replace("<image>",' ')
    gt_answer = item["conversations"][1]["value"]

    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

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

    template_new = copy.deepcopy(template)
    template_new["image"].append(item["image"])
    template_new["task_name"] = item["task_name"]
    template_new["messages"][0]["content"]+=prompt
    template_new["messages"][1]["content"]+=gt_answer
    template_new["model_answer"]=text_outputs[0]
    template_new["gt_answer"]=gt_answer
    rst_list.append(template_new)

    # import pdb; pdb.set_trace()
    

model_name = pretrained.split("/")[-1]
data_name = test_file_path.split("/")[-1].split(".")[0]
output_dir = f"inference_with_mplug/{dataset}_inference_output/{model_name}"
os.makedirs(output_dir.replace("inference_with_mplug","/group/40079/yinanzhou/LLaVA-NeXT/inference_with_mplug"), exist_ok=True)

with open(f"/group/40079/yinanzhou/LLaVA-NeXT/{output_dir}/{data_name}_result.jsonl", "w") as f:
    for line in rst_list:
        json.dump(line, f)
        f.write('\n')