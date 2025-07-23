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
import time

import os
import sys
import warnings
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class CustomDataset(Dataset):
    def __init__(self, image_dir, ann_file, image_processor, model_config):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.model_config = model_config
        with open(ann_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path)
        image_tensor = process_images([image], self.image_processor, self.model_config)

        return item, image, image_tensor

def custom_collate(batch):
    # 从批次中获取图像
    items = [item[0] for item in batch]
    images = [item[1] for item in batch]
    image_tensors = [item[2] for item in batch]

    return items, images, image_tensors

def main():
    parser = argparse.ArgumentParser(description='benchmark evaluation')
    parser.add_argument('--model_path', type=str, help='the directory path of model')
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--test_file_path", type=str)

    args = parser.parse_args()

    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(world_size)

    pretrained = args.model_path
    test_file_path = args.test_file_path

    image_dir = "[path_to:]/focus_benchmark_test/en_pdf_png"
   
    model_name = "llava_qwen"
    device = f"npu:{rank}"
    device_map = f"npu:{rank}"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
    model.tokenizer = tokenizer
    model.eval()

    dataset = CustomDataset(image_dir, test_file_path, image_processor, model.config)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4, shuffle=False, num_workers=4, collate_fn=custom_collate)

    print("test_length:",len(dataset))
    template ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
    # template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

    output = {}
    rst_list = []
    for batch_items, batch_image, batch_image_tensor in tqdm.tqdm(dataloader):
        for item, image, image_tensor in zip(batch_items, batch_image, batch_image_tensor):
            
            qid = item["image"].split(".")[0]
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            prompt =  item["conversations"][0]["value"].replace("<image>",' ').replace("[","<bbox>").replace("]","</bbox>")
            gt_answer = item["conversations"][1]["value"]
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
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

            output_json = {}
            output_json['image'] = item["image"]
            output_json['question'] = prompt 
            output_json['label'] = item["conversations"][1]["value"]
            output_json['answer'] = text_outputs[0]
            rst_list.append(output_json)


    output_dir = "/".join(args.output_file.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file.replace(".json", f"_{rank}.json"), 'w', encoding="utf-8") as file_obj:
        json.dump(rst_list, file_obj, ensure_ascii=False, indent=1)

    
    if rank == 0:
        final_result = []
        for rank_idx in range(world_size):
            while not os.path.exists(args.output_file.replace(".json", f"_{rank_idx}.json")):
                time.sleep(10)
                print(f"LLaVA-NeXT/{output_dir}/doge_result_{rank_idx}.json")
            
            with open(args.output_file.replace(".json", f"_{rank_idx}.json"), "r") as f:
                final_result += json.load(f)
            
            # os.remove(args.output_file.replace(".json", f"_{rank_idx}.json"))
        with open(args.output_file, 'w', encoding="utf-8") as file_obj:
            json.dump(final_result, file_obj, ensure_ascii=False, indent=1)
       

if __name__ == '__main__':
    main()