import os
import glob
import tqdm
import json
import time
import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

import argparse
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from PIL import Image

# from openai import OpenAI
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from qwen_vl_utils import process_vision_info
dist.init_process_group(backend='nccl')  # 根据你的硬件选择合适的backend


# prepare input data
def is_image_file(path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    _, extension = os.path.splitext(path)
    return extension.lower() in image_extensions

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, split="single"):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["image"][0])

        data_type = item["task_name"]
        if '?' in item["messages"][0]["content"]:
            text_prompt = item["messages"][0]["content"].replace("<|image|>", "").split("?")[0]+"?"
        else:
            text_prompt = item["messages"][0]["content"].replace("<|image|>", "").split(".")[0]+"?"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": text_prompt}, # 并将其转化为标准的latex代码
                ],
            }
        ]
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except:
            print("Reading Image Error!")
            image_inputs, video_inputs = None, None
        return item, text_prompt, image_inputs, video_inputs

def custom_collate(batch):
    # 从批次中获取图像
    items = [item[0] for item in batch]
    img_paths = [item[1] for item in batch]
    image_inputs = [item[2] for item in batch]
    video_inputs = [item[3] for item in batch]
    return items, img_paths, image_inputs, video_inputs
import re
def process_bbox(text_prompt, width, height):
    # 定义正则表达式来匹配 <bbox> 标签及其内容
    bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')

    # 查找所有匹配的 <bbox> 标签
    matches = bbox_pattern.findall(text_prompt)

    # 处理每个匹配的 <bbox> 标签
    for match in matches:
        # 将坐标字符串分割成四个浮点数
        # print(match)
        try:
            coords = list(map(int, match.split(',')))
        except:
            continue
        
        # 检查是否有四个坐标
        if len(coords) != 4:
            continue
        
        # 分别乘以 width 和 height
        x_min, y_min, x_max, y_max = coords
        x_min = int(x_min*width/1000)
        y_min= int(y_min*height/1000)
        x_max = int(x_max*width/1000)
        y_max= int(y_max*height/1000)

        # 创建新的 <bbox> 标签内容
        # new_bbox = f'<bbox>{x_min},{y_min},{x_max},{y_max}</bbox>'
        new_bbox = f'[{x_min},{y_min},{x_max},{y_max}]'


        # 替换原始的 <bbox> 标签
        text_prompt = text_prompt.replace(f'<bbox>{match}</bbox>', new_bbox)

    return text_prompt


def process_bbox_late(input_string,width,height):
    if input_string==None:
        return ''
    pattern = r'```json(.*?)```'
    matches = re.findall(pattern, input_string, re.DOTALL)

    for match in matches:
        # import pdb;pdb.set_trace()


        try:
            json_output = ast.literal_eval(match)
            for i, item in enumerate(json_output):
                # import pdb;pdb.set_trace()

                try:

                    bbox = item["bbox_2d"]
                    x_min, y_min, x_max, y_max = bbox
                    x_min = int(x_min*1000/width)
                    y_min= int(y_min*1000/height)
                    x_max = int(x_max*1000/width)
                    y_max= int(y_max*1000/height)
                    return f'<bbox>{x_min},{y_min},{x_max},{y_max}</bbox>'
                except:
                    return '<bbox>0,0,0,0</bbox>'
        except Exception as e:
            try:
                end_idx = match.rfind('"}') + len('"}')
                truncated_text = match[:end_idx] + "]"
                json_output = ast.literal_eval(truncated_text)
                for i, item in enumerate(json_output):
                    try:
                        bbox = item["bbox_2d"]
                        x_min, y_min, x_max, y_max = bbox
                        x_min = int(x_min*1000/width)
                        y_min= int(y_min*1000/height)
                        x_max = int(x_max*1000/width)
                        y_max= int(y_max*1000/height)
                        return f'<bbox>{x_min},{y_min},{x_max},{y_max}</bbox>'
                    except:
                        return '<bbox>0,0,0,0</bbox>'
            except:
                # # print(match)
                # import pdb;pdb.set_trace()
                continue

                # continue
    return ''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None, required=True)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--task_name", type=str, default= None)

    args = parser.parse_args()

    # dist.init_process_group(backend='hccl')


    # import pdb;pdb.set_trace()
    # print(args.local_rank)
    local_rank = args.local_rank
    task_name = args.task_name

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    # device = f"npu:{local_rank}"


    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


    output_file = f"temp_results_{local_rank}.jsonl"
    # if os.path.exists(output_file):
    #     answer_data = load_jsonl(output_file)
    #     answer_qid = [item.keys()[0] for item in answer_data]
    # else:
    #     answer_qid = []

    if task_name=="grounding":
        ann_file = "[path to :]/DocStruct4M/DocLocal4K/text_grounding.jsonl"
    else:
        ann_file = "[path to :]DocStruct4M/DocLocal4K/text_recognition.jsonl"

    image_dir = "[path to :]DocStruct4M/DocLocal4K"


    dataset = CustomImageDataset(img_dir=image_dir, ann_file=ann_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate, sampler=sampler)

    output = []
    for items, text_prompts, imgs, videos in tqdm.tqdm(dataloader):
        for item, text_prompt, img, video in zip(items, text_prompts, imgs, videos):

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": text_prompt}, # 并将其转化为标准的latex代码
                    ],
                }
            ]

            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=img,
                videos=video,
                padding=True,
                return_tensors="pt",
            )
            input_height = int(inputs['image_grid_thw'][0][1]*14)
            input_width = int(inputs['image_grid_thw'][0][2]*14)
            if task_name=="recognition":               
                text_prompt = process_bbox(text_prompt, input_width,input_height)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img,
                        },
                        {"type": "text", "text": text_prompt}, # 并将其转化为标准的latex代码
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=img,
                videos=video,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            text_outputs = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            item["model_answer"] = text_outputs[0]
            if task_name=="grounding":               
                item["model_answer"]  = process_bbox_late(item["model_answer"] , input_width,input_height)
            # except:
            #     item["model_answer"] = ""
            item["gt_answer"] = item["messages"][1]["content"]
            item["wh"] = (input_width,input_height)

            output.append(item)
            print("pred:", item["model_answer"],"gt:", item["gt_answer"])
            # print("gt:", item["gt_answer"])
            # import pdb; pdb.set_trace()
        #     break
        # break
    
    output_dir = "/".join(args.output_file.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file.replace(".json", f"_{local_rank}.json"), "w") as f:
        print(output)
        json.dump(output, f)

    # dist.barrier()
    if local_rank == 0:
        final_result = []
        for rank_idx in range(dist.get_world_size()):
            while not os.path.exists(args.output_file.replace(".json", f"_{rank_idx}.json")):
                time.sleep(10)
                print("File not exists:", args.output_file.replace(".json", f"_{rank_idx}.json"))

            with open(args.output_file.replace(".json", f"_{rank_idx}.json"), "r") as f:
                final_result += json.load(f)
            os.remove(args.output_file.replace(".json", f"_{rank_idx}.json"))

        with open(args.output_file, "w") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)





