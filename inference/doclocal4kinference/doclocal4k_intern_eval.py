import os
import glob
import tqdm
import json
import time
import torch

import argparse
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from PIL import Image

import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


dist.init_process_group(backend='nccl')  # 根据你的硬件选择合适的backend


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
generation_config = dict(max_new_tokens=1024, do_sample=True)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
   
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
    def __init__(self, img_dir, ann_file,task_name, transform=None, split="single"):
        self.img_dir = img_dir
        self.task_name = task_name

        self.transform = transform
        with open(ann_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["image"][0])


        text_prompt = item["messages"][0]["content"]
 
        if self.task_name=="recognition":
            text_prompt = f"Directly output the corresponding text of the bounding box {process_bbox(text_prompt)}, the bbox is in [x1,y1,x2,y2] format, the two corordinates((x1,y1),(x2,y2)) are the left-top and right-bottom position of the text in the image. x1, y1, x2, y2 in the normalized coordinates in [0,999] based on its width and height. Your output should in this format: <ocr> answer </ocr>, in which answer is the text of the given bounding box."
        elif self.task_name=="grounding":
            text_prompt = f"Directly output the bounding box coordinates of the text <ref>{process_ocr(text_prompt)}</ref>."


        pixel_values = load_image(img_path, max_num=12)

        question = "<image>\n"+text_prompt
        num_patch = pixel_values.size(0)
        return item, question,num_patch, pixel_values


def custom_collate(batch):
    # 从批次中获取图像
    items = [item[0] for item in batch]
    questions = [item[1] for item in batch]
    num_patches_list = [item[2] for item in batch]
    pixel_values = [item[3] for item in batch]


    return items, questions, num_patches_list, pixel_values
import re
def process_bbox(text_prompt):

    bbox_pattern = re.compile(r'<bbox>(.*?)</bbox>')
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
        x_min, y_min, x_max, y_max = coords

        new_bbox = f'[{x_min},{y_min},{x_max},{y_max}]'

        return new_bbox
def process_ocr(text_prompt):
    pattern = r"<ocr>(.*?)</ocr>"

    # Search for the pattern in the text
    match = re.search(pattern, text_prompt, re.DOTALL)

    # If a match is found, extract the content
    if match:
        content = match.group(1).strip()
        return content
# def process_bbox_late(input_string,width,height):
def process_bbox_late(input_string):

    if input_string==None:
        return ''
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, input_string, re.DOTALL)
    # import pdb;pdb.set_trace()
    for match in matches:
        # import pdb;pdb.set_trace()


        try:
            bbox = match.split(',')
            if len(bbox)==4:
                    x_min, y_min, x_max, y_max = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
                    # x_min = int(x_min*1000/width)
                    # y_min= int(y_min*1000/height)
                    # x_max = int(x_max*1000/width)
                    # y_max= int(y_max*1000/height)
                    return f'<bbox>{x_min},{y_min},{x_max},{y_max}</bbox>'
                
        except:
            return '<bbox>0,0,0,0</bbox>'
    return '<bbox>0,0,0,0</bbox>'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None, required=True)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--task_name", type=str, default= None)

    args = parser.parse_args()

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    # device = f"npu:{local_rank}"

    task_name = args.task_name

    if task_name=="grounding":
        ann_file = "[path to:]DocStruct4M/DocLocal4K/text_grounding.jsonl"
    else:
        ann_file = "[path to:]DocStruct4M/DocLocal4K/text_recognition.jsonl"

    image_dir = "[path to:]DocStruct4M/DocLocal4K"


    path = 'OpenGVLab/InternVL2_5-8B'
    # path = 'OpenGVLab/InternVL2-8B'

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # use_flash_attn=True,
        use_flash_attn=False,

        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)



    output_file = f"temp_results_{local_rank}.jsonl"


    dataset = CustomImageDataset(img_dir=image_dir, ann_file=ann_file,task_name=task_name)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate, sampler=sampler)

    output = []

    for items, questions,num_patches_list, pixel_values in tqdm.tqdm(dataloader):
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        print(questions)
        try:
            responses = model.batch_chat(tokenizer, pixel_values,
                                        num_patches_list=num_patches_list,
                                        questions=questions,
                                        generation_config=generation_config)
        except:
            continue
        for i,item in enumerate(items):
            res = process_bbox_late(responses[i])
            item["model_answer"] = res
            print(res)
            item["gt_answer"] = item["messages"][1]["content"]
            output.append(item)

    
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




