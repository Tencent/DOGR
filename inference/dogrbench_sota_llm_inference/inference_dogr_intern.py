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
    def __init__(self, img_dir, ann_file, transform=None, split="single"):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item["image"])

        data_type = item["task_name"]
        if '?' in item["conversations"][0]["value"]:
            text_prompt = item["conversations"][0]["value"].replace("<image>\n", "").split("?")[0]+"?"
        else:
            text_prompt = item["conversations"][0]["value"].replace("<image>\n", "").split(".")[0]+"?"

        if "Gr" in data_type:
            text_prompt += " You should give a reasoning. In the reasoning, you should wrap some text from the document using grounded blocks like <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the document image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. And you must give the final simple answer at the end following 'Answer:' . And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        elif "Ga" in data_type:
            text_prompt +=  "You should directly answer the question use a simple answer, and the answer should be given in the format of grounded blocks like  <ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>, in which the answer is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. "
        elif "Go" in data_type:
            text_prompt += " You should give a response. In the response, you should wrap some text from the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image.  <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        else:
            text_prompt = "Answer the following question related to the region <bbox>x1, y1, x2, y2</bbox> of the image. The two corordinates is the left-top and right-bottom position of the text in the image. Question:" + text_prompt
            if "GRr" in data_type:
                text_prompt += " You should give a reasoning. In the reasoning, you should wrap some text in the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. Make sure your output format is correct.  <bbox></bbox>  must following corresponding <ocr></ocr>. And you must give the final simple answer at the end following 'Answer:' . And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
            elif "GRa" in data_type:
                text_prompt +=  "You should answer the question use a simple answer, and the answer should be given in the format of grounded blocks like  <ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>, in which the answer is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. "
            elif "GRo" in data_type:
                text_prompt += "You should give a response. In the response, you should wrap some text from the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. Make sure your output format is correct. <bbox></bbox> must following corresponding <ocr></ocr>. And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
            elif "Rt" in data_type:
                text_prompt += " Give a simple and direct answer."
            else:
                import pdb; pdb.set_trace()

        # if "Gr" in data_type:
        #     text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning. At the end, you should give a short and simple final answer following 'Answer:'. " #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        # elif "Ga" in data_type:
        #     text_prompt +=  "Answer the question with a simple text answer following ' Answer: ', and then corresponding bbox coordinate in [x1, y1, x2, y2] format follwing ' Bbox: '. [x1, y1, x2, y2] is the bounding box of the answer."
        # elif "Go" in data_type:
        #     text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        # else:
        #     text_prompt = "Answer the following question related to the region [x1, y1, x2, y2] of the image. Question:" + text_prompt
        #     if "GRr" in data_type:
        #         text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning. At the end, you should give a short and simple final answer following 'Answer:'." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        #     elif "GRa" in data_type:
        #         text_prompt +=  "Answer the question with a simple text answer following ' Answer: ', and then corresponding bbox coordinate in [x1, y1, x2, y2] format follwing ' Bbox: '. [x1, y1, x2, y2] is the bounding box of the answer."
        #     elif "GRo" in data_type:
        #         text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        #     elif "Rt" in data_type:
        #         text_prompt += " Give a simple and direct answer."
        #     else:
        #         import pdb; pdb.set_trace()

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None, required=True)
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()

    # dist.init_process_group(backend='hccl')


    # import pdb;pdb.set_trace()
    # print(args.local_rank)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    # device = f"npu:{local_rank}"


    path = 'OpenGVLab/InternVL2_5-8B'
    # path = 'OpenGVLab/InternVL2-8B'

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)



    output_file = f"temp_results_{local_rank}.jsonl"
    image_dir = "source_data/doc_downstream"
    ann_file = "source_data/doge_bench.jsonl"

    dataset = CustomImageDataset(img_dir=image_dir, ann_file=ann_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate, sampler=sampler)

    output = []
    # for items, text_prompts, imgs, videos in tqdm.tqdm(dataloader):
    #     for item, text_prompt, img, video in zip(items, text_prompts, imgs, videos):


    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image",
    #                         "image": img,
    #                     },
    #                     {"type": "text", "text": text_prompt}, # 并将其转化为标准的latex代码
    #                 ],
    #             }
    #         ]
    #         ---------------------------------------------
    for items, questions,num_patches_list, pixel_values in tqdm.tqdm(dataloader):
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        responses = model.batch_chat(tokenizer, pixel_values,
                                    num_patches_list=num_patches_list,
                                    questions=questions,
                                    generation_config=generation_config)
        
        for i,item in enumerate(items):
            item["model_answer"] = responses[i]
            print(responses[i])
            item["gt_answer"] = item["conversations"][1]["value"]
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




