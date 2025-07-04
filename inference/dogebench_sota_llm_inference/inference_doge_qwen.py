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
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

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
        img_path = os.path.join(self.img_dir, item["image"])

        data_type = item["task_name"]
        if '?' in item["conversations"][0]["value"]:
            text_prompt = item["conversations"][0]["value"].replace("<image>\n", "").split("?")[0]+"?"
        else:
            text_prompt = item["conversations"][0]["value"].replace("<image>\n", "").split(".")[0]+"?"


        # if "Gr" in data_type:
        #     text_prompt += " You should give a reasoning including necessary grounded blocks like this '<ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>', in which the 'text' is from the document image and the two corordinates is the left-top and right-bottom position of the 'text' in the image.'<bbox></bbox>' must following corresponding '<ocr></ocr>'. And x1, y1, x2, y2 in the normalized coordinates in [0,999]. And you must give the final simple answer at the end following 'Answer:' ." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        # elif "Ga" in data_type:
        #     text_prompt +=  "You should directly answer the question use a simple answer, and the answer should be given in the format of grounded blocks like this '<ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>', in which the 'answer' is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. '<bbox></bbox>' must following corresponding '<ocr></ocr>'. And x1, y1, x2, y2 in the normalized coordinates in [0,999]. "
        # elif "Go" in data_type:
        #     text_prompt += " You should give a response including necessary grounded blocks like this '<ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>', in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. '<bbox></bbox>' must following corresponding '<ocr></ocr>'. And x1, y1, x2, y2 in the normalized coordinates in [0,999]." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        # else:
        #     text_prompt = "Answer the following question related to the region <bbox>x1, y1, x2, y2</bbox> of the image. x and y are normalized into [0,999]. The two corordinates is the left-top and right-bottom position of the text in the image. '<bbox></bbox>' must following corresponding '<ocr></ocr>'. Question:" + text_prompt
        #     if "GRr" in data_type:
        #         text_prompt += " You should give a reasoning including necessary grounded blocks like this '<ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>', in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. And x1, y1, x2, y2 in the normalized coordinates in [0,999]. '<bbox></bbox>' must following corresponding '<ocr></ocr>'. And you must give the final simple answer at the end following 'Answer:' ." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        #     elif "GRa" in data_type:
        #         text_prompt +=  "You should answer the question use a simple answer, and the answer should be given in the format of grounded blocks like this '<ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>', in which the 'answer' is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. '<bbox></bbox>' must following corresponding '<ocr></ocr>'. And x1, y1, x2, y2 in the normalized coordinates in [0,999]. "
        #     elif "GRo" in data_type:
        #         text_prompt += " You should give a response including necessary grounded blocks like this '<ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>', in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. And x1, y1, x2, y2 in the normalized coordinates in [0,999]. '<bbox></bbox>' must following corresponding '<ocr></ocr>'." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        #     elif "Rt" in data_type:
        #         text_prompt += " You should give a simple and direct answer with only text and without bounding boxes."
        #     else:
        #         import pdb; pdb.set_trace()


        # if "Gr" in data_type:
        #     text_prompt += " You should give a reasoning. In the reasoning, you should wrap some text from the document using grounded blocks like <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the document image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. And you must give the final simple answer at the end following 'Answer:' . And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        # elif "Ga" in data_type:
        #     text_prompt +=  "You should directly answer the question use a simple answer, and the answer should be given in the format of grounded blocks like  <ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>, in which the answer is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. "
        # elif "Go" in data_type:
        #     text_prompt += " You should give a response. In the response, you should wrap some text from the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image.  <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        # else:
        #     text_prompt = "Answer the following question related to the region <bbox>x1, y1, x2, y2</bbox> of the image. The two corordinates is the left-top and right-bottom position of the text in the image. Question:" + text_prompt
        #     if "GRr" in data_type:
        #         text_prompt += " You should give a reasoning. In the reasoning, you should wrap some text in the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. Make sure your output format is correct.  <bbox></bbox>  must following corresponding <ocr></ocr>. And you must give the final simple answer at the end following 'Answer:' . And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        #     elif "GRa" in data_type:
        #         text_prompt +=  "You should answer the question use a simple answer, and the answer should be given in the format of grounded blocks like  <ocr> answer </ocr><bbox>x1, y1, x2, y2</bbox>, in which the answer is the text from the image and the two corordinates is the left-top and right-bottom position of the text in the image. <bbox></bbox> must following corresponding <ocr></ocr>. Make sure your output format is correct. "
        #     elif "GRo" in data_type:
        #         text_prompt += "You should give a response. In the response, you should wrap some text from the document using grounded blocks like  <ocr> text </ocr><bbox>x1, y1, x2, y2</bbox>, in which the text is from the image and the two corordinates is the left-top and right-bottom position of the text in the image. Make sure your output format is correct. <bbox></bbox> must following corresponding <ocr></ocr>. And the grounded blocks should be as much as possible. The text and coordinates should be accurate." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        #     elif "Rt" in data_type:
        #         text_prompt += " Give a simple and direct answer."
        #     else:
        #         import pdb; pdb.set_trace()

        if "Gr" in data_type:
            text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning. At the end, you should give a short and simple final answer following 'Answer:'. " #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
        elif "Ga" in data_type:
            text_prompt +=  "Answer the question with a simple text answer following ' Answer: ', and then corresponding bbox coordinate in [x1, y1, x2, y2] format follwing ' Bbox: '. [x1, y1, x2, y2] is the bounding box of the answer."
        elif "Go" in data_type:
            text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
        else:
            text_prompt = "Answer the following question related to the region [x1, y1, x2, y2] of the image. Question:" + text_prompt
            if "GRr" in data_type:
                text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning. At the end, you should give a short and simple final answer following 'Answer:'." #  x1, y1, x2, y2 in the normalized coordinates in [0,999].
            elif "GRa" in data_type:
                text_prompt +=  "Answer the question with a simple text answer following ' Answer: ', and then corresponding bbox coordinate in [x1, y1, x2, y2] format follwing ' Bbox: '. [x1, y1, x2, y2] is the bounding box of the answer."
            elif "GRo" in data_type:
                text_prompt += "You should prepare the key text of your reasoning and corresponding bbox coordinate in JSON format, like```json\n[{\"bbox_2d\": [x1,y1,x2,y2], \"text\": \"\"}\n]```. The text should be exactly the content in bbox_2d. After the json output, give a reasoning." #  x1, y1, x2, y2 in the normalized coordinates in [0,999]
            elif "Rt" in data_type:
                text_prompt += " Give a simple and direct answer."
            else:
                import pdb; pdb.set_trace()

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


    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map=device,
    # )

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     device_map=device,
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        device_map=device,  # Use auto device placement
        torch_dtype=torch.bfloat16)

    # default processer
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct",  max_pixels=1280*28*28) # 
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=4, max_pixels=2000*28*28)
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


    output_file = f"temp_results_{local_rank}.jsonl"
    # if os.path.exists(output_file):
    #     answer_data = load_jsonl(output_file)
    #     answer_qid = [item.keys()[0] for item in answer_data]
    # else:
    #     answer_qid = []

    image_dir = "source_data/doc_downstream"
    ann_file = "source_data/doge_bench.jsonl"

    dataset = CustomImageDataset(img_dir=image_dir, ann_file=ann_file)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate, sampler=sampler)

    output = []
    for items, text_prompts, imgs, videos in tqdm.tqdm(dataloader):
        for item, text_prompt, img, video in zip(items, text_prompts, imgs, videos):
            # print(item["task_name"])
            # if "Ga" not in item["task_name"] and "GRa" not in item["task_name"] and "Rt" not in item["task_name"]:
            #     continue
            # try:
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
            # except:
            #     item["model_answer"] = ""
            item["gt_answer"] = item["conversations"][1]["value"]
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





