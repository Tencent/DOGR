import os
import io
import glob
import json
import tqdm
import base64
import numpy as np
from PIL import Image, ImageOps
from openai import OpenAI
import argparse
import time

from concurrent import futures

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def single_query(model_name, query, base64_image):

    response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
            temperature=0.2,
            )
    return response


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

        new_bbox = f'<bbox>{x_min},{y_min},{x_max},{y_max}</bbox>'

        return new_bbox
def process_ocr(text_prompt):
    bbox_pattern = re.compile(r'<ocr>(.*?)</ocr>')
    matches = bbox_pattern.findall(text_prompt)
    for match in matches:
        new_bbox = match
        return new_bbox

def multi_try_query(model_name, query, base64_image, max_try=3):
    # for _ in range(max_try):
        # try:
    fail=0
    while True:
        try:
            response = single_query(model_name, query, base64_image)
        except:
            response = None
        # this = response.choices[0].message.content
        if response!=None:
            return response
        else:
            fail+=1
            time.sleep(10)
            print("10s retry")
            if fail>6:
                return None
        #     break
        # except:
        #     response = None
        #     continue
    # return response

def get_response(item,task_name):
    img_path = os.path.join(test_img_dir, item["image"][0])
    data_type = item["task_name"]    
    base64_image = encode_image(img_path)

    text_prompt = item["messages"][0]["content"].replace("<|image|>", "")
    # print(text_prompt)

    if task_name=="recognition":
        text_prompt = f"Output the corresponding text of the bounding box {process_bbox(text_prompt)}, the bbox is in <bbox>x1,y1,x2,y2</bbox> format, the two corordinates((x1,y1),(x2,y2)) are the left-top and right-bottom position of the text in the image. x1, y1, x2, y2 in the normalized coordinates in [0,999] based on its width and height. Your output should in this format: <ocr> answer </ocr>, in which answer is the text of the given bounding box."
    elif task_name=="grounding":
        text_prompt = f"Directly output the bounding box of the text '{process_ocr(text_prompt)}' in <bbox>x1,y1,x2,y2</bbox> format, the two corordinates((x1,y1),(x2,y2)) are the left-top and right-bottom position of the text in the image. x1, y1, x2, y2 in the normalized coordinates in [0,999] based on its width and height."
    # print(text_prompt)
    # import pdb;pdb.set_trace()
    response = multi_try_query(model_name, text_prompt, base64_image)

    return item, response, text_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, default= None)
    parser.add_argument("task_name", type=str, default= None)

    args = parser.parse_args()
    task_name = args.task_name
    os.environ['OPENAI_API_KEY'] = "[OPENAI_API_KEY]"
    client = OpenAI(base_url="[URL]")
    if task_name=="grounding":
        src_file_path = "[Path to :]DocStruct4M/DocLocal4K/text_grounding.jsonl"
    else:
        src_file_path = "[Path to :]DocStruct4M/DocLocal4K/text_recognition.jsonl"

    test_img_dir = "[Path to :]DocStruct4M/DocLocal4K"

    with open(src_file_path, "r") as f:
        test_data = [json.loads(line) for line in f]
    

    # model_name = "gpt-4o-2024-08-06"
    # model_name = "gpt-4o-mini"
    # model_name ="gemini-1.5-flash"
    # model_name = "gemini-1.5-pro"
    # model_name = "qwen-vl-max-2025-01-25"
    # model_name = "gemini-2.5-flash"
    model_name = "gemini-2.5-pro"

    # model_name = "gemini-2.0-flash"

    result = []
    records = []

    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        process_futures = []
        for i, item in enumerate(test_data):
            process_futures.append(executor.submit(get_response, item,task_name))
        for future in tqdm.tqdm(futures.as_completed(process_futures), total=len(process_futures)):
            item, response, text_prompt = future.result()
            if response is not None:
                response = response.choices[0].message.content
            else:
                response=''
            item["model_answer"] = response
            item["gt_answer"] = item["messages"][1]["content"]
            result.append(item)
            # print(text_prompt)
            print(response)
    with open(f"[Save path]/eval_rst/{args.exp_name}_{model_name}.json", "w") as f:
        # json.dump(result, f, indent=2, ensure_ascii=False)
        json.dump(result, f, ensure_ascii=False)

            