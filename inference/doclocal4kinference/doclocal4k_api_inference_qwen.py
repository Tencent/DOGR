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
import ast
from concurrent import futures

from transformers import  AutoProcessor
from qwen_vl_utils import process_vision_info

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=1000*28*28)
def encode_image(image_path):
    # with open(image_path, "rb") as image_file:
    #     return base64.b64encode(image_file.read()).decode("utf-8")
    max_pixels = 1000 * 28 * 28

    with Image.open(image_path) as img:
        # Calculate the current number of pixels
        current_pixels = img.width * img.height

        # If the image has more pixels than the maximum allowed, resize it
        if current_pixels > max_pixels:
            # Calculate the scaling factor
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)

            # Resize the image while maintaining the aspect ratio
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the image to a bytes buffer
        buffered = io.BytesIO()
        img = img.convert('RGB')

        img.save(buffered, format="JPEG")

        # Encode the image to base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def single_query(model_name, query, base64_image):
    # You are an AI assistant that helps people find information.\n\nFor mathematical formulas, please wrap the formula with $$ or $.
    print(query)

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
def process_bbox(text_prompt,width,height):
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
        x_min = int(x_min*width/1000)
        y_min= int(y_min*height/1000)
        x_max = int(x_max*width/1000)
        y_max= int(y_max*height/1000)

        new_bbox = f'[{x_min},{y_min},{x_max},{y_max}]'

        return new_bbox

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


def process_ocr(text_prompt):
    pattern = r"<ocr>(.*?)</ocr>"

    # Search for the pattern in the text
    match = re.search(pattern, text_prompt, re.DOTALL)

    # If a match is found, extract the content
    if match:
        content = match.group(1).strip()
        return content
    # bbox_pattern = re.compile(r'<ocr>(.*?)</ocr>')
    # matches = bbox_pattern.findall(text_prompt, re.DOTALL)
    # for match in matches:
    #     new_bbox = match
    #     if new_bbox=='':
    #         import pdb;pdb.set_trace()
    #     return new_bbox
    # if matches==[]:
    #     bbox_pattern = re.compile(r'<ocr>(.*?)')
    #     matches = bbox_pattern.findall(text_prompt, re.DOTALL)
    #     for match in matches:
    #         new_bbox = match.replace("</ocr>","")
    #         if new_bbox=='':
    #             import pdb;pdb.set_trace()

    #         return new_bbox
    #     return "NONE!!!!!!!!!!!!!!!"+ text_prompt
    # import pdb;pdb.set_trace()


def multi_try_query(model_name, query, base64_image, max_try=3):
    response = single_query(model_name, query, base64_image)

    # for _ in range(max_try):
    #     try:
    #         response = single_query(model_name, query, base64_image)
    #         break
    #     except:
    #         response = None
    #         continue
    return response

def get_response(item,task_name):
    text_prompt = item["messages"][0]["content"].replace("<|image|>", "")

    img_path = os.path.join(test_img_dir, item["image"][0])
    data_type = item["task_name"]    
    base64_image = encode_image(img_path)
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
    img, video = process_vision_info(messages)
    inputs = processor(
                text=[text_prompt],
                images=img,
                videos=video,
                padding=True,
                return_tensors="pt",
            )

    input_height = int(inputs['image_grid_thw'][0][1]*14)
    input_width = int(inputs['image_grid_thw'][0][2]*14)



    if task_name=="recognition":
        text_prompt = f"Output the corresponding text of the bounding box {process_bbox(text_prompt,input_width,input_height)}. Your output should in this format: <ocr> answer </ocr>, in which answer is the text of the given bounding box."
        # print(text_prompt)
    elif task_name=="grounding":
        text_prompt = f"Directly output the one whole bounding box coordinates of the text '{process_ocr(text_prompt)}' in JSON format."
    # print(text_prompt)
    # import pdb;pdb.set_trace()
    response = multi_try_query(model_name, text_prompt, base64_image)
    # print(response)

    return item, response, text_prompt,input_width,input_height

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, default= None)
    parser.add_argument("task_name", type=str, default= None)

    args = parser.parse_args()
    task_name = args.task_name
    os.environ['OPENAI_API_KEY'] = "[OPENAI_API_KEY]"
    client = OpenAI(base_url="[URL]")
    if task_name=="grounding":
        src_file_path = "[path to :]DocStruct4M/DocLocal4K/text_grounding.jsonl"
    else:
        src_file_path = "[path to :]DocStruct4M/DocLocal4K/text_recognition.jsonl"

    test_img_dir = "[path to :]DocStruct4M/DocLocal4K"

    with open(src_file_path, "r") as f:
        test_data = [json.loads(line) for line in f]
    

    model_name ="qwen2.5-vl-7b-instruct"
    result = []
    records = []
   
        
    with futures.ThreadPoolExecutor(max_workers=5) as executor:
        process_futures = []
        for i, item in enumerate(test_data):
            process_futures.append(executor.submit(get_response, item,task_name))
        for future in tqdm.tqdm(futures.as_completed(process_futures), total=len(process_futures)):
            item, response, text_prompt,width,height = future.result()
            if response is not None:
                response = response.choices[0].message.content
            print(response)
            if task_name=="grounding":
                response = process_bbox_late(response,width,height)
            item["model_answer"] = response
            item["gt_answer"] = item["messages"][1]["content"]
            result.append(item)
            # print(text_prompt)
            print(response)
    with open(f"[save path]inference_output/{args.exp_name}_{model_name}.json", "w") as f:
        # json.dump(result, f, indent=2, ensure_ascii=False)
        json.dump(result, f, ensure_ascii=False)

            