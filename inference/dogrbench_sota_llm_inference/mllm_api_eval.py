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
    # You are an AI assistant that helps people find information.\n\nFor mathematical formulas, please wrap the formula with $$ or $.
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

def multi_try_query(model_name, query, base64_image, max_try=3):
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

def get_response(item):
    img_path = os.path.join(test_img_dir, item["image"])
    data_type = item["task_name"]    
    base64_image = encode_image(img_path)

    text_prompt = item["conversations"][0]["value"].replace("<image>\n", "")

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
    # print(text_prompt)
    response = multi_try_query(model_name, text_prompt, base64_image)
    # print(response)

    return item, response, text_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, default= None)
    args = parser.parse_args()

    os.environ['OPENAI_API_KEY'] = "[OPENAI_API_KEY]"
    client = OpenAI(base_url="[URL]")


    src_file_path = "[src file path]"

    test_img_dir = "[test img dir]"

    with open(src_file_path, "r") as f:
        test_data = [json.loads(line) for line in f]
    

    # model_name = "gpt-4o-2024-08-06"
    # model_name = "gpt-4o-mini"
    # model_name = "gemini-1.5-pro"
    # model_name = "gemini-2.0-pro"
    # model_name = "gemini-2.0-flash"
    # model_name = "gemini-2.5-flash"
    model_name = "gemini-2.5-pro"



    result = []
    # test_data = test_data["single"]
    records = []
    # 先过滤掉answer为无的样本
    
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        process_futures = []
        for i, item in enumerate(test_data):
            process_futures.append(executor.submit(get_response, item))
        for future in tqdm.tqdm(futures.as_completed(process_futures), total=len(process_futures)):
            item, response, text_prompt = future.result()
            if response is not None:
                response = response.choices[0].message.content
            item["model_answer"] = response
            item["gt_answer"] = item["conversations"][1]["value"]
            result.append(item)
            # print(text_prompt)
            print(response)
    with open(f"[save path]/{args.exp_name}_{model_name}.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
            