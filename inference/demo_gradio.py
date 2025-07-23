

import gradio as gr
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
from PIL import Image, ImageDraw
import re
import os
import sys
import warnings
from gradio_image_prompter import ImagePrompter
# from gradio_box_promptable_image import BoxPromptableImage

import argparse
MODEL_PATH="[MODEL_PATH]"
warnings.filterwarnings("ignore")
model_name = "llava_qwen"
device = "npu:0"
device_map = "npu:0"
tokenizer, model, image_processor, max_length = load_pretrained_model(MODEL_PATH, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
model.tokenizer = tokenizer
model.eval()


def split_string_by_list(s, lst):
    result = []
    temp = ""
    i = 0
    while i < len(s):
        match = False
        for item in lst:
            if s[i:i+len(item)] == item:
                if temp:
                    result.append(temp)
                    temp = ""
                result.append(item)
                i += len(item)
                match = True
                break
        if not match:
            temp += s[i]
            i += 1
    if temp:
        result.append(temp)
    return result

def infer_instance(prompter, qs,prompt_added):
    image = prompter['image']
    points = prompter['points']
    print(prompter)
    print(image.shape)
    image = Image.fromarray(image)

    width, height = image.size
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"


    prompt = qs+prompt_added
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
    print(text_outputs[0])
    # text_outputs[0] = text_outputs[0].replace("&lt;",'<').replace("&gt;",">").encode('utf-8').decode('utf-8')



    draw = ImageDraw.Draw(image)



        
    bbox_pattern = re.compile(r'<bbox>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</bbox>')
    ocr_pattern = re.compile(r'<ocr>((?:(?!</ocr>).)*)</ocr><bbox>((?:(?!</bbox>).)*)</bbox>')

    
    highlighted_text = []
    i=0
    ocrs = ocr_pattern.findall(text_outputs[0])
    print(ocrs)
    bboxes = []
    for ocr in ocrs:
        bboxes+=[ocr[1].split(',')]
    bboxes = [tuple(map(int, bbox)) for bbox in bboxes]


    colors = ["red", "orange", "yellow", "green", "blue", "pink", "purple"]
    # for i, bbox in enumerate(bboxes):
    #     # 计算真实坐标
    #     real_bbox = (
    #         bbox[0] / 999 * width,
    #         bbox[1] / 999 * height,
    #         bbox[2] / 999 * width,
    #         bbox[3] / 999 * height
    #     )
    #     # 绘制矩形
    #     draw.rectangle(real_bbox, outline=colors[i % len(colors)], width=4)



    
    drawed_bbox = []
    color_dict={}
    ocr_list = []
    i=0
    for ocr in ocrs:
        # import pdb;pdb.set_trace()
        name=re.escape(ocr[0])
        ocr_pattern2 = re.compile(f'<ocr>\s*{name}\s*</ocr>\s*<bbox>\s*{ocr[1]}\s*</bbox>')
        bbox= tuple(map(int, ocr[1].split(',')))
        real_bbox = (
            bbox[0] / 999 * width,
            bbox[1] / 999 * height,
            bbox[2] / 999 * width,
            bbox[3] / 999 * height
        )
        pcrs0 = ocr_pattern2.findall(text_outputs[0])
        ocr_list+=pcrs0
        for pcrs in pcrs0:
            if pcrs not in color_dict:
                color_dict.update({pcrs:str(i%7)})

        if real_bbox not in drawed_bbox:
            draw.rectangle(real_bbox, outline=colors[i % len(colors)], width=4)
            i+=1
            drawed_bbox+=[real_bbox]
    print(text_outputs[0])
    print(color_dict)


    line = []
    sentence = text_outputs[0]
    sentence_list = split_string_by_list(sentence,ocr_list)
    
    highlighted_text=[]
    for s in sentence_list:
        is_added = False
        if s in color_dict:
            is_added=True
            highlighted_text.append([s+" ",color_dict[s]])

        if not is_added:
            highlighted_text.append([s+" ",None])
    drawed = draw_new_bbox("Bbox in Prompt", prompter,prompt)
    if len(ocrs)==0:
        return highlighted_text,gr.Image(type="numpy", label="DOGR BBOX ANSWER",visible=False),drawed,'Bbox in Prompt'
    return  highlighted_text,gr.Image(type="numpy", label="DOGR BBOX ANSWER",visible=True,value= image),drawed,'Bbox in Prompt'
    
            




def add_bbox(prompter,prompt):
    image = prompter['image']
    points = prompter['points']
    print(points)
    print(image.shape)
    image = Image.fromarray(image)
    width, height = image.size
    bbox = []
    bbox_str = ""

    i=0
    # draw = ImageDraw.Draw(image)
    colors = ["red", "orange", "yellow", "green", "blue", "pink", "purple"]

    if len(points)>0:
        for point in points:
            if point[3]>0 and point[4]>0:
                bbox +=[[f'<bbox>{str(int(999*point[0]/width))},{str(int(999*point[1]/height))},{str(int(999*point[3]/width))},{str(int(999*point[4]/height))}</bbox> ',str(i%7)] ]
                bbox_str += f'<bbox>{str(int(999*point[0]/width))},{str(int(999*point[1]/height))},{str(int(999*point[3]/width))},{str(int(999*point[4]/height))}</bbox>'
                # draw.rectangle([point[0],point[1],point[3],point[4]], outline=colors[i % len(colors)], width=4)
                i+=1

            else:
                bbox +=[[" INVALID ","ERROR"]]
    

    return bbox,"Selected Bbox",add_bbox2(prompter,prompt)


def add_bbox2(prompter,prompt):
    image = prompter['image']
    points = prompter['points']
    print(points)
    print(image.shape)
    image = Image.fromarray(image)
    width, height = image.size
    bbox = []
    bbox_str = ""

    i=0
    # draw = ImageDraw.Draw(image)
    colors = ["red", "orange", "yellow", "green", "blue", "pink", "purple"]

    if len(points)>0:
        for point in points:
            if point[3]>0 and point[4]>0:
                bbox +=[[f'<bbox>{str(int(999*point[0]/width))},{str(int(999*point[1]/height))},{str(int(999*point[3]/width))},{str(int(999*point[4]/height))}</bbox> ',str(i%7)] ]
                bbox_str += f'<bbox>{str(int(999*point[0]/width))},{str(int(999*point[1]/height))},{str(int(999*point[3]/width))},{str(int(999*point[4]/height))}</bbox>'
                i+=1

            else:
                bbox +=[[" INVALID ","ERROR"]]
    return gr.Textbox(value = prompt+ " "+bbox_str)

def clear_prompt():
    return gr.Textbox(value = "")
# def high_light_selected(prompter,output_text):
#     for output in output_text[:]:
#         print(output)
#         if output['class_or_confidence']==None:
#             output_text.remove(output)
#     # image = draw_new_bbox("Selected Bbox",prompter,'')
    

#     return image,"Selected Bbox"

# def high_light_prompt(prompter,output_text):
#     for output in output_text[:]:
#         if output['class_or_confidence']==None:
#             output_text.remove(output)
#     # image = draw_new_bbox("Bbox in Prompt",prompter,'')
    

#     return image,"Bbox in Prompt"
def draw_new_bbox(bboxdraw_btn, prompter,prompt):
    if bboxdraw_btn =="Selected Bbox":
        image = prompter['image']
        points = prompter['points']
        print(points)
        print(image.shape)
        image = Image.fromarray(image)
        width, height = image.size
        bbox = []
        bbox_str = ""

        i=0
        draw = ImageDraw.Draw(image)
        colors = ["red", "orange", "yellow", "green", "blue", "pink", "purple"]

        if len(points)>0:
            for point in points:
                if point[3]>0 and point[4]>0:
                    draw.rectangle([point[0],point[1],point[3],point[4]], outline=colors[i % len(colors)], width=4)
                    i+=1
    
        return image

    else:
        image = prompter['image']
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        ocr_pattern = re.compile(r'<bbox>((?:(?!</bbox>).)*)</bbox>')

        ocrs = ocr_pattern.findall(prompt)
        bboxes = []
        for ocr in ocrs:
            bboxes+=[ocr.split(',')]
        bboxes = [tuple(map(int, bbox)) for bbox in bboxes]


        colors = ["red", "orange", "yellow", "green", "blue", "pink", "purple"]
        for i, bbox in enumerate(bboxes):
            # 计算真实坐标
            real_bbox = (
                bbox[0] / 999 * width,
                bbox[1] / 999 * height,
                bbox[2] / 999 * width,
                bbox[3] / 999 * height
            )
            # 绘制矩形
            draw.rectangle(real_bbox, outline=colors[i % len(colors)], width=2)
        return image


def prompt_coloring( prompter,prompt):
    drawed_image = draw_new_bbox('Bbox in Prompt',prompter,prompt)
    highlighted_text = []
    i=0
   
    color_dict={}
    i=0

    ocr_pattern2 = re.compile(r'<bbox>\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*</bbox>')
    ocr_list = ocr_pattern2.findall(prompt)
    print(ocr_list)

    for pcrs in ocr_list:
        if pcrs not in color_dict:
            color_dict.update({pcrs:i})
            i+=1
            
    line = []
    sentence = prompt
    sentence_list = split_string_by_list(sentence,ocr_list)
    
    highlighted_text=[]
    for s in sentence_list:
        is_added = False
        if s in color_dict:
            
            highlighted_text.append([s+" ",str(color_dict[s])])
            is_added= True
        if not is_added:
            highlighted_text.append([s+" ",None])
    print(highlighted_text)
    return  highlighted_text,drawed_image,'Bbox in Prompt'
# def refresh(prompter, task, prompt,bboxdraw_btn):
#     if bboxdraw_btn =="Selected Bbox":
#         image = prompter['image']
#         points = prompter['points']
#         print(points)
#         print(image.shape)
#         image = Image.fromarray(image)
#         width, height = image.size
#         bbox = []
#         bbox_str = ""

#         i=0
#         draw = ImageDraw.Draw(image)
#         colors = ["red", "orange", "yellow", "green", "blue", "pink", "violet"]

#         if len(points)>0:
#             for point in points:
#                 if point[3]>0 and point[4]>0:
#                     draw.rectangle([point[0],point[1],point[3],point[4]], outline=colors[i % len(colors)], width=4)
#                     i+=1
    
#         return image

#     else:
#         image = prompter['image']
#         image = Image.fromarray(image)
#         draw = ImageDraw.Draw(image)
#         width, height = image.size
#         ocr_pattern = re.compile(r'<bbox>((?:(?!</bbox>).)*)</bbox>')

#         ocrs = ocr_pattern.findall(prompt)
#         bboxes = []
#         for ocr in ocrs:
#             bboxes+=[ocr.split(',')]
#         bboxes = [tuple(map(int, bbox)) for bbox in bboxes]


#         colors = ["red", "orange", "yellow", "green", "blue", "pink", "violet"]
#         for i, bbox in enumerate(bboxes):
#             # 计算真实坐标
#             real_bbox = (
#                 bbox[0] / 999 * width,
#                 bbox[1] / 999 * height,
#                 bbox[2] / 999 * width,
#                 bbox[3] / 999 * height
#             )
#             # 绘制矩形
#             draw.rectangle(real_bbox, outline=colors[i % len(colors)], width=2)
#         return image
# def change_status():
#     return "Selected Bbox"
import random
def update_task2_options(task):
    if task == "Long Answer":
        return gr.update(choices=["With bbox"], value="With bbox")
    else:
        return gr.update(choices=["With bbox", "Without bbox"], value="Without bbox")
def edit_instruction(task,task2):
    short_grounded=["Provide a direct answer and include the answer's bounding box if available.",
        "Give a straightforward response and specify the bounding box of the answer if it exists.",
        "Respond directly and include the bounding box of the answer if it is present.",
        "Give a clear and simple answer and mention the answer's bounding box if it exists.",
        "Answer directly and provide the bounding box for the answer if applicable.",
        "Please respond succinctly and include the answer's bounding box if it is available.",
        "Deliver a direct answer and indicate the bounding box of the answer if it exists.",
        "Provide a concise response and include the answer's bounding box if it is available."]

    long_part1 = ["Answer the question in detail with bounding boxes.",
        "Give grounding and detailed answer.",
        "Provide a detailed answer with bounding boxes.",
        "Offer a comprehensive response using bounding boxes.",
        "Deliver an in-depth answer with bounding boxes.",
        "Give a thorough explanation with bounding boxes."]
    long_part2 = ["Provide the answer at the end.",
        "Deliver the answer at the conclusion.",
        "Present the answer at the end.",
        "Give the answer in the final part.",
        "State the answer at the conclusion.",
        "Reveal the answer at the end.",
        "Offer the answer at the finish."
        ]
    long = ["Answer the question in detail.",
        "Give detailed answer.",
        "Provide a detailed answer.",
        "Offer a comprehensive response.",
        "Deliver an in-depth answer.",
        "Give a thorough explanation."]
    if task =="Short Answer" and task2=="With bbox":
        return random.choice(short_grounded)
    elif task =="Short Answer" and task2=="Without bbox":
        return ''
    elif task =="Long Answer" and task2=="Without bbox":
        return random.choice(long)
    else:
        return random.choice(long_part1)+" "+random.choice(long_part2)


    return ""
def infer_model():
    # image = gr.Image(type="pil",label="Input Image")
    # prompt =  gr.Textbox(label="Prompt")
    # image_output_gt = gr.Image(type="numpy", label="Chart Image")
    
    # image.select(get_click_coords, [image], [image, image])
    with gr.Blocks() as demo:
        gr.Markdown('<h1 style="text-align: center;">DOGR: Towards Versatile Visual Document Grounding and Referring</h1>')
        # prompter = gr.Image(type="pil",label="Input Image")
        # prompter = BoxPromptableImage()
        refresh_trigger = gr.Number(value=0, visible=False)
        with gr.Row():

            with gr.Column(scale=2):
                prompter = ImagePrompter(label="1.Select your image, and your bbox(Optional).")
                prompt =  gr.Textbox(label="2. Edit your prompt")
                bboxdraw_btn = gr.Radio(
                        choices=["Selected Bbox", 'Bbox in Prompt'],
                        type="value",
                        value="Selected Bbox",
                        label="draw your bbox",
                        )

                drawed_image = gr.Image(type="pil",label="Image with your bbox")

            with gr.Column(scale=2):

                bbox_btn = gr.Button("2.(Optional)Add Bboxes into Prompt")
                output_text = gr.HighlightedText(
                    label="Your selected bbox corrdinates",
                    combine_adjacent=True,
                    # show_legend=True,
                    color_map={"0": "red", "1":"orange","2": "yellow", "3":"green", "4":"blue", "5":"pink", "6":"purple"},
                    # interactive=True
                    
                    )
                # bbox_btn2 = gr.Button("3.Add them into your PROMPT INSTRUCTION")
                output_text2 = gr.HighlightedText(
                    label="Your edited prompt with highlighted bboxes",
                    combine_adjacent=True,
                    # show_legend=True,
                    color_map={"0": "red", "1":"orange","2": "yellow", "3":"green", "4":"blue", "5":"pink", "6":"purple"},
                    # interactive=True
                    
                    )
                
                bbox_btn3 = gr.Button("CLEAR YOUR PROMPT INSTRUCTION")
                task = gr.Radio(
                choices=['Short Answer',"Long Answer"],
                type="value",
                value="Short Answer",
                label="Response Length",
                )

                task2 = gr.Radio(
                choices=["With bbox", 'Without bbox'],
                type="value",
                value="Without bbox",
                label="Response Type",
                )
                bbox_btn4 = gr.Button("Try another extra instruction")
                prompt_added =  gr.Textbox(label="3. Select your extra instruction type / Self-made Extra Instruction")        



                greet_btn = gr.Button("3.Submit")

            with gr.Column(scale=3):
            # import pdb; pdb.set_trace()
                image_output_gt = gr.Image(type="numpy", label="DOGR BBOX ANSWER")
                outtext = gr.HighlightedText(
                label="DOGR OUTPUT",
                combine_adjacent=True,
                show_legend=False,
                color_map={"0": "red", "1":"orange","2": "yellow", "3":"green", "4":"blue", "5":"pink", "6":"purple","ERROR":"gray"}
                )
        # image.select(,inputs=[image],outputs=None)
        s= gr.State()
        # prompter.change(fn=change_status,inputs=None,outputs=bboxdraw_btn)
        bbox_btn.click(fn=add_bbox,inputs=[prompter,prompt],outputs=[output_text,bboxdraw_btn,prompt])
        # bbox_btn2.click(fn=add_bbox2,inputs=[prompter,prompt],outputs=prompt)
        bbox_btn3.click(fn=clear_prompt,inputs=[],outputs=prompt)
        prompt.change(fn=prompt_coloring,inputs=[prompter,prompt],outputs=[output_text2,drawed_image,bboxdraw_btn])
        # output_text.change(fn=high_light_selected,inputs=[prompter,output_text],outputs=[drawed_image,bboxdraw_btn])
        # output_text2.change(fn=high_light_prompt,inputs=[prompter,output_text2],outputs=[drawed_image,bboxdraw_btn])
        task.change(fn=edit_instruction,inputs=[task,task2],outputs=prompt_added)
        task2.change(fn=edit_instruction,inputs=[task,task2],outputs=prompt_added)
        bbox_btn4.click(fn=edit_instruction,inputs=[task,task2],outputs=prompt_added)


        # import pdb; pdb.set_trace()
        greet_btn.click(fn=infer_instance, 
            inputs=[prompter, prompt,prompt_added], 
            outputs=[
            outtext,image_output_gt,drawed_image,bboxdraw_btn], 
        api_name="Submit")
        # import pdb; pdb.set_trace()

        bboxdraw_btn.change(fn=draw_new_bbox,
            inputs=[bboxdraw_btn, prompter,prompt],
            outputs=drawed_image
        )
        with gr.Column():
            gr.Examples(
                examples=[
                    [
                        {'image':"inference/demo_images/seafood.png","points":[]},
                        "Short Answer","With bbox",
                        "What are the main ingredients in the <bbox>421,108,577,134</bbox>?",
                        "Bbox in Prompt"
                        
                    ],
                    [
                        {'image':"inference/demo_images/doge_chart.png","points":[]},
                        "Long Answer","With bbox",
                        "What is the sum of the value for <bbox>131,876,283,912</bbox>, <bbox>527,719,689,749</bbox>, and <bbox>305,904,703,939</bbox>?",
                        "Bbox in Prompt"
                        
                    ],
                    [
                        {'image':"inference/demo_images/doc0.png","points":[]},
                        "Long Answer","With bbox",
                        "What was the price per share for the common stock acquired on 04/30/2021?",
                        "Bbox in Prompt"
                        
                    ],
                    [
                        
                        {'image':"inference/demo_images/chris.png","points":[]},
                        "Long Answer","With bbox",
                        "What is the holiday in <bbox>127,614,595,748</bbox>?",
                        "Bbox in Prompt"
                    ],
                    [
                        
                        {'image':"inference/demo_images/doge_page.png","points":[]},
                        "Short Answer","Without bbox",
                        "Translate the text in <bbox>86,311,485,392</bbox> into Chinese.",
                        "Bbox in Prompt"
                    ],
                    [
                        
                        {'image':"inference/demo_images/handwritten.png","points":[]},
                        "Long Answer","With bbox",
                        "Why is DOGE useful?",
                        "Bbox in Prompt"
                    ]



                ],
                inputs=[prompter, task,task2, prompt,bboxdraw_btn],
                # outputs=drawed_image,
                # fn=refresh,
                # cache_examples=True,

                outputs=None,
                fn=None,
                cache_examples=False,
            )
    # def process_input(input_dict):
    #     img, points = input_dict['image'], input_dict['points']
    #     box_inputs = get_box_inputs(points)

    #     for box in box_inputs:
    #         x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    #         cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 2)

    #     return img


    # demo = gr.Interface(
    #         process_input,
    #         BoxPromptableImage(),
    #         gr.Image(),
    #         # examples=examples,
    #     )
        
    demo.launch(server_name="0.0.0.0", server_port=12366)



if __name__ == "__main__":

    infer_model()
