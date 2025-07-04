import gradio as gr
import json
import cv2
import numpy as np
import os
import re
from PIL import Image, ImageDraw
from copy import deepcopy
# 读取 JSONL 文件并解析数据
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line)
            # if  "<bbox>" in json_data['messages'][0]['content'] and "<bbox>" not in json_data['messages'][1]['content'] and "CCMAIN" in line:
            data.append(json_data)
    return data

# 绘制边界框
def draw_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 红色边框
    return image

# 处理图像和数据
def process_data(data):
    img_path = data['image'][0]
    # img_path = img_path.replace("./imgs/CCMAIN",'/group/40034/yinanzhou/pdf_grounding_construction/ccmain_png_filtered')
    img_path = "/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/"+img_path
    question = data['messages'][0]['content']
    answer = data['model_answer']
    gt = data['gt_answer']

    image = Image.open(img_path)
    if image is not None:
        image_gt = deepcopy(image)
        draw = ImageDraw.Draw(image)
        draw1 = ImageDraw.Draw(image_gt)

        width, height = image.size
        
        bbox_pattern = re.compile(r'<bbox>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</bbox>')
        # 提取所有bbox
        bboxes = []
        bboxes1 = []

        bboxes+=bbox_pattern.findall(question)
        bboxes1+=bbox_pattern.findall(question)

        bboxes+=bbox_pattern.findall(data['model_answer'])
        bboxes1+=bbox_pattern.findall(data['gt_answer'])

        # print(bboxes)

        # 将字符串转换为整数
        bboxes = [tuple(map(int, bbox)) for bbox in bboxes]
        bboxes1 = [tuple(map(int, bbox)) for bbox in bboxes1]

        colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
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
        for i, bbox in enumerate(bboxes1):
            # 计算真实坐标
            real_bbox = (
                bbox[0] / 999 * width,
                bbox[1] / 999 * height,
                bbox[2] / 999 * width,
                bbox[3] / 999 * height
            )
            # 绘制矩形
            # draw1.rectangle(real_bbox, outline=colors[i % len(colors)], width=2)
    
    return image, question, answer,image_gt, gt


def load_all_data(folder_path):
    all_data = {}
    # Walk through all directories and files in the given folder path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                all_data[str(file_path)] = load_data(file_path)
    return all_data


def count_data(selected_file):
    data = all_data[selected_file]

    return len(data) - 1  # 返回最大索引

# 显示数据
def display_data(index, selected_file):
    data = all_data[selected_file][index]
    image, question, answer,image_gt, explanation = process_data(data)
    return image, question, answer,image_gt, explanation

# 加载所有数据
folder_path = '/group/40034/yinanzhou/cvpr_rebuttal'  # 替换为你的文件夹路径
all_data = load_all_data(folder_path)
# 获取文件名列表
file_names = list(all_data.keys())

# 创建 Gradio 界面
def update_slider(selected_file):
    # 更新滑块的最大值
    max_index = count_data(selected_file)
    return gr.update(maximum=max_index)

# 创建 Gradio 界面
with gr.Blocks() as demo:
    selected_file = gr.Dropdown(choices=file_names, label="Select JSONL File", interactive=True)
    example_index = gr.Slider(minimum=0, maximum=0, step=1, label="Select Example")
    
    # 更新滑块的最大值
    selected_file.change(update_slider, inputs=selected_file, outputs=example_index)
    question_output = gr.Textbox(label="Question")
    image_output = gr.Image(type="numpy", label="Chart Image")
    answer_output = gr.Textbox(label="Answer")
    image_output_gt = gr.Image(type="numpy", label="Chart Image")
    gt_output = gr.Textbox(label="GTs")
    
    # 显示数据
    example_index.change(display_data, inputs=[example_index, selected_file], outputs=[image_output, question_output, answer_output, image_output_gt,gt_output])

# 启动 Gradio 界面
demo.launch(server_name="0.0.0.0", server_port=12345)