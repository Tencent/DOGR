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

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class ExamDataset(Dataset):
    def __init__(self, image_dir, ann_file, image_processor, model_config, split="single",):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.model_config = model_config
        with open(ann_file, "r") as f:
            self.data = json.load(f)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        qid = item["qid"]
        img_path = os.path.join(self.image_dir, qid + ".png")
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
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 高分辨率resize+padding的ckpt:
    # pretrained = "/group/40079/uasonchen/projects/LLaVA-NeXT/checkpoints/high/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-64npu-qwen_1_5"

    # 高分辨率resize的ckpt:
    pretrained = "/group/40043/uasonchen/projects/LLaVA-NeXT/checkpoints/high/llavanext-model_zoo_InternViT-300M-448px-from-InternVL2-8b-model_zoo_Qwen_Qwen2-7B-Instruct-ln_mlp2x_gelu-mplug-pretrain-no-padding-64npu-qwen_1_5"
    model_name = "llava_qwen"
    device = f"npu:{rank}"
    device_map = f"npu:{rank}"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
    model.tokenizer = tokenizer
    model.eval()

    image_dir = "playground/data"
    image_dir = "/group/40033/public_datasets/DocStruct4M"
    test_file_path = "/group/40079/yinanzhou/LLaVA-NeXT/playground/annotations/mplug_text_grounding_llava.jsonl"

    with open(test_file_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    # template_bbox2t ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "bbox2t_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
    template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

    output = {}
    rst_list = []
    for item in tqdm.tqdm(test_data):
        qid = item["image"].split("/")[-1].split(".")[0]

        # qid = item["qid"]
        img_path = os.path.join(image_dir, item["image"])

        if os.path.exists(img_path):
            None
        elif os.path.exists( os.path.join(image_dir, item["image"].replace("DocStruct4M","DocDownstream-1.0"))):
            img_path = os.path.join(image_dir, item["image"].replace("DocStruct4M","DocDownstream-1.0"))
        else:
            continue
        # img_path = os.path.join(image_dir, qid + ".png")
        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        # question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
        prompt =  item["conversations"][0]["value"].replace("<image>",' ')
        gt_answer = item["conversations"][1]["value"]

        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        # question = DEFAULT_IMAGE_TOKEN + "\nGive the bounding box of the text <ocr> △ABC </ocr>"
        # question = DEFAULT_IMAGE_TOKEN + "\nplease answer the question in the image."
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

        template_t2bbox_new = copy.deepcopy(template_t2bbox)
        template_t2bbox_new["image"].append(item["image"])
        template_t2bbox_new["messages"][0]["content"]+=prompt
        template_t2bbox_new["messages"][1]["content"]+=gt_answer
        template_t2bbox_new["model_answer"]=text_outputs[0]
        template_t2bbox_new["gt_answer"]=gt_answer
        rst_list.append(template_t2bbox_new)




        # import pdb; pdb.set_trace()
        

    model_name = pretrained.split("/")[-1]
    data_name = test_file_path.split("/")[-1].split(".")[0]
    output_dir = f"grd_inference_output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)


    with open(f"{output_dir}/{data_name}_result_{rank}.jsonl", "w") as f:
        for line in rst_list:
            json.dump(line, f)
            f.write('\n')

    if rank == 0:
        final_result = []
        for rank_idx in range(world_size):
            with open(f"{output_dir}/{data_name}_result_{rank_idx}.jsonl", "r") as f:
                datas = []
                for line in f:
                    datas.append(json.loads(line))
                    
                final_result += datas
            os.remove(f"{output_dir}/{data_name}_result_{rank_idx}.jsonl")

        with open(f"{output_dir}/{data_name}_result.jsonl", "w") as f:
            for line in final_result:
                json.dump(line, f)
                f.wrire('\n')
        


if __name__ == '__main__':
    main()