from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import requests
import copy
import torch
import torch_npu
import json
import tqdm
import time

import os
import sys
import warnings
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

def all_reduce_tensor(t, op=dist.ReduceOp.SUM):
    dist.all_reduce(t, op=op)
    return t


class CustomDataset(Dataset):
    def __init__(self, image_dir, ann_file, image_processor, model_config):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.model_config = model_config
        with open(ann_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item["image"])
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
    parser = argparse.ArgumentParser(description='benchmark evaluation')
    parser.add_argument('--model_path', type=str, help='the directory path of model')
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--test_file_path", type=str)

    args = parser.parse_args()

    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pretrained = args.model_path
    test_file_path = args.test_file_path
    
    # path_list=
        # "/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/test/DOGE_test_llava.jsonl"
        # "#ccmain PDF
        #     #pretrain
        #        'pdf_pretrain_grd': '/group/40034/yinanzhou/pdf_grounding_construction/grd_pretrain_construction/ccmain_grounding_test_llava.jsonl',#grounding decode-ccamin 9k
        #         'pdf_pretrain_rcg':'/group/40034/yinanzhou/pdf_grounding_construction/grd_pretrain_construction/ccmain_recognition_test_llava.jsonl',#recogniton decode-ccamin 10k
        #     #decode
        #         'pdf_decode':'/group/40034/yinanzhou/pdf_grounding_construction/grd_pretrain_construction/ccmain_pdf_decode_new_tactic_test_llava.jsonl',#grounded decode-ccamin  636
        #     #sft
        #         'pdf_qa_finegrained':'/group/40034/yinanzhou/pdf_grounding_construction/GPT4oConstructor/ccamin_fine_grained_qa_new_test_llava.jsonl',#open test --ccmain 4k
        #         'pdf_qa':'/group/40034/yinanzhou/pdf_grounding_construction/GPT4oConstructor/ccamin_fine_grained_qa_simple_test_llava.jsonl',#short answer test -ccmain 6k
        #         'pdf_qa_open':'/group/40034/yinanzhou/pdf_grounding_construction/GPT4oConstructor/ccamin_long_text_qa_new_test_llava.jsonl',#long-text test-ccmain 418

        # #ChartQA CHART
        #     #pretrain
        #         'chart_pretrain_grd':'/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/grd_test_chart_llava.jsonl',
        #         'chart_pretrain_rcg':'/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/rcg_test_chart_llava.jsonl',
        #     #decode
        #         'chart_decode_json':'/group/40034/yinanzhou/chart_grounding_consruction/ChartQA/reconsructor/dataset_constructor/generated_data/test_chart_to_json_with_random_mask_pad_llava.jsonl',
        #     #sft need to add prompt
        #         'chart_qa':'/group/40034/yinanzhou/chart_grounding_consruction/ChartQA/reconsructor/4o_examples_self_generated_test/chartqa_test_llava.jsonl',
        # #Crello POSTER
        #     #pretrain
        #         'poster_pretrain_grd':'/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/grd_test_llava.jsonl',
        #         'poster_pretrain_rcg':'/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/rcg_test_llava.jsonl',
        #     #decode
        #         'poster_decode':'/group/40034/yinanzhou/crello/pretrain_crello_tolist_new_test_llava.jsonl',
        #     #sft need to add prompt
        #         'poster_qa_open':'/group/40034/yinanzhou/crello/crello_qa_new_test_llava.jsonl',
        #         'poster_qa':'/group/40034/yinanzhou/crello/crello_informat_constructor/crello_in_format_qa_test_llava.jsonl'"
    # }
    image_dir = "/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0"
    # test_file_path = os.path.join(image_dir, 'test', dataset+'_llavanext_test.jsonl')
    # test_file_path = "/group/40033/public_datasets/DocStruct4M/DocDownstream-1.0/test/DOGE_test_new_pq_llava.jsonl"
    # "0,1,2,3"
    model_name = "llava_qwen"
    device = f"npu:{rank}"
    device_map = f"npu:{rank}"

    total_latency   = torch.zeros(1, device=device)   # 累积时间 (s)
    total_tokens    = torch.zeros(1, device=device)   # 累积生成 token 数
    peak_mem_mb     = torch.zeros(1, device=device)   # 各 rank 观察到的峰值显存 (MB)


    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="eager")  # Add any other thing you want to pass in llava_model_args
    model.tokenizer = tokenizer
    model.eval()

    dataset = CustomDataset(image_dir, test_file_path, image_processor, model.config)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)

    print("test_length:",len(dataset))
    template ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}
    # template_t2bbox ={"image": [], "messages": [{"role": "user", "content": "<|image|>"}, {"role": "assistant", "content": ""}], "task_name": "t2bbox_sft", "dataset_name": "ChartQA_PCG", "model_answer": "", "gt_answer": ""}

    output = {}
    rst_list = []
    for batch_items, batch_image, batch_image_tensor in tqdm.tqdm(dataloader):
        for item, image, image_tensor in zip(batch_items, batch_image, batch_image_tensor):
            

            # if "CHART" not in item["task_name"] or "long" not in item["task_name"]:
            #     continue

            # print("CHART" not in item["task_name"],"long" not in item["task_name"])
            # print(item["image"])
            # print(item["task_name"])
            qid = item["image"].split("/")[-1].split(".")[0]
            # if("Chart") not in item["image"]:
            #     continue
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            prompt =  item["conversations"][0]["value"].replace("<image>",' ')
            gt_answer = item["conversations"][1]["value"]
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size]

            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats(device)

            start_evt = torch.npu.Event(enable_timing=True)
            end_evt   = torch.npu.Event(enable_timing=True)
            start_evt.record()


            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=1024,
            )


            end_evt.record()
            torch.npu.synchronize()

            # -------------- 单条样本统计 ----------------
            latency = start_evt.elapsed_time(end_evt) / 1000.0         # ms -> s
            gen_tokens = cont.shape[1] - input_ids.shape[1]

            total_latency += latency
            total_tokens  += gen_tokens

            cur_peak_mb = torch.npu.max_memory_allocated(device) / 1024 / 1024
            peak_mem_mb = torch.max(peak_mem_mb, torch.tensor(cur_peak_mb, device=device))

            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            output[qid] = {"pred": text_outputs[0], "gt": gt_answer, "question": prompt}
            print(f"qid: {qid}\npred: {text_outputs[0]}\ngt: {gt_answer}")

            template_new = copy.deepcopy(template)
            template_new["image"].append(item["image"])
            template_new["task_name"] = item["task_name"]
            template_new["messages"][0]["content"]+=prompt
            template_new["messages"][1]["content"]+=gt_answer
            template_new["model_answer"]=text_outputs[0]
            template_new["gt_answer"]=gt_answer
            if "necessary bbox" in item.keys():
                template_new.update({"necessary bbox":item["necessary bbox"]})
            rst_list.append(template_new)            

    total_latency = all_reduce_tensor(total_latency, op=dist.ReduceOp.SUM)
    total_tokens  = all_reduce_tensor(total_tokens,  op=dist.ReduceOp.SUM)
    peak_mem_mb   = all_reduce_tensor(peak_mem_mb,   op=dist.ReduceOp.MAX)  # 取最大峰值


    output_dir = "/".join(args.output_file.split("/")[:-1])
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file.replace(".jsonl", f"_{rank}.jsonl"), "w") as f:
        for line in rst_list:
            json.dump(line, f)
            f.write('\n')
    
    if rank == 0:
        final_result = []
        for rank_idx in range(world_size):
            while not os.path.exists(args.output_file.replace(".jsonl", f"_{rank_idx}.jsonl")):
                time.sleep(10)
                print(f"/group/40079/yinanzhou/LLaVA-NeXT/{output_dir}/doge_result_{rank_idx}.jsonl")
            
            with open(args.output_file.replace(".jsonl", f"_{rank_idx}.jsonl"), "r") as f:
                final_result += [json.loads(line) for line in f]
            
            os.remove(args.output_file.replace(".jsonl", f"_{rank_idx}.jsonl"))
        with open(args.output_file, "w") as f:
            for line in final_result:
                json.dump(line, f)
                f.write('\n')
        avg_latency  = (total_latency / len(dataset)).item()
        throughput   = (total_tokens  / total_latency).item()
        avg_peak_mem = peak_mem_mb.item()
        flops_token  = 2 * sum(p.numel() for p in model.parameters()) / 1e9  # GFLOPs/token

        print(f"\n================  BENCHMARK  =================")
        print(f"Average Latency (s) : {avg_latency:.4f}")
        print(f"Avg. Peak memory(MB): {avg_peak_mem:.2f}")
        print(f"Throughput (tok/s)  : {throughput:.2f}")
        print(f"FLOPs / token (≈)   : {flops_token:.2f}  GFLOPs")
        print("==============================================")


if __name__ == '__main__':
    main()