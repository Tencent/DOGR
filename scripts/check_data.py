import os
import yaml
import json
import tqdm
import copy
import glob
import random


def is_valid_convs(convs):
    conv_cat = ""
    for i in range(1, len(convs), 2):
        conv_cat += convs[i]["value"]
    if "<image>" in conv_cat:
        import pdb; pdb.set_trace()
        return False
    else:
        return True

def update_json(filename, data):
    bad_data = []
    new_data = []
    for item in data:
        if not is_valid_convs(item["conversations"]):
            bad_data.append(copy.deepcopy(item))
            continue
        if 'image' in item and item['image'] is not None:
            if "<image>" in item["conversations"][0]["value"]:
                new_data.append(item)
            elif "<|image|>" in item["conversations"][0]["value"]:
                bad_data.append(copy.deepcopy(item))
                item["conversations"][0]["value"] = item["conversations"][0]["value"].replace("<|image|>", "<image>")
                new_data.append(item)
            else:
                bad_data.append(copy.deepcopy(item))
                item["conversations"][0]["value"] = "<image>\n" + item["conversations"][0]["value"]
                new_data.append(item)
        else:
            new_data.append(item)
    if len(bad_data) == 0:
        return data
    else:
        import pdb; pdb.set_trace()
        if filename.endswith("jsonl"):
            print("update:", filename.replace(".jsonl", "_filter.jsonl"))
            with open(filename.replace(".jsonl", "_filter.jsonl"), "w") as f:
                for item in new_data:
                    f.write(json.dumps(item) + '\n')
        elif filename.endswith("json"):
            print("update:", filename.replace(".json", "_filter.json"))
            with open(filename.replace(".json", "_filter.json"), "w") as f:
                json.dump(new_data, f)
        return new_data


with open("scripts/data/stage2.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)
datasets = yaml_data.get("datasets")

list_data_dict = []

dataset_paths = [dataset.get("json_path") for dataset in datasets]

for dataset in datasets:
    json_path = dataset.get("json_path")
    if not os.path.exists(json_path):
        print(json_path)
        import pdb; pdb.set_trace()
for dataset in datasets:
    json_path = dataset.get("json_path")
    sampling_strategy = dataset.get("sampling_strategy", "all")
    sampling_number = None

    print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

    if json_path.endswith(".jsonl"):
        cur_data_dict = []
        with open(json_path, "r") as json_file:
            for line in json_file:
                cur_data_dict.append(json.loads(line.strip()))
    elif json_path.endswith(".json"):
        with open(json_path, "r") as json_file:
            cur_data_dict = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {json_path}")

    cur_data_dict = update_json(json_path, cur_data_dict)
    if ":" in sampling_strategy:
        sampling_strategy, sampling_number = sampling_strategy.split(":")
        if "%" in sampling_number:
            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
        else:
            sampling_number = int(sampling_number)

    # Apply the sampling strategy
    if sampling_strategy == "first" and sampling_number is not None:
        cur_data_dict = cur_data_dict[:sampling_number]
    elif sampling_strategy == "end" and sampling_number is not None:
        cur_data_dict = cur_data_dict[-sampling_number:]
    elif sampling_strategy == "random" and sampling_number is not None:
        random.shuffle(cur_data_dict)
        cur_data_dict = cur_data_dict[:sampling_number]
    
    list_data_dict.extend(cur_data_dict)

image_dir = "./playground/data"
dataset_dirs = []
all_img_paths = []
for item in list_data_dict:
    if 'image' in item and item['image'] is not None:
        img_path = os.path.join(image_dir, item['image'])
        img_dir = "/".join(img_path.split("/")[:-1])
        dataset_dirs.append(img_dir)
        all_img_paths.append(img_path)

glob_all_img_paths = []
for img_dir in list(set(dataset_dirs)):
    glob_all_img_paths.extend(glob.glob(os.path.join(img_dir, '*')))

import pdb; pdb.set_trace()
invalid_imgs = set(all_img_paths).difference(set(glob_all_img_paths))
for item in tqdm.tqdm(list_data_dict):
    if 'image' in item and item['image'] is not None:
        if not os.path.exists(os.path.join(image_dir, item['image'])):
            print(item['image'])
            import pdb; pdb.set_trace()
import pdb; pdb.set_trace()