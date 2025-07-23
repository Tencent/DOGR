import re
import json
import argparse

from evaluation.utils import compute_grd_f1, compute_text_sim, get_raw_string
from evaluation.doge_evaluator import doc_evaluate
import ast


iou_threshold=0.5
def extract_number(string):
    match = re.search(r'\d+(\.\d+)?', string)
    if match:
        return match.group()
    return string

def preprocess_lower_case(texts):
    texts = [" ".join(t.split()) for t in texts]
    return [t.strip(".").strip().lower() for t in texts]

def preprocess_short_answer(texts, types=None,extract=False):
    for i,t in enumerate(texts):
        if t==None:
            texts[i]=''
    texts = [" ".join(t.split()) for t in texts]
    if extract:
        texts = [t.split("Answer:")[-1] for t in texts]
    # remove <OCR>
    texts = [get_raw_string(t) for t in texts]
    # replace \n with " "
    texts = [t.replace("\n", " ") for t in texts]


    # replace multiple space with single space
    texts = [re.sub(r'\s+', ' ', t) for t in texts]

    texts = [t.strip(".").strip().lower() for t in texts]
    if types is not None:
        for i,t  in enumerate(texts):
            if types[i]=="math":
                texts[i] = extract_number(t)

    return texts

# def evalute_opened_qa(preds, gts, metric_names=["BLEU1", "BLEU4", "RougeL", "Meteor"]):
def evalute_opened_qa(preds, gts, metric_names=["BLEU1", "BLEU4", "RougeL"],qs=None,iou_threshold=iou_threshold):

    pred_texts = preprocess_short_answer(preds)
    gt_texts = [preprocess_short_answer(gt) for gt in gts]
    metric2score = {}
    for metric_name in metric_names:

        
        metric2score[metric_name] = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)

    F1all, F1loc,F1rcg = compute_grd_f1(preds, gts,iou_threshold=iou_threshold,qs=qs)
    F1all_detail, F1loc_detail,F1rcg_detail = compute_grd_f1(preds, gts,iou_threshold=iou_threshold, verbose=True,qs=qs)
    metric2score["F1all"] = (F1all, F1all_detail)
    metric2score["F1loc"] = (F1loc, F1loc_detail)
    metric2score["F1rcg"] = (F1rcg, F1rcg_detail)


    return metric2score



def evaluate_short_answer(preds, gts,types, metric_names=["RelaxedAccuracy"]):
    preds = preprocess_short_answer(preds,types)
    gts = [preprocess_short_answer(gt) for gt in gts]
    # gts = preprocess_short_answer(gts,types)

    metric2score = {}
    for metric_name in metric_names:
        score, scores = doc_evaluate(metric=metric_name, targets=gts, predictions=preds)
        metric2score[metric_name] = (score, scores)
    return metric2score
    
def evaluate_short_grounding_answer(preds, gts, metric_names=["RelaxedAccuracy"],iou_threshold=iou_threshold):
    metric2score = {}
    for metric_name in metric_names:
        pred_texts = preprocess_short_answer(preds)
        gt_texts = [preprocess_short_answer(gt) for gt in gts]
        metric2score[metric_name] = doc_evaluate(metric=metric_name, targets=gt_texts, predictions=pred_texts)
    
    # grounding accuracy
    F1all, F1loc,F1rcg = compute_grd_f1(preds, gts,iou_threshold=iou_threshold)
    F1all_detail, F1loc_detail,F1rcg_detail = compute_grd_f1(preds, gts,iou_threshold=iou_threshold, verbose=True)
    metric2score["F1all"] = (F1all, F1all_detail)
    metric2score["F1loc"] = (F1loc, F1loc_detail)
    metric2score["F1rcg"] = (F1rcg, F1rcg_detail)

    return metric2score
    
def evaluate_long_reasoning_qa(preds, gts,types, box_list=None, metric_names=["ExactAccuracy"],qs=None,iou_threshold=iou_threshold):

    metric2score = {}
    for metric_name in metric_names:
        pred_texts = preprocess_short_answer(preds, extract=True)
        gt_texts = [preprocess_short_answer(gt, extract=True) for gt in gts]
        metric2score[metric_name] = doc_evaluate(metric=metric_name, targets=gt_texts, predictions=pred_texts)
    
    preds = [t.split("Answer:")[0] for t in preds]
    gts = [[t.split("Answer:")[0] for t in gt] for gt in gts]
    F1all, F1loc,F1rcg = compute_grd_f1(preds, gts,iou_threshold=iou_threshold, box_list=box_list,qs=qs)
    F1all_detail, F1loc_detail,F1rcg_detail = compute_grd_f1(preds, gts,iou_threshold=iou_threshold,box_list=box_list, verbose=True,qs=qs)
    metric2score["F1all"] = (F1all, F1all_detail)
    metric2score["F1loc"] = (F1loc, F1loc_detail)
    metric2score["F1rcg"] = (F1rcg, F1rcg_detail)

    return metric2score

def parse_predict_results(pred_results):

    data_dict = {
        "POSTER": {"Rt": {"preds": [], "gts": []}, "Ga": {"preds": [], "gts": []},"GRa": {"preds": [], "gts": []}, "Gr": {"preds": [], "gts": [], "bboxes": []},"GRr": {"preds": [], "gts": [], "bboxes": [],'qs':[]}, "Go": {"preds": [], "gts": []},"GRo": {"preds": [], "gts": [],'qs':[]}}, 
        "CHART": {"Rt": {"preds": [], "gts": [],"types":[]}, "Ga": {"preds": [], "gts": []},"GRa": {"preds": [], "gts": []}, "Gr": {"preds": [], "gts": [], "bboxes": [],"types":[]},"GRr": {"preds": [], "gts": [], "bboxes": [],"types":[],'qs':[]}, "Go": {"preds": [], "gts": [],"types":[]},"GRo": {"preds": [], "gts": [],'qs':[]}}, 
        "PDF": {"Rt": {"preds": [], "gts": []}, "Ga": {"preds": [], "gts": []},"GRa": {"preds": [], "gts": []}, "Gr": {"preds": [], "gts": []},"GRr": {"preds": [], "gts": [],'qs':[]}, "Go": {"preds": [], "gts": []},"GRo": {"preds": [], "gts": [],'qs':[]}}, 
        }
    

    for item in pred_results:
        # task_name = item["task_name"]
        data_name = item["task_name"].split("_")[1]
        task_name = item["task_name"].split("_")[2]

        # print(item)
        if "POSTER" in data_name:
            if "Ga" in task_name or "GRa" in task_name or "Rt" in task_name or "Go" in task_name or "GRo" in task_name:
                data_dict[data_name][task_name]["preds"].append(item["model_answer"])
                data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
            elif "Gr" in task_name or "GRr" in task_name:
                data_dict[data_name][task_name]["preds"].append(item["model_answer"])
                data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
                data_dict[data_name][task_name]["bboxes"].append(item["necessary bbox"])
            if "GRr" in task_name or "GRo" in task_name:
                data_dict[data_name][task_name]["qs"].append(item["messages"][0]["content"])

        elif "CHART" in data_name:
            if "Ga" in task_name or "GRa" in task_name or "Go" in task_name or "GRo" in task_name:
                data_dict[data_name][task_name]["preds"].append(item["model_answer"])
                data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
            elif "Rt" in task_name :
                data_dict[data_name][task_name]["preds"].append(item["model_answer"])
                data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
                # data_dict[data_name][task_name]["types"].append([item["task_type"]])
                data_dict[data_name][task_name]["types"].append(["reasoning"])
            elif "Gr" in task_name or "GRr" in task_name:
                data_dict[data_name][task_name]["preds"].append(item["model_answer"])
                data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
                data_dict[data_name][task_name]["bboxes"].append(item["necessary bbox"])
                # data_dict[data_name][task_name]["types"].append([item["task_type"]])
                data_dict[data_name][task_name]["types"].append(["reasoning"])
            if "GRr" in task_name or "GRo" in task_name:
                try:
                    data_dict[data_name][task_name]["qs"].append(item["messages"][0]["content"])
                except:
                    data_dict[data_name][task_name]["qs"].append(item["conversations"][0]["value"])

        elif "PDF" in data_name:
            data_dict[data_name][task_name]["preds"].append(item["model_answer"])
            data_dict[data_name][task_name]["gts"].append([item["gt_answer"]])
            if "GRr" in task_name or "GRo" in task_name:
                try:
                    data_dict[data_name][task_name]["qs"].append(item["messages"][0]["content"])
                except:
                    data_dict[data_name][task_name]["qs"].append(item["conversations"][0]["value"])

            
    return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="path of model prediction results", required=True)
    parser.add_argument("--iou_threshold", type=float, help="path of model prediction results", required=True)


    args = parser.parse_args()
    iou_threshold = args.iou_threshold
    with open(args.pred_file, "r") as f:
        try:
            pred_results = [json.loads(line) for line in f]
        except:
            pred_results = json.load(f) 
    
    data_dict = parse_predict_results(pred_results)

    import re
    pattern = r"<ocr>.*?</bbox>"
    
    
    for i,item in enumerate(data_dict["CHART"]["Gr"]["bboxes"]):
        if isinstance(item,str):
            data_dict["CHART"]["Gr"]["bboxes"][i] = re.findall(pattern, item)
            # import pdb; pdb.set_trace()

    for i,item in enumerate(data_dict["CHART"]["GRr"]["bboxes"]):
        if isinstance(item,str):
            data_dict["CHART"]["GRr"]["bboxes"][i] = re.findall(pattern, item)
    for i,item in enumerate(data_dict["POSTER"]["Gr"]["bboxes"]):
        if isinstance(item,str):
            data_dict["POSTER"]["Gr"]["bboxes"][i] = re.findall(pattern, item)

    score_dict = {
        "POSTER": {"Rt": None, "Ga": None, "Gr": None, "Go": None,"GRo": None}, #"GRa": None  "GRr": None,
        "CHART": {"Rt": None, "Ga": None,"GRa": None, "Gr": None,"GRr": None, "Go": None,"GRo": None}, 
        "PDF": {"Rt": None, "Ga": None,"GRa": None, "Gr": None,"GRr": None, "Go": None,"GRo": None}, 
        }
    final_dict = {
        "avg":{"Rt": {}, "Ga": {},"GRa": {}, "Gr": {},"GRr": {}, "Go": {},"GRo": {}}
    }
    

    for data_type, type_items in data_dict.items():
       
        for task_type, task_items in type_items.items():
            print(task_type,len(task_items["gts"]))

            if task_type == "Rt":
                score_dict[data_type][task_type] = evaluate_short_answer(task_items["preds"], task_items["gts"],types=task_items["types"] if "types" in task_items else None, metric_names=["RelaxedAccuracy"])
            elif task_type == "Gr" or task_type == "GRr":
                score_dict[data_type][task_type] = evaluate_long_reasoning_qa(task_items["preds"], task_items["gts"],types=task_items["types"] if "types" in task_items else None, metric_names=["RelaxedAccuracy"], box_list=task_items["bboxes"] if "bboxes" in task_items else None,qs=task_items["qs"] if "qs" in task_items else None,iou_threshold=iou_threshold)
            elif task_type == "Ga" or task_type == "GRa":
                score_dict[data_type][task_type] = evaluate_short_grounding_answer(task_items["preds"], task_items["gts"], metric_names=["RelaxedAccuracy"],iou_threshold=iou_threshold)
            else:
                score_dict[data_type][task_type] = evalute_opened_qa(task_items["preds"], task_items["gts"],qs=task_items["qs"] if "qs" in task_items else None,iou_threshold=iou_threshold)
    
    final_results = []
    total_score = {}
    for data_type, type_items in score_dict.items():
        print("DATA TYPE:", data_type)
        total_score[data_type] = {}
        for task_type, task_score in type_items.items():
            total_score[data_type][task_type] = {}
            print("\tTASK TYPE:", task_type)
            for metric_name, scores in task_score.items():
                total_score[data_type][task_type][metric_name] = scores[0]
                print(f"\t\t{metric_name}: {scores[0]}")
                for idx, item_score in enumerate(scores[1]):
                    final_results.append({"DATA TYPE": data_type, "TASK TYPE": task_type, "METRIC": metric_name, "PRED": data_dict[data_type][task_type]["preds"][idx], "GT": data_dict[data_type][task_type]["gts"][idx], "score": item_score})
    for data_type, type_items in final_dict.items():
        for task_type, task_items in type_items.items():
            for item in score_dict["POSTER"][task_type]:
                if score_dict["POSTER"][task_type][item][0]!=0 and score_dict["POSTER"][task_type][item][0]!=None:
                    print(score_dict["POSTER"][task_type][item][0])
                    print(score_dict["CHART"][task_type][item][0])
                    print(score_dict["PDF"][task_type][item][0])
                    
                    final_dict[data_type][task_type][item] = (score_dict["POSTER"][task_type][item][0]+score_dict["CHART"][task_type][item][0]+score_dict["PDF"][task_type][item][0])/3
                else:
                    final_dict[data_type][task_type][item] = (score_dict["CHART"][task_type][item][0]+score_dict["PDF"][task_type][item][0])/2
    print(final_dict)
    final_results = [total_score] + final_results

    with open(args.pred_file.split(".")[0] + "_eval.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    

    
    