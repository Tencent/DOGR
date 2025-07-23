import re
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import ast
# Ensure the necessary NLTK data files are downloaded
nltk.download('wordnet')

def calculate_bleu_scores(reference, candidate):
    """Calculate BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores."""
    reference = [reference.split()]  # BLEU expects a list of references
    candidate = candidate.split()
    smoothing_function = SmoothingFunction().method1

    # Calculate BLEU scores for different n-grams
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
    # bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    # bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

    return bleu1, bleu4 # bleu2, bleu3, 

def calculate_rouge_l(reference, candidate):
    """Calculate ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def calculate_meteor(reference, candidate):
    """Calculate METEOR score."""
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    return meteor_score([reference_tokens], candidate_tokens)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : list or tuple
        The first bounding box [x1, y1, x2, y2].
    box2 : list or tuple
        The second bounding box [x1, y1, x2, y2].

    Returns
    -------
    float
        The IoU of the two bounding boxes.
    """
    # Extract the coordinates of the bounding boxes
    try:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate the coordinates of the intersection rectangle
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate the area of intersection rectangle
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate the area of both bounding boxes
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate the union area
        union_area = box1_area + box2_area - inter_area

        # Calculate the IoU
        iou = inter_area / union_area if union_area != 0 else 0
        
        return iou
    except:
        return 0
def calculate_iou_qwen(box1, box2,wh):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    box1 : list or tuple
        The first bounding box [x1, y1, x2, y2].
    box2 : list or tuple
        The second bounding box [x1, y1, x2, y2].

    Returns
    -------
    float
        The IoU of the two bounding boxes.
    """

    try:
        # Extract the coordinates of the bounding boxes
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        width = wh[0]
        height = wh[1]
        if x1_min>1 and y1_min>1 and x1_max>1 and y1_max>1:
            x1_min = (x1_min/width)
            y1_min = (y1_min/height)
            x1_max = (x1_max/width)
            y1_max = (y1_max/height)
        else:
            x1_min = (x1_min*999/width)
            y1_min = (y1_min*999/height)
            x1_max = (x1_max*999/width)
            y1_max = (y1_max*999/height)


        # Calculate the coordinates of the intersection rectangle
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Calculate the area of intersection rectangle
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Calculate the area of both bounding boxes
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Calculate the union area
        union_area = box1_area + box2_area - inter_area

        # Calculate the IoU
        iou = inter_area / union_area if union_area != 0 else 0
        
        # import pdb;pdb.set_trace()
        return iou
    except:
        return 0

def extract_ocr_bbox(input_string):
    """
    Extract tuples of (text, [x1, y1, x2, y2]) from the input string.

    Parameters
    ----------
    input_string : str
        The input string containing <ocr> and <bbox> tags.

    Returns
    -------
    list of tuples
        A list of tuples, each containing the text and a list of bounding box coordinates.
    """
    if "```json" in input_string:
        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, input_string, re.DOTALL)
        result = []
        for match in matches:
            # import pdb;pdb.set_trace()
            try:
                json_output = ast.literal_eval(match)
            except Exception as e:
                try:
                    end_idx = match.rfind('"}') + len('"}')
                    truncated_text = match[:end_idx] + "]"
                    json_output = ast.literal_eval(truncated_text)
                except:
                    # # print(match)
                    # import pdb;pdb.set_trace()
                    continue

                # continue
            for i, item in enumerate(json_output):
                try:
                    text = item["text"]
                    bbox = item["bbox_2d"]
                    result.append((text, bbox))
                except:
                    continue
        return result



    # Define the regular expression pattern
    pattern = r'<ocr>(.*?)</ocr><bbox>(.*?)</bbox>'

    # Find all matches in the input string
    matches = re.findall(pattern, input_string)

    # Process each match to extract the text and bounding box coordinates
    result = []
    for text, bbox in matches:
        # Split the bbox string by comma and strip spaces, then convert to float
        try:
            coordinates = [max(min(float(coord.strip()) / 999, 1.0), 0.0) for coord in bbox.split(',')]
        except:
            coordinates = [0.] * 4
        if len(coordinates) < 4:
            coordinates = [0.] * 4
        # Append the tuple to the result list
        result.append((text, coordinates))

    return result


def extract_bbox(input_string):
    
    # Define the regular expression pattern
    pattern = r'<bbox>(.*?)</bbox>'

    # Find all matches in the input string
    matches = re.findall(pattern, input_string)

    # Process each match to extract the text and bounding box coordinates
    result = []
    for  bbox in matches:
        # Split the bbox string by comma and strip spaces, then convert to float
        try:
            coordinates = [max(min(float(coord.strip()) / 999, 1.0), 0.0) for coord in bbox.split(',')]
        except:
            coordinates = [0.] * 4
        if len(coordinates) < 4:
            coordinates = [0.] * 4
        # Append the tuple to the result list
        result.append(coordinates)

    return result

def get_raw_string(input_string):
    # 替换<ocr>xxx</ocr>为xxx
    result = input_string
    if '<ocr>' in input_string:
        result = re.sub(r'<ocr>(.*?)</ocr>', r'\1', input_string, flags=re.DOTALL)
        # 删除<bbox>xxx</bbox>
        result = re.sub(r'<bbox>.*?</bbox>', '', result)
    elif "```json" in input_string:\
        result = re.sub(r'```json.*?```', '', input_string, flags=re.DOTALL)
    if "Answer:" in input_string:
        result = input_string.split("Answer:")[0]
    
    return result

def preprocess_short_answer(text):
    text = " ".join(text.split(" "))
    # replace \n with " "
    text = text.replace("\n", "")
    # replace multiple space with single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(".").strip().lower()
    return text

def compute_grd_f1(preds, gts, iou_threshold=0.5, verbose=False, box_list=None,qs=None):
    """
    preds: list of prediction texts
    gts: list of ground truth texts
    verbose: return score of all samples if True, return mean score if False
    """
    all_f1all = []
    all_f1loc = []
    all_f1rcg = []
    if qs!=None and len(qs)==len(preds):
        for idx, (pred, gt,q) in enumerate(zip(preds, gts,qs)):
            # 提取ocr word及对应的bbox
            # 返回格式: 
            bbox_in_q = extract_bbox(q)

            pred_ocr_words = extract_ocr_bbox(pred)
            if isinstance(gt, list):
                gt = gt[0]
            gt_ocr_words = extract_ocr_bbox(gt)
            
            for text,bbox in gt_ocr_words[:]:
                if bbox in bbox_in_q:
                    gt_ocr_words.remove((text,bbox))
            for text,bbox in pred_ocr_words[:]:
                if bbox in bbox_in_q:
                    pred_ocr_words.remove((text,bbox))        

                
            if box_list is not None:
                if len(preds) != len(box_list):
                    import pdb; pdb.set_trace()
                gt_ocr_words = extract_ocr_bbox(",".join(box_list[idx]))
                # import pdb; pdb.set_trace()

                # print(box_list[idx])
            
            total_gen_words = len(pred_ocr_words)
            total_gt_words = len(gt_ocr_words)

            correct_text_in_pred_count = 0
            correct_text_in_gt_count = 0
            correct_text_accurate_bbox_count = 0

            correct_bbox_in_pred_count = 0
            correct_bbox_in_gt_count = 0
            correct_bbox_accurate_text_count = 0


            gt_dict = {preprocess_short_answer(text): bbox for text, bbox in gt_ocr_words}

            # Check each prediction
            for pred_text, pred_bbox in pred_ocr_words:
                pred_text = preprocess_short_answer(pred_text)
                if pred_text in gt_dict:
                    correct_text_in_pred_count += 1
                    gt_bbox = gt_dict[pred_text]
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    if iou > iou_threshold:
                        correct_text_accurate_bbox_count += 1

            for pred_text, pred_bbox in pred_ocr_words:
                for _,gt_bbox in gt_ocr_words:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    if iou > iou_threshold:
                        correct_bbox_in_pred_count += 1
                        pred_text = preprocess_short_answer(pred_text)
                        if pred_text in gt_dict:
                            correct_bbox_accurate_text_count+=1


            # Check each ground truth
            for gt_text, _ in gt_ocr_words:
                gt_text = preprocess_short_answer(gt_text)
                if gt_text in [preprocess_short_answer(pred_text) for pred_text, _ in pred_ocr_words]:
                    correct_text_in_gt_count += 1
            for _, gt_bbox in gt_ocr_words:
                for _, pred_bbox in pred_ocr_words:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                    if iou > iou_threshold:
                        correct_bbox_in_gt_count+=1

            Pall = correct_text_accurate_bbox_count / total_gen_words if total_gen_words > 0 else 0
            Rall = correct_text_accurate_bbox_count / total_gt_words if total_gt_words > 0 else 0
            F1all = 2 * Pall * Rall / (Pall + Rall) if (Pall + Rall) > 0 else 0
            # import pdb;pdb.set_trace()

            Ploc = correct_text_accurate_bbox_count / correct_text_in_pred_count if correct_text_in_pred_count > 0 else 0
            Rloc = correct_text_accurate_bbox_count / correct_text_in_gt_count if correct_text_in_gt_count > 0 else 0
            F1loc = 2 * Ploc * Rloc / (Ploc + Rloc) if (Ploc + Rloc) > 0 else 0

            Prcg = correct_bbox_accurate_text_count / correct_bbox_in_pred_count if correct_bbox_in_pred_count > 0 else 0
            Rrcg = correct_bbox_accurate_text_count / correct_bbox_in_gt_count if correct_bbox_in_gt_count > 0 else 0
            F1rcg = 2 * Prcg * Rrcg / (Prcg + Rloc) if (Prcg + Rrcg) > 0 else 0
            if total_gt_words==0:
                continue

            all_f1all.append(F1all)
            all_f1loc.append(F1loc)
            all_f1rcg.append(F1rcg)
    else:
        for idx, (pred, gt) in enumerate(zip(preds, gts)):
            # 提取ocr word及对应的bbox
            # 返回格式: 
            pred_ocr_words = extract_ocr_bbox(pred)
            if isinstance(gt, list):
                gt = gt[0]
            gt_ocr_words = extract_ocr_bbox(gt)
            if box_list is not None:
                if len(preds) != len(box_list):
                    import pdb; pdb.set_trace()
                    
                # if isinstance(box_list[idx],str):
                #     try:
                #         box_list[idx] = ast.literal_eval(box_list[idx].replace("/",'//'))
                #     except:
                #         import pdb; pdb.set_trace()


                gt_ocr_words = extract_ocr_bbox(",".join(box_list[idx]))
                # import pdb; pdb.set_trace()

                # print(box_list[idx])

            total_gen_words = len(pred_ocr_words)
            total_gt_words = len(gt_ocr_words)

            correct_text_in_pred_count = 0
            correct_text_in_gt_count = 0
            correct_text_accurate_bbox_count = 0

            correct_bbox_in_pred_count = 0
            correct_bbox_in_gt_count = 0
            correct_bbox_accurate_text_count = 0


            gt_dict = {preprocess_short_answer(text): bbox for text, bbox in gt_ocr_words}

            # Check each prediction
            for pred_text, pred_bbox in pred_ocr_words:
                pred_text = preprocess_short_answer(pred_text)
                if pred_text in gt_dict:
                    correct_text_in_pred_count += 1
                    gt_bbox = gt_dict[pred_text]
                    try:
                        iou = calculate_iou(pred_bbox, gt_bbox)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_text_accurate_bbox_count += 1

            for pred_text, pred_bbox in pred_ocr_words:
                for _,gt_bbox in gt_ocr_words:
                    try:
                        iou = calculate_iou(pred_bbox, gt_bbox)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_bbox_in_pred_count += 1
                        pred_text = preprocess_short_answer(pred_text)
                        if pred_text in gt_dict:
                            correct_bbox_accurate_text_count+=1


            # Check each ground truth
            for gt_text, _ in gt_ocr_words:
                gt_text = preprocess_short_answer(gt_text)
                if gt_text in [preprocess_short_answer(pred_text) for pred_text, _ in pred_ocr_words]:
                    correct_text_in_gt_count += 1
            for _, gt_bbox in gt_ocr_words:
                for _, pred_bbox in pred_ocr_words:
                    try:
                        iou = calculate_iou(pred_bbox, gt_bbox)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_bbox_in_gt_count+=1

            Pall = correct_text_accurate_bbox_count / total_gen_words if total_gen_words > 0 else 0
            Rall = correct_text_accurate_bbox_count / total_gt_words if total_gt_words > 0 else 0
            F1all = 2 * Pall * Rall / (Pall + Rall) if (Pall + Rall) > 0 else 0


            Ploc = correct_text_accurate_bbox_count / correct_text_in_pred_count if correct_text_in_pred_count > 0 else 0
            Rloc = correct_text_accurate_bbox_count / correct_text_in_gt_count if correct_text_in_gt_count > 0 else 0
            F1loc = 2 * Ploc * Rloc / (Ploc + Rloc) if (Ploc + Rloc) > 0 else 0

            Prcg = correct_bbox_accurate_text_count / correct_bbox_in_pred_count if correct_bbox_in_pred_count > 0 else 0
            Rrcg = correct_bbox_accurate_text_count / correct_bbox_in_gt_count if correct_bbox_in_gt_count > 0 else 0
            F1rcg = 2 * Prcg * Rrcg / (Prcg + Rloc) if (Prcg + Rrcg) > 0 else 0

            if total_gt_words==0:
                continue
            all_f1all.append(F1all)
            all_f1loc.append(F1loc)
            all_f1rcg.append(F1rcg)
    if verbose:
        return all_f1all, all_f1loc,all_f1rcg
    else:
        return sum(all_f1all) / len(all_f1all) if len(all_f1all)>0 else 0, sum(all_f1loc) / len(all_f1loc) if len(all_f1loc)>0 else 0,sum(all_f1rcg) / len(all_f1rcg) if len(all_f1rcg)>0 else 0



def compute_grd_f1_qwen(preds, gts,whs, iou_threshold=0.5, verbose=False, box_list=None,qs=None):
    """
    preds: list of prediction texts
    gts: list of ground truth texts
    verbose: return score of all samples if True, return mean score if False
    """
    all_f1all = []
    all_f1loc = []
    all_f1rcg = []

    if qs!=None and len(qs)==len(preds):
        for idx, (pred, gt,q,wh) in enumerate(zip(preds, gts,qs,whs)):
            # 提取ocr word及对应的bbox
            # 返回格式: 
            bbox_in_q = extract_bbox(q)

            pred_ocr_words = extract_ocr_bbox(pred)
            if isinstance(gt, list):
                gt = gt[0]
            gt_ocr_words = extract_ocr_bbox(gt)
            for text,bbox in gt_ocr_words[:]:
                if bbox in bbox_in_q:
                    gt_ocr_words.remove((text,bbox))
            for text,bbox in pred_ocr_words[:]:
                if bbox in bbox_in_q:
                    pred_ocr_words.remove((text,bbox))        

                
            if box_list is not None:
                if len(preds) != len(box_list):
                    import pdb; pdb.set_trace()
                gt_ocr_words = extract_ocr_bbox(",".join(box_list[idx]))
                # import pdb; pdb.set_trace()

                # print(box_list[idx])
           
            total_gen_words = len(pred_ocr_words)
            total_gt_words = len(gt_ocr_words)

            correct_text_in_pred_count = 0
            correct_text_in_gt_count = 0
            correct_text_accurate_bbox_count = 0

            correct_bbox_in_pred_count = 0
            correct_bbox_in_gt_count = 0
            correct_bbox_accurate_text_count = 0


            gt_dict = {preprocess_short_answer(text): bbox for text, bbox in gt_ocr_words}
            # if box_list is not None:
            #     import pdb;pdb.set_trace()

            # Check each prediction
            # import pdb;pdb.set_trace()
            for pred_text, pred_bbox in pred_ocr_words:
                pred_text = preprocess_short_answer(pred_text)
                if pred_text in gt_dict:
                    correct_text_in_pred_count += 1
                    gt_bbox = gt_dict[pred_text]
                    iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    if iou > iou_threshold:
                        correct_text_accurate_bbox_count += 1

            for pred_text, pred_bbox in pred_ocr_words:
                for _,gt_bbox in gt_ocr_words:
                    iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    if iou > iou_threshold:
                        correct_bbox_in_pred_count += 1
                        pred_text = preprocess_short_answer(pred_text)
                        if pred_text in gt_dict:
                            correct_bbox_accurate_text_count+=1


            # Check each ground truth
            for gt_text, _ in gt_ocr_words:
                gt_text = preprocess_short_answer(gt_text)
                if gt_text in [preprocess_short_answer(pred_text) for pred_text, _ in pred_ocr_words]:
                    correct_text_in_gt_count += 1
            for _, gt_bbox in gt_ocr_words:
                for _, pred_bbox in pred_ocr_words:
                    iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    if iou > iou_threshold:
                        correct_bbox_in_gt_count+=1

            Pall = correct_text_accurate_bbox_count / total_gen_words if total_gen_words > 0 else 0
            Rall = correct_text_accurate_bbox_count / total_gt_words if total_gt_words > 0 else 0
            F1all = 2 * Pall * Rall / (Pall + Rall) if (Pall + Rall) > 0 else 0


            Ploc = correct_text_accurate_bbox_count / correct_text_in_pred_count if correct_text_in_pred_count > 0 else 0
            Rloc = correct_text_accurate_bbox_count / correct_text_in_gt_count if correct_text_in_gt_count > 0 else 0
            F1loc = 2 * Ploc * Rloc / (Ploc + Rloc) if (Ploc + Rloc) > 0 else 0

            Prcg = correct_bbox_accurate_text_count / correct_bbox_in_pred_count if correct_bbox_in_pred_count > 0 else 0
            Rrcg = correct_bbox_accurate_text_count / correct_bbox_in_gt_count if correct_bbox_in_gt_count > 0 else 0
            F1rcg = 2 * Prcg * Rrcg / (Prcg + Rloc) if (Prcg + Rrcg) > 0 else 0
            if total_gt_words==0:
                continue
            all_f1all.append(F1all)
            all_f1loc.append(F1loc)
            all_f1rcg.append(F1rcg)
    else:
        for idx, (pred, gt,wh) in enumerate(zip(preds, gts,whs)):
            # 提取ocr word及对应的bbox
            # 返回格式: 
            pred_ocr_words = extract_ocr_bbox(pred)
            if isinstance(gt, list):
                gt = gt[0]
            gt_ocr_words = extract_ocr_bbox(gt)
            if box_list is not None:
                if len(preds) != len(box_list):
                    import pdb; pdb.set_trace()

                gt_ocr_words = extract_ocr_bbox(",".join(box_list[idx]))

            total_gen_words = len(pred_ocr_words)
            total_gt_words = len(gt_ocr_words)

            correct_text_in_pred_count = 0
            correct_text_in_gt_count = 0
            correct_text_accurate_bbox_count = 0

            correct_bbox_in_pred_count = 0
            correct_bbox_in_gt_count = 0
            correct_bbox_accurate_text_count = 0


            gt_dict = {preprocess_short_answer(text): bbox for text, bbox in gt_ocr_words}
            # Check each prediction
            for pred_text, pred_bbox in pred_ocr_words:
                pred_text = preprocess_short_answer(pred_text)

                if pred_text in gt_dict:
                    correct_text_in_pred_count += 1
                    gt_bbox = gt_dict[pred_text]
                    try:
                        iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_text_accurate_bbox_count += 1

            for pred_text, pred_bbox in pred_ocr_words:
                for _,gt_bbox in gt_ocr_words:
                    try:
                        iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_bbox_in_pred_count += 1
                        pred_text = preprocess_short_answer(pred_text)
                        if pred_text in gt_dict:
                            correct_bbox_accurate_text_count+=1


            # Check each ground truth
            for gt_text, _ in gt_ocr_words:
                gt_text = preprocess_short_answer(gt_text)
                if gt_text in [preprocess_short_answer(pred_text) for pred_text, _ in pred_ocr_words]:
                    correct_text_in_gt_count += 1
            for _, gt_bbox in gt_ocr_words:
                for _, pred_bbox in pred_ocr_words:
                    try:
                        iou = calculate_iou_qwen(pred_bbox, gt_bbox,wh)
                    except:
                        iou=0
                    if iou > iou_threshold:
                        correct_bbox_in_gt_count+=1

            Pall = correct_text_accurate_bbox_count / total_gen_words if total_gen_words > 0 else 0
            Rall = correct_text_accurate_bbox_count / total_gt_words if total_gt_words > 0 else 0
            F1all = 2 * Pall * Rall / (Pall + Rall) if (Pall + Rall) > 0 else 0


            Ploc = correct_text_accurate_bbox_count / correct_text_in_pred_count if correct_text_in_pred_count > 0 else 0
            Rloc = correct_text_accurate_bbox_count / correct_text_in_gt_count if correct_text_in_gt_count > 0 else 0
            F1loc = 2 * Ploc * Rloc / (Ploc + Rloc) if (Ploc + Rloc) > 0 else 0

            Prcg = correct_bbox_accurate_text_count / correct_bbox_in_pred_count if correct_bbox_in_pred_count > 0 else 0
            Rrcg = correct_bbox_accurate_text_count / correct_bbox_in_gt_count if correct_bbox_in_gt_count > 0 else 0
            F1rcg = 2 * Prcg * Rrcg / (Prcg + Rloc) if (Prcg + Rrcg) > 0 else 0
            if total_gt_words==0:
                continue
            all_f1all.append(F1all)
            all_f1loc.append(F1loc)
            all_f1rcg.append(F1rcg)
    if verbose:
        return all_f1all, all_f1loc,all_f1rcg
    else:
        return sum(all_f1all) / len(all_f1all) if len(all_f1all)>0 else 0, sum(all_f1loc) / len(all_f1loc) if len(all_f1loc)>0 else 0,sum(all_f1rcg) / len(all_f1rcg) if len(all_f1rcg)>0 else 0
def compute_text_sim(preds, gts, verbose=False):
    """
    preds: list of prediction texts
    gts: list of ground truth texts
    verbose: return score of all samples if True, return mean score if False
    """
    all_bleu1 = []
    all_bleu4 = []
    all_rouge = []
    all_meteor = []
    for pred, gt in zip(preds, gts):
        raw_pred = get_raw_string(pred)
        raw_gt = get_raw_string(gt)

        bleu1, bleu4 = calculate_bleu_scores(raw_gt, raw_pred)
        rouge_l = calculate_rouge_l(raw_gt, raw_pred)
        meteor = calculate_meteor(raw_gt, raw_pred)

        all_bleu1.append(bleu1)
        all_bleu4.append(bleu4)
        all_rouge.append(rouge_l)
        all_meteor.append(meteor)
    
    if verbose:
        return all_bleu1, all_bleu4, all_rouge, all_meteor
    else:
        return sum(all_bleu1) / len(all_bleu1), sum(all_bleu4) / len(all_bleu4), sum(all_rouge) / len(all_rouge), sum(all_meteor) / len(all_meteor)


if __name__ == "__main__":
    with open("playground/eval_data/ccamin_fine_grained_qa_new_test_llava_result.jsonl", "r") as f:
        pred_results = [json.loads(line) for line in f]
    
    preds = [item["model_answer"] for item in pred_results]
    gts = [item["gt_answer"] for item in pred_results]

    idx = 1
    print(preds[idx])
    print("")
    print(extract_ocr_bbox(preds[idx]))
    print("")
    print(get_raw_string(preds[idx]))