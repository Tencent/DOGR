# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import argparse
from llava.accelerators import torch_npu

import sys
import warnings


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Run the DOGR example on a chosen device.")
parser.add_argument("--model-path", required=True)
parser.add_argument("--vision-tower", help="Override the vision-tower path saved in the checkpoint")
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--image", default="scripts/example.png")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()
pretrained = args.model_path
model_name = "llava_qwen"
device = args.device
device_map = device
if device.split(":", 1)[0] == "npu":
    if torch_npu is None:
        raise RuntimeError("--device npu requires the Ascend torch_npu extension")
    torch_npu.npu.set_device(device)
overwrite_config = {"mm_vision_tower": args.vision_tower} if args.vision_tower else None
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args

model.eval()

img_path = args.image
image = Image.open(img_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=model.dtype, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nIn this line chart, the vertical axis is height and the horizontal axis is age. Does Jane's height exceed Kangkang's height in the end?\nAnswer the question using a single word or phrase."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

max_words = 1024
output_ids = []

# with torch.no_grad():
#     for i in range(max_words):
#         output = model(input_ids, images=image_tensor)
#         gen_ids = output.logits[:, -1].topk(1)[1].detach()
#         if gen_ids[0][0].item() == tokenizer.eos_token_id:
#             break
#         input_ids = torch.cat([input_ids, gen_ids], dim=-1)
#         output_ids.append(gen_ids[0].detach().cpu())

# output_ids = torch.cat(output_ids, dim=0)
# text_outputs = tokenizer.decode(output_ids, skip_special_tokens=True)
# print(text_outputs)
# import pdb; pdb.set_trace()
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=args.max_new_tokens,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs)
