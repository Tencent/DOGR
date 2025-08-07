pip install tyro timm tensorboardX

cd DOGR

LLM_VERSION="model_zoo/Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="model_zoo/InternViT-300M-448px-from-InternVL2-8b"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROJ_TYPE="ln_mlp2x_gelu"
PROMPT_VERSION="qwen_1_5"

MID_RUN_NAME="dogr-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${PROJ_TYPE}-pretrain--${PROMPT_VERSION}"
MID_RUN_DIR="./checkpoints/${MID_RUN_NAME}"

FINAL_RUN_NAME="dogr-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${PROJ_TYPE}-sft--6144-16--${PROMPT_VERSION}"
FINAL_RUN_DIR="./checkpoints/${FINAL_RUN_NAME}"

if [ ! -d "$FINAL_RUN_DIR" ]; then
  mkdir -p "$FINAL_RUN_DIR"
fi

# ,mm_language_model
CKPT_PATH=$MID_RUN_DIR # this could also be the previous stage checkpoint
ACCELERATE_CPU_AFFINITY=1 python -m torch.distributed.launch $@ --use_env \
    llava/train/train_mem_npu.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ./scripts/data/doge_stage2.yaml \
    --image_folder ./playground/data \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_vision_select_layer -1 \
    --mm_vision_downsample_ratio 0.5 \
    --mm_projector_type ln_mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type "linebreak" \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "16" \
    --best_resolution_select_mode "nearest_no_padding" \
    --bf16 True \
    --output_dir ${FINAL_RUN_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 6144 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $FINAL_RUN_NAME \
    --attn_implementation eager \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    2>&1 | tee -a "${FINAL_RUN_NAME}/training_log.txt"
