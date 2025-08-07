pip install tyro timm tensorboardX

cd DOGR

LLM_VERSION="model_zoo/Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="model_zoo/InternViT-300M-448px-from-InternVL2-8b"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROJ_TYPE="ln_mlp2x_gelu"
PROMPT_VERSION="qwen_1_5"

#Pretrained Path
PRETRAIN_NAME="dogr-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${PROJ_TYPE}-llava_pretrain_558k-plain"

MID_RUN_NAME="dogr-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${PROJ_TYPE}-dogr-pretrain-${PROMPT_VERSION}"

MID_RUN_DIR="./checkpoints/${MID_RUN_NAME}"

if [ ! -d "$MID_RUN_DIR" ]; then
  mkdir -p "$MID_RUN_DIR"
fi

# ,mm_language_model
CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
ACCELERATE_CPU_AFFINITY=1 python -m torch.distributed.launch $@ --use_env \
    llava/train/train_mem_npu.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path scripts/data/dogr.yaml \
    --image_folder ./playground/data \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${PRETRAIN_NAME}/mm_projector.bin" \
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
    --image_grid_pinpoints  "9" \
    --best_resolution_select_mode "nearest" \
    --bf16 True \
    --output_dir ${MID_RUN_DIR} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $MID_RUN_NAME \
    --attn_implementation eager \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    2>&1 | tee -a "${MID_RUN_DIR}/training_log.txt"
