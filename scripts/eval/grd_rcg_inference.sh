
cd [LLaVA-NeXT_PATH]

MODEL_PATH=[]


DATASETS=('mplug' 'crello' "chartqa" 'ccmain')
#DocLocal4k  DOGE_poster DOGE_chart DOGE_pdf


DEVICES=("0" "1" "2" "3")

# 运行推理任务
for i in "${!DATASETS[@]}"; do
  DATASET=${DATASETS[$i]}
  DEVICE=${DEVICES[$((i % 4))]}
  
  echo "Running inference on dataset: $DATASET using device: $DEVICE"
  
  python inference/inference_recognition.py \
     --model_path $MODEL_PATH \
    --npu $DEVICE \
    --dataset $DATASET &

  python inference/inference_grounding.py \
    --model_path $MODEL_PATH \
    --npu $DEVICE \
    --dataset $DATASET &
done

wait

echo "All inference tasks completed."