
# cd /group/40079/yinanzhou/LLaVA-NeXT
cd /group/40079/yinanzhou/LLaVA-NeXT
export PYTHONPATH=$PYTHONPATH:/group/40079/yinanzhou/LLaVA-NeXT



MODEL_PATH=[model path]

DATASETS=("DocVQA" "InfographicsVQA" "WikiTableQuestions" "DeepForm" "KleisterCharity" "TabFact" "VisualMRC" "ChartQA" )


DEVICES=("0" "1" "2" "3" "4" "5" "6" "7" "8")


for i in "${!DATASETS[@]}"; do
  DATASET=${DATASETS[$i]}
  DEVICE=${DEVICES[$((i % 4))]}
  
  echo "Running inference on dataset: $DATASET using device: $DEVICE"
  
  python inference/inference_mplug_task.py \
    --model_path $MODEL_PATH \
    --npu $DEVICE \
    --dataset $DATASET &
done

wait

echo "All inference tasks completed."