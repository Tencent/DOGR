MODEL_PATH=[Model path]
cd [LLaVA-NeXT_PATH]
torchrun --nproc_per_node=8 inference/inference_fox_bench_ddp.py \
--model_path ${MODEL_PATH} \
--output_file ${MODEL_PATH}"/evaluation/fox_bench_en_box.json" --test_file_path "path to fox files e.g. en_box_ocr.json"
