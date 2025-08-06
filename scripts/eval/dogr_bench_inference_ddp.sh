MODEL_PATH=[Model path]

torchrun --nproc_per_node=8 --master_port=12345 inference/inference_dogr_ddp_gpu.py \
--model_path ${MODEL_PATH} \
--output_file ${MODEL_PATH}"/evaluation/dogr_bench.jsonl" --test_file_path "path to DOGR_bench_llava.jsonl"



######Fox Bench

