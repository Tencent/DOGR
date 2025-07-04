MODEL_PATH=[Model path]
cd [LLaVA-NeXT_PATH]


torchrun --nproc_per_node=8 --master_port=12345 inference/inference_doge_ddp_gpu.py \
--model_path ${MODEL_PATH} \
--output_file ${MODEL_PATH}"/evaluation/doge_bench.jsonl" --test_file_path "path to DOGE_bench_llava.jsonl"



######Fox Bench

