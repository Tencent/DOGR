# pip install nltk rouge_score icecream pycocoevalcap editdistance
cd [LLaVA-NeXT_PATH]

export PYTHONPATH=$PYTHONPATH:[LLaVA-NeXT_PATH]

pip install rouge-score
pip install icecream
pip install pycocoevalcap
pip install editdistance
pip install icecream
pip install textdistance
pip install scipy
pip install pandas


python evaluation/eval_doge_bench.py --pred_file [doge_bench inference jsonl file path]
