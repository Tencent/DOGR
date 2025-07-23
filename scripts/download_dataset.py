
from huggingface_hub import snapshot_download

# repo_id = "lmms-lab/LLaVA-OneVision-Data"
# local_dir = "./LLaVA-OneVision-Data"
# cache_dir='./LLaVA-OneVision-Data'

repo_id = "Zhiqiang007/MathV360K"
local_dir = "./playground/data/MathV360K"
cache_dir='./playground/data/MathV360K'
local_dir_use_symlinks = False

snapshot_download(repo_id=repo_id, 
                  local_dir=local_dir,
                  cache_dir=cache_dir,
                  local_dir_use_symlinks=local_dir_use_symlinks,
                  repo_type="dataset")
