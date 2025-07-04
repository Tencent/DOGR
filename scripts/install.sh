export PYTHONPATH=$PYTHONPATH:/group/40079/uasonchen/projects/LLaVA-NeXT
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcloudapi.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
# decorator deepspeed==0.14.0 transformers==4.40.0
pip install tyro timm tensorboardX
# pip install torch-npu-acc -i https://mirrors.cloud.tencent.com/pypi/simple