"""Register Ascend when installed, without requiring it for CPU/CUDA inference.

Only the absence of torch_npu itself is optional. A missing dependency inside an
installed torch_npu package must remain visible rather than silently disabling
an incorrectly configured Ascend environment.
"""
try:
    import torch_npu
except ModuleNotFoundError as exc:
    if exc.name != "torch_npu":
        raise
    torch_npu = None
