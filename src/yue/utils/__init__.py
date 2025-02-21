import torch


def empty_gpu_cache(is_cuda) -> None:
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
