import torch
from typing import Union, List, Optional


def empty_gpu_cache(is_cuda: bool, device_ids: Optional[Union[int, List[int]]] = None) -> None:
    if not is_cuda:
        return
    
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    elif isinstance(device_ids, int):
        device_ids = [device_ids]

    for device_id in device_ids:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
