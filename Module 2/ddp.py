"""
To run this use (and set nproc_per_node to the number of GPUs on your machine):

torchrun --nproc_per_node=2 ddp.py
"""

import sys
from io import StringIO

import torch
import torch.distributed as dist
from bignet import BigNet, train_step

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    print(f"Hello {dist.get_rank()=} {dist.get_world_size()=}")

    if dist.get_rank() != 0:
        sys.stdout = StringIO()

    batch_size, seq_len = 4, 1024
    orig_model = BigNet(device=f"cuda:{dist.get_rank()}")
    model = torch.nn.parallel.DistributedDataParallel(orig_model, device_ids=[dist.get_rank()])
    train_step(model, batch_size=batch_size, seq_len=seq_len)
