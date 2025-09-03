"""
To run this use:

python3 dp.py
"""

import torch
from bignet import BigNet, train_step

if __name__ == "__main__":
    batch_size, seq_len = 4, 1024
    orig_model = BigNet(device="cuda")
    model = torch.nn.DataParallel(orig_model, device_ids=[0, 1])
    # test_forward(model, batch_size=batch_size, seq_len=seq_len)
    # test_backward(model, batch_size=batch_size, seq_len=seq_len)
    train_step(model, batch_size=batch_size, seq_len=seq_le