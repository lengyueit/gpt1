batch_size = 1
epoch = 10
max_len = 512
lr = 0.001
head_num = 12
top_k = 5

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
