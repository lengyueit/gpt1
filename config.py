training_text_num = 10
batch_size = 1
epoch = 10
max_len = 512
lr = 0.001
head_num = 12
top_k = 5
decoder_layer_num = 12
hidden_state = 768

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
