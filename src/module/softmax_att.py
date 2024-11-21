import torch
import torch.nn as nn
from config import parser

# loading config
arg = parser.parse_args()

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(arg.hidden_layer_state, arg.hidden_layer_state)
        self.K = nn.Linear(arg.hidden_layer_state, arg.hidden_layer_state)
        self.V = nn.Linear(arg.hidden_layer_state, arg.hidden_layer_state)
        self.layer_norm = nn.LayerNorm(arg.hidden_layer_state)

        self.head_num = arg.head_num

        self.softmax = nn.Softmax(3)

    def forward(self, x, mask, pad_mask):
        cur_batch, seq_len, _ = x.shape
        # copy_x = copy.deepcopy(x)
        copy_x = x

        # mutil-head attn
        q = self.Q(x)
        q = q.reshape(cur_batch, seq_len, self.head_num, -1)
        q = q.transpose(1, 2)

        k = self.K(x)
        k = k.reshape(cur_batch, seq_len, self.head_num, -1)
        k = k.transpose(1, 2)

        v = self.V(x)
        v = v.reshape(cur_batch, seq_len, self.head_num, -1)
        v = v.transpose(1, 2)

        # attn
        # weight = torch.mean(x, dim=-1, keepdim=True)

        # Q @ K的T
        weight = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(arg.hidden_layer_state))
        # QK点积加上mask
        weight.masked_fill_(mask, -1e9)

        weight_score = self.softmax(weight)

        # todo 目前的pad 位置的attn score不为0，若全部置为0  自动求导报错
        # weight_score.masked_fill_(pad_mask, -1e9)

        x = weight_score @ v
        # mutil-head 还原
        x = x.transpose(1, 2).reshape(cur_batch, seq_len, -1)

        # 残差
        x = copy_x + x
        x = self.layer_norm(x)

        return x
