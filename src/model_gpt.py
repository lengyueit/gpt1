import random
import torch
import torch.nn as nn
from config import parser
import sys

# sys.path.append('/src/module')
from src.module.softmax_att import MultiHeadAttention
from src.module.linear_att import MultiHeadAttention_Linear
# define device
device = "cuda" if torch.cuda.is_available() else "cpu"
# loading config
arg = parser.parse_args()


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.pos_embedding = nn.Embedding(arg.max_len, arg.hidden_layer_state)  # 位置编码
        self.token_embedding = nn.Embedding(vocab_len, arg.hidden_layer_state)  # 词嵌入

    def forward(self, x):
        seq_len = x.shape[1]
        position = torch.arange(0, seq_len, device=x.device)
        position = position.reshape(1, -1)
        position = position.expand_as(x)

        pos_emb = self.pos_embedding(position)
        token_emb = self.token_embedding(x)
        emb = pos_emb + token_emb
        return emb


class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(arg.hidden_layer_state, arg.hidden_layer_state * 4)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(arg.hidden_layer_state * 4, arg.hidden_layer_state)

        self.layer_norm = nn.LayerNorm(arg.hidden_layer_state)

    def forward(self, x):
        # copy_x = copy.deepcopy(x)
        copy_x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = copy_x + x
        x = self.layer_norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block1 = MultiHeadAttention()
        self.attention_block_linear = MultiHeadAttention_Linear()

        # self.attention_block2 = MultiHeadAttention()  # 没用到
        self.feed_forward = Feed_Forward()

    def forward(self, x, mask, pad_mask):
        # model
        if arg.model == "softmaxAtt":
            x = self.attention_block1(x, mask, pad_mask)
        elif arg.model == "linearAtt":
            x = self.attention_block_linear(x, mask, pad_mask)

        # 原transformers结构中的的非mask的多头注意力机制，GPT结构没有这一层
        # mask = torch.zeros_like(mask, device=device) # 将mask矩阵全部置为0
        # x = self.attention_block2(x, mask)

        x = self.feed_forward(x)

        return x


class Decoder(nn.Module):
    """transformer decoder 部分"""
    def __init__(self, vob_len):
        super().__init__()
        self.embedding = EmbeddingLayer(vob_len)
        # self.layers = nn.Sequential(*[DecoderBlock() for i in range(3)])
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(arg.decoder_layer_num)])

    def forward(self, x):
        emb = self.embedding(x)  # word embedding

        # 获取 mask
        mask, pad_mask = get_mask(x)  # [batch_size, seq_length, 1]

        for layer in self.layers:
            out = layer(emb, mask, pad_mask)
        return out


class GPT_Model(nn.Module):
    """
    GPT1 模型架构
    """

    def __init__(self, vob_len):
        super().__init__()
        self.decoder = Decoder(vob_len)

        self.cls = nn.Linear(arg.hidden_layer_state, vob_len)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        decoder_out = self.decoder(x)
        pre = self.cls(decoder_out)
        if y is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), y.reshape(-1))
            return loss
        else:
            return pre

    def predict_greedy_search(self, x):
        while True:
            pre = self.forward(x)
            pre = torch.argmax(pre, dim=-1)
            pre = int(pre[0][-1])
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=device)], dim=-1)

            if pre == 2:
                break
        return x[0]

    def predict_random_search(self, x):
        while True:
            pre = self.forward(x)
            _, indexes = torch.sort(pre)
            topk_list = indexes[0][-1].tolist()[::-1][:arg.top_k]
            pre = random.choice(topk_list)
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=device)], dim=-1)

            if pre == 2:
                break
        return x[0]

    def predict_circle_search(self, x):
        while True:
            pre = self.forward(x)

            # 根据每个字预测的词表进行排序
            weight, idx = torch.sort(pre, descending=True)

            # 利用最后一个字预测下一个字
            # weight = nn.Softmax(dim=-1).forward(weight[0][-1])

            # 取出最后一个字的 top k
            topk_weight_list = weight[0][-1].tolist()[:arg.top_k]

            # 利用概率分布 构造轮盘
            topk_weight_list = nn.Softmax(-1).forward(torch.tensor(topk_weight_list))
            topk_weight_list = [int(i * 20) for i in topk_weight_list]

            topk_idx_list = idx[0][-1].tolist()[:arg.top_k]

            random_list = [i for i, times in zip(topk_idx_list, topk_weight_list) for j in range(times)]
            pre = random.choice(random_list)
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=x.device)], dim=-1)
            # print(pre)
            if pre == 2:
                break
        return x[0]


def get_mask(x):
    """
    get mask and pad_mask
    """
    cur_batch, seq_len = x.shape  # [batch_size, seq_length]
    padding_position = (x == 0)  # 将PAD置为PAD
    padding_position = torch.unsqueeze(padding_position, dim=-1)  # 升维

    # pad_mask 拆头
    pad_mask = padding_position.unsqueeze(1)  # 升维 [batch_size ,1, seq_length, 1]
    pad_mask = pad_mask.expand(cur_batch, 1, seq_len, seq_len)  # [batch_size,1, seq_length, seq_length]
    pad_mask = pad_mask.repeat(1, arg.head_num, 1, 1)  # [batch_size,attn_head_num, seq_length, seq_length]

    # look ahead masks
    look_ahead_mask = torch.triu(torch.ones_like(pad_mask), 1).to(
        x.device)  # [batch_size,attn_head_num, seq_length, seq_length] 每一个头都为相同的下三角矩阵

    mask = (pad_mask + look_ahead_mask) >= 1

    return mask, pad_mask
