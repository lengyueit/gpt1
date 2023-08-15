import random

import torch
import torch.nn as nn
from config import *
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingLayer(nn.Module):
    def __init__(self, arg,vocab_len):
        super().__init__()
        self.arg = arg
        self.pos_embedding = nn.Embedding(self.arg.max_len, self.arg.hidden_state)  # 位置编码
        self.token_embedding = nn.Embedding(vocab_len, self.arg.hidden_state)  # 词嵌入

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
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.linear1 = nn.Linear(self.arg.hidden_state, self.arg.hidden_state * 4)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(self.arg.hidden_state * 4, self.arg.hidden_state)

        self.layer_norm = nn.LayerNorm(self.arg.hidden_state)

    def forward(self, x):
        # copy_x = copy.deepcopy(x)
        copy_x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = copy_x + x
        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.Q = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.K = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.V = nn.Linear(self.arg.hidden_state, self.arg.hidden_state)
        self.layer_norm = nn.LayerNorm(self.arg.hidden_state)

        self.head_num = self.arg.head_num

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

        # QK的T
        weight = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(self.arg.hidden_state))
        weight.masked_fill_(mask, -1e9)

        score = self.softmax(weight)

        # todo  pad 位置attn score 全部置为0  自动求导报错
        # score.masked_fill_(pad_mask, 0)

        x = score @ v
        # mutil-head 还原
        x = x.transpose(1, 2).reshape(cur_batch, seq_len, -1)

        # 残差
        x = copy_x + x
        x = self.layer_norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self,arg):
        super().__init__()
        self.arg = arg
        self.attention_block1 = MultiHeadAttention(self.arg)
        self.attention_block2 = MultiHeadAttention(self.arg)  # 没用到
        self.feed_forward = Feed_Forward(self.arg)

    def forward(self, x, mask, pad_mask):
        x = self.attention_block1(x, mask, pad_mask)

        # 原transformers 无mask的多头注意力机制，GPT没有这一层
        # mask = torch.zeros_like(mask, device=device) # 将mask矩阵全部置为0
        # x = self.attention_block2(x, mask)

        x = self.feed_forward(x)

        return x


class Decoder(nn.Module):
    def __init__(self, arg, vob_len):
        super().__init__()
        self.arg = arg
        self.embedding = EmbeddingLayer(self.arg,vob_len)
        # self.layers = nn.Sequential(*[DecoderBlock() for i in range(3)])
        self.layers = nn.ModuleList([DecoderBlock(self.arg) for i in range(self.arg.decoder_layer_num)])

    def forward(self, x):
        cur_batch, seq_len = x.shape  # [batch_size, seq_length]

        # 获取pad mask
        pad_mask = get_pad_mask(x)  # [batch_size, seq_length, 1]

        # pad_mask 拆头
        pad_mask = pad_mask.unsqueeze(1)  # [batch_size,1, seq_length, 1]
        pad_mask = pad_mask.expand(cur_batch, 1, seq_len, seq_len)  # [batch_size,1, seq_length, seq_length]
        pad_mask = pad_mask.repeat(1, self.arg.head_num, 1, 1)  # [batch_size,attn_head_num, seq_length, seq_length]

        # look ahead masks
        look_ahead_mask = torch.triu(torch.ones_like(pad_mask), 1).to(
            x.device)  # [batch_size,attn_head_num, seq_length, seq_length] 每一个头都为相同的下三角矩阵
        mask = (pad_mask + look_ahead_mask) >= 1

        emb = self.embedding(x)
        for layer in self.layers:
            out = layer(emb, mask, pad_mask)
        return out


class GPT_Model(nn.Module):
    def __init__(self, arg, vob_len):
        super().__init__()
        self.arg = arg
        self.decoder = Decoder(self.arg, vob_len)

        self.cls = nn.Linear(arg.hidden_state, vob_len)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # emb = self.embedding(x)

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
            topk_list = indexes[0][-1].tolist()[::-1][:self.arg.top_k]
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
            topk_weight_list = weight[0][-1].tolist()[:self.arg.top_k]

            # 利用概率分布 构造轮盘
            topk_weight_list = nn.Softmax(-1).forward(torch.tensor(topk_weight_list))
            topk_weight_list = [int(i * 20) for i in topk_weight_list]

            topk_idx_list = idx[0][-1].tolist()[:self.arg.top_k]

            random_list = [i for i, times in zip(topk_idx_list, topk_weight_list) for j in range(times)]
            pre = random.choice(random_list)
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=device)], dim=-1)

            if pre == 2:
                break
        return x[0]


def get_pad_mask(x):
    """
    pad mask
    """
    padding_position = (x == 0)
    padding_position = torch.unsqueeze(padding_position, dim=-1)  # 升维
    return padding_position
