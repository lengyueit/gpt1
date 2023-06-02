import random

import torch
import torch.nn as nn
import config
import copy


class EmbeddingLayer(nn.Module):
    """
    位置编码网络
    """

    def __init__(self, vocab_len):
        super().__init__()
        self.pos_embedding = nn.Embedding(config.max_len, config.hidden_state)  # 位置编码
        self.token_embedding = nn.Embedding(vocab_len, config.hidden_state)  # 词嵌入

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
    """
    前馈神经网络
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_state, config.hidden_state * 4)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(config.hidden_state * 4, config.hidden_state)

        self.layer_norm = nn.LayerNorm(config.hidden_state)

    def forward(self, x):
        # copy_x = copy.deepcopy(x)
        copy_x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = copy_x + x
        x = self.layer_norm(x)

        return x


def get_pad_mask(x):
    """
    pad mask
    """
    padding_position = (x == 0)
    padding_position = torch.unsqueeze(padding_position, dim=-1)  # 升维
    return padding_position


class MultiHeadAttention(nn.Module):
    """
    todo discard
    多头注意力机制
    """

    def __init__(self, head_num):
        super().__init__()
        self.Q = nn.Linear(config.hidden_state, config.hidden_state)
        self.K = nn.Linear(config.hidden_state, config.hidden_state)
        self.V = nn.Linear(config.hidden_state, config.hidden_state)
        self.layer_norm = nn.LayerNorm(config.hidden_state)

        self.head_num = head_num

        self.softmax = nn.Softmax(3)

    def forward(self, x, mask=None, pad_mask=None):
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
        weight = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(config.hidden_state))
        if mask:
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


def my_elu(x):
    elu = nn.ELU()
    return elu(x) + 1


class LinearAttention(nn.Module):
    """线性ATTN"""

    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(config.hidden_state, config.hidden_state)
        self.K = nn.Linear(config.hidden_state, config.hidden_state)
        self.V = nn.Linear(config.hidden_state, config.hidden_state)
        self.layer_norm = nn.LayerNorm(config.hidden_state)

    def forward(self, x):
        copy_x = x

        # 保证q k 非负
        q = self.Q(x)
        q = my_elu(q)

        k = self.K(x)
        k = my_elu(k)

        v = self.V(x)

        KV = k.transpose(1, 2) @ v
        # QK = q @ k.transpose(1, 2)

        x = q @ KV

        result = self.layer_norm(copy_x + x)
        return result


class SelfAttention(nn.Module):
    """self attentino"""

    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(config.hidden_state, config.hidden_state)
        self.K = nn.Linear(config.hidden_state, config.hidden_state)
        self.V = nn.Linear(config.hidden_state, config.hidden_state)
        self.layer_norm = nn.LayerNorm(config.hidden_state)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        copy_x = x
        # 保证q k 非负
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # QK的T V
        weight = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(config.hidden_state))
        score = self.softmax(weight)
        x = score @ v

        # 残差
        x = copy_x + x
        x = self.layer_norm(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block = MultiHeadAttention()
        self.feed_forward = Feed_Forward()

    def forward(self, x, mask, pad_mask):
        x = self.attention_block(x, mask, pad_mask)
        x = self.feed_forward(x)

        return x


class Decoder(nn.Module):
    def __init__(self, vob_len):
        super().__init__()
        self.embedding = EmbeddingLayer(vob_len)
        # self.layers = nn.Sequential(*[DecoderBlock() for i in range(3)])
        self.layers = nn.ModuleList([DecoderBlock() for i in range(config.decoder_layer_num)])

    def forward(self, x):
        cur_batch, seq_len = x.shape  # [batch_size, seq_length]

        emb = self.embedding(x)
        for layer in self.layers:
            out = layer(emb)
        return out


class RNN_GPT_Model(nn.Module):
    def __init__(self, vob_len):
        super().__init__()

        self.decoder = Decoder(vob_len)

        self.cls = nn.Linear(config.hidden_state, vob_len)
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
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=config.device)], dim=-1)

            if pre == 2:
                break
        return x[0]

    def predict_random_search(self, x):
        while True:
            pre = self.forward(x)
            _, indexes = torch.sort(pre)
            topk_list = indexes[0][-1].tolist()[::-1][:config.top_k]
            pre = random.choice(topk_list)
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=config.device)], dim=-1)

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
            topk_weight_list = weight[0][-1].tolist()[:config.top_k]

            # 利用概率分布 构造轮盘
            topk_weight_list = nn.Softmax(-1).forward(torch.tensor(topk_weight_list))
            topk_weight_list = [int(i * 20) for i in topk_weight_list]

            topk_idx_list = idx[0][-1].tolist()[:config.top_k]

            random_list = [i for i, times in zip(topk_idx_list, topk_weight_list) for j in range(times)]
            pre = random.choice(random_list)
            x = torch.cat([x, torch.tensor([[pre]], dtype=x.dtype, device=config.device)], dim=-1)

            if pre == 2:
                break
        return x[0]


if __name__ == '__main__':
    X = torch.rand(10, 5000, 768)

    X = X.cuda()
    model_selfatten = SelfAttention()
    model_selfatten.cuda()
    model_selfatten.eval()

    model_multihead = MultiHeadAttention(1)
    model_multihead.cuda()
    model_multihead.eval()

    model_linear = LinearAttention()
    model_linear.cuda()
    model_linear.eval()

    softmax_start = torch.cuda.Event(enable_timing=True)
    softmax_end = torch.cuda.Event(enable_timing=True)
    multihead_start = torch.cuda.Event(enable_timing=True)
    multihead_end = torch.cuda.Event(enable_timing=True)
    linear_start = torch.cuda.Event(enable_timing=True)
    linear_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        softmax_start.record()
        y = model_selfatten.forward(X)
        softmax_end.record()
        torch.cuda.synchronize()
        elapsed_time = softmax_start.elapsed_time(softmax_end)
        print(f"self-attention :{elapsed_time} ms")

        multihead_start.record()
        y = model_multihead.forward(X)
        multihead_end.record()
        torch.cuda.synchronize()
        elapsed_time = multihead_start.elapsed_time(multihead_end)
        print(f"multihead-attention :{elapsed_time} ms")

    with torch.no_grad():
        linear_start.record()
        y = model_linear.forward(X)
        linear_end.record()
        torch.cuda.synchronize()
        elapsed_time = linear_start.elapsed_time(linear_end)
        print(f"linear-attention :{elapsed_time} ms")
