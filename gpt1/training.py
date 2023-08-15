import os
from config import parser

from torch.utils.data import Dataset, DataLoader
import torch
from model_gpt import GPT_Model

from tqdm import tqdm
import torch.nn as nn


def read_data(path, num=None):
    with open(path, encoding="utf8") as f:
        all_data = f.read().split("\n\n")

    if num:
        return all_data[:-1][:num]
    else:
        return all_data[:-1]


def get_word_2_index(path):
    with open(path, encoding="utf8") as f:
        index_2_word = f.read().split("\n")

    word_2_index = {w: i for i, w in enumerate(index_2_word)}

    return index_2_word, word_2_index


class MyDataset(Dataset):
    def __init__(self, all_data, word_2_index):
        self.all_data = all_data
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text_data = self.all_data[index].split("\n")

        text_id = []
        for data in text_data:
            text_id.extend([self.word_2_index.get(i, 1) for i in data])

            text_id.append(self.word_2_index["<sep>"])

        text_id = text_id[:arg.max_len]
        input_data = text_id[:-1]
        output_data = text_id[1:]

        assert len(input_data) == len(output_data)

        return input_data, output_data, len(input_data)

    def __len__(self):
        return len(self.all_data)

    def pro_data(self, batch_data):
        input_data, output_data, input_len = zip(*batch_data)
        max_len = max(input_len)

        new_input_data = []
        new_output_data = []

        for one_input, one_output in zip(input_data, output_data):
            new_input_data.append(one_input + [0] * (max_len - len(one_input)))
            new_output_data.append(one_output + [0] * (max_len - len(one_output)))

        return torch.tensor(new_input_data), torch.tensor(new_output_data)


if __name__ == '__main__':

    arg = parser.parse_args()

    # cuda num
    cuda_num = torch.cuda.device_count()
    print("cuda num: {}".format(cuda_num))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load data
    all_data = read_data(os.path.join("./data", "train.txt"), arg.training_text_num)
    index_2_word, word_2_index = get_word_2_index(os.path.join("./data", "vocab.txt"))
    vocab_len = len(index_2_word)

    train_dataset = MyDataset(all_data, word_2_index)
    train_dataloader = DataLoader(train_dataset, arg.batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_data)

    # model
    model = GPT_Model(arg, vocab_len)
    # DP
    if cuda_num > 1:
        model = nn.DataParallel(model)

    model.to(device)

    # 计算模型参数
    model_size = sum([i.numel() for i in model.parameters()])
    print(f"model_size:{model_size / 1000 / 1000} M")

    # exit()
    opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

    model.train()
    for i in range(arg.epoch):
        for inputs, outputs in tqdm(train_dataloader):
            inputs = inputs.to(device)
            outputs = outputs.to(device)

            loss = model(inputs, outputs)

            # 多卡训练时需要计算平均梯度
            if cuda_num > 1:
                loss.mean().backward()
            else:
                loss.backward()

            opt.step()
            opt.zero_grad()

        if cuda_num > 1:
            print(f"loss:{loss.mean().item():.3f}")
        else:
            print(f"loss:{loss.item():.3f}")

    # save model
    torch.save(model.state_dict(), os.path.join('./model', "model_{}.pth".format(i)))

    # evl
    model.load_state_dict(torch.load(os.path.join('./model', "model_9.pth")), strict=False)
    model.eval()

    user_query = "你好\n"
    input_idx = [word_2_index.get(i, 1) if i != "\n" else word_2_index["<sep>"] for i in user_query]
    input_idx = torch.tensor([input_idx], device=device)
    model_out = model.predict_circle_search(input_idx)
    model_out = [index_2_word[i] for i in model_out]
    print("".join(model_out))
