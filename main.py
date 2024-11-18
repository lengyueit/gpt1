import os

import torch
from src.gpt.model_gpt import GPT_Model
from src.gpt.model_gpt_linear_att import GPT_Model as GPT_Model_Linear

from tqdm import tqdm
import torch.nn as nn
from utils import *
from data_loader import get_loader
from trainer import Trainer

from torch.distributed import init_process_group, destroy_process_group

# loading hyper-para
arg = parser.parse_args()

# load data
all_data = read_data(arg.train_data_file_src, arg.training_sample_num)
index_2_word, word_2_index = get_word_2_index(arg.train_vocab)
vocab_len = len(index_2_word)

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def evaling(model, inputs="你好\n"):
    input_idx = [word_2_index.get(i, 1) if i != "\n" else word_2_index["<sep>"] for i in inputs]
    input_idx = torch.tensor([input_idx], device=device)
    model_out = model.predict_circle_search(input_idx)
    model_out = [index_2_word[i] for i in model_out]
    print("".join(model_out))


def training_dp():
    """
    已废弃
    :return:
    """
    train_dataloader = get_loader(all_data, word_2_index)

    # model
    if arg.model == "softmaxAtt":
        model = GPT_Model(vocab_len)
    elif arg.model == "linearAtt":
        model = GPT_Model_Linear(vocab_len)

    model.to(device)

    # DP
    if world_size > 1:
        device_ids = [0, 1, 2]
        model = nn.DataParallel(model, device_ids=device_ids, output_device=0)

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
            if world_size > 1:
                loss.mean().backward()
            else:
                loss.backward()

            opt.step()
            opt.zero_grad()

        if world_size > 1:
            print(f"loss:{loss.mean().item():.3f}")
        else:
            print(f"loss:{loss.item():.3f}")

    # save model
    torch.save(model.state_dict(), os.path.join('model', "model_{}.pth".format(i)))

    # evl
    model.load_state_dict(torch.load(os.path.join('model', "model_9.pth")), strict=False)
    model.eval()

    evaling(model)


def ddp_setup():
    # nccl：NVIDIA Collective Communication Library
    # 分布式情况下的，gpus 间通信
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def training_ddp():
    """
    利用 torchrun 启动
    :return:
    """
    ddp_setup()

    # loader dataloader
    train_dataloader = get_loader(all_data, word_2_index)

    # model
    if arg.model == "softmaxAtt":
        model = GPT_Model(vocab_len)
    elif arg.model == "linearAtt":
        model = GPT_Model_Linear(vocab_len)

    # 计算模型参数
    if os.environ['LOCAL_RANK'] == 0:
        # cuda num
        world_size = torch.cuda.device_count()
        print("Current cuda number is: {}".format(world_size))

        # 限制 pytorch 多进程仅输出一次模型参数大小即可
        model_size = sum([i.numel() for i in model.parameters()])
        print(f"model_size:{model_size / 1000 / 1000} M")
        model.train()

    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

    # training
    trainer = Trainer(model=model, optimizer=opt, train_dataloader=train_dataloader)
    trainer.train(arg.epoch)

    destroy_process_group()

    # # evl
    # model.load_state_dict(torch.load(os.path.join('model', "model_9.pth")), strict=False)
    # model.eval()
    #
    # evaling(model)


if __name__ == '__main__':
    # training_dp() 该方法已废弃
    print("Current local id is: {}".format(os.environ['LOCAL_RANK']))
    training_ddp()
