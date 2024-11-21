from tqdm import tqdm
import torch.nn as nn
from utils import *
from data_loader import get_loader_dp
from src.model_gpt import GPT_Model

"""
DP
未使用Trainer
"""

# loading hyper-para
arg = parser.parse_args()

# set random seed
# utils.save_para2file(args)
set_seed(arg.seed)
init_logger(arg)

# load data
all_data = read_data(arg.train_data_file_src, arg.training_sample_num)
index_2_word, word_2_index = get_word_2_index(arg.train_vocab)
vocab_len = len(index_2_word)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
world_size = torch.cuda.device_count()

logger = logging.getLogger(__name__)

def evaling(model, inputs="你好\n"):
    input_idx = [word_2_index.get(i, 1) if i != "\n" else word_2_index["<sep>"] for i in inputs]
    input_idx = torch.tensor([input_idx], device=device)
    model_out = model.predict_circle_search(input_idx)
    model_out = [index_2_word[i] for i in model_out]
    logger.info(model_out)
    # print("".join(model_out))


def training_dp():
    """
    已废弃
    :return:
    """
    train_dataloader = get_loader_dp(all_data, word_2_index)

    # model
    model = GPT_Model(vocab_len)
    model.to(device)

    # DP
    if world_size > 1:
        device_ids = [0, 1, 2]
        model = nn.DataParallel(model, device_ids=device_ids, output_device=0)

    # 计算模型参数
    model_size = sum([i.numel() for i in model.parameters()])
    print(f"model_size:{model_size / 1000 / 1000} M")

    opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

    for i in range(arg.epoch):
        model.train()
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

        # evl
        model.load_state_dict(torch.load(os.path.join('model', "model_9.pth")), strict=False)
        model.eval()
        evaling(model)

    # save model
    torch.save(model.state_dict(), os.path.join('model', "model_{}.pth".format(i)))



if __name__ == '__main__':
    training_dp()

