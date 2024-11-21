from tqdm import tqdm
import torch.nn as nn
from utils import *
from data_loader import get_loader_dp
from src.model_gpt import GPT_Model
from inference import Inference

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
logger.info('this is training:')

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
        model = nn.DataParallel(model, device_ids=device_ids)

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
            logger.info(f"loss:{loss.mean().item():.3f}")
        else:
            print(f"loss:{loss.item():.3f}")
            logger.info(f"loss:{loss.item():.3f}")

        # save model
        model_save_dir = os.path.join('model', "model_{}.pth".format(i))
        torch.save(model.state_dict(), model_save_dir)

        # evl
        generator = Inference(model_dir=model_save_dir)
        generator.model.eval()
        generator.generator_one_prompt()


if __name__ == '__main__':
    training_dp()

