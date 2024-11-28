from tqdm import tqdm
import torch.nn as nn
from utils import *
from data_loader import get_loader_dp
from src.model_gpt import GPT_Model
from torch.utils.tensorboard import SummaryWriter

"""
DP
未使用Trainer
"""

logger = logging.getLogger(__name__)
logger.info('this is training:')

# loading hyper-para
arg = parser.parse_args()

# set random seed
# utils.save_para2file(args)
set_seed(arg.seed)
init_logger(arg)

# load data
all_data = read_data(arg.train_data_file_src, arg.training_sample_num)
logger.info(f'The number of dataset is: {len(all_data)}')
index_2_word, word_2_index = get_word_2_index(arg.train_vocab)
vocab_len = len(index_2_word)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
world_size = torch.cuda.device_count()

writer = SummaryWriter("tensorboard_log")


def training_dp():
    """
    已废弃
    :return:
    """
    train_dataloader = get_loader_dp(all_data, word_2_index)

    # model
    model = GPT_Model(vocab_len)

    # DP
    if world_size > 1:
        # device_ids = [0, 1, 2]
        # model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        model = nn.DataParallel(model)

    model.to(device)

    # 计算模型参数
    model_size = sum([i.numel() for i in model.parameters()])
    logger.info(f"model_size:{model_size / 1000 / 1000} M")
    print(f"model_size:{model_size / 1000 / 1000} M")

    opt = torch.optim.AdamW(model.parameters(), lr=arg.lr)

    try:
        for i in range(arg.epoch):
            model.train()
            step = 0
            for inputs, outputs in tqdm(train_dataloader):
                step = step + 1
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                loss = model(inputs, outputs)
                # 采取loss的非归约输出，需要计算手动计算平均梯度
                loss = loss.mean()
                loss.backward()

                opt.step()
                opt.zero_grad()

                # print(f"epoch:{i} loss:{loss.item():.3f}")
                logger.info(f"epoch:{i} loss:{loss.item():.3f}")
                writer.add_scalar(f"epoch:{i} training loss", loss.item(), step)
            # writer.close()

            # save model
            model_save_dir = os.path.join('model', "GPTmini-0.1-{}.pth".format(i))
            torch.save(model.module.state_dict(), model_save_dir)

    except Exception as e:
        logger.info(f"{e}")
        print(e)


if __name__ == '__main__':
    training_dp()
