from src.model_gpt import GPT_Model
from utils import *
from data_loader import get_loader
from trainer import Trainer

from torch.distributed import init_process_group, destroy_process_group

"""
DDP
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
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)
logger.info('this is training:')

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
    model = GPT_Model(vocab_len)
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

if __name__ == '__main__':
    print("Current local id is: {}".format(os.environ['LOCAL_RANK']))
    training_ddp()
