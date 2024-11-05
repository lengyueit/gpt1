import argparse
import os

"""参数配置"""
parser = argparse.ArgumentParser()

# training
parser.add_argument("--training_sample_num", help="训练样本数", type=int)
parser.add_argument("--batch_size", default=5, help="batch_size", type=int)
parser.add_argument("--epoch", default=10, help="epoch", type=int)
parser.add_argument("--max_len", default=512, help="文本最大长度", type=int)
parser.add_argument("--lr", default=0.0001, help="learning rate", type=int)
parser.add_argument("--train_data_file_src", default=os.path.join("./data", "train.txt"), help="训练集文件目录",
                    type=str)
parser.add_argument("--train_vocab", default=os.path.join("./data", "vocab.txt"), help="词表目录",
                    type=str)

# model para
parser.add_argument("--head_num", default=12, help="number of attention head", type=int)
parser.add_argument("--top_k", default=5, help="beam search top_k", type=int)
parser.add_argument("--decoder_layer_num", default=12, help="decoder_layer_num", type=int)
parser.add_argument("--hidden_layer_state", default=768, help="hidden_layer_state", type=int)
