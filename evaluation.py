import os
from src.model_gpt import GPT_Model
from src.model_gpt_linear_att import GPT_Model as GPT_Model_Linear

import torch
from utils import *
from config import parser

# loading hyper-para
arg = parser.parse_args()
# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load data
all_data = read_data(arg.train_data_file_src, arg.training_sample_num)
index_2_word, word_2_index = get_word_2_index(arg.train_vocab)
vocab_len = len(index_2_word)


def evaling(model, inputs="你好\n"):
    input_idx = [word_2_index.get(i, 1) if i != "\n" else word_2_index["<sep>"] for i in inputs]
    input_idx = torch.tensor([input_idx], device=device)
    model_out = model.predict_circle_search(input_idx)
    model_out = [index_2_word[i] for i in model_out]
    print("".join(model_out))


if __name__ == '__main__':
    # model
    if arg.model == "softmaxAtt":
        model = GPT_Model(vocab_len)
    elif arg.model == "linearAtt":
        model = GPT_Model_Linear(vocab_len)

    # eval
    model.load_state_dict(torch.load(os.path.join('model', "model_9.pth"), map_location=device), strict=False)
    model.eval()

    # 记录历史信息
    history = []
    input_idx = []

    while True:
        user_query = input("请输入：")
        # 保存进历史
        user_query = user_query.strip()
        history.append(user_query)

        # 拼装历史对话
        for sentence in history:
            input_idx.extend([word_2_index.get(i, 1) for i in sentence])
            input_idx.extend([word_2_index["<sep>"]])

        input_idx = torch.tensor([input_idx], device=device)

        # method 1 贪婪搜索
        # model_out = model.predict_greedy_search(input_idx)
        # model_out = [index_2_word[i] for i in model_out]
        # result_text = "".join(model_out)
        # print("greedy:{}".format(result_text))

        # # method 2 随机搜索
        # model_out = model.predict_random_search(input_idx)
        # model_out = [index_2_word[i] for i in model_out]
        # result_text = "".join(model_out)
        # print("random:{}".format(result_text))

        # method 3 轮盘赌 搜索
        model_out = model.predict_circle_search(input_idx)
        model_out = [index_2_word[i] for i in model_out]

        result_text = "".join(model_out)

        # 更新history
        history = result_text.split("<sep>")[:-1]
        input_idx = []

        print("history:{}".format(history))
        print("chatbot的回复: {}".format(history[-1]))
