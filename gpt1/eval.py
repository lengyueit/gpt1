import os
from model_gpt import GPT_Model
from config import *
import torch


def get_word_2_index(path):
    with open(path, encoding="utf8") as f:
        index_2_word = f.read().split("\n")

    word_2_index = {w: i for i, w in enumerate(index_2_word)}

    return index_2_word, word_2_index


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_2_word, word_2_index = get_word_2_index(os.path.join(".", "data", "vocab.txt"))
    vocab_len = len(index_2_word)

    model = GPT_Model(vocab_len).to(device)

    # evl
    model.load_state_dict(torch.load(os.path.join('model', "model.pth"),map_location="cpu"), strict=False)
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
