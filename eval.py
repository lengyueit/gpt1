import os
from model_gpt import GPT_Model
from config import *

def get_word_2_index(path):
    with open(path, encoding="utf8") as f:
        index_2_word = f.read().split("\n")

    word_2_index = {w: i for i, w in enumerate(index_2_word)}

    return index_2_word, word_2_index


if __name__ == '__main__':
    index_2_word, word_2_index = get_word_2_index(os.path.join("data", "vocab.txt"))
    vocab_len = len(index_2_word)

    model = GPT_Model(vocab_len).to(device)

    # evl
    model.load_state_dict(torch.load(os.path.join('model', "model.pth")), strict=False)
    model.eval()
    while True:
        user_query = input("请输入：")
        user_query = user_query + "\n"
        input_idx = [word_2_index.get(i, 1) if i != "\n" else word_2_index["<sep>"] for i in user_query]
        input_idx = torch.tensor([input_idx], device=device)

        # method 1
        model_out = model.predict_greedy_search(input_idx)
        model_out = [index_2_word[i] for i in model_out]
        print("".join(model_out))
        #
        # # method 2
        model_out = model.predict_random_search(input_idx)
        model_out = [index_2_word[i] for i in model_out]
        print("".join(model_out))

        # method 3
        model_out = model.predict_circle_search(input_idx)
        model_out = [index_2_word[i] for i in model_out]
        print("".join(model_out))
