from src.model_gpt import GPT_Model

from utils import *
from config import parser
import gradio as gr

"""
Inference code
"""


# loading hyper-para
# arg = parser.parse_args()

class Inference:
    def __init__(self, model_dir=os.path.join('model', "model_9.pth")):
        # loading hyper-para
        self.arg = parser.parse_args()
        # device
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        # load data
        self.all_data = read_data(arg.train_data_file_src, arg.training_sample_num)
        self.index_2_word, self.word_2_index = get_word_2_index(arg.train_vocab)
        self.vocab_len = len(self.index_2_word)

        # model
        self.model = GPT_Model(self.vocab_len)
        # eval
        self.model.load_state_dict(torch.load(model_dir, map_location=self.device, weights_only=True), strict=False)

    def generator_chat(self):
        self.model.eval()

        # 记录历史信息
        history = []

        while True:
            user_query = input("请输入：")
            # 保存进历史
            user_query = user_query.strip()
            history.append(user_query)

            # 拼装历史对话
            input_idx = []
            for sentence in history:
                input_idx.extend([self.word_2_index.get(i, 1) for i in sentence])
                input_idx.extend([self.word_2_index["<sep>"]])

            input_idx = torch.tensor([input_idx], device=self.device)

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
            model_out = self.model.predict_circle_search(input_idx)
            model_out = [self.index_2_word[i] for i in model_out]

            result_text = "".join(model_out)

            # 更新history
            history = result_text.split("<sep>")

            print("history:{}".format(history))
            print("Chatbot: {}".format(history[-1]))

    def generator_chat_gradio(self, inputs, history):
        self.model.eval()

        history_texts = []
        for one_history in history:
            for i in one_history:
                history_texts.append(i.strip())
        history_texts.append(inputs.strip())
        print(history_texts)
        # 拼装历史对话
        input_idx = []
        for sentence in history_texts:
            input_idx.extend([self.word_2_index.get(i, 1) for i in sentence])
            input_idx.extend([self.word_2_index["<sep>"]])

        input_idx = torch.tensor([input_idx], device=self.device)

        # method 1 贪婪搜索
        # model_out = self.model.predict_greedy_search(input_idx)
        # model_out = [self.index_2_word[i] for i in model_out]

        # # method 2 随机搜索
        # model_out = model.predict_random_search(input_idx)
        # model_out = [index_2_word[i] for i in model_out]

        # method 3 轮盘赌 搜索
        model_out = self.model.predict_circle_search(input_idx)
        model_out = [self.index_2_word[i] for i in model_out]

        result_text = "".join(model_out)
        # print("Chatbot: {}".format(result_text))
        result_text = result_text.split("<sep>")
        return result_text[-1]

    def chat_by_gradio(self):
        gr.ChatInterface(title="KDDE GPT mini",
                         description="GPT mini 模型，目前仅支持中文聊天",
                         fn=self.generator_chat_gradio
                         ).queue(default_concurrency_limit=10).launch(share=True)


if __name__ == '__main__':
    model_dir = os.path.join('model', "GPTmini-0.1-{}.pth".format(9))

    generator = Inference(model_dir)
    # generator.generator_chat()
    generator.chat_by_gradio()
