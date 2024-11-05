from config import parser
arg = parser.parse_args()

def read_data(path, num=None):
    with open(path, encoding="utf8") as f:
        all_data = f.read().split("\n\n")

    if num:
        return all_data[:num]
    else:
        return all_data[:-1]


def get_word_2_index(path):
    with open(path, encoding="utf8") as f:
        index_2_word = f.read().split("\n")

    word_2_index = {w: i for i, w in enumerate(index_2_word)}

    return index_2_word, word_2_index
