from torch.utils.data import Dataset, DataLoader
import torch
from config import parser
from torch.utils.data.distributed import DistributedSampler

# loading hyper-para
arg = parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, all_data, word_2_index):
        self.all_data = all_data
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        text_data = self.all_data[index].split("\n")

        text_id = []
        for data in text_data:
            text_id.extend([self.word_2_index.get(i, 1) for i in data])

            text_id.append(self.word_2_index["<sep>"])

        text_id = text_id[:arg.max_len]
        input_data = text_id[:-1]
        output_data = text_id[1:]

        assert len(input_data) == len(output_data)

        return input_data, output_data, len(input_data)

    def __len__(self):
        return len(self.all_data)

    def pro_data(self, batch_data):
        input_data, output_data, input_len = zip(*batch_data)
        max_len = max(input_len)

        new_input_data = []
        new_output_data = []

        for one_input, one_output in zip(input_data, output_data):
            new_input_data.append(one_input + [0] * (max_len - len(one_input)))
            new_output_data.append(one_output + [0] * (max_len - len(one_output)))

        return torch.tensor(new_input_data), torch.tensor(new_output_data)


def get_loader_dp(all_data, word_2_index, mode='train'):
    shuffle = False
    if mode == 'train':
        shuffle = True

    _dataset = MyDataset(all_data, word_2_index)
    data_loader = DataLoader(_dataset, arg.batch_size, shuffle=shuffle, pin_memory=True,
                             collate_fn=_dataset.pro_data)

    return data_loader


def get_loader(all_data, word_2_index):
    _dataset = MyDataset(all_data, word_2_index)
    data_loader = DataLoader(_dataset, arg.batch_size, shuffle=False, pin_memory=True,
                             collate_fn=_dataset.pro_data, sampler=DistributedSampler(_dataset))

    return data_loader
