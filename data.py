import jieba
import pandas as pd
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self, file_path, vocab, tag_dict):
        self.data = pd.read_csv(file_path)
        self.vocab, self.tag_dict = vocab, tag_dict
        self.unk_id = vocab["[UNK]"]
        self.text_as = []
        self.text_bs = []
        self.labels = []

        for idx, row in self.data.iterrows():
            text_a = []
            for word in jieba.cut(row['text_a']):
                word = word.strip()
                if not word or word not in vocab:
                    continue
                text_a.append(vocab[word])
            self.text_as.append(text_a)

            text_b = []
            for word in jieba.cut(row['text_b']):
                word = word.strip()
                if not word or word not in vocab:
                    continue
                text_b.append(vocab[word])
            self.text_bs.append(text_a)
            self.labels.append(self.tag_dict[row['label']])

    def __getitem__(self, idx):
        return self.text_as[idx], self.text_bs[idx], self.labels[idx]

    def __len__(self):
        return len(self.text_as)
