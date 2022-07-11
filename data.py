import jieba
import pandas as pd
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self, file_path, is_test=False):
        self.is_test = is_test
        self.data = pd.read_csv(file_path)

    def __getitem__(self, idx):
        if self.is_test:
            return self.data.loc[idx]["text_a"], self.data.loc[idx]["text_b"], self.data.loc[idx]["Document Number"]
        else:
            return self.data.loc[idx]["text_a"], self.data.loc[idx]["text_b"],\
                   self.data.loc[idx]["part"], self.data.loc[idx]["label"]

    def __len__(self):
        return self.data.shape[0]
