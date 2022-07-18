import re

import jieba
import pandas as pd
from torch.utils.data import Dataset


class MyDataSet(Dataset):

    def __init__(self, file_path, is_test=False):
        self.is_test = is_test
        self.data = pd.read_csv(file_path)

    def __getitem__(self, idx):
        if self.is_test:
            text_a, text_b, part = self.data.loc[idx]["Defect Found"], self.data.loc[idx]["Work Executed"], self.data.loc[idx]['QM Part Structure Text']
            part = part.strip().lower()
            if not (isinstance(text_a, str)):
                text_a = ""
            if not (isinstance(text_b, str)):
                text_b = ""
            text_a = re.sub(r"sj[0-9A-Za-z]*[-/]*\d*|SJ[0-9A-Za-z]*[-/]*\d*|//\d*-|\d*-\d*-", "", text_a)
            text_b = re.sub(r"</br>|br|\sbr\s|\s</br>\s", "。", text_b)
            return text_a, text_b, part, None
        else:
            return self.data.loc[idx]["text_a"], self.data.loc[idx]["text_b"],\
                   self.data.loc[idx]["part"], self.data.loc[idx]["label"]

    def __len__(self):
        return self.data.shape[0]
