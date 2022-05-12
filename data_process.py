import json
import random
import re

import numpy as np
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "/Users/xuefegzh/Desktop/program/bsh_washingmachine/data/Classification_2210.xlsx"

data = pd.read_excel(file_path, sheet_name="数据")
tags = pd.read_excel(file_path, sheet_name="分类")
tags1 = set()
for idx, row in tags.iterrows():
    if not isinstance(row['QM Parts'], str) or not isinstance(row["Defect"], str):
        continue
    part = row['QM Parts'].strip().lower()
    defect = row['Defect'].strip().lower()
    label = part + "-" + defect
    tags1.add(label)

tags2 = set()
cleaned_text_a, cleaned_text_b, labels = [], [], []
all_data = []
for idx, row in data.iterrows():
    part = row['QM Part Structure Text']
    tag = row["FSB-Text"]
    if not (isinstance(part, str) and isinstance(tag, str)):
        continue
    part = part.strip().lower()
    tag = tag.strip().lower()
    label = part + "-" + tag
    tags2.add(label)

    text_a = row['Defect Found']
    text_b = row['Work Executed']
    if not (isinstance(text_a, str) and isinstance(text_b, str)):
        continue
    cleaned_a = re.sub(r"sj[0-9A-Za-z]*[-/]*\d*|SJ[0-9A-Za-z]*[-/]*\d*|//\d*-|\d*-\d*-", "", text_a)
    cleaned_b = re.sub(r"</br>|br|\sbr\s|\s</br>\s", "。", text_b)
    cleaned_text_a.append(cleaned_a)
    cleaned_text_b.append(cleaned_b)
    all_data.append((cleaned_a, cleaned_b, label))
    labels.append(label)

d = {"text_a": cleaned_text_a, "text_b": cleaned_text_b, "label": labels}
df = pd.DataFrame(data=d)
path = "/Users/xuefegzh/Desktop/program/bsh_washingmachine/data/cleaned_data.csv"
df.to_csv(path, index=False, encoding="utf8")

tags_file = "/Users/xuefegzh/Desktop/program/bsh_washingmachine/data/tags.txt"
tag_dict = {}
for idx, tag in enumerate(tags2):
    tag_dict[tag] = idx
with open(tags_file, "w", encoding="utf8") as f:
    f.write(json.dumps(tag_dict))

random.shuffle(all_data)
num_train_data = int(len(all_data) * 0.8)
train_data = all_data[:num_train_data]
test_data = all_data[num_train_data:]
train_file = "/Users/xuefegzh/Desktop/program/bsh_washingmachine/data/train.csv"
text_a, text_b, label = [], [], []
for a, b, l in train_data:
    text_a.append(a)
    text_b.append(b)
    label.append(l)
train_set = pd.DataFrame({"text_a": text_a, "text_b": text_b, "label": label})
train_set.to_csv(train_file, index=False, encoding="utf8")

text_a, text_b, label = [], [], []
for a, b, l in test_data:
    text_a.append(a)
    text_b.append(b)
    label.append(l)
test_file = "/Users/xuefegzh/Desktop/program/bsh_washingmachine/data/dev.csv"
test_set = pd.DataFrame({"text_a": text_a, "text_b": text_b, "label": label})
test_set.to_csv(test_file, index=False, encoding="utf8")
