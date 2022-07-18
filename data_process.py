import json
import os
import random
import re
from collections import Counter
from collections import defaultdict

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def read_stopword(path):
    stopwords = set()
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            word = line.strip()
            stopwords.add(word)
    return stopwords


def write_vocab(filepath, vocab):
    with open(filepath, "w", encoding="utf8") as f:
        f.write(json.dumps(vocab))


def build_vocab(texts, stopwords=set(), num_words=None, min_freq=5, unk_token="[UNK]", pad_token="[PAD]"):
    """
    According to the texts, it is to build vocabulary.
    Args:
        texts (obj:`List[str]`): The raw corpus data.
        num_words (obj:`int`): the maximum size of vocabulary.
        stopwords (obj:`List[str]`): The list where each element is a word that will be
            filtered from the texts.
        min_freq (obj:`int`): the minimum word frequency of words to be kept.
        unk_token (obj:`str`): Special token for unknow token.
        pad_token (obj:`str`): Special token for padding token.
    Returns:
        word_index (obj:`Dict`): The vocabulary from the corpus data.
    """
    word_counts = defaultdict(int)
    for text in texts:
        if not text:
            continue
        for word in jieba.cut(text):
            word = word.strip()
            if word in stopwords or not word:
                continue
            word_counts[word] += 1

    wcounts = []
    for word, count in word_counts.items():
        if count < min_freq:
            continue
        wcounts.append((word, count))
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # -2 for the pad_token and unk_token which will be added to vocab.
    if num_words is not None and len(wcounts) > (num_words - 2):
        wcounts = wcounts[:(num_words - 2)]
    # add the special pad_token and unk_token to the vocabulary
    sorted_voc = [pad_token, unk_token]
    sorted_voc.extend(wc[0] for wc in wcounts)
    word_index = dict(zip(sorted_voc, list(range(len(sorted_voc)))))
    return word_index


def clean_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Training Data")
    c_dids, cleaned_text_a, cleaned_text_b = [], [], []
    c_raw_text_a, c_raw_text_b = [], []
    l_dids, labeled_a, labeled_b, labels = [], [], [], []
    l_raw_text_a, l_raw_text_b = [], []
    c_parts, l_parts = [], []
    l_tags = []

    for idx, row in data.iterrows():
        text_a = row['Defect Found']
        text_b = row['Work Executed']
        doc_id = row["Document Number"]
        if not (isinstance(text_a, str) and isinstance(text_b, str)):
            continue
        cleaned_a = re.sub(r"sj[0-9A-Za-z]*[-/]*\d*|SJ[0-9A-Za-z]*[-/]*\d*|//\d*-|\d*-\d*-", "", text_a)
        cleaned_b = re.sub(r"</br>|br|\sbr\s|\s</br>\s", "。", text_b)

        part = row['QM Part Structure Text']
        tag = row["FSB-Text"]
        if isinstance(tag, str):
            part = part.strip().lower()
            tag = tag.strip().lower()
            label = part + "-" + tag
            labeled_a.append(cleaned_a)
            labeled_b.append(cleaned_b)
            labels.append(label)
            l_dids.append(doc_id)
            l_raw_text_a.append(text_a)
            l_raw_text_b.append(text_b)
            l_parts.append(part)
            l_tags.append(tag)
        else:
            cleaned_text_a.append(cleaned_a)
            cleaned_text_b.append(cleaned_b)
            c_dids.append(doc_id)
            c_raw_text_a.append(text_a)
            c_raw_text_b.append(text_b)
            c_parts.append(part)

    cleaned_data = pd.DataFrame({
        "Document Number": c_dids,
        'Defect Found': c_raw_text_a,
        "Work Executed": c_raw_text_b,
        'QM Part Structure Text': c_parts,
        'text_a': cleaned_text_a,
        'text_b': cleaned_text_b
    })
    labeled_data = pd.DataFrame({
        "Document Number": l_dids,
        'Defect Found': l_raw_text_a,
        "Work Executed": l_raw_text_b,
        'QM Part Structure Text': l_parts,
        "FSB-Text": l_tags,
        # 'text_a': labeled_a,
        # 'text_b': labeled_b
    })
    return cleaned_data, labeled_data


def check_and_clean_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Training Data")
    tags = pd.read_excel(file_path, sheet_name="label")
    tags1 = set()
    for idx, row in tags.iterrows():
        if not isinstance(row['QM Parts'], str) or not isinstance(row["Defect"], str):
            continue
        part = row['QM Parts'].strip().lower()
        defect = row['Defect'].strip().lower()
        label = part + "-" + defect
        tags1.add(label)

    tags2 = set()
    cleaned_text_a, cleaned_text_b, labels, parts = [], [], [], []
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
        all_data.append((cleaned_a, cleaned_b, part, label))
        labels.append(label)
        parts.append(part)

    assert tags1 == tags2, "The data tag set is inconsistent! Please check the data."
    return cleaned_text_a, cleaned_text_b, labels, parts, all_data, tags2


def write_to_file(save_dir, cleaned_text_a, cleaned_text_b, parts, labels, tag_set):
    d = {"text_a": cleaned_text_a, "text_b": cleaned_text_b, "part": parts, "label": labels}
    df = pd.DataFrame(data=d)
    path = os.path.join(save_dir, "cleaned_data.csv")
    df.to_csv(path, index=False, encoding="utf8")

    tags_file = os.path.join(save_dir, "tags.txt")
    tag_dict = {}
    for idx, tag in enumerate(tag_set):
        tag_dict[tag] = idx
    with open(tags_file, "w", encoding="utf8") as f:
        f.write(json.dumps(tag_dict))


def split_to_train_and_test(all_data, save_dir):

    def write_data(data, file_path):
        text_a, text_b, label, parts = [], [], [], []
        for a, b, p, l in data:
            text_a.append(a)
            text_b.append(b)
            parts.append(p)
            label.append(l)
        data_set = pd.DataFrame({"text_a": text_a, "text_b": text_b, "part": parts, "label": label})
        data_set.to_csv(file_path, index=False, encoding="utf8")

    random.shuffle(all_data)
    num_train_data = int(len(all_data) * 0.8)
    train_data = all_data[:num_train_data]
    test_data = all_data[num_train_data:]
    train_file = os.path.join(save_dir, "all_train.csv")
    write_data(all_data, train_file)
    test_file = os.path.join(save_dir, "test.csv")
    write_data(test_data, test_file)


def oversample(file_path):
    data = pd.read_csv(file_path)
    labels = list(data["label"])
    doc_ids = list(range(data.shape[0]))
    doc_ids = np.array(doc_ids).reshape(-1, 1)
    ros = RandomOverSampler(random_state=100)
    x_resampled, y_resampled = ros.fit_resample(doc_ids, labels)

    aug_data = pd.DataFrame(columns=['text_a', 'text_b', 'part', "label"])
    idx = 0
    for did, label in zip(x_resampled, y_resampled):
        aug_data.loc[idx] = data.loc[did].values.tolist()[0]
        idx += 1
    path = os.path.join(data_dir, "aug_balanced_data.csv")
    aug_data.to_csv(path, index=False, encoding="utf8")


def read_prediction_data(file_path):
    data = data = pd.read_csv(file_path)
    pattern = re.compile(r'(?:e|E)\d+')
    cnt = 0
    ecodes = []
    for idx, row in data.iterrows():
        part = row['QM Part Structure Text']
        if not isinstance(part, str):
            continue
        part = part.strip().lower()

        text_a = row['Defect Found']
        text_b = row['Work Executed']
        if not (isinstance(text_a, str)):
            text_a = ""
        if not (isinstance(text_b, str)):
            text_b = ""
        cleaned_a = re.sub(r"sj[0-9A-Za-z]*[-/]*\d*|SJ[0-9A-Za-z]*[-/]*\d*|//\d*-|\d*-\d*-", "", text_a)
        cleaned_b = re.sub(r"</br>|br|\sbr\s|\s</br>\s", "。", text_b)
        e_code = pattern.findall(cleaned_b)

        if not e_code:
            e_code = pattern.findall(cleaned_a)

        if not e_code:
            ecodes.append("")
            cnt += 1
        else:
            ecodes.append(e_code[0])
            l = list(set([i.lower() for i in e_code]))
            if len(l) != 1:
                print(idx, e_code)
    data['FSB-Text'] = ecodes
    print(cnt, data.shape)
    data.to_csv("./data/prediction_0717_2.csv", index=False, encoding="utf8")


if __name__ == "__main__":
    data_dir = "./data"
    file_path = os.path.join(data_dir, "prediction_0717.csv")
    read_prediction_data(file_path)

    # cleaned_text_a, cleaned_text_b, labels, parts, all_data, tag_set = check_and_clean_data(file_path)
    # file_path = os.path.join(data_dir, "aug_train.csv")
    # oversample(file_path)

    # write_to_file(data_dir, cleaned_text_a, cleaned_text_b, parts, labels, tag_set)
    # split_to_train_and_test(all_data, data_dir)

    # cleaned_text_a.extend(cleaned_text_b)
    # stopwords_file = os.path.join(data_dir, "stopwords.txt")
    # stopwords = read_stopword(stopwords_file)
    # vocab = build_vocab(cleaned_text_a, stopwords)
    # vocab_file = os.path.join(data_dir, "vocab.json")
    # write_vocab(vocab_file, vocab)
    # valid_set = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    # tag_set = set(["lower rack-bearing danaged", "lower rack-clamp damaged/fallen", "lower rack-others",
    #                "lower rack-paint dropping", "lower rack-rack deformation/damaged", "upper rack-clamp damaged",
    #                "upper rack-handle damaged", "upper rack-others", "upper rack-part missing", "upper rack-rack deformation/damaged",
    #                "upper rack-rack rust", "upper rack-wheel damaged", "upper rack-wheel loosen",])
    #
    # text_a, text_b, labels = [], [], []
    # for idx, row in valid_set.iterrows():
    #     if row['label'] in tag_set:
    #         text_a.append(row['text_a'])
    #         text_b.append(row['text_b'])
    #         labels.append(row['label'])
    # data = pd.DataFrame({"text_a": text_a, "text_b": text_b, "label": labels})
    # data.to_csv(os.path.join(data_dir, "analysis_data.csv"))

    # file_path = os.path.join(data_dir, "cleaned_data.csv")
    # data = pd.read_csv(file_path)
    # print(data.shape)
    # # tag_file = os.path.join(data_dir, "tags.txt")
    # # with open(tag_file, "r", encoding="utf8") as f:
    # #     tad_dict = json.load(f)
    #
    # label_num = defaultdict(int)
    # for label in data["label"]:
    #     label_num[label] += 1
    # label_num = sorted(label_num.items(), key=lambda x:x[1], reverse=True)
    # labels = [item[0] for item in label_num if item[1] > 100]
    # cnt = [item[1] for item in label_num if item[1] > 100]
    #
    # # label_ana = pd.DataFrame({"label": labels, "cnt": cnt})
    # # label_ana.to_csv(os.path.join(data_dir, "label_analysis.csv"), index=False, encoding="utf8")
    #
    # plt.figure(figsize=(20, 13))
    # patches,l_text,p_text = plt.pie(cnt, labels=labels, autopct="%0.0f%%", radius=1, )
    # plt.title("Label Analysis", fontsize=28)
    # # plt.legend(loc='center right')
    # plt.axis('equal')
    #
    # # 设置饼图内文字大小
    # for t in p_text:
    #     t.set_size(20)
    #
    # for t in l_text:
    #     t.set_size(20)
    # plt.savefig("./data/label analysis.png")
    # plt.show()
