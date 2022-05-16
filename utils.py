import json
from collections import defaultdict

import jieba


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


def preprocess_data(data, vocab, label_dict):
    text_as, text_bs, labels = [], [], []
    unk_id = vocab["[UNK]"]
    for idx, row in data.iterrows():
        text_a = []
        for word in jieba.cut(row["text_a"]):
            word = word.strip()
            if not word or word not in vocab:
                continue
            text_a.append(word)
        text_as.append(text_a)

        text_b = []
        for word in jieba.cut(row["text_b"]):
            word = word.strip()
            if not word:
                continue
            text_b.append(word)
        text_bs.append(text_b)
        labels.append(row['label'])

    return text_as, text_bs, labels
