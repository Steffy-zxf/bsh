import argparse
import json
import os
import time
from functools import partial

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import MyDataSet
from model import BiLSTM
from utils import build_vocab
from utils import read_stopword
from utils import write_vocab

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=15, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--data_dir", type=str, default='data/', help="Directory to data.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def collate_batch(batch, padding_idx=0):
    texts, labels = [], []
    seq_lens = []
    for text_a, text_b, label in batch:
        texts.append(text_a + text_b)
        seq_lens.append(len(text_a) + len(text_b))
        labels.append(label)
    batch_max_len = max(seq_lens)
    for idx, text in enumerate(texts):
        if len(text) < batch_max_len:
            texts[idx].extend([padding_idx] * (batch_max_len - len(text)))
    texts = torch.tensor(texts)
    labels = torch.tensor(labels)
    seq_lens = torch.tensor(seq_lens)
    return texts, seq_lens, labels


def train(dataloader, model, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 10

    for idx, (texts, seq_lens, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        probs = model(texts, seq_lens)
        loss = criterion(probs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (probs.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            print("| epoch {:3d} | {:5d}/{:5d} batches "
                  "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count))
            total_acc, total_count = 0, 0


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (texts, seq_lens, labels) in enumerate(dataloader):
            probs = model(texts, seq_lens)
            total_acc += (probs.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    raw_data_path = os.path.join(args.data_dir, "cleaned_data.csv")
    data = pd.read_csv(raw_data_path)
    texts = data['text_a'].tolist()
    texts.extend(data['text_b'].tolist())

    stopwords_path = os.path.join(args.data_dir, "stopwords.txt")
    stopwords = read_stopword(stopwords_path)

    tags_file = os.path.join(args.data_dir, "tags.txt")
    with open(tags_file, "r", encoding="utf8") as f:
        label_dict = json.load(f)

    vocab = build_vocab(texts)
    vocab_file = os.path.join(args.data_dir, "vocab.json")
    write_vocab(vocab_file, vocab)

    train_file = os.path.join(args.data_dir, "train.csv")
    train_ds = MyDataSet(train_file, vocab, label_dict)
    dev_file = os.path.join(args.data_dir, "dev.csv")
    dev_ds = MyDataSet(dev_file, vocab, label_dict)

    trans_fn = partial(collate_batch, padding_idx=vocab["[PAD]"])
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=32, collate_fn=trans_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=True, collate_fn=trans_fn)

    model = BiLSTM(len(vocab), len(label_dict), padding_idx=vocab["[PAD]"])
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    num_epoch = 200
    for i in range(num_epoch):
        epoch_start_time = time.time()
        train(train_loader, model, optimizer, criterion, epoch=i)
        if local_rank == 0:
            accu_val = evaluate(dev_loader, model)
            print("-" * 59)
            print("| end of epoch {:3d} | time: {:5.2f}s | "
                  "valid accuracy {:8.3f} ".format(i,
                                                   time.time() - epoch_start_time, accu_val))
            print("-" * 59)
            path = os.path.join(args.save_dir, "epoch_" + str(i) + ".pth")
            torch.save(model.state_dict(), path)
