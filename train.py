import argparse
import json
import os
import time
from functools import partial

import pandas as pd
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from data import MyDataSet
from model import BiLSTM
from utils import build_vocab
from utils import read_stopword
from utils import write_vocab

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=200, help="Number of epoches for training.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_prop", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate used to train.")
parser.add_argument("--data_dir", type=str, default='data/', help="Directory to data.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=16, help="Total examples' number of a batch for training.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument('--seed', type=int, default=1000, help='random seed (default: 1000)')
args = parser.parse_args()
# yapf: enable


def collate_batch(batch, tokenizer, label_dict):
    text_a = [item[0] for item in batch]
    text_b = [item[1] for item in batch]
    labels = torch.tensor([label_dict[item[2]] for item in batch])

    encoded_inputs = tokenizer(text=text_a, text_pair=text_b, padding=True, return_tensors="pt")
    labels = torch.tensor(labels)
    return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], encoded_inputs["attention_mask"], labels

    # texts, labels = [], []
    # seq_lens = []
    # for text_a, text_b, label in batch:
    #     texts.append(text_a + text_b)
    #     seq_lens.append(len(text_a) + len(text_b))
    #     labels.append(label)
    # batch_max_len = max(seq_lens)
    # for idx, text in enumerate(texts):
    #     if len(text) < batch_max_len:
    #         texts[idx].extend([padding_idx] * (batch_max_len - len(text)))
    # texts = torch.tensor(texts)
    # labels = torch.tensor(labels)
    # seq_lens = torch.tensor(seq_lens)
    # return texts, seq_lens, labels


def train(device, dataloader, model, optimizer, scheduler, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 10

    for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)["logits"]
        probs = torch.softmax(logits, dim=-1)
        loss = criterion(probs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        total_acc += (probs.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            print("| epoch {:3d} | {:5d}/{:5d} batches "
                  "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count))
            total_acc, total_count = 0, 0


def evaluate(
    device,
    dataloader,
    model,
):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(input_ids, attention_mask, token_type_ids)["logits"]
            probs = torch.softmax(logits, dim=-1)
            total_acc += (probs.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl")
    device = torch.device(args.device)

    # raw_data_path = os.path.join(args.data_dir, "cleaned_data.csv")
    # data = pd.read_csv(raw_data_path)
    # texts = data['text_a'].tolist()
    # texts.extend(data['text_b'].tolist())

    stopwords_path = os.path.join(args.data_dir, "stopwords.txt")
    stopwords = read_stopword(stopwords_path)

    tags_file = os.path.join(args.data_dir, "tags.txt")
    with open(tags_file, "r", encoding="utf8") as f:
        label_dict = json.load(f)

    # vocab = build_vocab(texts)
    vocab_file = os.path.join(args.data_dir, "vocab.json")
    # write_vocab(vocab_file, vocab)

    with open(vocab_file, "r", encoding="utf8") as f:
        vocab = json.load(f)

    train_file = os.path.join(args.data_dir, "train.csv")
    train_ds = MyDataSet(train_file)  #, vocab, label_dict)
    dev_file = os.path.join(args.data_dir, "dev.csv")
    dev_ds = MyDataSet(dev_file)  #, vocab, label_dict)

    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=len(label_dict))
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    trans_fn = partial(collate_batch, tokenizer=tokenizer, label_dict=label_dict)
    # train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=32, collate_fn=trans_fn)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=True, collate_fn=trans_fn)

    # model = BiLSTM(len(vocab), len(label_dict), padding_idx=vocab["[PAD]"])
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = total_steps * args.warmup_prop
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,  # Default value in run_glue.py
        num_training_steps=total_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    for i in range(args.epochs):
        epoch_start_time = time.time()
        train(device, train_loader, model, optimizer, scheduler, criterion, epoch=i)
        if local_rank == 0:
            accu_val = evaluate(device, dev_loader, model)
            print("-" * 59)
            print("| end of epoch {:3d} | time: {:5.2f}s | "
                  "valid accuracy {:8.3f} ".format(i,
                                                   time.time() - epoch_start_time, accu_val))
            print("-" * 59)
            path = os.path.join(args.save_dir, "epoch_" + str(i) + ".pth")
            torch.save(model.state_dict(), path)
