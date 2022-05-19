import argparse
import json
import os
import time
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from data import MyDataSet
from utils import collate_batch

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=50, help="Number of epoches for training.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_prop", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate used to train.")
parser.add_argument("--data_dir", type=str, default='data/', help="Directory to data.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint and logger.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number of a batch for training.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument('--seed', type=int, default=1000, help='random seed (default: 1000)')
args = parser.parse_args()
# yapf: enable


def train(device, dataloader, model, optimizer, scheduler, criterion, epoch, inv_label, writer):
    model.train()
    all_preds, all_labels = [], []
    total_acc, total_count = 0, 0
    total_loss = 0
    log_interval = 10

    global_step = epoch * len(dataloader)
    for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader):
        global_step += 1
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)["logits"]
        preds = torch.argmax(logits, dim=-1)
        preds = preds.cpu().numpy().tolist()
        preds = [inv_label[index] for index in preds]
        true_labels = [inv_label[index] for index in labels.cpu().numpy().tolist()]
        all_preds.extend(preds)
        all_labels.extend(true_labels)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_acc += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            print("| epoch {:3d} | {:5d}/{:5d} batches "
                  "| accuracy {:8.3f}".format(epoch, idx, len(dataloader), total_acc / total_count))
            writer.add_scalar("loss", total_loss / total_count, global_step)
            writer.add_scalar("train_acc", total_acc / total_count, global_step)
            t = classification_report(all_labels, all_preds)
            print(t)

            total_acc, total_count = 0, 0
            all_preds, all_labels = [], []


def evaluate(device, dataloader, model, inv_label):
    model.eval()
    total_acc, total_count = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)["logits"]
            preds = torch.argmax(logits, dim=-1)
            preds = preds.cpu().numpy().tolist()
            preds = [inv_label[index] for index in preds]
            true_labels = [inv_label[index] for index in labels.cpu().numpy().tolist()]
            all_preds.extend(preds)
            all_labels.extend(true_labels)
            total_acc += (logits.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
        t = classification_report(all_labels, all_preds)
        print(t)
    return total_acc / total_count


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    local_rank = os.environ['LOCAL_RANK']
    dist.init_process_group(backend="nccl")
    device = torch.device(args.device)

    tags_file = os.path.join(args.data_dir, "tags.txt")
    with open(tags_file, "r", encoding="utf8") as f:
        label_dict = json.load(f)
        inv_label = {value: key for key, value in label_dict.items()}

    train_file = os.path.join(args.data_dir, "train.csv")
    train_ds = MyDataSet(train_file)
    dev_file = os.path.join(args.data_dir, "dev.csv")
    dev_ds = MyDataSet(dev_file)

    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=len(label_dict))
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    trans_fn = partial(collate_batch, tokenizer=tokenizer, label_dict=label_dict)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=32, collate_fn=trans_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=True, collate_fn=trans_fn)

    model = model.to(device)
    model = nn.DataParallel(model)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    total_steps = args.epochs * len(train_loader)
    warmup_steps = total_steps * args.warmup_prop
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    log_dir = os.path.join(args.save_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    for i in range(args.epochs):
        epoch_start_time = time.time()
        train(device, train_loader, model, optimizer, scheduler, criterion, i, inv_label, writer)
        if local_rank == 0:
            accu_val = evaluate(device, dev_loader, model, inv_label)
            print("-" * 59)
            print("| end of epoch {:3d} | time: {:5.2f}s | "
                  "valid accuracy {:8.3f} ".format(i,
                                                   time.time() - epoch_start_time, accu_val))
            writer.add_scalar("test_acc", accu_val, (i + 1) * len(train_loader))
            print("-" * 59)
            path = os.path.join(args.save_dir, "epoch_" + str(i) + ".pth")
            torch.save(model.module.state_dict(), path)
