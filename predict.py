import argparse
import json
import os
import re
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import BertTokenizer

from data import MyDataSet
from model import Model
from utils import collate_batch

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--data_path", type=str, default='data/prediction_0717.csv', help="Directory to data.")
parser.add_argument("--save_dir", type=str, default='data/', help="Directory to save model checkpoint and logger.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number of a batch for predicting.")
parser.add_argument("--init_from_ckpt", type=str, default="./epoch_9.pth", help="The path of model to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, device, dataloader, inv_label_dict):
    model.eval()
    results = []
    confs = []
    with torch.no_grad():
        for idx, (input_ids, token_type_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            max_probs = max_probs.cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=-1)
            preds = preds.cpu().numpy().tolist()
            results.extend([inv_label_dict[item] for item in preds])
            confs.extend(max_probs)

    return results, confs


if __name__ == "__main__":
    device = torch.device(args.device)

    tags_file = os.path.join(args.save_dir, "tags.txt")
    with open(tags_file, "r", encoding="utf8") as f:
        label_dict = json.load(f)
        inv_label = {value: key for key, value in label_dict.items()}

    pred_ds = MyDataSet(args.data_path, is_test=True)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    trans_fn = partial(collate_batch, tokenizer=tokenizer, label_dict=label_dict, is_test=True)
    data_loader = DataLoader(pred_ds, shuffle=False, batch_size=args.batch_size, collate_fn=trans_fn)

    ptm = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=len(label_dict))
    model = Model(ptm, num_labels=len(label_dict), pooling='last-avg')
    state_dict = torch.load(args.init_from_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    labels, confs = predict(model, device, data_loader, inv_label)
    # res = []
    # t = classification_report(labels, true_labels)
    # pattern = re.compile(r'(?:e|E)\d+')
    # idx = 0
    # for text_a, text_b, part, ground_truth in pred_ds:
    #     if idx == 2724:
    #         break
    #     label = labels[idx]
    #     if not label.startswith(part):
    #         print(part, label, confs[idx], ground_truth, text_a, text_b)
    #         e_code = pattern.findall(text_b)
    #         if not e_code:
    #             e_code = pattern.findall(text_a)
    #         if not e_code:
    #             print("no ecode")
    #         else:
    #             labels[idx] = part + "-" + e_code[0]
    #
    #     idx += 1
    # t = classification_report(labels, true_labels)
    # print(t)
    # idx = 0
    # from collections import defaultdict
    # part_map = defaultdict(int)
    # label_map = defaultdict(int)
    # for text_a, text_b, part, ground_truth in pred_ds:
    #     if idx == 2724:
    #         break
    #     label = labels[idx]
    #     if label != ground_truth:
    #         print(part, label, confs[idx], ground_truth, text_a, text_b)
    #         part_map[part] += 1
    #         label_map[ground_truth] += 1
    #     idx += 1
    # print(sorted(dict(part_map).items(), key=lambda x: x[1], reverse=True))
    # print(sorted(dict(label_map).items(), key=lambda x: x[1], reverse=True))
    # for idx, data in pred_ds:
    #     part = data.loc[idx]["part"]
    #
    # print(dids)
    #
    pred_ds.data["confidence"] = confs
    pred_ds.data["label"] = labels
    # data = pred_ds.data.drop(labels=["text_a", "text_b"])
    # print(pred_ds.data["Document Number"])
    #
    pred_ds.data.to_csv(args.data_path, index=False, encoding="utf8")
