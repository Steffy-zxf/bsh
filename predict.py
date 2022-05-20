import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from data import MyDataSet
from utils import collate_batch

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'cuda'], default="cuda", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--data_path", type=str, default='data/unlabeled_data.csv', help="Directory to data.")
parser.add_argument("--save_dir", type=str, default='data/', help="Directory to save model checkpoint and logger.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number of a batch for predicting.")
parser.add_argument("--init_from_ckpt", type=str, default="./ckpt_epoch50_LR5e-5/epoch_49.pth", help="The path of model to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, device, dataloader, inv_label_dict):
    model.eval()
    results = []
    confs = []
    dids = []
    with torch.no_grad():
        for idx, (input_ids, token_type_ids, attention_mask, doc_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask, token_type_ids)["logits"]
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            probs = probs.cpu().numpy().tolist()
            max_probs = max_probs.cpu().numpy().tolist()
            preds = torch.argmax(logits, dim=-1)
            preds = preds.cpu().numpy().tolist()
            results.extend([inv_label_dict[item] for item in preds])
            confs.extend(max_probs)
            dids.extend(doc_ids)

    return results, confs, dids


if __name__ == "__main__":
    device = torch.device(args.device)

    tags_file = os.path.join(args.save_dir, "tags.txt")
    with open(tags_file, "r", encoding="utf8") as f:
        label_dict = json.load(f)
        inv_label = {value: key for key, value in label_dict.items()}

    pred_ds = MyDataSet(args.data_path, is_test=True)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    trans_fn = partial(collate_batch, tokenizer=tokenizer, is_test=True)
    data_loader = DataLoader(pred_ds, shuffle=False, batch_size=args.batch_size, collate_fn=trans_fn)

    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=len(label_dict))
    model = nn.DataParallel(model)
    state_dict = torch.load(args.init_from_ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    labels, confs, dids = predict(model, device, data_loader, inv_label)
    print(dids)

    save_path = os.path.join(args.save_dir, "predictions.csv")
    pred_ds.data["confidence"] = confs
    pred_ds.data["label"] = labels
    data = pred_ds.data.drop(labels=["text_a", "text_b"])
    print(pred_ds.data["Document Number"])

    pred_ds.data.to_csv(save_path, index=False, encoding="utf8")
