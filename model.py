import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class BiLSTM(nn.Module):

    def __init__(self,
                 vocab_size,
                 num_classes,
                 embedding_dim=128,
                 lstm_units=198,
                 lstm_layers=2,
                 hidden_dim=96,
                 padding_idx=0):
        super(BiLSTM, self).__init__()

        self.embedder = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        # bidirectional
        num_directions = 2
        self.fc = nn.Linear(lstm_units * num_directions, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, text_lengths):
        embedded = self.embedder(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed_embedded)
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]
        out = torch.relu(out)
        dense = torch.tanh(self.fc(out))
        logits = self.classifier(dense)
        probs = torch.softmax(logits, dim=1)
        return probs


class Similarity(nn.Module):

    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Model(nn.Module):

    def __init__(self, pretrained_model, num_labels, pooling='cls'):
        super(Model, self).__init__()
        self.ptm = pretrained_model
        self.pooling = pooling
        self.classifier = nn.Linear(self.ptm.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.ptm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            pooled_output = out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            pooled_output = out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pooled_output = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        logits = self.classifier(pooled_output)
        return logits
