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
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed_embedded)
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]
        out = torch.relu(out)
        dense = torch.tanh(self.fc(out))
        logits = self.classifier(dense)
        probs = torch.softmax(logits, dim=1)
        return probs
