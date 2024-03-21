import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMGRUClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super(BiLSTMGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.bigru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        # LSTM
        lstm_output, _ = self.bilstm(embedded)

        # GRU
        gru_output, _ = self.bigru(lstm_output)

        # Concatenate the final hidden state from both directions
        hidden = torch.cat((gru_output[:, -1, :self.hidden_dim], gru_output[:, 0, self.hidden_dim:]), dim=1)

        # Classifier
        return F.log_softmax(self.fc(hidden), dim=-1)
