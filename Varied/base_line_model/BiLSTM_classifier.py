import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BiLSTMGRUClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pretrained_embeddings, num_layers=2,
                 dropout=0.5):
        super(BiLSTMGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False  # 设置为True以在训练中更新权重
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.bigru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.hidden_dim = hidden_dim

        # Initialize LSTM and GRU weights orthogonally
        self._init_orthogonal(self.bilstm)
        self._init_orthogonal(self.bigru)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_output, _ = self.bilstm(embedded)
        gru_output, _ = self.bigru(lstm_output)
        hidden = torch.cat((gru_output[:, -1, :self.hidden_dim], gru_output[:, 0, self.hidden_dim:]), dim=1)
        return F.log_softmax(self.fc(hidden), dim=-1)

    def _init_orthogonal(self, layer):
        for name, parameter in layer.named_parameters():
            if 'weight_ih' in name:
                init.orthogonal_(parameter.data)
            elif 'weight_hh' in name:
                init.orthogonal_(parameter.data)
            elif 'bias' in name:
                parameter.data.fill_(0)
