import torch
import torch.nn as nn

class AWDLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, weight_dropout=0.5, embedding_size=None):
        super(AWDLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if embedding_size is not None:
            self.embedding = nn.Embedding(input_size, embedding_size)
            lstm_input_size = embedding_size
        else:
            self.embedding = None
            lstm_input_size = input_size

        self.lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers, dropout=dropout)

        self.weight_drop = nn.Dropout(weight_dropout)

    def forward(self, x, hidden=None):
        if self.embedding is not None:
            x = self.embedding(x)

        lstm_out, hidden = self.lstm(x, hidden)

        return self.weight_drop(lstm_out), hidden
