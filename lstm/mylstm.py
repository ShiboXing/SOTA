import torch
import torch.nn as nn


class LSTM_Cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size = feature_length
        # hidden_size is the # of hidden units in a hidden layer

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM gates parameters
        # Forget gate
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size)

        # Input gate
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size)

        # Input node(c_tilde)
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)

        # Output gate
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size)


    def forward(self, inputs):
        # inputs shape is (batch_size, seq_size, feature_length)
        X, (h_prev, c_prev) = inputs

        outputs = []
        f_t = torch.sigmoid(self.W_xf(X) + self.W_hf(h_prev))
        i_t = torch.sigmoid(self.W_xi(W) + self.W_hi(h_prev))
        c_tilde_t = torch.tanh(self.W_xc(X) + self.W_hc(h_prev))
        c_t = f_t * c_prev + i_t * c_tilde_t
        o_t = torch.sigmoid(self.W_xo(X) + self.W_ho(h_prev))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstms = nn.ModuleList(
            (LSTM_Cell(hidden_size, hidden_size) if i else LSTM_Cell(input_size, hidden_size) for i in range(layer_num))
        )

    def forward(self, inputs):
        """
        inputs (BS, Seq, in_len)
        """
        device = inputs.device
        h_prev = torch.zeros(
            (inputs.shape[0], inputs.shape[1], self.hidden_size), device=device
        )
        c_prev = torch.zeros(
            (inputs.shape[0], inputs.shape[1], self.hidden_size), device=device
        )

        for i in range(inputs.shape[0]):
            X = inputs[:, i, :]
            for lstm in self.lstms:
                h_prev, c_prev = lstm(X, (h_prev, c_prev))
                inputs = h_prev

        return inputs, (h_prev, c_prev)