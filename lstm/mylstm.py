import torch
import torch.nn as nn
import math


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        # input_size = feature_length
        # hidden_size is the # of hidden units in a hidden layer

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM gates parameters
        # Forget gate
        self.W_xf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Input gate
        self.W_xi_lin = nn.Linear(input_size, hidden_size)
        self.W_xi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Input node(c_tilde)
        self.W_xc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate
        self.W_xo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # initialze parameters
        self.init_params()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, init_states=None):
        # inputs shape is (batch_size, seq_size, feature_length)
        if init_states is None:
            h_prev = torch.zeros(
                (inputs.shape[0], self.hidden_size), device=inputs.device
            )
            c_prev = torch.zeros(
                (inputs.shape[0], self.hidden_size), device=inputs.device
            )
        else:
            h_prev, c_prev = init_states

        outputs = []
        for X in inputs:
            f_t = torch.sigmoid(
                torch.matmul(X, self.W_xf) + torch.matmul(h_prev, self.W_hf) + self.b_f
            )
            i_t = torch.sigmoid(
                torch.matmul(X, self.W_xi) + torch.matmul(h_prev, self.W_hi) + self.b_i
            )
            c_tilde_t = torch.tanh(
                torch.matmul(X, self.W_xc) + torch.matmul(h_prev, self.W_hc) + self.b_c
            )
            c_t = f_t * c_prev + i_t * c_tilde_t
            o_t = torch.sigmoid(
                torch.matmul(X, self.W_xo) + torch.matmul(h_prev, self.W_ho) + self.b_o
            )
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)

        return torch.concat(outputs, axis=0), (h_t, c_t)
