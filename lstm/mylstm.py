import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ipdb import set_trace
from lstm_cell import lstm_cell_act_forward


class LSTM_Cell(nn.Module):

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def __init__(self, input_size, hidden_size, use_ext=False):
        super().__init__()
        self.use_ext = use_ext
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Forget gate
        self.Wi = nn.Parameter(torch.empty(hidden_size * 4, input_size))
        self.Wh = nn.Parameter(torch.empty(hidden_size * 4, hidden_size))
        self.bi = nn.Parameter(torch.empty(hidden_size * 4))
        self.bh = nn.Parameter(torch.empty(hidden_size * 4))

        self.reset_parameters()

    def forward(self, inputs):
        # inputs shape is (batch_size, 1, feature_length)
        X, (h_prev, c_prev) = inputs
        gates = F.linear(X, self.Wi, self.bi) + F.linear(h_prev, self.Wh, self.bh)
        # gates = torch.mm(X, self.Wi.t()) + torch.mm(h_prev, self.Wh.t()) + self.bi + self.bh

        if self.use_ext:
            return lstm_cell_act_forward(gates, c_prev)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 2)
        i_gate = torch.sigmoid(i_gate)  # input gate
        f_gate = torch.sigmoid(f_gate)  # forget gate
        c_gate = torch.tanh(c_gate)  # cell gate
        o_gate = torch.sigmoid(o_gate)  # output gate

        c_next = f_gate * c_prev + i_gate * c_gate

        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, use_ext=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstms = nn.ModuleList(
            (
                (
                    LSTM_Cell(hidden_size, hidden_size, use_ext=use_ext)
                    if i
                    else LSTM_Cell(input_size, hidden_size, use_ext=use_ext)
                )
                for i in range(layer_num)
            )
        )

    def forward(self, inputs):
        """
        inputs (BS, Seq, in_len)
        """
        bs, seq_len, _ = inputs.shape
        device = inputs.device
        outputs, h_outputs, c_outputs = [], [], []

        # layer by layer
        for lstm in self.lstms:
            h_prev = torch.zeros(
                (bs, 1, self.hidden_size), requires_grad=False, device=device
            ).to(inputs.device)
            c_prev = torch.zeros(
                (bs, 1, self.hidden_size), requires_grad=False, device=device
            ).to(inputs.device)

            # token by token
            for i in range(seq_len):
                X = inputs[:, [i], :]
                h_prev, c_prev = lstm((X, (h_prev, c_prev)))
                outputs.append(h_prev)

            h_outputs.append(h_prev)
            c_outputs.append(c_prev)
            inputs = torch.concat(outputs, dim=1)
            outputs = []

        return inputs, (
            torch.concat(h_outputs, dim=1).permute(1, 0, 2),
            torch.concat(c_outputs, dim=1).permute(1, 0, 2),
        )
