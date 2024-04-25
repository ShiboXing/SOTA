import torch
import torch.nn as nn

class CustomLSTM(nn.module):
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
        stdv = 1.0 / torch.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform(-stdv, stdv)

    def forward(self, inputs, init_states=None):
        # inputs shape is (batch_size, seq_size, feature_length)
        if init_states is None:
            pass

        outputs = []
        for X in inputs:
            pass
            # I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H))
            





