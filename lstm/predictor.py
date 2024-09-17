from torch.nn import LSTM, Transformer
from torch import nn
import torch


class Predictor(nn.Module):

    def __init__(self, I, H, LSTM_LAYER, TRANSFORMER_LAYER, HEAD):
        super(Predictor, self).__init__()
        self.lstm = LSTM(I, H, num_layers=LSTM_LAYER, batch_first=True).cuda()
        self.trans = Transformer(
            d_model=H,
            nhead=HEAD,
            num_encoder_layers=TRANSFORMER_LAYER,
            num_decoder_layers=TRANSFORMER_LAYER,
            batch_first=True,
        )
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        o1, (_, _) = self.lstm(x1)
        o2 = self.trans(o1, x2)
        return o1, self.relu(o2)
