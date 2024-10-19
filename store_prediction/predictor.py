from torch.nn import LSTM, Transformer
from torch import nn
from typing import Tuple
import torch


class Predictor(nn.Module):

    def __init__(self, I1, I2, H, LSTM_LAYER, TRANSFORMER_LAYER, HEAD, max_seq):
        super(Predictor, self).__init__()
        # use H*2 to include the positional embedding
        self.lstm1 = LSTM(I1 * 2 + I2 * 2, H, num_layers=LSTM_LAYER, batch_first=True)
        self.lstm2 = LSTM(I1 * 2 + I2 * 2, H, num_layers=LSTM_LAYER, batch_first=True)

        # self.trans = Transformer(
        #     d_model=H * 2,
        #     nhead=HEAD,
        #     num_encoder_layers=TRANSFORMER_LAYER,
        #     num_decoder_layers=TRANSFORMER_LAYER,
        #     batch_first=True,
        # )
        # self.relu = nn.ReLU()
        self.freqs1 = self.precompute_freqs_cis(I1, 365)
        self.freqs2 = self.precompute_freqs_cis(I2, 365)

        self.freqs1 = torch.concat(
            [self.freqs1, self.freqs1]
        )  # calculate cross-year encodings
        self.freqs2 = torch.concat(
            [self.freqs2, self.freqs2]
        )  # calculate cross-year encodings
        self.max_seq = max_seq

    def forward(self, x1, x2, day_info: Tuple):
        """
        x1, x2: (B, seq_len, feature)
        day_info: (start_day, end_day)
        """
        # use the absolute slice of dates within a year to encode inputs
        # _x1, _x2 = x1, x2
        _x1 = self.apply_rotary_emb(x1, self.freqs1, day_info[0])
        _x2 = self.apply_rotary_emb(x2, self.freqs2, day_info[0])
        # _x = torch.stack((_x1, _x2), dim=-1).reshape(
        #     _x1.shape[:-1] + (_x1.shape[-1] + _x2.shape[-1],)
        # )  # stack and interleave the feature dimension
        _x = torch.concat((_x1, _x2), dim=-1)
        o1, (_, _) = self.lstm1(_x)
        o2, (_, _) = self.lstm2(_x)
        # o3 = self.trans(_x1, _x2)

        return o1, o2
        # return torch.nan_to_num(o1), torch.nan_to_num(o3)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
        and the end index 'end'. The 'theta' parameter scales the frequencies.
        The returned tensor contains complex values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex exponentials.
        """
        device = torch.device("cuda")  # next(self.lstm.parameters())[0].device
        freqs = 1.0 / (
            theta
            ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(end, device=device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
        for the purpose of broadcasting the frequency tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected shape.
            AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor.

        This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
        frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
        returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        assert x.shape[0] == len(start)
        # x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        tmp_freqs = []
        for i in range(x.shape[0]):
            tmp_freqs.append(freqs_cis[start[i] : start[i] + self.max_seq].unsqueeze(0))
        freqs_cis = torch.concat(tmp_freqs)
        # freqs_cis = Predictor.reshape_for_broadcast(freqs_cis, xq_)
        # xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        # xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        freqs_cis = torch.view_as_real(freqs_cis).flatten(2)
        x_out = torch.concat((freqs_cis, x), axis=2)

        return x_out.type_as(x)
