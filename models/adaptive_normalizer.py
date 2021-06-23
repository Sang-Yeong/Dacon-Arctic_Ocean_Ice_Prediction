import torch


class AdaptiveNormalizer:
    def __init__(self, output_dim, seasonality=''):
        self.output_dim = output_dim
        self.seasonality = seasonality
        self.min_max = []

    def norm(self, x):
        """
        Normalizes input tensor.

        :param torch.tensor input_tensor: (T, M, N, D)
        :return: normalized tensor
        :rtype: torch.Tensor
        """
        batch_size, seq_len, height, width, d_dim = x.shape
        out = []
        for d in range(d_dim):
            a = x[..., d]
            aa = a.contiguous().view(a.size(0), -1)

            min_a = aa.min(dim=1, keepdim=True)[0]
            aa -= min_a

            max_a = aa.max(dim=1, keepdim=True)[0]
            aa /= max_a

            aa = aa.view(batch_size, seq_len, height, width)
            out.append(aa)
            self.min_max.append((min_a, max_a))

        out = torch.stack(out, dim=-1)

        return out

    def inv_norm(self, x, device):
        """
        Take inverse normalization of the input tensor.

        :param torch.tensor input_tensor: (b, t, d, m, n)
        :return:
        """
        x = x.permute(0, 1, 3, 4, 2)
        batch_size, seq_len, height, width, d_dim = x.shape
        x = x.contiguous().view(x.size(0), -1)
        min_, max_ = self.min_max[self.output_dim]
        x = x * max_.to(device) + min_.to(device)
        x = x.view(batch_size, seq_len, height, width, d_dim)
        x = x.permute(0, 1, 4, 2, 3)
        return x
