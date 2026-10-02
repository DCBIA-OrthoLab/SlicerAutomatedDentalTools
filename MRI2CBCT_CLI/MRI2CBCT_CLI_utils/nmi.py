import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

# ===== Logging Configuration =====
from ADTLib.logging_setup import get_logger

logger = get_logger("MRI2CBCT_CLI_utils_nmi")

def nmi_gauss(x1, x2, x1_bins, x2_bins, sigma=1e-3, e=1e-10):
    assert x1.shape == x2.shape, "Inputs are not of similar shape"

    def gaussian_window(x, bins, sigma):
        assert x.ndim == 2, "Input tensor should be 2-dimensional."
        return torch.exp(
            -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
        ) / (math.sqrt(2 * math.pi) * sigma)

    x1_windowed = gaussian_window(x1.flatten(1), x1_bins, sigma)
    x2_windowed = gaussian_window(x2.flatten(1), x2_bins, sigma)
    p_xy = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
    p_xy += e  # deal with numerical instability

    p_xy = p_xy / p_xy.sum((1, 2))[:, None, None]

    p_x = p_xy.sum(1)
    p_y = p_xy.sum(2)

    I = (p_xy * torch.log(p_xy / (p_x[:, None] * p_y[:, :, None]))).sum((1, 2))

    marg_ent_0 = (p_x * torch.log(p_x)).sum(1)
    marg_ent_1 = (p_y * torch.log(p_y)).sum(1)

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


def nmi_gauss_mask(x1, x2, x1_bins, x2_bins, mask, sigma=1e-3, e=1e-10):
    def gaussian_window_mask(x, bins, sigma):

        assert x.ndim == 1, "Input tensor should be 2-dimensional."
        return torch.exp(-((x[None, :] - bins[:, None]) ** 2) / (2 * sigma ** 2)) / (
            math.sqrt(2 * math.pi) * sigma
        )

    x1_windowed = gaussian_window_mask(torch.masked_select(x1, mask), x1_bins, sigma)
    x2_windowed = gaussian_window_mask(torch.masked_select(x2, mask), x2_bins, sigma)
    p_xy = torch.mm(x1_windowed, x2_windowed.transpose(0, 1))
    p_xy = p_xy + e  # deal with numerical instability

    p_xy = p_xy / p_xy.sum()

    p_x = p_xy.sum(0)
    p_y = p_xy.sum(1)

    I = (p_xy * torch.log(p_xy / (p_x[None] * p_y[:, None]))).sum()

    marg_ent_0 = (p_x * torch.log(p_x)).sum()
    marg_ent_1 = (p_y * torch.log(p_y)).sum()

    normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1)  # harmonic mean

    return normalized


class NMI(_Loss):
    """Normalized mutual information metric.

    As presented in the work by `De Vos 2020: <https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11313/113130R/Mutual-information-for-unsupervised-deep-learning-image-registration/10.1117/12.2549729.full?SSO=1>`_

    """

    def __init__(
        self,
        intensity_range: Optional[Tuple[float, float]] = None,
        nbins: int = 32,
        sigma: float = 0.1,
        use_mask: bool = False,
    ):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma
        if use_mask:
            self.forward = self.masked_metric
        else:
            self.forward = self.metric

    def metric(self, fixed: Tensor, warped: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss(
            fixed, warped, bins_fixed, bins_warped, sigma=self.sigma
        ).mean()

    def masked_metric(self, fixed: Tensor, warped: Tensor, mask: Tensor) -> Tensor:
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0],
            fixed_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )
        bins_warped = torch.linspace(
            warped_range[0],
            warped_range[1],
            self.nbins,
            dtype=fixed.dtype,
            device=fixed.device,
        )

        return -nmi_gauss_mask(
            fixed, warped, bins_fixed, bins_warped, mask, sigma=self.sigma
        )