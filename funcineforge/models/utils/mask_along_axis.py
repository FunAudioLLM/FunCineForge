import torch
from typing import Sequence
from typing import Union


class MaskTailVariableMaxWidth(torch.nn.Module):
    def __init__(
            self,
            mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
            replace_value: float = 0.0,
    ):
        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.replace_value = replace_value

    def extra_repr(self):
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, "
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        bb, tt, _ = spec.shape

        mask_width_ratio = torch.rand((bb, 1), device=spec.device)
        ratio_st, ratio_ed = self.mask_width_ratio_range
        mask_width_ratio = mask_width_ratio * (ratio_ed - ratio_st) + ratio_st
        mask_length = (mask_width_ratio * spec_lengths.unsqueeze(1)).to(spec_lengths)

        # mask_pos: (B, 1)
        mask_start_pos = spec_lengths.unsqueeze(-1) - mask_length

        aran = torch.arange(tt, device=spec.device)[None, :]
        # mask: (Batch, L)
        mask = aran < mask_start_pos
        # (Batch, L) -> (Batch, L, 1)
        mask = mask.unsqueeze(2)

        return mask

class PrefixMaskVariableMaxWidth(torch.nn.Module):
    def __init__(
            self,
            mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
            replace_value: float = 0.0,
    ):
        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.replace_value = replace_value

    def extra_repr(self):
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, "
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None, return_mask: bool = False):
        bb, tt, _ = spec.shape

        mask_width_ratio_range = torch.tensor(self.mask_width_ratio_range, dtype=torch.float32, device=spec.device)
        mask_width_range = (mask_width_ratio_range * tt).long()
        mask_length = torch.randint(
            mask_width_range[0],
            mask_width_range[1],
            (bb, 1),
            device=spec.device,
        ).unsqueeze(2)

        # mask_pos: (B, num_mask, 1)
        mask_pos = tt - mask_length

        aran = torch.arange(tt, device=spec.device)[None, None, :]
        # mask: (Batch, num_mask, L)
        mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
        # Multiply masks: (Batch, num_mask, L) -> (Batch, L, 1)
        mask = mask.any(dim=1).unsqueeze(2)

        return mask
