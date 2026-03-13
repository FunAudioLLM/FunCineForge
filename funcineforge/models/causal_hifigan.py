# Copyright 2023 KaiHu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HIFI-GAN"""

from typing import Dict
from typing import Tuple, List

import numpy as np
from scipy.signal import get_window
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.nn.utils.parametrizations import weight_norm
import logging
from funcineforge.register import tables
from funcineforge.utils.device_funcs import to_device
import os
from torch.nn.utils.rnn import pad_sequence
from funcineforge.models.utils import dtype_map
from funcineforge.models.modules.hifigan import init_weights
from funcineforge.models.modules.hifigan.activations import Snake


class LookRightConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(LookRightConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.size(2) == 0:
            x = F.pad(x, (0, self.causal_padding), value=0.0)
        else:
            assert context.size(2) == self.causal_padding
            x = torch.concat([x, context], dim=2)
        x = super(LookRightConv1d, self).forward(x)
        return x

class LookLeftConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(LookLeftConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1 and dilation == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        # NOTE 兼容kernel_size=1的情况
        if self.causal_padding == 0:
            cache_new = x[:, :, :0]
        else:
            cache_new = x[:, :, -self.causal_padding:]
        x = super(LookLeftConv1d, self).forward(x)
        return x, cache_new


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self,
                 num_class: int = 1,
                 in_channels: int = 80,
                 cond_channels: int = 512
                 ):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                LookRightConv1d(in_channels, cond_channels, kernel_size=4)
            ),
            nn.ELU(),
            weight_norm(
                LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)
            ),
            nn.ELU(),
            weight_norm(
                LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)
            ),
            nn.ELU(),
            weight_norm(
                LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)
            ),
            nn.ELU(),
            weight_norm(
                LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0, 0), finalize: bool = True) -> torch.Tensor:
        if finalize is False:
            x, context = x[:, :, :-self.condnet[0].causal_padding], x[:, :, -self.condnet[0].causal_padding:]
        else:
            x, context = x, x[:, :, :0]
        x = self.condnet[0](x, context)
        x = self.condnet[1](x)
        if cache.size(0) != 0:
            x, cache[0] = self.condnet[2](x, cache[0])
        else:
            x, _ = self.condnet[2](x)
        x = self.condnet[3](x)
        if cache.size(0) != 0:
            x, cache[1] = self.condnet[4](x, cache[1])
        else:
            x, _ = self.condnet[4](x)
        x = self.condnet[5](x)
        if cache.size(0) != 0:
            x, cache[2] = self.condnet[6](x, cache[2])
        else:
            x, _ = self.condnet[6](x)
        x = self.condnet[7](x)
        if cache.size(0) != 0:
            x, cache[3] = self.condnet[8](x, cache[3])
        else:
            x, _ = self.condnet[8](x)
        x = self.condnet[9](x)
        x = x.transpose(1, 2)
        x = torch.abs(self.classifier(x).squeeze(-1))
        return x, cache

    def init_cache(self, device):
        return torch.zeros(4, 1, 512, 2).to(device)

    def remove_weight_norm(self):
        print('Removing weight norm...')
        try:
            remove_weight_norm(self.condnet[0])
            remove_weight_norm(self.condnet[2])
            remove_weight_norm(self.condnet[4])
            remove_weight_norm(self.condnet[6])
            remove_weight_norm(self.condnet[8])
        except:
            remove_parametrizations(self.condnet[0], 'weight')
            remove_parametrizations(self.condnet[2], 'weight')
            remove_parametrizations(self.condnet[4], 'weight')
            remove_parametrizations(self.condnet[6], 'weight')
            remove_parametrizations(self.condnet[8], 'weight')


class LookLeftConvTranspose1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(LookLeftConvTranspose1d, self).__init__(in_channels, out_channels,
                                           kernel_size, 1,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert dilation == 1 and stride != 1
        self.causal_padding = kernel_size - 1
        self.upsample = torch.nn.Upsample(scale_factor=stride, mode='nearest')

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.upsample(x)
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding:]
        x = super(LookLeftConvTranspose1d, self).forward(x)
        return x, cache_new


class LookLeftConv1dWithStride(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(LookLeftConv1dWithStride, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding:]
        x = super(LookLeftConv1dWithStride, self).forward(x)
        return x, cache_new


class LookLeftConv1dWithDilation(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(LookLeftConv1dWithDilation, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        # NOTE(lyuxiang.lx) 这个causal_padding仅在kernel_size为奇数时才成立
        assert kernel_size // 2 * dilation * 2 == int((kernel_size * dilation - dilation) / 2) * 2
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0)) -> Tuple[torch.Tensor, torch.Tensor]:
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding:]
        x = super(LookLeftConv1dWithDilation, self).forward(x)
        return x, cache_new


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""
    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
    ):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    LookLeftConv1dWithDilation(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation
                    ) if dilation != 1 else
                    LookLeftConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    LookLeftConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1
                    )
                )
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs1))
        ])
        self.activations2 = nn.ModuleList([
            Snake(channels, alpha_logscale=False)
            for _ in range(len(self.convs2))
        ])

    def forward(self, x: torch.Tensor, cache: torch.Tensor = torch.zeros(0, 0, 0, 0, 0)) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt, _ = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt, _ = self.convs2[idx](xt)
            x = xt + x
        return x, cache

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            try:
                remove_weight_norm(self.convs1[idx])
                remove_weight_norm(self.convs2[idx])
            except:
                remove_parametrizations(self.convs1[idx], 'weight')
                remove_parametrizations(self.convs2[idx], 'weight')


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, upsample_scale, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.rand_ini = torch.rand(1, 9)
        self.rand_ini[:, 0] = 0
        self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
#             # for normal case

#             # To prevent torch.cumsum numerical overflow,
#             # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
#             # Buffer tmp_over_one_idx indicates the time step to add -1.
#             # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

#             phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            rad_values = torch.nn.functional.interpolate(rad_values.transpose(1, 2),
                                                         scale_factor=1/self.upsample_scale,
                                                         mode="linear").transpose(1, 2)

#             tmp_over_one = torch.cumsum(rad_values, 1) % 1
#             tmp_over_one_idx = (padDiff(tmp_over_one)) < 0
#             cumsum_shift = torch.zeros_like(rad_values)
#             cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = torch.nn.functional.interpolate(phase.transpose(1, 2) * self.upsample_scale,
                                                    scale_factor=self.upsample_scale, mode="nearest").transpose(1, 2)
            sines = torch.sin(phase)

        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim,
                             device=f0.device)
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * self.sine_waves[:, :sine_waves.shape[1]].to(sine_waves.device)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()
        self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = self.uv[:, :uv.shape[1]] * self.sine_amp / 3
        return sine_merge, noise, uv


class CausalHiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """
    def __init__(
            self,
            in_channels: int = 80,
            base_channels: int = 512,
            nb_harmonics: int = 8,
            sampling_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = [8, 8],
            upsample_kernel_sizes: List[int] = [16, 16],
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = [7, 11],
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            f0_predictor: torch.nn.Module = None,
    ):
        super(CausalHiFTGenerator, self).__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold)
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"], mode='nearest')

        self.conv_pre = weight_norm(
            LookRightConv1d(in_channels, base_channels, 5, 1)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    LookLeftConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2**(i + 1)),
                        k,
                        u
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    LookLeftConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    LookLeftConv1dWithStride(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u)
                )

            self.source_resblocks.append(
                ResBlock(base_channels // (2 ** (i + 1)), k, d)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2**(i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(LookLeftConv1d(ch, istft_params["n_fft"] + 2, 7, 1))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor
        # f0回退3帧，hift回退5帧
        self.context_size = 8

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            try:
                remove_weight_norm(l)
            except:
                remove_parametrizations(l, 'weight')
        for l in self.resblocks:
            l.remove_weight_norm()
        try:
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except:
            remove_parametrizations(self.conv_pre, 'weight')
            remove_parametrizations(self.conv_post, 'weight')
        self.f0_predictor.remove_weight_norm()
        for l in self.source_resblocks:
            l.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
            return_complex=True)
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(0, 0, 0), finalize: bool = True) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        # NOTE(lyuxiang.lx) 回退4帧
        if finalize is False:
            s_stft_real, s_stft_imag = s_stft_real[:, :, :-int(480 * 4 / self.istft_params["hop_len"])], s_stft_imag[:, :, :-int(480 * 4 / self.istft_params["hop_len"])]
            x = self.conv_pre(x[:, :, :-4], x[:, :, -4:])
        else:
            x = self.conv_pre(x)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x, _ = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si, _ = self.source_downs[i](s_stft)
            si, _ = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                this_xs, _ = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = this_xs
                else:
                    xs += this_xs
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x, _ = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        # NOTE(lyuxiang.lx) 回退1帧
        if finalize is False:
            x = x[:, :-480]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat: torch.Tensor, f0_cpu: bool = False, finalize: bool = True) -> torch.Tensor:
        # mel->f0->source
        if f0_cpu is True:
            self.f0_predictor.to('cpu')
            f0, _ = self.f0_predictor(speech_feat.cpu(), finalize=finalize)
            f0 = f0.to(speech_feat.device)
        else:
            self.f0_predictor.to(speech_feat.device)
            f0, _ = self.f0_predictor(speech_feat, finalize=finalize)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if finalize is False:
            generated_speech = self.decode(speech_feat[:, :, :-3], s, finalize=finalize)
        else:
            generated_speech = self.decode(speech_feat, s, finalize=finalize)
        return generated_speech, []


@tables.register("model_classes", "CausalHifiGan")
class CausalHifiGan(nn.Module):
    """HIFIGAN-style vocoders (generator [stack of time-level-upsampling blocks] + discriminator).
       NSF-HIFIGAN, HiFTNet Optional.
    """

    def __init__(
            self,
            CausalHiFTGenerator_conf: dict = {},
            CausalConvRNNF0Predictor_conf: dict = {},
            sample_rate: float = 24000,
            **kwargs
    ):
        super().__init__()
        self.generator = CausalHiFTGenerator(**CausalHiFTGenerator_conf)
        self.generator.f0_predictor = CausalConvRNNF0Predictor(**CausalConvRNNF0Predictor_conf)
        self.generator.remove_weight_norm()
        self.sample_rate = sample_rate

    def inference_prepare(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        feat_list = []
        feat_len_list = []
        for i, feat in enumerate(data_in):
            if isinstance(feat, str) and os.path.exists(feat):
                feat = np.load(feat)
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat)

            feat_list.append(feat)
            feat_len_list.append(feat.shape[0])

        batch = {
            "x": pad_sequence(feat_list, batch_first=True),
            "x_lengths": torch.tensor(feat_len_list, dtype=torch.int64),
        }
        batch = to_device(batch, kwargs["device"])

        return batch

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        f0_cpu: bool = True,
        finalize: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run inference.

        Args:
            x (torch.Tensor): input representation, B x T x C

        Returns:
            Dict[str, Tensor]:
                * recon_speech (Tensor): Reconstructed waveform tensor (B, T_wav).

        """
        uttid = key[0]
        batch = self.inference_prepare(data_in, data_lengths, key, **kwargs)
        voc_dtype = dtype_map[kwargs.get("voc_dtype", "fp32")]
        x = batch["x"].to(voc_dtype)
        recon_speech = self.generator.inference(x.transpose(1, 2), f0_cpu=f0_cpu, finalize=finalize)[0].squeeze(1)
        recon_speech = recon_speech.float()
        logging.info(f"{uttid}: wav lengths {recon_speech.shape[1]}")

        output_dir = kwargs.get("output_dir", None)
        output_sr = kwargs.get("output_sr", None)
        if output_dir is not None:
            wav_out_dir = os.path.join(output_dir, "wav")
            os.makedirs(wav_out_dir, exist_ok=True)
            wav_sr = self.sample_rate
            if output_sr is not None and output_sr != self.sample_rate:
                recon_speech = torchaudio.functional.resample(
                    recon_speech,
                    orig_freq=self.sample_rate,
                    new_freq=output_sr
                )
                wav_sr = output_sr
            torchaudio.save(
                os.path.join(wav_out_dir, f"{key[0]}.wav"), recon_speech.cpu(),
                sample_rate=wav_sr, encoding='PCM_S', bits_per_sample=16
            )

        return recon_speech