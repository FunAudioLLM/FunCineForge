import os.path

import torch
import torch.nn as nn
from funcineforge.register import tables
from typing import Dict
import logging
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F
from funcineforge.models.utils.nets_utils import make_pad_mask
from funcineforge.utils.device_funcs import to_device
import numpy as np
from funcineforge.utils.load_utils import extract_campp_xvec
import time
from funcineforge.models.utils import dtype_map
from funcineforge.utils.hinter import hint_once
from funcineforge.models.utils.masks import add_optional_chunk_mask
from .modules.dit_flow_matching.dit_model import DiT


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        center=False,
        device='cuda',
        feat_type="power_log",
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float().to(device)
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmax = mel_fmax
        self.center = center
        self.feat_type = feat_type

    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        if self.feat_type == "mag_log10":
            power_spec = torch.sqrt(torch.pow(fft.imag, 2) + torch.pow(fft.real, 2))
            mel_output = torch.matmul(self.mel_basis, power_spec)
            return torch.log10(torch.clamp(mel_output, min=1e-5))
        power_spec = torch.pow(fft.imag, 2) + torch.pow(fft.real, 2)
        mel_spec = torch.matmul(self.mel_basis, torch.sqrt(power_spec + 1e-9))
        return self.spectral_normalize(mel_spec)

    @classmethod
    def spectral_normalize(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_compression(spec, C, clip_val)
        return output

    @classmethod
    def spectral_de_normalize_torch(cls, spec, C=1, clip_val=1e-5):
        output = cls.dynamic_range_decompression(spec, C, clip_val)
        return output

    @staticmethod
    def dynamic_range_compression(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def dynamic_range_decompression(x, C=1):
        return torch.exp(x) / C


class LookaheadBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels, channels,
            kernel_size=pre_lookahead_len+1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, in_channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs, ilens, context: torch.Tensor = torch.zeros(0, 0, 0)):
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0)
        else:
            assert context.size(2) == self.pre_lookahead_len
            outputs = torch.concat([outputs, context], dim=2)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode='constant', value=0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        mask = (~make_pad_mask(ilens).unsqueeze(-1).to(inputs.device))
        # residual connection
        outputs = (outputs + inputs) * mask
    
        return outputs, ilens


@tables.register("model_classes", "CosyVoiceFlowMatching")
class CosyVoiceFlowMatching(nn.Module):
    def __init__(
            self,
            codebook_size: int,
            model_size: int,
            xvec_size: int = 198,
            dit_conf: Dict = {},
            mel_feat_conf: Dict = {},
            prompt_conf: Dict = None,
            **kwargs):
        super().__init__()

        # feat related
        self.feat_token_ratio = kwargs.get("feat_token_ratio", None)
        try:
            self.mel_extractor = Audio2Mel(**mel_feat_conf)
            self.sample_rate = self.mel_extractor.sampling_rate
        except:
            self.mel_extractor = None
            self.sample_rate = 24000
        self.mel_norm_type = kwargs.get("mel_norm_type", None)
        self.num_mels = num_mels = mel_feat_conf["n_mel_channels"]
        self.token_rate = kwargs.get("token_rate", 25)
        self.model_dtype = kwargs.get("model_dtype", "fp32")
        self.codebook_size = codebook_size

        # condition related
        self.prompt_conf = prompt_conf
        if self.prompt_conf is not None:
            self.prompt_masker = self.build_prompt_masker()

        # codec related
        self.codec_embedder = nn.Embedding(codebook_size, num_mels)
        lookahead_length = kwargs.get("lookahead_length", 4)
        self.lookahead_conv1d = LookaheadBlock(num_mels, model_size, lookahead_length)

        # spk embed related
        if xvec_size is not None:
            self.xvec_proj = torch.nn.Linear(xvec_size, num_mels)

        # dit model related
        self.dit_conf = dit_conf
        self.dit_model = DiT(**dit_conf)

        self.training_cfg_rate = kwargs.get("training_cfg_rate", 0)
        self.only_mask_loss = kwargs.get("only_mask_loss", True)

        # NOTE fm需要右看的下文
        self.context_size = self.lookahead_conv1d.pre_lookahead_len

    def build_prompt_masker(self):
        prompt_type = self.prompt_conf.get("prompt_type", "free")
        if prompt_type == "prefix":
            from funcineforge.models.utils.mask_along_axis import MaskTailVariableMaxWidth
            masker = MaskTailVariableMaxWidth(
                mask_width_ratio_range=self.prompt_conf["prompt_width_ratio_range"],
            )
        else:
            raise NotImplementedError

        return masker

    @staticmethod
    def norm_spk_emb(xvec):
        xvec_mask = (~xvec.norm(dim=-1).isnan()) * (~xvec.norm(dim=-1).isinf())
        xvec = xvec * xvec_mask.unsqueeze(-1)
        xvec = xvec.mean(dim=1)
        xvec = F.normalize(xvec, dim=1)

        return xvec

    def select_target_prompt(self, y: torch.Tensor, y_lengths: torch.Tensor):
        # cond_mask: 1, 1, 1, ..., 0, 0, 0
        cond_mask = self.prompt_masker(y, y_lengths, return_mask=True)

        return cond_mask

    @torch.no_grad()
    def normalize_mel_feat(self, feat, feat_lengths):
        # feat in B,T,D
        if self.mel_norm_type == "mean_std":
            max_length = feat.shape[1]
            mask = (~make_pad_mask(feat_lengths, maxlen=max_length))
            mask = mask.unsqueeze(-1).to(feat)
            mean = ((feat * mask).sum(dim=(1, 2), keepdim=True) /
                    (mask.sum(dim=(1, 2), keepdim=True) * feat.shape[-1]))
            var = (((feat - mean)**2 * mask).sum(dim=(1, 2), keepdim=True) /
                   (mask.sum(dim=(1, 2), keepdim=True) * feat.shape[-1] - 1))  # -1 for unbiased estimation
            std = torch.sqrt(var)
            feat = (feat - mean) / std
            feat = feat * mask
            return feat
        if self.mel_norm_type == "min_max":
            bb, tt, dd = feat.shape
            mask = (~make_pad_mask(feat_lengths, maxlen=tt))
            mask = mask.unsqueeze(-1).to(feat)
            feat_min = (feat * mask).reshape([bb, tt * dd]).min(dim=1, keepdim=True).values.unsqueeze(-1)
            feat_max = (feat * mask).reshape([bb, tt * dd]).max(dim=1, keepdim=True).values.unsqueeze(-1)
            feat = (feat - feat_min) / (feat_max - feat_min)
            # noise ~ N(0, I), P(x >= 3sigma) = 0.001, 3 is enough.
            feat = (feat * 3) * mask  # feat in [-3, 3]
            return feat
        else:
            raise NotImplementedError

    @torch.no_grad()
    def extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        mel_extractor = self.mel_extractor.float()
        feat = mel_extractor(y)
        feat = feat.transpose(1, 2)
        feat_lengths = (y_lengths / self.mel_extractor.hop_length).to(y_lengths)
        if self.mel_norm_type is not None:
            feat = self.normalize_mel_feat(feat, feat_lengths)
        return feat, feat_lengths

    def load_data(self, contents: dict, **kwargs):
        fm_use_prompt = kwargs.get("fm_use_prompt", True)

        # codec
        codec = contents["codec"]
        if isinstance(codec, np.ndarray):
            codec = torch.from_numpy(codec)
            # codec = torch.from_numpy(codec)[None, :]
        codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64)
        
        # prompt codec (optional)
        prompt_codec = kwargs.get("prompt_codec", None)
        prompt_codec_lengths = None
        if prompt_codec is not None and fm_use_prompt:
            if isinstance(prompt_codec, str) and os.path.exists(prompt_codec):
                prompt_codec = np.load(prompt_codec)
            if isinstance(prompt_codec, np.ndarray):
                prompt_codec = torch.from_numpy(prompt_codec)[None, :]
            prompt_codec_lengths = torch.tensor([prompt_codec.shape[1]], dtype=torch.int64)
        else:
            prompt_codec = None
        spk_emb = kwargs.get("spk_emb", None)
        spk_emb_lengths = None
        if spk_emb is not None:
            if isinstance(spk_emb, str) and os.path.exists(spk_emb):
                spk_emb = np.load(spk_emb)
            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)[None, :]
            spk_emb_lengths = torch.tensor([spk_emb.shape[1]], dtype=torch.int64)

        # prompt wav as condition
        prompt_wav = contents["vocal"]
        prompt_wav_lengths = None
        if prompt_wav is not None and fm_use_prompt and os.path.exists(prompt_wav):
            if prompt_wav.endswith(".npy"):
                spk_emb = np.load(prompt_wav)
                spk_emb_lengths = torch.tensor([spk_emb.shape[1]], dtype=torch.int64)
            else:
                spk_emb = extract_campp_xvec(prompt_wav, **kwargs)
                spk_emb = torch.from_numpy(spk_emb)
                spk_emb_lengths = torch.tensor([spk_emb.shape[1]], dtype=torch.int64)
            # prompt_wav = load_audio_text_image_video(prompt_wav, fs=self.sample_rate)   
            # prompt_wav = prompt_wav[None, :]
            # prompt_wav_lengths = torch.tensor([prompt_wav.shape[1]], dtype=torch.int64)
        else:
            logging.info("[error] prompt_wav is None or not path or path not exists! Please provide the correct speaker embedding.")
        
        output = {
            "codec": codec,
            "codec_lengths": codec_lengths,
            "prompt_codec": prompt_codec,
            "prompt_codec_lengths": prompt_codec_lengths,
            "prompt_wav": None,
            "prompt_wav_lengths": None,
            "xvec": spk_emb,
            "xvec_lengths": spk_emb_lengths,
        }

        return output

    @torch.no_grad()
    def inference(
            self,
            data_in,
            data_lengths=None,
            key: list = None,
            chunk_size: int = -1,
            finalize: bool = True,
            **kwargs,
    ):
        uttid = key[0]
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        batch = self.load_data(data_in[0], **kwargs)
        batch = to_device(batch, kwargs["device"])
        batch.update({'finalize': finalize, 'chunk_size': chunk_size})
        feat = self._inference(**batch, **kwargs)
        feat = feat.float()
        logging.info(f"{uttid}: feat lengths {feat.shape[1]}")

        return feat

    @torch.no_grad()
    def _inference(
            self,
            codec, codec_lengths,
            prompt_codec=None, prompt_codec_lengths=None,
            prompt_wav=None, prompt_wav_lengths=None,
            xvec=None, xvec_lengths=None, chunk_size=-1, finalize=False,
            **kwargs
    ):
        fm_dtype = dtype_map[kwargs.get("fm_dtype", "fp32")]
        rand_xvec = None
        if xvec is not None:
            if xvec.dim() == 2:
                xvec = xvec.unsqueeze(1)
                xvec_lens = torch.ones_like(xvec_lengths)
            rand_xvec = self.norm_spk_emb(xvec)
            self.xvec_proj.to(fm_dtype)
            rand_xvec = self.xvec_proj(rand_xvec.to(fm_dtype))
            rand_xvec = rand_xvec.unsqueeze(1)

        if (codec >= self.codebook_size).any():
            new_codec = codec[codec < self.codebook_size].unsqueeze(0)
            logging.info(f"remove out-of-range token for FM: from {codec.shape[1]} to {new_codec.shape[1]}.")
            codec_lengths = codec_lengths - (codec.shape[1] - new_codec.shape[1])
            codec = new_codec
        if prompt_codec is not None:
            codec, codec_lengths = self.concat_prompt(prompt_codec, prompt_codec_lengths, codec, codec_lengths)
        mask = (codec != -1).float().unsqueeze(-1)
        codec_emb = self.codec_embedder(torch.clamp(codec, min=0)) * mask

        self.lookahead_conv1d.to(fm_dtype)
        if finalize is True:
            context = torch.zeros(1, 0, self.codec_embedder.embedding_dim).to(fm_dtype)
        else:
            codec_emb, context = codec_emb[:, :-self.context_size].to(fm_dtype), codec_emb[:, -self.context_size:].to(fm_dtype)
            codec_lengths = codec_lengths - self.context_size
        mu, _ = self.lookahead_conv1d(codec_emb, codec_lengths, context)
        mu = mu.repeat_interleave(self.feat_token_ratio, dim=1)
        # print(mu.size())
        conditions = torch.zeros([mu.size(0), mu.shape[1], self.num_mels]).to(mu)
        # get conditions
        if prompt_wav is not None:
            if prompt_wav.ndim == 2:
                prompt_wav, prompt_wav_lengths = self.extract_feat(prompt_wav, prompt_wav_lengths)
            # NOTE 在fmax12k fm中，尝试mel interploate成token 2倍shape，而不是强制截断
            prompt_wav = prompt_wav.to(fm_dtype)
            for i, _len in enumerate(prompt_wav_lengths):
                conditions[i, :_len] = prompt_wav[i]

        feat_lengths = codec_lengths * self.feat_token_ratio
        # NOTE add_optional_chunk_mask支持生成-1/1/15/30不同chunk_size的mask
        mask = add_optional_chunk_mask(mu, torch.ones([1, 1, mu.shape[1]]).to(mu).bool(), False, False, 0, chunk_size, -1)
        feat = self.solve_ode(mu, rand_xvec, conditions.to(fm_dtype), mask, **kwargs)

        if prompt_codec is not None and prompt_wav is not None:
            feat, feat_lens = self.remove_prompt(None, prompt_wav_lengths, feat, feat_lengths)

        return feat

    @staticmethod
    def concat_prompt(prompt, prompt_lengths, text, text_lengths):
        xs_list, x_len_list = [], []
        for idx, (_prompt_len, _text_len) in enumerate(zip(prompt_lengths, text_lengths)):
            xs_list.append(torch.concat([prompt[idx, :_prompt_len], text[idx, :_text_len]], dim=0))
            x_len_list.append(_prompt_len + _text_len)

        xs = torch.nn.utils.rnn.pad_sequence(xs_list, batch_first=True, padding_value=0.0)
        x_lens = torch.tensor(x_len_list, dtype=torch.int64).to(xs.device)

        return xs, x_lens

    @staticmethod
    def remove_prompt(prompt, prompt_lengths, padded, padded_lengths):
        xs_list = []
        for idx, (_prompt_len, _x_len) in enumerate(zip(prompt_lengths, padded_lengths)):
            xs_list.append(padded[idx, _prompt_len: _x_len])

        xs = torch.nn.utils.rnn.pad_sequence(xs_list, batch_first=True, padding_value=0.0)

        return xs, padded_lengths - prompt_lengths

    def get_rand_noise(self, mu: torch.Tensor, **kwargs):
        use_fixed_noise_infer = kwargs.get("use_fixed_noise_infer", True)
        max_len = kwargs.get("max_len", 50*300)
        if use_fixed_noise_infer:
            if not hasattr(self, "rand_noise") or self.rand_noise is None or self.rand_noise.shape[2] < mu.shape[2]:
                self.rand_noise = torch.randn([1, max_len, mu.shape[2]]).to(mu)
                logging.info("init random noise for Flow")
            # return self.rand_noise[:, :mu.shape[1], :]
            return torch.concat([self.rand_noise[:, :mu.shape[1], :] for _ in range(mu.size(0))], dim = 0)
        else:
            return torch.randn_like(mu)

    def solve_ode(self, mu, rand_xvec, conditions, mask, **kwargs):
        fm_dtype = dtype_map[kwargs.get("fm_dtype", "fp32")]
        temperature = kwargs.get("temperature", 1.0)
        n_timesteps = kwargs.get("n_timesteps", 10)
        infer_t_scheduler = kwargs.get("infer_t_scheduler", "cosine")
        z = self.get_rand_noise(mu) * temperature
        # print("z", z.size(), "mu", mu.size())
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(mu)
        # print("t_span", t_span)
        if infer_t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        fm_time = time.time()
        self.dit_model.to(fm_dtype)
        feat = self.solve_euler(
            z.to(fm_dtype), t_span=t_span.to(fm_dtype), mu=mu.to(fm_dtype), mask=mask,
            spks=rand_xvec.to(fm_dtype), cond=conditions.to(fm_dtype), **kwargs
        )
        escape_time = (time.time() - fm_time) * 1000.0
        logging.info(f"fm dec {n_timesteps} step time: {escape_time:.2f}, avg {escape_time/n_timesteps:.2f} ms")
        return feat

    def solve_euler(self, x, t_span, mu, mask, spks=None, cond=None, **kwargs):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        inference_cfg_rate = kwargs.get("inference_cfg_rate", 0.7)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        # print("solve_euler cond", cond.size())
        steps = 1
        z, bz = x, x.shape[0]
        while steps <= len(t_span) - 1:
            if inference_cfg_rate > 0:
                x_in = torch.concat([x, x], dim=0)
                spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
                mask_in = torch.concat([mask, mask], dim=0)
                mu_in = torch.concat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.concat([t.unsqueeze(0) for _ in range(mu_in.size(0))], dim=0)
                if isinstance(cond, torch.Tensor):
                    cond_in = torch.concat([cond, torch.zeros_like(cond)], dim=0)
                else:
                    cond_in = dict(
                        prompt=[
                            torch.concat([cond["prompt"][0], torch.zeros_like(cond["prompt"][0])], dim=0),
                            torch.concat([cond["prompt"][1], cond["prompt"][1]], dim=0),
                        ]
                    )
            else:
                x_in, mask_in, mu_in, spks_in, t_in, cond_in = x, mask, mu, spks, t, cond

            # if spks is not None:
            #     cond_in = cond_in + spks

            infer_causal_mask_type = kwargs.get("infer_causal_mask_type", 0)
            chunk_mask_value = self.dit_model.causal_mask_type[infer_causal_mask_type]["prob_min"]
            hint_once(
                f"flow mask type: {infer_causal_mask_type}, mask_rank value: {chunk_mask_value}.",
                "chunk_mask_value"
            )
            # print("dit_model cond", x_in.size(), cond_in.size(), mu_in.size(), spks_in.size(), t_in.size())
            # print(t_in)
            dphi_dt = self.dit_model(
                x_in, cond_in, mu_in, spks_in, t_in,
                mask=mask_in,
                mask_rand=torch.ones_like(t_in).reshape(-1, 1, 1) * chunk_mask_value
            )
            if inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt -
                           inference_cfg_rate * cfg_dphi_dt)

            x = x + dt * dphi_dt
            t = t + dt
            # sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return x