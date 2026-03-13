#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import time
import torch
import logging
import os
from tqdm import tqdm
from funcineforge.utils.misc import deep_update
from funcineforge.register import tables
from funcineforge.utils.set_all_random_seed import set_all_random_seed
from funcineforge.utils.load_pretrained_model import load_pretrained_model
from funcineforge.download.download_model_from_hub import download_model


def prepare_data_iterator(data_in, input_len):
    """ """
    data_list = []
    key_list = []
    for idx in range(input_len):
        item = data_in[idx]
        utt = item["utt"]
        data_list.append(item)
        key_list.append(utt)
    return key_list, data_list


class AutoModel:

    def __init__(self, **kwargs):
        log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
        logging.basicConfig(level=log_level)
        model, kwargs = self.build_model(**kwargs)
        self.kwargs = kwargs
        self.model = model
        self.model_path = kwargs.get("model_path")

    @staticmethod
    def build_model(**kwargs):
        assert "model" in kwargs
        if "model_conf" not in kwargs:
            logging.info("download models from {} or local dir".format(kwargs.get("hub", "ms")))
            kwargs = download_model(**kwargs)
        
        set_all_random_seed(kwargs.get("seed", 0))

        device = kwargs.get("device", "cuda")
        if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
            device = "cpu"
            kwargs["batch_size"] = 1
        kwargs["device"] = device

        torch.set_num_threads(kwargs.get("ncpu", 4))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer
        
        # build face_encoder
        face_encoder = kwargs.get("face_encoder", None)
        if face_encoder is not None:
            face_classes = tables.face_classes.get(face_encoder)
            face_encoder = face_classes(**kwargs.get("face_encoder_conf", {}))
        kwargs["face_encoder"] = face_encoder

        model_class = tables.model_classes.get(kwargs["model"])
        assert model_class is not None, f'{kwargs["model"]} is not registered'
        model_conf = {}
        deep_update(model_conf, kwargs.get("model_conf", {}))
        deep_update(model_conf, kwargs)
        model = model_class(**model_conf, vocab_size=vocab_size)

        # init_param
        init_param = kwargs.get("init_param", None)
        if init_param is not None and os.path.exists(init_param):
            logging.info(f"Loading pretrained params from ckpt: {init_param}")
            load_pretrained_model(
                path=init_param,
                model=model,
                ignore_init_mismatch=kwargs.get("ignore_init_mismatch", True),
                scope_map=kwargs.get("scope_map", []),
                excludes=kwargs.get("excludes", None),
                use_deepspeed=kwargs.get("train_conf", {}).get("use_deepspeed", False),
                save_deepspeed_zero_fp32=kwargs.get("save_deepspeed_zero_fp32", True),
            )

        # fp16
        if kwargs.get("fp16", False):
            model.to(torch.float16)
        elif kwargs.get("bf16", False):
            model.to(torch.bfloat16)
        model.to(device)

        if not kwargs.get("disable_log", True):
            tables.print()

        return model, kwargs

    def __call__(self, *args, **cfg):
        kwargs = self.kwargs
        deep_update(kwargs, cfg)
        res = self.model(*args, kwargs)
        return res
    

    def inference(self, input, input_len=None, model=None, kwargs=None, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        deep_update(kwargs, cfg)
        model = self.model if model is None else model
        model.eval()
        batch_size = kwargs.get("batch_size", 1)
        key_list, data_list = prepare_data_iterator(
            input, input_len=input_len
        )

        speed_stats = {}
        num_samples = len(data_list)
        disable_pbar = self.kwargs.get("disable_pbar", False)
        pbar = (
            tqdm(colour="blue", total=num_samples, dynamic_ncols=True) if not disable_pbar else None
        )
        time_speech_total = 0.0
        time_escape_total = 0.0
        count = 0
        log_interval = kwargs.get("log_interval", None)
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "data_lengths": end_idx - beg_idx, "key": key_batch}

            time1 = time.perf_counter()
            with torch.no_grad():
                res = model.inference(**batch, **kwargs)
                if isinstance(res, (list, tuple)):
                    results = res[0] if len(res) > 0 else [{"text": ""}]
                    meta_data = res[1] if len(res) > 1 else {}
            time2 = time.perf_counter()

            batch_data_time = meta_data.get("batch_data_time", -1)
            time_escape = time2 - time1
            speed_stats["forward"] = f"{time_escape:0.3f}"
            speed_stats["batch_size"] = f"{len(results)}"
            speed_stats["rtf"] = f"{(time_escape) / batch_data_time:0.3f}"
            description = f"{speed_stats}, "
            if pbar:
                pbar.update(batch_size)
                pbar.set_description(description)
            else:
                if log_interval is not None and count % log_interval == 0:
                    logging.info(
                        f"processed {count*batch_size}/{num_samples} samples: {key_batch[0]}"
                    )
            time_speech_total += batch_data_time
            time_escape_total += time_escape
            count += 1

        if pbar:
            pbar.set_description(f"rtf_avg: {time_escape_total/time_speech_total:0.3f}")
        torch.cuda.empty_cache()
        return
