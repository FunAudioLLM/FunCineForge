import logging
import torch
import pickle
import numpy as np
from funcineforge.register import tables
from funcineforge.utils.hinter import hint_once

@tables.register("dataset_classes", "FunCineForgeDataset")
class FunCineForgeDataset(torch.utils.data.Dataset):
    """
    Dataset for Mixed LM of FunCineForge
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        face_encoder=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        self.tokenizer = tokenizer
        self.face_encoder = face_encoder

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.retry = kwargs.get("retry", 100)

        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 1)
        self.face_size = kwargs.get("face_size", 512)

        self.codebook_size = kwargs.get("codebook_size", 6561)
        self.sos = kwargs.get("sos", self.codebook_size)
        self.eos = kwargs.get("eos", self.codebook_size + 1)
        self.turn_of_speech = kwargs.get("turn_of_speech", self.codebook_size + 2)
        self.ignore_id = kwargs.get("ignore_id", -100)
        
        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug

        self.set_invalid_xvec_zeros = kwargs.get("set_invalid_xvec_zeros", False)
        self.use_emotion_clue = kwargs.get("use_emotion_clue", False)
        logging.info(f"use_emotion_clue: {self.use_emotion_clue}")

    def get_source_len(self, index):
        item = self.index_ds[index]
        source_len = self.index_ds.get_source_len(item)
        return source_len

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)
    
    def mixup_text_codec(self, text: torch.Tensor, aug_codec: torch.Tensor, timespk_ids: torch.Tensor, type_id: int):
        text_len = text.shape[0]
        timespk_len = timespk_ids.shape[0]
        sequence = [self.sos, *text.tolist(), type_id, *timespk_ids.tolist(), self.turn_of_speech, *aug_codec.tolist(), self.eos]
        # sequence = [self.sos, *text.tolist(), type_id, self.turn_of_speech, *aug_codec.tolist(), self.eos]
        input_ids = torch.tensor(sequence, dtype=torch.int64)
        text_flag = torch.zeros(len(sequence), dtype=torch.float32)
        text_flag[1:text_len+1] = 1
        timespk_flag = torch.zeros(len(sequence), dtype=torch.float32)
        timespk_flag[text_len+1:text_len+2+timespk_len] = 1
        # timespk_flag[text_len+1:text_len+2] = 1
        codec_flag = 1 - (text_flag + timespk_flag)
        labels = torch.tensor(sequence, dtype=torch.int64)
        labels[:text_len+timespk_len+3] = self.ignore_id
        # labels[:text_len+3] = self.ignore_id

        return input_ids, labels, text_flag, codec_flag, timespk_flag
    
    def __getitem__(self, index):
        output = None
        for idx in range(self.retry):
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()
            item = self.index_ds[index_cur]
            
            # clue + text
            text = item["text"]
            clue = "<|startofclue|>" + item["clue"] + "<|endofclue|>"
            if self.use_emotion_clue:
                text = clue + text
            text_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.int32)
            hint_once(f"raw text: {text}", "log_text")
            
            # speech tokens
            target_out = item["token"]
            codec = torch.from_numpy(np.load(target_out))
            codec_len = codec.shape[0] # 可用数据集中的 speech_length 代替
            aug_codec = codec.clone()
            if self.specaug is not None:  # aug_codec是随机mask的codec增强鲁棒性
                aug_codec, _ = self.specaug(aug_codec.float().unsqueeze(0).unsqueeze(-1))
                aug_codec = aug_codec.squeeze(0).squeeze(-1).long()            
            
            # dialogue
            timespk_ids = torch.from_numpy(item["timespk_ids"])
            
            # mixup
            type_id = item["type_id"]
            input_ids, labels, text_flag, codec_flag, timespk_flag = self.mixup_text_codec(
                text_ids, aug_codec, timespk_ids, type_id
            )
            
            # face
            face_features = item["face"]
            face_emb = torch.zeros((codec_len, self.face_size), dtype=torch.float32) # face_emb 长度与 codec_len 相同
            with open(face_features, 'rb') as f:
                stat_obj = pickle.load(f)
                embeddings = stat_obj['embeddings']
                faceI = stat_obj['faceI']
                for emb, frameI in zip(embeddings, faceI):
                    fi = int(frameI)
                    if 0 <= fi < codec_len:
                        end = min(fi + 5, codec_len)
                        face_emb[fi:end] = torch.from_numpy(emb).expand(end - fi, -1)
            
            # attention_mask 对应序列长度包括input_id=(sos, <|startofclue|>, clue, <|endofclue|>, text, type_id, timespk_ids, turn_of_speech, speech, eos)
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            codec_len = torch.tensor([codec_len], dtype=torch.int32)
            output = {
                "input_ids": input_ids,
                "face_emb": face_emb,
                "attention_mask": attention_mask,
                "labels_ids": labels,
                "text_flag": text_flag,
                "codec_flag": codec_flag,
                "timespk_flag": timespk_flag,
                "codec_len": codec_len,
            }
            break
        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_token_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs