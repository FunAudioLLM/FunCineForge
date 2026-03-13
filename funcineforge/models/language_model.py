import logging
import os
import torch
import torch.nn as nn
from funcineforge.models.utils.llm_decoding import LLMDecoder
from funcineforge.register import tables
from funcineforge.utils.device_funcs import to_device
import numpy as np
from funcineforge.models.utils import dtype_map
from transformers import AutoModelForCausalLM
import pickle


@tables.register("model_classes", "FunCineForgeLM")
class FunCineForgeLM(nn.Module):
    def __init__(
        self,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        # llm
        self.llm_conf = llm_conf
        self.llm = None

        init_param_path = llm_conf.get("init_param_path", "")
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        self.sample_rate = kwargs.get("sample_rate", 24000)
        self.token_rate = kwargs.get("token_rate", 25)

        if kwargs.get("infer_lora_merged", False):
            llm_conf["use_qlora"] = False
            llm_conf["use_lora"] = False
            kwargs["infer_use_lora"] = False
        

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            **llm_load_kwargs,
        )

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}, use_qlora: {llm_conf.get('use_qlora', False)}, infer_use_lora: {kwargs.get('infer_use_lora',False)}, infer_lora_merged: {kwargs.get('infer_lora_merged',False)}")

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]
        
        if (not llm_conf.get("use_lora", False)) and (not kwargs.get("infer_use_lora",False)):
            del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 6761)
        self.timespk_unit = kwargs.get("timespk_unit", 1550)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim, 0)
        self.timespk_embed = nn.Embedding(self.timespk_unit, llm_dim, 0)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)
        self.face_size = kwargs.get("face_size", 512)
        self.face_linear = nn.Linear(self.face_size, llm_dim)

        self.length_normalized_loss = length_normalized_loss
        self.ignore_id = kwargs.get("ignore_id", -100)

        specaug = kwargs.get("specaug", None)
        specaug_conf = kwargs.get("specaug_conf", {})
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        self.specaug = specaug
        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")


    def insert_face_embeddings(
        self, inputs_embeds, face_emb, attention_mask, labels_ids, 
        codec_len, insert_pos, device
    ):
        """
        将face_emb插入到inputs_embeds中的指定位置, 同步更新attention_mask和labels_ids
        Args:
            inputs_embeds: (batch_size, token_num, dims) 输入embedding
            face_emb: (batch_size, max_face_len, dims) 面部embedding
            attention_mask: (batch_size, token_num) 注意力mask
            labels_ids: (batch_size, token_num) 标签ID
            codec_len: (batch_size,) 每个样本的实际face_emb长度
            insert_pos: int 插入位置, SOS token之后
            device
        Returns:
            padded_inputs_embeds: 插入face_emb并padding后的inputs_embeds
            padded_attention_mask: 更新后的attention_mask
            padded_labels: 更新后的labels_ids
        """
        batch_size, token_num, dims = inputs_embeds.shape
        max_face_len = face_emb.size(1)
        
        # 预计算新序列的最大长度
        new_max_length = token_num + max_face_len
        
        # 预分配输出张量
        padded_inputs_embeds = torch.zeros(batch_size, new_max_length, dims, device=device)
        padded_attention_mask = torch.zeros(batch_size, new_max_length, device=device, dtype=attention_mask.dtype)
        padded_labels = torch.full((batch_size, new_max_length), self.ignore_id, device=device, dtype=labels_ids.dtype)
        
        for i in range(batch_size):
            current_face_len = codec_len[i].item()
            
            # 直接填充，避免中间拼接
            padded_inputs_embeds[i, :insert_pos] = inputs_embeds[i, :insert_pos]
            padded_inputs_embeds[i, insert_pos:insert_pos+current_face_len] = face_emb[i, :current_face_len]
            padded_inputs_embeds[i, insert_pos+current_face_len:token_num+current_face_len] = inputs_embeds[i, insert_pos:]
            
            # 同样处理mask和labels
            padded_attention_mask[i, :insert_pos] = attention_mask[i, :insert_pos]
            padded_attention_mask[i, insert_pos:insert_pos+current_face_len] = 1
            padded_attention_mask[i, insert_pos+current_face_len:token_num+current_face_len] = attention_mask[i, insert_pos:]
            
            padded_labels[i, :insert_pos] = labels_ids[i, :insert_pos]
            padded_labels[i, insert_pos:insert_pos+current_face_len] = self.ignore_id
            padded_labels[i, insert_pos+current_face_len:token_num+current_face_len] = labels_ids[i, insert_pos:]
        
        return padded_inputs_embeds, padded_attention_mask, padded_labels


    def load_data(self, contents: dict, **kwargs):
        lm_use_prompt = kwargs.get("lm_use_prompt", True)
        tokenizer = kwargs.get("tokenizer")
        # text + clue
        text = contents["text"]
        clue = "<|startofclue|>" + contents["clue"] + "<|endofclue|>"
        if lm_use_prompt:
            text = clue + text
        text_ids = tokenizer.encode(text)
        text_len = len(text_ids)
        # timespk_ids
        timespk_ids = contents["timespk_ids"].tolist()
        type_id = contents["type_id"]
        # sequence
        sequence = [
            kwargs['dataset_conf']["sos"],
            *text_ids,
            type_id,
            *timespk_ids,
            kwargs['dataset_conf']["turn_of_speech"]
        ]
        input_ids = torch.tensor(sequence, dtype=torch.int64)
        
        # flag tensors
        text_flag = torch.zeros(len(sequence), dtype=torch.float32)
        timespk_flag = torch.zeros(len(sequence), dtype=torch.float32)
        codec_flag = torch.zeros(len(sequence), dtype=torch.float32)
        text_flag[1: text_len+1] = 1
        timespk_flag[text_len+1: -1] = 1
        codec_flag = 1 - text_flag - timespk_flag
        
        # face embs
        speech_len = contents["speech_len"]
        face_embs = torch.zeros((speech_len, self.face_size), dtype=torch.float32)
        face_path = contents.get("face")
        with open(face_path, 'rb') as f:
            stat_obj = pickle.load(f)
            embeddings = stat_obj['embeddings']
            faceI = stat_obj['faceI']
            for emb, frameI in zip(embeddings, faceI):
                fi = int(frameI)
                if 0 <= fi < speech_len:
                    end = min(fi + 5, speech_len)
                    face_embs[fi:end] = torch.from_numpy(emb).expand(end - fi, -1)
                    
        # batch dimension
        input_ids = input_ids[None, :]
        text_flag = text_flag[None, :]
        timespk_flag = timespk_flag[None, :]
        codec_flag = codec_flag[None, :]
        face_embs = face_embs[None, :, :]
        output = {
            "input_ids": input_ids,
            "face_embs": face_embs,
            "text_flag": text_flag > 0,
            "timespk_flag": timespk_flag > 0,
            "codec_flag": codec_flag > 0,
            "prompt_codec": None, # you can add prompt codec here if needed
        }
        return output

    def inference_prepare(self, data_in, **kwargs):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        output = self.load_data(data_in[0], **kwargs)
        batch = to_device(output, kwargs["device"])
        input_ids = batch["input_ids"]
        text_flag = batch["text_flag"]
        timespk_flag = batch["timespk_flag"]
        codec_flag = batch["codec_flag"]
        face_embs = batch["face_embs"]
        
        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
            text_embeds = self.llm.base_model.model.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        else:
            text_embeds = self.llm.model.get_input_embeddings()(input_ids * text_flag) * text_flag.unsqueeze(-1)
        timespk_embeds = self.timespk_embed(input_ids * timespk_flag) * timespk_flag.unsqueeze(-1)
        codec_embs = self.codec_embed(input_ids * codec_flag) * codec_flag.unsqueeze(-1)
        face_embs = self.face_linear(face_embs)

        inputs_embeds = text_embeds + timespk_embeds + codec_embs

        inputs_embeds = torch.cat([
            inputs_embeds[:, 0:1, :],   # sos token
            face_embs,                  # face embeddings
            inputs_embeds[:, 1:, :]     # inputs_embeds after sos
        ], dim=1)
        
        prompt_codec = batch.get("prompt_codec", None)
        if prompt_codec is not None:
            codec_emb = self.codec_embed(prompt_codec)
            inputs_embeds = torch.cat((inputs_embeds, codec_emb), dim=1)

        return inputs_embeds

    @torch.no_grad()
    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        **kwargs,
    ):
        uttid = key[0]
        inputs_emb = self.inference_prepare(data_in, **kwargs)

        logging.info(f"{uttid}: min length: {kwargs['min_length']}, max length: {kwargs['max_length']}")

        dtype = dtype_map[kwargs.get("llm_dtype", "fp32")]
        if not hasattr(self, "llm_generator"):
            llm_generator_conf = kwargs.get("dataset_conf", {})
            self.llm_generator = LLMDecoder(
                token_embeder=self.codec_embed,
                **llm_generator_conf
            ).to(dtype)

        if (kwargs.get("use_qlora",False) or kwargs.get("infer_use_lora",False)) and (not kwargs.get("infer_lora_merged",False)):
            self.llm.base_model.model.lm_head = self.codec_head.to(dtype)
        else:
            self.llm.lm_head = self.codec_head.to(dtype)

        gen_codec, hit_eos, states = self.llm_generator(
            inputs_emb.to(dtype),
            self.llm,
            states=kwargs.get("states", {}),
            **kwargs
        )

        output_dir = kwargs.get("output_dir", None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, "codec")
            os.makedirs(output_dir, exist_ok=True)
            np.save(
                os.path.join(output_dir, f"{key[0]}.npy"),
                gen_codec[0].cpu().numpy()
            )

        return gen_codec, hit_eos, states