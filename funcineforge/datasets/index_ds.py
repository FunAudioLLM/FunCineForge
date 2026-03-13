import json
import torch
import logging
import numpy as np
from funcineforge.register import tables


@tables.register("index_ds_classes", "FunCineForgeDS")
class FunCineForgeDS(torch.utils.data.Dataset):

    def __init__(self, data_jsonl: str, **kwargs):
        super().__init__()

        self.max_source_length = kwargs.get("max_source_length", None)
        self.max_text_length = kwargs.get("max_text_length", None)
        self.max_token_length = kwargs.get("max_token_length", None)
        self.ignore_id = kwargs.get("ignore_id", -100)
        self.frame_shift = kwargs.get("frame_shift", 25)
        self.timebook_size = kwargs.get("timebook_size", 1500)
        self.type_map = {"旁白": kwargs.get("pangbai", self.timebook_size),
                         "独白": kwargs.get("dubai", self.timebook_size + 1),
                         "对话": kwargs.get("duihua", self.timebook_size + 2),
                         "多人": kwargs.get("duoren", self.timebook_size + 3),}
        self.gender_map = {"男": kwargs.get("male", self.timebook_size + 4), 
                           "male": kwargs.get("male", self.timebook_size + 4), 
                           "女": kwargs.get("female", self.timebook_size + 5),
                           "female": kwargs.get("female", self.timebook_size + 5),}
        self.age_map = {"儿童": kwargs.get("child", self.timebook_size + 6), 
                        "child": kwargs.get("child", self.timebook_size + 6), 
                        "青年": kwargs.get("youth", self.timebook_size + 7), 
                        "teenager": kwargs.get("youth", self.timebook_size + 7), 
                        "中年": kwargs.get("adult", self.timebook_size + 8), 
                        "adult": kwargs.get("adult", self.timebook_size + 8), 
                        "中老年": kwargs.get("middle", self.timebook_size + 9), 
                        "middle-aged": kwargs.get("middle", self.timebook_size + 9), 
                        "老年": kwargs.get("elderly", self.timebook_size + 10),
                        "elderly": kwargs.get("elderly", self.timebook_size + 10)}
        self.speaker_id_start = kwargs.get("speaker_id_start", self.timebook_size + 11)
        
        load_meta_data_key = kwargs.get("load_meta_data_key").split(",")

        if not (data_jsonl.endswith(".jsonl") or data_jsonl.endswith(".json")):
            # jsonl list file
            with open(data_jsonl, encoding="utf-8") as fin:
                file_list = fin.readlines()
                logging.info(f"file_list: {file_list}")
        else:
            file_list = [data_jsonl]
            
        contents = []
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    utt = data_dict["utt"]
                    data_type = data_dict.get("type")
                    type_id = self.type_map[data_type] if data_type in self.type_map else 1500
                    data = data_dict["messages"]
                    speech_length = data_dict.get("speech_length", -1)
                    # 2 for startofclue, endofclue
                    text_length = data_dict.get("text_length", -1) + data_dict.get("clue_length", -1) + 2
                    if self.max_token_length is not None and (speech_length > self.max_token_length or speech_length <= 0):
                        logging.info(
                            f"speech_length: {speech_length} > {self.max_token_length}, drop it: {data_dict}"
                        )
                        continue
                    if self.max_text_length is not None and (text_length > self.max_text_length or text_length <= 0):
                        logging.info(
                            f"text_length: {text_length} > {self.max_text_length}, drop it: {data_dict}"
                        )
                        continue

                    skip_flag = None
                    roles = {item.get("role") for item in data}
                    for key in load_meta_data_key:
                        if key not in roles:
                            skip_flag = key
                            break
                    if skip_flag is not None:
                        logging.info(
                            f"doesn't have {skip_flag}, drop it: {data_dict}")
                        continue

                    contents_i = {}
                    timespk_ids_len = 0
                    for i, item in enumerate(data):
                        role = item["role"]
                        content = item["content"]
                        for key in load_meta_data_key:
                            if role == key:
                                if key == "dialogue":
                                    timespk_ids = self.timespk_to_codec(content)
                                    timespk_ids_len = len(timespk_ids)
                                    if timespk_ids_len == 0:
                                        logging.info(f"[WARNING] len of timespk_ids is 0: {data_dict}")
                                    contents_i["timespk_ids"] = timespk_ids
                                else:
                                    contents_i[role] = content
                    contents_i["utt"] = utt
                    contents_i["type_id"] = type_id
                    # face embs len = speech tokens len, so need * 2;
                    # 4: sos, tos, eos; type_id
                    contents_i["source_len"] = speech_length * 2 + text_length + timespk_ids_len + 4
                    contents_i["speech_len"] = speech_length
                    contents_i["text_len"] = text_length # include clue_length
                    contents.append(contents_i)

        self.contents = contents

        logging.info("total_num of samplers: {}, {}".format(len(self.contents), data_jsonl))


    def timespk_to_codec(self, dialogue):
        # tuple tokens (start, spk, gender, age, end) * n_parts
        n_parts = len(dialogue)
        if n_parts == 0:
            return np.array([], dtype=np.int64)
        starts = np.array([part["start"] for part in dialogue])
        durations = np.array([part["duration"] for part in dialogue])
        speakers = np.array([int(part["spk"]) for part in dialogue])
        genders = [part["gender"] for part in dialogue]
        ages = [part["age"] for part in dialogue]
        
        start_idxs = (starts * self.frame_shift + 1).astype(np.int64)
        end_idxs = ((starts + durations) * self.frame_shift + 1).astype(np.int64)
        spk_ids = (self.speaker_id_start + speakers - 1).astype(np.int64)
        gender_ids = [self.gender_map.get(g, self.ignore_id) for g in genders]
        age_ids = [self.age_map.get(a, self.ignore_id) for a in ages]
        
        sequence = np.full(n_parts * 5, self.ignore_id, dtype=np.int64)
        sequence[0::5] = start_idxs
        sequence[1::5] = spk_ids
        sequence[2::5] = gender_ids
        sequence[3::5] = age_ids
        sequence[4::5] = end_idxs
        return sequence

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", 0)
        return source_len

    def get_target_len(self, data_dict):
        target_len = data_dict.get("speech_len", 0)
        return target_len