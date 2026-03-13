import logging
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import hydra
from omegaconf import DictConfig
from funcineforge import AutoModel
from funcineforge.utils.hinter import get_logger
from funcineforge.models.utils import dtype_map
from funcineforge.register import tables

def main(**kwargs):
    # device related
    node_rank = kwargs.get("node_rank", 1)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_gpus = kwargs.get("num_gpus", 1)
    world_size = kwargs.get("world_size", 1)
    num_tasks = world_size * num_gpus
    torch.cuda.set_device(local_rank)
    task_id = node_rank * num_gpus + local_rank
    logger = get_logger(log_level=logging.INFO, local_rank=task_id, world_size=num_tasks)

    # task related
    output_dir = kwargs.get("output_dir")
    data_jsonl = kwargs.get("data_jsonl")
    if local_rank == 0:
        tables.print()

    # build LM model
    lm_ckpt_path = kwargs.get("lm_ckpt_path", "")
    lm_exp_dir, lm_model_name, lm_ckpt_id, _ = lm_ckpt_path.rsplit("/", 3)
    logger.info(f"init LM model form {lm_ckpt_path}")
    lm_model = (AutoModel(
        model=os.path.join(lm_exp_dir, lm_model_name),
        init_param=lm_ckpt_path,
        output_dir=None,
        device=f"cuda:{local_rank}",
    ))
    lm_model.model.to(dtype_map[kwargs.get("llm_dtype", "fp32")])

    # build FM model
    fm_ckpt_path = kwargs.get("fm_ckpt_path", "")
    fm_exp_dir, fm_model_name, fm_ckpt_id, _ = fm_ckpt_path.rsplit("/", 3)
    logger.info(f"build FM model form {fm_ckpt_path}")
    fm_model = AutoModel(
        model=os.path.join(fm_exp_dir, fm_model_name),
        init_param=fm_ckpt_path,
        output_dir=None,
        device=f"cuda:{local_rank}",
    )
    fm_model.model.to(dtype_map[kwargs.get("fm_dtype", "fp32")])

    # build voc model
    voc_ckpt_path = kwargs.get("voc_ckpt_path", "")
    voc_exp_dir, voc_model_name, voc_ckpt_id, _ = voc_ckpt_path.rsplit("/", 3)
    logger.info(f"build VOC model form {voc_ckpt_path}")
    voc_model = AutoModel(
        model=os.path.join(voc_exp_dir, voc_model_name),
        init_param=voc_ckpt_path,
        output_dir=None,
        device=f"cuda:{local_rank}",
    )
    voc_model.model.to(dtype_map[kwargs.get("voc_dtype", "fp32")])

    # build inference model
    logger.info(f"build inference model {kwargs.get('model')}")
    kwargs["output_dir"] = output_dir
    kwargs["tokenizer"] = None
    model = AutoModel(
        **kwargs,
        lm_model=lm_model,
        fm_model=fm_model,
        voc_model=voc_model,
    )
    index_ds_class = tables.index_ds_classes.get(kwargs.get('index_ds'))
    dataset_conf = kwargs.get("dataset_conf")
    index_ds = index_ds_class(data_jsonl, **dataset_conf)
    
    model.inference(input=index_ds, input_len=len(index_ds))


@hydra.main(config_path="decode_conf", config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    command_line = ' '.join(sys.argv)
    logging.info(command_line)
    if kwargs.get("debug", False):
        import pdb
        pdb.set_trace()

    assert "model" in kwargs
    main(**kwargs)


if __name__ == "__main__":
    main_hydra()
