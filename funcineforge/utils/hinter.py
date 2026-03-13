import sys
import logging
import os
import torch

HINTED = set()


def hint_once(content, uid, rank=None):
    if (rank is None) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == rank:
        if uid not in HINTED:
            logging.info(content, stacklevel=3)
            HINTED.add(uid)


def get_logger(fpath=None, log_level=logging.INFO, local_rank=0, world_size=1):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    formatter = logging.Formatter(
        f"[{os.uname()[1].split('.')[0]}]({local_rank}/{world_size})"
        f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.basicConfig(
        level=log_level,
        format=f"[{os.uname()[1].split('.')[0]}]({local_rank}/{world_size})"
               f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("Pyobj, f")
    if fpath is not None:
        # Dump log to file
        fh = logging.FileHandler(fpath)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_current_command():
    # Get the command-line arguments (including the script name)
    command_line_args = sys.argv

    # Get the full path of the Python interpreter
    python_interpreter = os.path.abspath(sys.executable)

    # Combine the interpreter and command-line arguments to reconstruct the command
    full_command = ' '.join([python_interpreter] + command_line_args)

    return full_command


def get_gpu_info():
    gpu_info = (
        "GPU, memory: usage: {:.3f} GB, "
        "peak: {:.3f} GB, "
        "cache: {:.3f} GB, "
        "cache_peak: {:.3f} GB".format(
            torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
            torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
        )
    )
    return gpu_info
