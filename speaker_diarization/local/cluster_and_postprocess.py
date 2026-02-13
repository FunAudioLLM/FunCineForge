"""
This script is designed to cluster speaker embeddings and generate RTTM result files as output.
"""

import os
import argparse
import pickle
import numpy as np
from glob import glob
import torch.distributed as dist

from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build

parser = argparse.ArgumentParser(description='Cluster embeddings and output rttm files')
parser.add_argument('--conf', default=None, help='Config file')
parser.add_argument('--root', default=None, help='Wav list file')
parser.add_argument('--wav_list_name', default='clean_wav.list', help='Name of wav list files')


def find_wav_lists(root_dir, pattern='clean_wav.list'):
    """递归查找所有匹配的 wav list 文件"""
    return glob(os.path.join(root_dir, '**', pattern), recursive=True)
    
    

def make_rttms(seg_list, out_rttm, rec_id):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed = seg[0]
        seg_st = float(seg_st)
        seg_ed = float(seg_ed)
        cluster_id = seg[1] + 1
        if i == 0:
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][3]:
            if seg_st > new_seg_list[-1][2]:
                new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][2] = seg_ed
        else:
            if seg_st < new_seg_list[-1][2]:
                p = (new_seg_list[-1][2]+seg_st) / 2
                new_seg_list[-1][2] = p
                seg_st = p
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])

    line_str ="SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
    with open(out_rttm,'w') as f:
        for seg in new_seg_list:
            seg_id, seg_st, seg_ed, cluster_id = seg
            f.write(line_str.format(seg_id, seg_st, seg_ed-seg_st, cluster_id))
            

def audio_only_func(wav_file, rec_id, audio_embs_file, rttm_dir, config):
    cluster = build('cluster', config)
    if not os.path.exists(audio_embs_file):
        print("[Audio Embs] does not exist, it is possible that vad model did not detect valid speech in file %s, please check it."%(wav_file))
        return
    with open(audio_embs_file, 'rb') as f:
        stat_obj = pickle.load(f)
        embeddings = stat_obj['embeddings']
        times = stat_obj['times']
    # cluster
    labels = cluster(embeddings)
    # output rttm
    new_labels = np.zeros(len(labels), dtype=int)
    uniq = np.unique(labels)
    for i in range(len(uniq)):
        new_labels[labels==uniq[i]] = i 
    seg_list = [(i,j) for i, j in zip(times, new_labels)]
    out_rttm = os.path.join(rttm_dir, rec_id+'.rttm')
    make_rttms(seg_list, out_rttm, rec_id)


def audio_vision_func(wav_file, rec_id, audio_embs_file, visual_embs_file, rttm_dir, config):
    cluster = build('cluster', config)
    if not os.path.exists(audio_embs_file):
        print("[Audio Embs] does not exist, it is possible that vad model did not detect valid speech in file %s, please check it."%(wav_file))
        return
    with open(audio_embs_file, 'rb') as f:
        stat_obj = pickle.load(f)
        audio_embeddings = stat_obj['embeddings']
        audio_times = stat_obj['times']
    with open(visual_embs_file, 'rb') as f:
        stat_obj = pickle.load(f)
        visual_embeddings = stat_obj['embeddings']
        frameI = stat_obj['frameI']
        faceI = stat_obj['faceI']
        visual_times = frameI * 0.04
        frame_indices = [np.where(faceI == frame)[0][0] for frame in frameI]
        speak_embeddings = visual_embeddings[frame_indices]
        speak_embeddings_normalized = speak_embeddings / np.sqrt(np.sum(speak_embeddings**2, axis=-1, keepdims=True)) # normalize to only save speaker characteristics
        
    # cluster
    labels = cluster(audio_embeddings, speak_embeddings_normalized, audio_times, visual_times, config)
    # output rttm
    new_labels = np.zeros(len(labels), dtype=int)
    uniq = np.unique(labels)
    for i in range(len(uniq)):
        new_labels[labels==uniq[i]] = i 
    seg_list = [(i,j) for i, j in zip(audio_times, new_labels)]
    out_rttm = os.path.join(rttm_dir, rec_id+'.rttm')
    make_rttms(seg_list, out_rttm, rec_id)


def main():
    args = parser.parse_args()
    config = build_config(args.conf)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    print(f"[INFO] Cluster and postprocess using total threads num: {threads_num}")
    dist.init_process_group(backend='gloo')
    
    wav_lists = find_wav_lists(args.root, pattern=args.wav_list_name)
    if not wav_lists:
        raise Exception(f"No wav_list files found in the directory: {args.root}")
    
    all_wavs = []
    wav_metadata = {}
    for wav_list_path in wav_lists:
        episode_dir = os.path.dirname(wav_list_path)
        visual_embs_dir = os.path.join(episode_dir, 'embs_video')
        audio_embs_dir = os.path.join(episode_dir, 'embs_wav')
        rttm_dir = os.path.join(episode_dir, 'rttm')

        # 读取 wav list
        with open(wav_list_path, 'r') as f:
            episode_wavs = [i.strip() for i in f.readlines()]
            
        for wpath in episode_wavs:
            if not os.path.isabs(wpath):
                raise ValueError(f"Wav path {wpath} is not an absolute path.")
            all_wavs.append(wpath)
            wav_metadata[wpath] = {
                'audio_embs_dir': audio_embs_dir,
                'visual_embs_dir': visual_embs_dir,
                'rttm_dir': rttm_dir
            }
            

    local_wavs = all_wavs[rank::threads_num]
    for wpath in local_wavs:
        metadata = wav_metadata[wpath]
        audio_embs_dir = metadata['audio_embs_dir']
        visual_embs_dir = metadata['visual_embs_dir']
        rttm_dir = metadata['rttm_dir']

        os.makedirs(rttm_dir, exist_ok=True)
        # 取 rec_id
        rec_id = os.path.splitext(os.path.basename(wpath))[0]
        rttm_file = os.path.join(rttm_dir, rec_id + '.rttm')
        if os.path.exists(rttm_file):
            print(f"[SKIP] RTTM already exists {rttm_file}. Skipping.")
            continue
        
        # wav and video emb pkl path
        visual_embs_file = os.path.join(visual_embs_dir, rec_id + '.pkl')
        audio_embs_file = os.path.join(audio_embs_dir, rec_id + '.pkl')

        if os.path.exists(visual_embs_file):
            audio_vision_func(wpath, rec_id, audio_embs_file, visual_embs_file, rttm_dir, config)
        else:
            print("[Visual Embs] does not exist in file %s, now only use audio cluster."%(wpath))
            audio_only_func(wpath, rec_id, audio_embs_file, rttm_dir, config)
  


if __name__ == "__main__":
    main()
