#!/usr/bin/env python3
import os
import json
import pickle
import torch
import gc
import time
import queue
from pathlib import Path
import argparse
import numpy as np
from contextlib import contextmanager
from pyannote.audio import Inference, Model as PA_Model
from pydub import AudioSegment
from typing import Dict, Any, Optional, Callable
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
except ImportError:
    raise ImportError("Please install modelscope: pip install modelscope")
from local.utils.utils import circle_pad
from local.utils.config import yaml_config_loader, build_config
from local.utils.builder import build
from local.utils.fileio import load_audio
import local.vision_tools.face_detection as face_detection
import local.vision_tools.active_speaker_detection as active_speaker_detection
import local.vision_tools.face_recognition as face_recognition
import local.vision_tools.face_quality_assessment as face_quality_assessment
import local.vision_tools.lip_detection as lip_detection
from local.vision_processer import VisionProcesser

class ModelPool:
    def __init__(self, creator: Callable, pool_size: int = 1):
        self._q = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._q.put(creator())
    @contextmanager
    def borrow(self, timeout: Optional[float] = None):
        try:
            inst = self._q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"Timeout ({timeout}s) when borrowing model instance")
        try:
            yield inst
        finally:
            self._q.put(inst)

        
class GlobalModels:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalModels, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        config_path: Optional[str] = None,
        pretrained_dir: Optional[str] = None,
        device: Optional[str] = None,
        device_id: int = 0,
        pool_sizes: Optional[Dict[str, int]] = None,
        batch_size: int = 32,
        preload: bool = True,
    ):
        if hasattr(self, "initialized"):
            return
        self.hf_token = hf_token
        self.config_path = config_path
        self.conf = yaml_config_loader(config_path)
        self.pretrained_dir = Path(pretrained_dir) if pretrained_dir else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_str = "cuda" if self.device.type == "cuda" else "cpu"
        self.device_id = device_id
        self.batch_size = batch_size
        self.pool_sizes = pool_sizes or {}
        self.visual_pools: Dict[str, ModelPool] = {}
        self.audio_models: Dict[str, Any] = {
            "segmentation": None,
            "vad_pipeline": None,
            "feature_extractor": None,
            "embedding_model": None,
        }
        if preload:
            self.preload()
        self.initialized = True

    def preload(self):
        """预加载所有模型（音频 + 视觉）"""
        if not all(self.audio_models.values()) and self.hf_token and self.config_path and self.pretrained_dir:
            self._init_audio_models()
        if not self.visual_pools and self.pretrained_dir:
            self._init_visual_pools()
        
    def _init_audio_models(self):
        """初始化音频模型"""
        if all(self.audio_models.values()):
            return
        start_time = time.time()
        
        # 1. Pyannote Segmentation
        print("[INFO] Loading segmentation model (overlap detection)...")
        self.audio_models["segmentation"] = PA_Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=self.hf_token
        ).to(self.device)

        # 2. VAD: ModelScope FSMN-VAD
        print("[INFO] Loading VAD model...")
        vad_model_path = self.pretrained_dir / "speech_fsmn_vad"
        self.audio_models["vad_pipeline"] = pipeline(
            task=Tasks.voice_activity_detection,
            model=str(vad_model_path),
            device=self.device_str,
        )

        # 3. Speaker Embedding: CAMPPlus
        print("[INFO] Loading CAMPPlus speaker embedding model...")
        feature_extractor = build('feature_extractor', self.conf)
        embedding_model = build('embedding_model', self.conf)

        ckpt = self.pretrained_dir / "speech_campplus" / "campplus_cn_en_common.pt"
        state_dict = torch.load(ckpt, map_location=self.device)
        embedding_model.load_state_dict(state_dict)
        embedding_model.eval().to(self.device)
        self.audio_models["feature_extractor"] = feature_extractor
        self.audio_models["embedding_model"] = embedding_model

        print(f"[SUCCESS] Audio models loaded in {time.time() - start_time:.2f}s.") 

    def _init_visual_pools(self):
        """初始化视觉模型池"""
        if self.visual_pools:
            return
        print("[INFO] Initializing visual model pools...")
        self.visual_pools['face'] = ModelPool(
            lambda: face_detection.Predictor(self.pretrained_dir, self.device_str, self.device_id),
            pool_size=self.pool_sizes.get('face', 1)
        )
        self.visual_pools['asd'] = ModelPool(
            lambda: active_speaker_detection.ASDTalknet(self.pretrained_dir, self.device_str, self.device_id),
            pool_size=self.pool_sizes.get('asd', 1)
        )
        self.visual_pools['fr'] = ModelPool(
            lambda: face_recognition.FaceRecIR101(self.pretrained_dir, self.device_str, self.device_id),
            pool_size=self.pool_sizes.get('fr', 1)
        )
        self.visual_pools['fq'] = ModelPool(
            lambda: face_quality_assessment.FaceQualityAssess(self.pretrained_dir, self.device_str, self.device_id),
            pool_size=self.pool_sizes.get('fq', 1)
        )
        self.visual_pools['lip'] = ModelPool(
            lambda: lip_detection.LipDetector(self.pretrained_dir, self.device_str, self.device_id),
            pool_size=self.pool_sizes.get('lip', 1)
        )
        print("[SUCCESS] Visual model pools initialized.")

    # === 音频模型获取接口 ===
    def get_segmentation_model(self):
        if self.audio_models["segmentation"] is None:
            raise RuntimeError("Segmentation model not loaded. Call preload() first.")
        return self.audio_models["segmentation"]

    def get_vad_pipeline(self):
        if self.audio_models["vad_pipeline"] is None:
            raise RuntimeError("VAD pipeline not loaded.")
        return self.audio_models["vad_pipeline"]

    def get_embedding_components(self):
        if self.audio_models["feature_extractor"] is None or self.audio_models["embedding_model"] is None:
            raise RuntimeError("Embedding models not loaded.")
        return self.audio_models["feature_extractor"], self.audio_models["embedding_model"]

    # === 视觉模型调用接口 ===
    def detect_faces(self, image, top_k=10, prob_threshold=0.9, borrow_timeout=1000):
        with self.visual_pools['face'].borrow(timeout=borrow_timeout) as model:
            return model(image, top_k=top_k, prob_threshold=prob_threshold)

    def asd_score(self, audio_feature, video_feature, borrow_timeout=1000):
        with self.visual_pools['asd'].borrow(timeout=borrow_timeout) as model:
            return model(audio_feature, video_feature)

    def get_face_embedding(self, face_image, borrow_timeout=2000):
        with self.visual_pools['fr'].borrow(timeout=borrow_timeout) as model:
            return model(face_image)
        
    def face_quality_score(self, face_image, borrow_timeout=1000):
        with self.visual_pools['fq'].borrow(timeout=borrow_timeout) as model:
            return model(face_image)
        
    def detect_lip(self, face_image, borrow_timeout=3000):
        with self.visual_pools['lip'].borrow(timeout=borrow_timeout) as model:
            return model.detect_lip(face_image)

    def release(self):
        for k in self.audio_models:
            self.audio_models[k] = None
        for _, pool in self.visual_pools.items():
            del pool
        self.visual_pools.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
model_pool: Optional[GlobalModels] = None

# =======================
# 工具函数
# =======================

def extract_audio_from_video(video_path: str, wav_path: str, sample_rate: int = 16000):
    """Extract mono 16kHz WAV from video."""
    print(f"[INFO] Extracting audio from {video_path} to {wav_path}")
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(wav_path, format="wav")


def detect_overlap(wav_path: str, threshold: float = 0.5) -> bool:
    """Detect speaker overlap using preloaded segmentation model."""
    print("[INFO] Running overlap detection...")
    model = model_pool.get_segmentation_model()
    device = model_pool.device

    inference = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=model_pool.batch_size,
        device=device,
    )
    try:
        segmentations = inference({"audio": Path(wav_path)})
        frame_windows = inference.model.receptive_field

        # Aggregate and count active speakers
        count_feat = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count_feat.data = np.rint(count_feat.data).astype(np.uint8)
        count_data = count_feat.data.squeeze()
        sliding_window = count_feat.sliding_window
        total_overlap_duration = 0.0
        current_start = None
        for i, val in enumerate(count_data):
            timestamp = sliding_window[i].start
            if val >= 2:
                if current_start is None:
                    current_start = timestamp
            else:
                if current_start is not None:
                    current_end = timestamp
                    duration = current_end - current_start
                    if duration >= threshold:
                        total_overlap_duration += duration
                    current_start = None

        if current_start is not None:
            current_end = sliding_window[-1].end
            duration = current_end - current_start
            if duration >= threshold:
                total_overlap_duration += duration
        has_overlap = total_overlap_duration > 0
        return has_overlap

    finally:
        del inference, segmentations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_vad(wav_path: str, out_file: str):
    """Run VAD using preloaded model."""
    print("[INFO] Running voice activity detection...")
    vad_pipeline = model_pool.get_vad_pipeline()
    result = vad_pipeline(wav_path)[0]
    vad_time = [[round(v[0] / 1000, 3), round(v[1] / 1000, 3)] for v in result['value']]

    basename = Path(wav_path).stem
    json_dict = {}
    for start, end in vad_time:
        seg_id = f"{basename}_{start}_{end}"
        json_dict[seg_id] = {
            "file": wav_path,
            "start": start,
            "stop": end
        }
    os.makedirs(Path(out_file).parent, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(json_dict, f, indent=2)
    print(f"[INFO] VAD saved to {out_file}")
    return json_dict


def generate_subsegments(vad_json_path: str, out_file: str, dur: float = 1.5, shift: float = 0.75):
    """Generate overlapping subsegments from VAD output."""
    print("[INFO] Generating sub-segments...")
    with open(vad_json_path, 'r') as f:
        vad_json = json.load(f)

    subseg_json = {}
    for segid in vad_json:
        wavid = segid.rsplit('_', 2)[0]
        st = vad_json[segid]['start']
        ed = vad_json[segid]['stop']
        subseg_st = st
        while subseg_st + dur < ed + shift:
            subseg_ed = min(subseg_st + dur, ed)
            item = vad_json[segid].copy()
            item.update({
                'start': round(subseg_st, 2),
                'stop': round(subseg_ed, 2)
            })
            subsegid_new = f"{wavid}_{round(subseg_st, 2)}_{round(subseg_ed, 2)}"
            subseg_json[subsegid_new] = item
            subseg_st += shift

    os.makedirs(Path(out_file).parent, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(subseg_json, f, indent=2)
    print(f"[INFO] Subsegments saved to {out_file}")


def merge_overlap_region(vad_time_list):
    if not vad_time_list:
        return []
    vad_time_list.sort(key=lambda x: x[0])
    out_vad_time_list = []
    for t in vad_time_list:
        if len(out_vad_time_list) == 0 or t[0] > out_vad_time_list[-1][1]:
            out_vad_time_list.append(t[:])
        else:
            out_vad_time_list[-1][1] = max(out_vad_time_list[-1][1], t[1])
    return out_vad_time_list

def create_debug_path(debug_dir, name):
    if not debug_dir:
        return None
    path = Path(debug_dir) / f"{name}_DEBUG.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def make_rttms(seg_list, out_rttm, rec_id):
    """
    Merge overlapping segments and write RTTM format.
    seg_list: list of [(start_time, end_time), label]
    """
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed = float(seg[0][0]), float(seg[0][1])
        cluster_id = int(seg[1]) + 1  # 1-indexed

        if not new_seg_list:
            new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
        else:
            last = new_seg_list[-1]
            if cluster_id == last[3]:  # Same speaker
                if seg_st > last[2]:
                    new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])
                else:
                    last[2] = max(last[2], seg_ed)  # Extend end time
            else:  # Different speaker
                if seg_st < last[2]:  # Overlap → split at midpoint
                    mid = (last[2] + seg_st) / 2
                    last[2] = mid
                    seg_st = mid
                new_seg_list.append([rec_id, seg_st, seg_ed, cluster_id])

    line_str = "SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
    with open(out_rttm, 'w') as f:
        for seg in new_seg_list:
            f.write(line_str.format(seg[0], seg[1], seg[2] - seg[1], seg[3]))
    print(f"[INFO] RTTM saved to {out_rttm}")


def extract_wav_embeddings(subseg_json_path: str, wav_emb_path: str):
    """Extract embeddings using preloaded embedding models."""
    print("[INFO] Extracting speaker embeddings...")
    device = model_pool.device
    batch_size = model_pool.batch_size
    feature_extractor, embedding_model = model_pool.get_embedding_components()

    with open(subseg_json_path, 'r') as f:
        subseg_json = json.load(f)
    if not subseg_json:
        print("[WARNING] No segments found. Skipping embedding extraction.")
        return
    all_keys = list(subseg_json.keys())
    if Path(wav_emb_path).exists():
        print(f"[INFO] Embedding already exists: {wav_emb_path}, skipping.")
        return

    wav_path = subseg_json[all_keys[0]]['file']
    wav = load_audio(wav_path, obj_fs=16000)

    wavs = []
    times = []
    for key in subseg_json:
        start = int(subseg_json[key]['start'] * 16000)
        end = int(subseg_json[key]['stop'] * 16000)
        wavs.append(wav[0, start:end])  # mono
        times.append([subseg_json[key]['start'], subseg_json[key]['stop']])

    max_len = max(w.shape[0] for w in wavs)
    wavs = [circle_pad(w, max_len) for w in wavs]
    wavs_tensor = torch.stack(wavs).unsqueeze(1)  # (B, 1, T)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(wavs_tensor), batch_size):
            batch = wavs_tensor[i:i + batch_size].to(device)
            feats = torch.vmap(feature_extractor)(batch)
            embs_batch = embedding_model(feats).cpu()
            embeddings.append(embs_batch)

    embeddings = torch.cat(embeddings, dim=0).numpy()

    result = {
        'embeddings': embeddings,
        'times': times
    }
    with open(wav_emb_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"[INFO] Embeddings saved to {wav_emb_path}")

def extract_visual_embeddings(
    vad_data: json, 
    video_path: str, 
    wav_path: str, 
    face_emb_pkl: str, 
    debug_dir:str
):
    rec_id = video_path.stem
    subset = {k: v for k, v in vad_data.items() if k.rsplit('_', 2)[0] == rec_id}
    if len(subset) == 0:
        print(f"[WARNING] No VAD segments for {rec_id}.")
        return None
    rec_vad_time_list = [[v['start'], v['stop']] for v in subset.values()]
    rec_vad_time_list = merge_overlap_region(rec_vad_time_list)
    debug_video = create_debug_path(debug_dir, rec_id)
    
    try:
        vp = VisionProcesser(
            video_file_path = video_path,
            audio_file_path = wav_path,
            audio_vad = rec_vad_time_list,
            out_feat_path = face_emb_pkl,
            visual_models = model_pool,
            conf = model_pool.conf,
            out_video_path=debug_video
        )
        vp.run()
    except Exception as e:
        print(f"[ERROR] Failed to process {video_path}: {e}")
        raise
    finally:
        if 'vp' in locals():
            vp.close()

def audio_only_cluster(audio_embs_file, rttm_file, rec_id, config):
    print("[INFO] Running audio-only clustering...")
    cluster = build('audio_cluster', config)
    if not os.path.exists(audio_embs_file):
        print(f"[ERROR] Audio embedding file not found: {audio_embs_file}")
        return False

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
    make_rttms(seg_list, rttm_file, rec_id)
    return True


def audio_visual_cluster(audio_embs_file, visual_embs_file, rttm_file, rec_id, config):
    print("[INFO] Running audio-visual joint clustering...")
    cluster = build('cluster', config)
    if not os.path.exists(audio_embs_file):
        print(f"[ERROR] Audio embedding file not found: {audio_embs_file}")
        return False
    if not os.path.exists(visual_embs_file):
        print(f"[ERROR] Visual embedding file not found: {visual_embs_file}")
        return False

    # Load audio embeddings
    with open(audio_embs_file, 'rb') as f:
        a_data = pickle.load(f)
        audio_embeddings = a_data['embeddings']
        audio_times = a_data['times']

    # Load visual embeddings
    with open(visual_embs_file, 'rb') as f:
        v_data = pickle.load(f)
        visual_embeddings = v_data['embeddings']
        frameI = v_data['frameI']
        faceI = v_data['faceI']
        visual_times = frameI * 0.04
        frame_indices = [np.where(faceI == frame)[0][0] for frame in frameI]
        speak_embeddings = visual_embeddings[frame_indices]
        visual_embeddings_normalized = speak_embeddings / np.sqrt(np.sum(speak_embeddings**2, axis=-1, keepdims=True))

    labels = cluster(audio_embeddings, visual_embeddings_normalized, audio_times, visual_times, config)
    # output rttm
    new_labels = np.zeros(len(labels), dtype=int)
    uniq = np.unique(labels)
    for i in range(len(uniq)):
        new_labels[labels==uniq[i]] = i 
    seg_list = [(i,j) for i, j in zip(audio_times, new_labels)]
    make_rttms(seg_list, rttm_file, rec_id)
    return True


def main():
    parser = argparse.ArgumentParser(description="Process a single video for speaker embedding extraction.")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video file")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory to save intermediate files")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace access token for pyannote")
    parser.add_argument("--config", default="diar.yaml", help="YAML config file")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to local pretrained models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cuda' or 'cpu'.")
    parser.add_argument("--jointcluster", action="store_true", help="Use audio-visual joint clustering. If not set, use audio-only clustering.")
    parser.add_argument("--debug_dir", type=str, default="", help="Optional: save debug video")
    args = parser.parse_args()

    global model_pool
    model_pool = GlobalModels(
        hf_token = args.hf_token,
        config_path = args.config,
        pretrained_dir= args.pretrained,
        device= args.device,
        pool_sizes = {"face": 1, "asd": 8, "fr": 3},
        batch_size = args.batch_size,
        preload = True
    )

    video_path = Path(args.video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rec_id = video_path.stem
    wav_path = work_dir / f"{rec_id}.wav"
    vad_json = work_dir / "vad.json"
    subseg_json = work_dir / "subseg.json"
    wav_emb_pkl = work_dir / "audio.pkl"
    face_emb_pkl = work_dir / "face.pkl"
    rttm_file = work_dir / f"{rec_id}.rttm"

    # Pipeline Start
    infer_start = time.time()
    
    # 1. Extract audio
    extract_audio_from_video(video_path, wav_path)

    # 2. Overlap detection
    if detect_overlap(str(wav_path), threshold=1.0):
        print("[WARNING] Speaker overlap detected. Skipping this video.")
        os.remove(wav_path)
        return
    
    # 3. VAD
    vad_data = run_vad(str(wav_path), str(vad_json))

    # 4. Sub-segment
    generate_subsegments(str(vad_json), str(subseg_json), dur=1.5, shift=0.75)

    # 5. Extract audio embeddings
    extract_wav_embeddings(str(subseg_json), str(wav_emb_pkl))
    
    # 6. Extract visual embeddings
    extract_visual_embeddings(vad_data, video_path, str(wav_path), str(face_emb_pkl), args.debug_dir)
    
    # 7. Cluster audio and visual embeddings
    config = build_config(args.config)
    if args.jointcluster and face_emb_pkl.exists():
        success = audio_visual_cluster(
            str(wav_emb_pkl),
            str(face_emb_pkl),
            str(rttm_file),
            rec_id,
            config
        )
    else:
        print("[INFO] Visual embeddings not found, using audio-only mode.")
        success = audio_only_cluster(
            str(wav_emb_pkl),
            str(rttm_file),
            rec_id,
            config
        )
    
    inference_time = time.time() - infer_start
    if success:
        print("✅ PROCESSING COMPLETED")
    else:
        print("[FAILED] Clustering failed.")
    print(f"Inference Time: {inference_time:.2f}s")


if __name__ == "__main__":
    main()
