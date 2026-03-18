#!/usr/bin/env python3
"""
FunCineForge Gradio Demo
Run from the repo root:  python gradio_demo.py
"""

import os, sys, json, pickle, tempfile, uuid, shutil, logging
import numpy as np
import cv2
import gradio as gr

# ── make sure the repo root is on the path ─────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
EXPS_DIR = os.path.join(ROOT, "exps")

# ── checkpoint paths (relative to exps/) ───────────────────────────────────
LM_CKPT  = "funcineforge_zh_en/llm/ds-model.pt.best/mp_rank_00_model_states.pt"
FM_CKPT  = "funcineforge_zh_en/flow/ds-model.pt.best/mp_rank_00_model_states.pt"
VOC_CKPT = "funcineforge_zh_en/vocoder/ds-model.pt.best/avg_5_removewn.pt"
FACE_ONNX = os.path.join(ROOT, "speaker_diarization/pretrained_models/face_recog_ir101.onnx")
DECODE_CONF = os.path.join(EXPS_DIR, "decode_conf")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("FunCineForge-Demo")

# ── lazy model loading ──────────────────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from funcineforge import AutoModel
    from funcineforge.models.utils import dtype_map

    os.chdir(EXPS_DIR)

    def _load(exp_dir, model_name, ckpt_path, device="cuda:0"):
        return AutoModel(
            model=os.path.join(exp_dir, model_name),
            init_param=ckpt_path,
            output_dir=None,
            device=device,
        )

    log.info("Loading LM model …")
    lm_exp_dir, lm_model_name, _, _ = LM_CKPT.rsplit("/", 3)
    lm_model = _load(lm_exp_dir, lm_model_name, LM_CKPT)
    lm_model.model.to(dtype_map["fp32"])

    log.info("Loading FM model …")
    fm_exp_dir, fm_model_name, _, _ = FM_CKPT.rsplit("/", 3)
    fm_model = _load(fm_exp_dir, fm_model_name, FM_CKPT)
    fm_model.model.to(dtype_map["fp32"])

    log.info("Loading Vocoder …")
    voc_exp_dir, voc_model_name, _, _ = VOC_CKPT.rsplit("/", 3)
    voc_model = _load(voc_exp_dir, voc_model_name, VOC_CKPT)
    voc_model.model.to(dtype_map["fp32"])

    log.info("Building inference model …")
    from funcineforge.register import tables

    infer_model = AutoModel(
        model="FunCineForgeInferModel",
        model_conf={},
        output_dir=None,
        device="cuda:0",
        lm_model=lm_model,
        fm_model=fm_model,
        voc_model=voc_model,
        tokenizer=None,
        # decode params
        sampling="ras",
        lm_use_prompt=True,
        fm_use_prompt=True,
        use_llm_cache=True,
        max_length=1500,
        min_length=50,
        llm_dtype="fp32",
        fm_dtype="fp32",
        voc_dtype="fp32",
        batch_size=1,
        xvec_model="funcineforge_zh_en/camplus.onnx",
        dataset_conf={
            "load_meta_data_key": "text,clue,face,dialogue,vocal,video",
            "sos": 6561, "eos": 6562,
            "turn_of_speech": 6563, "fill_token": 6564,
            "ignore_id": -100,
            "startofclue_token": 151646, "endofclue_token": 151647,
            "frame_shift": 25, "timebook_size": 1500,
            "pangbai": 1500, "dubai": 1501, "duihua": 1502, "duoren": 1503,
            "male": 1504, "female": 1505,
            "child": 1506, "youth": 1507, "adult": 1508,
            "middle": 1509, "elderly": 1510,
            "speaker_id_start": 1511,
        },
        index_ds="FunCineForgeDS",
        disable_pbar=True,
        random_seed=0,
    )

    _pipeline = infer_model
    log.info("Pipeline ready ✓")
    return _pipeline


# ── face embedding extraction using the ONNX model ─────────────────────────
_face_ort = None

def get_face_ort():
    global _face_ort
    if _face_ort is None:
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
        _face_ort = onnxruntime.InferenceSession(
            FACE_ONNX, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _face_ort


def preprocess_face(img_bgr: np.ndarray) -> np.ndarray:
    """Resize & normalise a face crop for the IR101 face-rec model (112×112)."""
    face = cv2.resize(img_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = (face - 127.5) / 127.5
    face = face.transpose(2, 0, 1)[np.newaxis]          # (1, 3, 112, 112)
    return face


def extract_face_embeddings_from_video(video_path: str, every_n_frames: int = 5):
    """
    Sample frames from a video every `every_n_frames` frames,
    detect the dominant face bounding box (using simple haar or centre-crop),
    and compute 512-d embeddings with the ONNX face-rec model.

    Returns a dict with keys: embeddings, faceI, frameI  (matching the .pkl format).
    """
    ort = get_face_ort()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    embeddings, faceI, frameI = [], [], []
    speech_token_idx = 0   # maps to speech-token index (25 tokens/sec → one token per 40 ms)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            # Map video-frame idx → speech-token idx
            token_idx = int(frame_idx / fps * 25)   # 25 tokens/s

            # Try face detection; fall back to centre crop
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
                crop = frame[y: y + h, x: x + w]
            else:
                # centre-crop fallback
                h, w = frame.shape[:2]
                side = min(h, w)
                y0, x0 = (h - side) // 2, (w - side) // 2
                crop = frame[y0: y0 + side, x0: x0 + side]

            inp = preprocess_face(crop)
            emb = ort.run(None, {ort.get_inputs()[0].name: inp})[0][0]  # (512,)
            embeddings.append(emb.astype(np.float32))
            faceI.append(token_idx)
            frameI.append(token_idx)
        frame_idx += 1

    cap.release()
    return {"embeddings": embeddings, "faceI": np.array(faceI), "frameI": np.array(frameI),
            "face": [], "face_bbox": [], "lip": [], "lip_bbox": []}


# ── inference helper ────────────────────────────────────────────────────────
def run_inference(
    text: str,
    clue: str,
    vocal_path: str,
    video_path: str,
    gender: str,
    age: str,
    speech_type: str,
    duration: float,
    output_dir: str,
    utt_id: str,
):
    """Build the data dict the pipeline expects and run inference."""
    from funcineforge.register import tables

    # ── face embeddings ─────────────────────────────────────────────────────
    log.info("Extracting face embeddings …")
    face_pkl_path = os.path.join(output_dir, f"{utt_id}.pkl")
    face_data = extract_face_embeddings_from_video(video_path)
    with open(face_pkl_path, "wb") as f:
        pickle.dump(face_data, f)

    # ── dialogue metadata ───────────────────────────────────────────────────
    dialogue = [{"start": 0.0, "duration": duration, "spk": "1",
                 "gender": gender.lower(), "age": age.lower()}]

    # ── dataset conf (must match decode.yaml) ──────────────────────────────
    TIMEBOOK_SIZE = 1500
    TYPE_MAP = {"monologue": TIMEBOOK_SIZE + 1, "dialogue": TIMEBOOK_SIZE + 2,
                "narration": TIMEBOOK_SIZE, "multi-speaker": TIMEBOOK_SIZE + 3}
    type_id = TYPE_MAP.get(speech_type, TIMEBOOK_SIZE + 1)

    GENDER_MAP = {"male": TIMEBOOK_SIZE + 4, "female": TIMEBOOK_SIZE + 5}
    AGE_MAP = {"child": TIMEBOOK_SIZE + 6, "teenager": TIMEBOOK_SIZE + 7,
               "adult": TIMEBOOK_SIZE + 8, "middle-aged": TIMEBOOK_SIZE + 9,
               "elderly": TIMEBOOK_SIZE + 10}
    SPEAKER_ID_START = TIMEBOOK_SIZE + 11
    FRAME_SHIFT = 25

    starts   = np.array([d["start"]    for d in dialogue])
    durations= np.array([d["duration"] for d in dialogue])
    speakers = np.array([int(d["spk"]) for d in dialogue])
    start_idxs = (starts * FRAME_SHIFT + 1).astype(np.int64)
    end_idxs   = ((starts + durations) * FRAME_SHIFT + 1).astype(np.int64)
    spk_ids    = (SPEAKER_ID_START + speakers - 1).astype(np.int64)
    gender_ids = [GENDER_MAP.get(d["gender"], -100) for d in dialogue]
    age_ids    = [AGE_MAP.get(d["age"],    -100) for d in dialogue]

    n = len(dialogue)
    timespk_ids = np.full(n * 5, -100, dtype=np.int64)
    timespk_ids[0::5] = start_idxs
    timespk_ids[1::5] = spk_ids
    timespk_ids[2::5] = gender_ids
    timespk_ids[3::5] = age_ids
    timespk_ids[4::5] = end_idxs

    # ── build data dict ─────────────────────────────────────────────────────
    data = {
        "utt": utt_id,
        "text": text,
        "clue": clue,
        "vocal": vocal_path,
        "video": video_path,
        "face": face_pkl_path,
        "type_id": type_id,
        "timespk_ids": timespk_ids,
        "speech_len": int(duration * FRAME_SHIFT),
        "source_len": int(duration * FRAME_SHIFT) * 2 + 200,
    }

    # ── run the pipeline ────────────────────────────────────────────────────
    pipeline = get_pipeline()

    class _SingleDS:
        def __len__(self): return 1
        def __getitem__(self, _): return data

    pipeline.kwargs["output_dir"] = output_dir
    pipeline.inference(input=_SingleDS(), input_len=1)

    wav_path = os.path.join(output_dir, "wav", f"{utt_id}.wav")
    mp4_path = os.path.join(output_dir, "mp4", f"{utt_id}.mp4")
    return (wav_path if os.path.exists(wav_path) else None,
            mp4_path if os.path.exists(mp4_path) else None)


# ── Gradio callback ─────────────────────────────────────────────────────────
def infer_gradio(
    text, clue, vocal_file, video_file,
    gender, age, speech_type,
    duration, progress=gr.Progress(track_tqdm=True)
):
    if not text.strip():
        raise gr.Error("Please enter the script text.")
    if vocal_file is None:
        raise gr.Error("Please upload a reference audio file.")
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    # Get the actual file paths (Gradio passes temp file paths)
    vocal_path = vocal_file if isinstance(vocal_file, str) else vocal_file.name
    video_path = video_file if isinstance(video_file, str) else video_file.name

    # Create a unique temp output dir
    run_id = str(uuid.uuid4())[:8]
    out_dir = os.path.join(tempfile.gettempdir(), f"fcf_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        progress(0.1, desc="Loading models …")
        get_pipeline()       # ensure models are loaded

        progress(0.3, desc="Extracting face embeddings …")
        wav_out, mp4_out = run_inference(
            text=text.strip(),
            clue=clue.strip() if clue.strip() else "A speaker delivers the dialogue naturally.",
            vocal_path=vocal_path,
            video_path=video_path,
            gender=gender,
            age=age,
            speech_type=speech_type,
            duration=float(duration),
            output_dir=out_dir,
            utt_id=f"demo_{run_id}",
        )
        progress(1.0, desc="Done!")

        if wav_out is None:
            raise gr.Error("Inference completed but no output audio was found.")

        # Copy outputs to stable temp paths so Gradio can serve them
        final_wav = os.path.join(tempfile.gettempdir(), f"fcf_{run_id}_out.wav")
        shutil.copy2(wav_out, final_wav)

        final_mp4 = None
        if mp4_out and os.path.exists(mp4_out):
            final_mp4 = os.path.join(tempfile.gettempdir(), f"fcf_{run_id}_out.mp4")
            shutil.copy2(mp4_out, final_mp4)

        return final_wav, final_mp4, "✅ Generation complete!"

    except Exception as e:
        log.exception("Inference failed")
        shutil.rmtree(out_dir, ignore_errors=True)
        raise gr.Error(f"Inference failed: {e}")


# ── UI ──────────────────────────────────────────────────────────────────────
TITLE = "Video Dubbing Demo"
DESCRIPTION = """
This demo showcases **FunCineForge**, a cinematic speech synthesis system. 

It uses an a pipeline with LLM for prosody modeling, Flow-Matching for acoustic modeling, and a high-fidelity Vocoder. By leveraging both text and visual cues (from the reference video's face embeddings), it generates character-accurate, emotionally charged speech perfectly suited for film and video dubbing.
"""

CUSTOM_CSS = """
    #generate-btn { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; font-size: 1.1rem; }
    #status-box { font-size: 0.9rem; color: #6b7280; }
    .gr-panel { border-radius: 12px !important; }
    footer { display: none !important; }
"""
DEMO_THEME = gr.themes.Ocean()

with gr.Blocks(title="FunCineForge Demo") as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")

    with gr.Row():
        # ── LEFT: inputs ────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### Script & Prompt")
            text_input = gr.Textbox(
                label="Script Text",
                placeholder="Enter the dialogue or narration text here …",
                lines=3,
            )
            clue_input = gr.Textbox(
                label="Voice / Emotion Clue",
                placeholder=(
                    "E.g., 'A calm, 40-year-old male speaker with a deep voice, "
                    "delivering a professional and thoughtful statement.'"
                ),
                lines=3,
            )

            gr.Markdown("### Reference Files")
            with gr.Row():
                vocal_input = gr.Audio(
                    label="Reference Audio (WAV — same character)",
                    type="filepath",
                )
                video_input = gr.Video(
                    label="Reference Video (MP4 — scene to dub)",
                )

            gr.Markdown("### Speaker Metadata")
            with gr.Row():
                gender_input = gr.Radio(
                    ["male", "female"],
                    value="male",
                    label="Gender",
                )
                age_input = gr.Dropdown(
                    choices=["child", "teenager", "adult", "middle-aged", "elderly"],
                    value="adult",
                    label="Age Group",
                )
                type_input = gr.Dropdown(
                    choices=["monologue", "dialogue", "narration", "multi-speaker"],
                    value="monologue",
                    label="Scene Type",
                )
            duration_input = gr.Slider(
                minimum=1.0, maximum=30.0, value=5.0, step=0.5,
                label="Approximate Speech Duration (seconds)",
                info="Estimate of the expected audio length",
            )

            generate_btn = gr.Button("Generate Speech", elem_id="generate-btn", variant="primary")

        # ── RIGHT: outputs ───────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Generated Output")
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=False,
            )
            video_output = gr.Video(
                label="Dubbed Video (if video was provided)",
                interactive=False,
            )
            status_output = gr.Textbox(
                label="Status",
                elem_id="status-box",
                interactive=False,
            )

    # ── examples ─────────────────────────────────────────────────────────────
    def get_example_path(rel_path):
        p = os.path.join(EXPS_DIR, rel_path)
        return p if os.path.exists(p) else None

    examples = [
        [
            "Every closet on a Carnival cruise ship. To make the numbers work, I needed a lot of cedar, fast and cheap.",
            "A single middle-aged male speaker describes a business or construction requirement with a practical and matter-of-fact tone. His voice is deep and slightly gravelly, maintaining a professional and informative demeanor.",
            get_example_path("data/clipped/en_monologue_1.wav"),
            get_example_path("data/clipped/en_monologue_1.mp4"),
            "male",
            "middle-aged",
            "monologue",
            5.74,
        ],
        [
            "Oh my God. Do you remember that bottle of wine we put aside the night Haley was born?",
            "An adult female speaker expresses a sense of sudden realization and excitement. Her tone is bright and nostalgic as she recalls a significant event from the past. The overall emotion is one of pleasant surprise and anticipation.",
            get_example_path("data/clipped/en_monologue_2.wav"),
            get_example_path("data/clipped/en_monologue_2.mp4"),
            "female",
            "adult",
            "monologue",
            4.06,
        ],
        [
            "I was just letting you know that if you were having any problems, you could come to me with them.",
            "A single adult female speaker delivers a supportive and reassuring message. Her tone is friendly and caring, with a hint of advice, and her emotions fluctuate greatly. She speaks clearly and at a moderate pace, offering assistance to the listener.",
            get_example_path("data/clipped/en_monologue_3.wav"),
            get_example_path("data/clipped/en_monologue_3.mp4"),
            "female",
            "adult",
            "monologue",
            4.94,
        ],
    ]
    
    # Filter out examples where files are missing
    valid_examples = [ex for ex in examples if ex[2] and ex[3]]

    if valid_examples:
        gr.Examples(
            examples=valid_examples,
            inputs=[text_input, clue_input, vocal_input, video_input,
                    gender_input, age_input, type_input, duration_input],
            label="📂 Try an example from the demo dataset",
        )

    generate_btn.click(
        fn=infer_gradio,
        inputs=[
            text_input, clue_input, vocal_input, video_input,
            gender_input, age_input, type_input, duration_input,
        ],
        outputs=[audio_output, video_output, status_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=False,
        theme=DEMO_THEME,
        css=CUSTOM_CSS,
    )
