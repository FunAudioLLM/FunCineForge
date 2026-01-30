import os
import torch
import librosa
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from moviepy.editor import VideoFileClip
from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state, convert_pcm_to_float
from funasr import AutoModel
# If you find that the generated srt file is too fragmented, you need to add "if punc_id > 2:" to line 167 after "sentence_text += punc_list[punc_id - 2]" in funasr 1.2.7 funasr.utils.timestamp_tools file.

class VideoClipper:
    def __init__(self, funasr_model):
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0
        self.lang = 'zh'

    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input
        data = convert_pcm_to_float(data)

        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(data.shape) == 2:
            data = data.mean(axis=1)
        state['audio_input'] = (sr, data)

        if sd_switch == 'yes':
            rec_result = self.funasr_model.generate(
                data,
                return_spk_res=True,
                return_raw_text=True,
                is_final=True,
                output_dir=output_dir,
                hotword=hotwords,
                pred_timestamp=self.lang == 'en',
                en_post_proc=self.lang == 'en',
                cache={},
                merge_vad=True,
                merge_length_s=30
            )
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        else:
            rec_result = self.funasr_model.generate(
                data,
                return_spk_res=False,
                sentence_timestamp=True,
                return_raw_text=True,
                is_final=True,
                hotword=hotwords,
                output_dir=output_dir,
                pred_timestamp=self.lang == 'en',
                en_post_proc=self.lang == 'en',
                cache={},
                merge_vad=True,
                merge_length_s=30
            )
            res_srt = generate_srt(rec_result[0]['sentence_info'])

        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        res_text = rec_result[0]['text']
        del data
        return res_text, res_srt, state

    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        video = VideoFileClip(video_filename)
        base_name, _ = os.path.splitext(os.path.basename(video_filename))
        audio_file = os.path.join(output_dir or '.', base_name + '.wav')

        if video.audio is None:
            raise ValueError("No audio information found.")

        video.audio.write_audiofile(audio_file, verbose=False, logger=None)
        wav, sr = librosa.load(audio_file, sr=16000)
        if os.path.exists(audio_file):
            os.remove(audio_file)
        video.close()
        del video
        return self.recog((sr, wav), sd_switch, {'video_filename': video_filename}, hotwords, output_dir)

    def video_clip(self, state, output_dir=None):
        sentences = state['sentences']
        video_filename = state['video_filename']
        ts = []
        for s in sentences:
            start_time = s['start'] / 1000.0
            end_time = s['end'] / 1000.0
            speaker_id = s.get('spk', 'unknown')
            ts.append([start_time, end_time, speaker_id])

        srt_index = 1
        time_acc_ost = 0.0
        clipped_folder = os.path.join(output_dir, 'clipped')
        os.makedirs(clipped_folder, exist_ok=True)

        for i, (start, end, speaker_id) in enumerate(ts):
            srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index - 1, time_acc_ost=time_acc_ost)
            base_name = os.path.basename(video_filename)
            video_name_without_ext, _ = os.path.splitext(base_name)
            h, m, s = int(subs[0][0][0] // 3600), int((subs[0][0][0] % 3600) // 60), int(subs[0][0][0] % 60)
            ms = int((subs[0][0][0] - int(subs[0][0][0])) * 100)
            clip_filename = f"{video_name_without_ext}_{h:02}_{m:02}_{s:02}_{ms:02}_spk{speaker_id}"
            clip_filepath = os.path.join(clipped_folder, clip_filename)

            video_filepath = clip_filepath + '.mp4'
            audio_filepath = clip_filepath + '.wav'
            clip_srt_file = clip_filepath + '.srt'

            if not (os.path.exists(video_filepath) and os.path.exists(audio_filepath) and os.path.exists(clip_srt_file)):
                with VideoFileClip(video_filename) as video:
                    sub = video.subclip(start, end)
                    sub.write_videofile(video_filepath, audio_codec="aac", verbose=False, logger=None)
                    sub.audio.write_audiofile(audio_filepath, codec='pcm_s16le', verbose=False, logger=None)
                    sub.close()
                    del sub
                with open(clip_srt_file, 'w') as f:
                    f.write(srt_clip)
            time_acc_ost += (end - start)

        return f"{len(ts)} clips created."


def init_models(lang='zh', device='cpu'):
    if lang == 'zh':
        model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device
        )
    elif lang == 'en':
        model = AutoModel(
            model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device
        )
    else:
        raise ValueError(f"Unsupported language: {lang}")
    return model


_process_local = None

def _init_worker(lang: str, device: str):
    global _process_local
    if _process_local is not None:
        return

    # 获取当前 worker ID
    worker_id = mp.current_process().name
    try:
        pid = int(worker_id.split('-')[-1])
    except:
        pid = 1

    # 分配设备
    if device == 'cuda':
        gpu_id = (pid - 1) % torch.cuda.device_count()
        dev = f'cuda:{gpu_id}'
        print(f"[Worker-{pid}] Using GPU {gpu_id}")
    else:
        dev = 'cpu'

    _process_local = {
        'model': init_models(lang, dev),
        'device': dev
    }


def runner(stage, file, sd_switch, output_dir, lang):
    global _process_local
    funasr_model = None
    
    if stage == 1:
        if _process_local is None:
            raise RuntimeError("Model not initialized!")
        funasr_model = _process_local['model']

    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = lang

    audio_suffixes = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixes = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    ext = os.path.splitext(file)[1].lower()

    if ext in audio_suffixes:
        mode = 'audio'
    elif ext in video_suffixes:
        mode = 'video'
    else:
        print(f"❌ Unsupported format: {file}")
        return

    output_dir = output_dir.rstrip('/')
    os.makedirs(output_dir, exist_ok=True)

    if stage == 1:
        if mode == 'audio':
            wav, sr = librosa.load(file, sr=16000)
            res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch)
        else:
            res_text, res_srt, state = audio_clipper.video_recog(file, sd_switch)

        total_srt_file = os.path.join(output_dir, 'total.srt')
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
        write_state(output_dir, state)
        print(f"✅ Stage 1 success: {total_srt_file}")

    elif stage == 2 and mode == 'video':
        state = load_state(output_dir)
        state['video_filename'] = file
        message = audio_clipper.video_clip(state, output_dir=output_dir)
        print(f"✅ Stage 2 clip: {message}")


def find_all_videos(folder, base_output_dir=None, stage=1, skip_processed=True, suffixes=None):
    if suffixes is None:
        suffixes = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    all_videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in suffixes):
                file_path = os.path.join(root, file)
                if skip_processed and base_output_dir:
                    parent_dir_name = os.path.basename(root)
                    video_name = os.path.splitext(file)[0]
                    output_subdir = os.path.join(base_output_dir, parent_dir_name, video_name)
                    total_srt = os.path.join(output_subdir, 'total.srt')
                    clipped_dir = os.path.join(output_subdir, 'clipped')
                    if stage == 1 and os.path.exists(total_srt):
                        print(f"Skipping already processed: {total_srt}")
                        continue
                    if stage == 2 and os.path.isdir(clipped_dir) and len(os.listdir(clipped_dir)) > 3:
                        print(f"Skipping already processed: {clipped_dir}")
                        continue
                all_videos.append(file_path)
    return all_videos


def process_single_video(file, stage, sd_switch, base_output_dir, lang):
    try:
        video_name = os.path.splitext(os.path.basename(file))[0]
        parent_dir_name = os.path.basename(os.path.dirname(file))
        output_dir = os.path.join(base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        runner(stage, file, sd_switch, output_dir, lang)
    except Exception as e:
        print(f"❌ Failed: {file}, error: {e}")

def _wait_futures(futures, timeout=3600):
    for future in as_completed(futures, timeout=timeout):
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            print("[ERROR] Task timed out after 1 hours.")
        except Exception as e:
            print(f"[ERROR] {e}")

def get_parser():
    parser = ArgumentParser(description="ClipVideo Argument", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True, help="Stage: 1=ASR & VAD, 2=Clip")
    parser.add_argument("--file", type=str, required=True, help="Input file or folder")
    parser.add_argument("--sd_switch", type=str, default="no", choices=["no", "yes"], help="Enable speaker diarization")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed videos")
    parser.add_argument("--lang", type=str, default="zh", help="Language: zh or en")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Device to use")
    parser.add_argument("--machine_rank", type=int, default=0, help="Rank of this machine (0-indexed)")
    parser.add_argument("--total_machines", type=int, default=1, help="Total number of machines")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    file_or_folder = kwargs['file']
    stage = kwargs['stage']
    sd_switch = kwargs['sd_switch']
    output_dir = kwargs['output_dir']
    skip_processed = kwargs['skip_processed']
    lang = kwargs['lang']
    device = kwargs['device']
    machine_rank = kwargs['machine_rank']
    total_machines = kwargs['total_machines']

    if device == 'gpu':
        device = 'cuda'

    # 设置 worker 数量
    if device == 'cuda' and torch.cuda.is_available():
        max_workers = torch.cuda.device_count()
        print(f"[INFO] Machine {machine_rank+1}/{total_machines}: Using {max_workers} GPU(s)")
    else:
        max_workers = max(1, mp.cpu_count())
        print(f"[INFO] Machine {machine_rank+1}/{total_machines}: Using CPU with {max_workers} worker(s)")

    # 单文件直接处理
    if not os.path.isdir(file_or_folder):
        if stage == 1:
            _init_worker(lang, device)
        runner(stage, file_or_folder, sd_switch, output_dir, lang)
        print(f"✅ Done single file: {file_or_folder}")
        return

    # 获取所有任务并按机器分片
    all_videos = find_all_videos(file_or_folder, output_dir, stage, skip_processed)
    my_videos = [v for i, v in enumerate(all_videos) if i % total_machines == machine_rank]
    print(f"Machine {machine_rank+1}/{total_machines} assigned {len(my_videos)} out of {len(all_videos)} videos.")

    if not my_videos:
        print("No tasks assigned.")
        return

    # 多进程处理
    if stage == 1:
        with ProcessPoolExecutor(
            max_workers = max_workers,
            initializer = _init_worker,
            initargs = (lang, device)
        ) as executor:
            futures = [
                executor.submit(process_single_video, file, stage, sd_switch, output_dir, lang)
                for file in my_videos
            ]
            _wait_futures(futures)
    else:   # stage == 2
        with ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(process_single_video, file, stage, sd_switch, output_dir, lang)
                for file in my_videos
            ]
            _wait_futures(futures)

    print(f"✅ Machine {machine_rank+1}/{total_machines} completed.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
