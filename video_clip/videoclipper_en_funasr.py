import os
import torch
import librosa
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from moviepy.editor import VideoFileClip, AudioFileClip
from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser
from utils.trans_utils import write_state, load_state, convert_pcm_to_float
from funasr import AutoModel
# If you find that the generated srt file is too fragmented, you need to add "if punc_id > 2:" to line 167 after "sentence_text += punc_list[punc_id - 2]" in funasr 1.2.7 funasr.utils.timestamp_tools file.
_MODEL_CACHE = {}
_process_local = None

class VideoClipper():
    def __init__(self, funasr_model):
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0
        self.lang = 'zh'

    def recog(self, audio_input, sd_switch='yes', state=None):
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
                sentence_timestamp=True, 
                return_raw_text=True, 
                is_final=True,
                pred_timestamp=self.lang=='en',
                en_post_proc=self.lang=='en',
                cache={},
                merge_vad=True,               # 开启 VAD 合并
                merge_length_s=30             # 设置合并目标段最大长度（秒）
            )
        else:
            rec_result = self.funasr_model.generate(
                data, 
                return_spk_res=False, 
                sentence_timestamp=True, 
                return_raw_text=True, 
                is_final=True, 
                pred_timestamp=self.lang=='en',
                en_post_proc=self.lang=='en',
                cache={},
                merge_vad=True,               # 开启 VAD 合并
                merge_length_s=30             # 设置合并目标段最大长度（秒）
            )
        res_srt = generate_srt(rec_result[0]['sentence_info'])
            
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        # res_text = rec_result[0]['text']
        del data
        return res_srt, state
    

    def video_recog(self, video_filename, sd_switch='yes', output_dir=None):
        # Extract the base name, add '_clip.mp4', and 'wav'
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)
        else:
            base_name, _ = os.path.splitext(video_filename)
            audio_file = base_name + '.wav'
            
        with VideoFileClip(video_filename) as video:
            if video.audio is None:
                raise ValueError("No audio information found.")
            video.audio.write_audiofile(audio_file, codec='pcm_s16le', verbose=False, logger=None)
            video.close()
            del video
            
        wav, sr = librosa.load(audio_file, sr=16000)
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return self.recog((sr, wav), sd_switch, {'video_filename': video_filename})

    def video_clip(self, state, output_dir=None):
        """
        Clip the video based on the given dest_text or provided timestamps in the state.
        """
        # Retrieve data from the state
        sentences = state['sentences']
        video_file = state['video_file']
        vocal_file = state['vocal_file']
        raw_audio_file = state['raw_audio_file']
        instrumental_file = state['instrumental_file']
        
        # timestamps
        ts = []
        for sentence in sentences:
            start_time = sentence['start'] / 1000.0  # Convert to seconds
            end_time = sentence['end'] / 1000.0  # Convert to seconds
            speaker_id = sentence.get('spk', 'unknown')  # Get speaker id from the sentence
            ts.append([start_time, end_time, speaker_id])
        srt_index = 1
        time_acc_ost = 0.0

        if len(ts):
            time_acc_ost = 0.0
            for i, (start, end, speaker_id) in enumerate(ts):
                clipped_folder = os.path.join(output_dir, 'clipped')
                vocal_folder = os.path.join(output_dir, 'vocals')
                instrumental_folder = os.path.join(output_dir, 'instrumental')
                os.makedirs(clipped_folder, exist_ok=True)
                os.makedirs(vocal_folder, exist_ok=True)
                os.makedirs(instrumental_folder, exist_ok=True)

                # Create filename
                srt_clip, subs, srt_index = generate_srt_clip(
                    sentences, start, end, begin_index=srt_index-1, time_acc_ost=time_acc_ost
                )
                if not subs:
                    print(f"[WARNING] 空片段跳过生成")
                    continue
                base_name = os.path.basename(video_file)
                video_name_without_ext, _ = os.path.splitext(base_name)
                start_hours = int(subs[0][0][0] // 3600)
                start_minutes = int((subs[0][0][0] % 3600) // 60)
                start_seconds = int(subs[0][0][0] % 60)
                start_milliseconds = int((subs[0][0][0] - int(subs[0][0][0])) * 100)  # Extract milliseconds
                if speaker_id != 'unknown':
                    clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}_spk{speaker_id}"
                else:
                    clip_filename = f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_{start_milliseconds:02}"
        
                # Clip the audio and video and Write the clips
                clip_filepath = os.path.join(clipped_folder, clip_filename)
                video_clip = clip_filepath + '.mp4'
                audio_clip = clip_filepath + '.wav'
                clip_srt_file = clip_filepath + '.srt'
                vocal_clip = os.path.join(vocal_folder, clip_filename) + '.wav'
                instrumental_clip = os.path.join(instrumental_folder, clip_filename) + '.wav'
                
                if not (os.path.exists(video_clip) and os.path.exists(audio_clip) and os.path.exists(clip_srt_file)):
                    # clip the raw video and audio of this video
                    if vocal_file is None:
                        with VideoFileClip(video_file) as video:
                            sub = video.subclip(start, end)
                            sub.write_videofile(video_clip, audio_codec="aac", verbose=False, logger=None)
                            sub.audio.write_audiofile(audio_clip, codec='pcm_s16le', verbose=False, logger=None)
                            sub.close()
                            del sub
                    else:   # clip the raw video, raw audio, vocal and instrumental
                        with VideoFileClip(video_file) as video:
                            sub_v = video.subclip(start, end)
                            sub_v.write_videofile(video_clip, audio_codec="aac", verbose=False, logger=None)
                            sub_v.close()
                            del sub_v
                        with AudioFileClip(vocal_file) as vocal:
                            sub_vo = vocal.subclip(start, end)
                            sub_vo.write_audiofile(vocal_clip, codec='pcm_s16le', verbose=False, logger=None)
                            sub_vo.close()
                            del sub_vo
                        if instrumental_file:
                            with AudioFileClip(instrumental_file) as instrumental:
                                sub_in = instrumental.subclip(start, end)
                                sub_in.write_audiofile(instrumental_clip, codec='pcm_s16le', verbose=False, logger=None)
                                sub_in.close()
                                del sub_in
                        if raw_audio_file:
                            with AudioFileClip(raw_audio_file) as audio:
                                sub_a = audio.subclip(start, end)
                                sub_a.write_audiofile(audio_clip, codec='pcm_s16le', verbose=False, logger=None)
                                sub_a.close()
                                del sub_a
                    # Write the SRT file
                    with open(clip_srt_file, 'w') as fout:
                        fout.write(srt_clip)

                time_acc_ost += (end - start)
            
            message = f"{len(ts)} periods found in the speech, clips created."
        else:
            message = "[WARNING] No valid periods found in the speech."

        return message
    
def init_models(lang='zh', device='cpu'):
    """初始化模型，使用缓存避免大量下载"""
    cache_key = f"{lang}_{device}"
    
    # 检查缓存中是否已有模型
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    # 下载并缓存模型
    if lang == 'zh':
        funasr_model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device
        )
    elif lang == 'en':
        funasr_model = AutoModel(
            # model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
            model="iic/speech_paraformer_asr-en-16k-vocab4199-1B-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_kwargs={"speech_to_sil_time_thres": 300, "max_end_silence_time": 1200, "speech_noise_thres": 0.7, "decibel_thres": -80.0},
            punc_model="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
            device=device
        )
    else:
        raise ValueError(f"Unsupported language: {lang}")
    
    _MODEL_CACHE[cache_key] = funasr_model
    return funasr_model


def _init_worker(lang: str, device: str):
    global _process_local
    if _process_local is None:
        worker_id = mp.current_process().name
        if '-' in worker_id:
            try:
                pid = int(worker_id.split('-')[-1])
            except:
                pid = 1
        else:
            pid = 1
        if device == 'cuda' and torch.cuda.is_available():
            gpu_id = (pid - 1) % torch.cuda.device_count()  # 循环分配 GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # 验证隔离是否生效
            visible_gpus = torch.cuda.device_count()
            if visible_gpus != 1:
                raise RuntimeError(
                    f"[CRITICAL] Worker-{pid} 应仅见1张GPU，实际见{visible_gpus}张！"
                    f"检查 CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
                )
            
            print(f"✅ [Worker-{pid}] 硬件绑定: 物理GPU {gpu_id}")
            _process_local = {'model': init_models(lang, f'cuda:{gpu_id}')}
        else:
            _process_local = {'model': init_models(lang, 'cpu')}


def runner(stage, video_file, raw_audio_file, vocal_file, instrumental_file, sd_switch, output_dir, lang):
    global _process_local
    funasr_model = None
    if stage == 1:
        if _process_local is None:
            raise RuntimeError("Model not initialized!")
        funasr_model = _process_local['model']

    audio_clipper = VideoClipper(funasr_model)
    audio_clipper.lang = lang
    if vocal_file:
        mode = 'audio'
    else:
        mode = 'video'
        
    while output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    os.makedirs(output_dir, exist_ok=True)
        
    if stage == 1:
        if mode == 'audio':
            wav, sr = librosa.load(vocal_file, sr=16000)
            res_srt, state = audio_clipper.recog((sr, wav), sd_switch, {'audio_filename': vocal_file})
        elif mode == 'video':
            res_srt, state = audio_clipper.video_recog(video_file, sd_switch, output_dir)
        total_srt_file = output_dir + '/total.srt'
        with open(total_srt_file, 'w') as fout:
            fout.write(res_srt)
        write_state(output_dir, state)
        print(f"✅ Stage 1 success: {total_srt_file}")
        
    if stage == 2:
        state = load_state(output_dir)
        state['video_file'] = video_file
        state['raw_audio_file'] = raw_audio_file
        state['vocal_file'] = vocal_file
        state['instrumental_file'] = instrumental_file
        message = audio_clipper.video_clip(state, output_dir=output_dir)
        print(f"✅ Stage 2 clip: {message}")
     
def count_srt_entries(srt_path):
    for enc in ['utf-8', 'gbk', 'gb2312']:
        try:
            with open(srt_path, 'r', encoding=enc) as f:
                return sum(1 for line in f if line.strip())
        except UnicodeDecodeError:
            continue
    return 0    
       
def find_all_videos(folder, base_output_dir=None, stage=1, skip_processed=True):
    all_videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                file_path = os.path.join(root, file)
                if skip_processed and base_output_dir:
                    parent_dir_name = os.path.basename(root)
                    video_name = os.path.splitext(file)[0]
                    output_subdir = os.path.join(base_output_dir, parent_dir_name, video_name)
                    total_srt = os.path.join(output_subdir, 'total.srt')
                    clipped_dir = os.path.join(output_subdir, 'clipped')
                    if stage == 1 and os.path.exists(total_srt):
                        if count_srt_entries(total_srt) != 0:
                            print(f"Skipping already processed: {total_srt}")
                            continue
                    elif stage == 2:
                        if os.path.exists(total_srt) and os.path.isdir(clipped_dir):
                            expected_count = count_srt_entries(total_srt)
                            actual_count = len(os.listdir(clipped_dir))
                            if actual_count > 0 and actual_count >= expected_count - 30:
                                print(f"Skipping stage 2: {clipped_dir} ({actual_count}/{expected_count} clips)")
                                continue
                all_videos.append(file_path)
    return all_videos


def process_single_video(video_file, stage, sd_switch, base_output_dir, lang):
    try:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        parent_dir = os.path.dirname(video_file)
        vocal_file = os.path.join(parent_dir, "vocals", f"{video_name}.wav")
        instrumental_file = os.path.join(parent_dir, "instrumental", f"{video_name}.wav")
        raw_audio_file = os.path.join(parent_dir, f"{video_name}.wav")
        if not os.path.exists(vocal_file):
            vocal_file = None
            print(f"Use audio of video {video_name}")
        else:
            print(f"Use audio of vocal {vocal_file}")
        if not os.path.exists(instrumental_file): instrumental_file = None
        if not os.path.exists(raw_audio_file): raw_audio_file = None
        parent_dir_name = os.path.basename(parent_dir)
        output_dir = os.path.join(base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)
        runner(stage, video_file, raw_audio_file, vocal_file, instrumental_file, sd_switch, output_dir, lang)
    except Exception as e:
        print(f"❌ failed: {video_file}, error: {e}")

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
    parser.add_argument("--file", type=str, required=True, help="Input video file or folder")
    parser.add_argument("--sd_switch", type=str, default="yes", choices=["no", "yes"], help="Enable speaker diarization")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed videos")
    parser.add_argument("--lang", type=str, default="zh", help="Language: zh or en")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "gpu"], help="Device to use")
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
    
    if device == 'gpu':
        device = 'cuda'

    if stage == 1 and device=='cpu':
        print(f"⏳ Pre-downloading models for {lang} on {device}...")
        init_models(lang, device)
        print("✅ Models pre-downloaded and cached")
        
    if device == 'cuda' and torch.cuda.is_available():
        max_workers = torch.cuda.device_count()
        print(f"[INFO] Using {max_workers} GPU(s)")
    else:
        max_workers = max(1, mp.cpu_count() // 2)
        print(f"[INFO] Using CPU with {max_workers} worker(s)")
    
    # 单文件测试
    if not os.path.isdir(file_or_folder):
        if stage == 1:
            _init_worker(lang, device)
        process_single_video(file_or_folder, stage, sd_switch, output_dir, lang)
        print(f"✅ Done single file: {file_or_folder}")
        return
        

    all_videos = find_all_videos(file_or_folder, output_dir, stage, skip_processed)
    print(f"Found {len(all_videos)} video files.")

    # 多进程处理
    if stage == 1:
        with ProcessPoolExecutor(
            max_workers = max_workers, 
            initializer = _init_worker, 
            initargs = (lang, device)
        ) as executor:
            futures = [
                executor.submit(process_single_video, file, stage, sd_switch, output_dir, lang)
                for file in all_videos
            ]
            _wait_futures(futures)
    else:   # stage == 2
        with ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(process_single_video, file, stage, sd_switch, output_dir, lang)
                for file in all_videos
            ]
            _wait_futures(futures)
            
    print("✅ All videos processed.")



if __name__ == '__main__':
    mp.set_start_method('spawn')
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']
    main()
