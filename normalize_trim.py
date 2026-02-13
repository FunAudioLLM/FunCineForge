#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import multiprocessing as mp
from tqdm import tqdm

# 支持的视频扩展名
VIDEO_EXTENSIONS = ('.mkv', '.avi', '.mov', '.flv', '.wmv', '.rmvb', '.webm', '.mp4')

def ensure_dependencies():
    """检查 ffprobe 和 ffmpeg 是否可用"""
    for cmd in ["ffprobe", "ffmpeg"]:
        result = subprocess.run(["which", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERROR] 系统未安装 '{cmd}'，请先安装 ffmpeg 工具包。")
            exit(1)

def normalize_filename(filename: str) -> str:
    name, _ = os.path.splitext(filename)
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name

def get_video_duration(video_path):
    """获取视频时长"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=True
        )
        return float(result.stdout.strip())
    except:
        return None

def process_video(task):
    input_path, target_mp4, target_wav, intro, outro = task
    
    duration = get_video_duration(input_path)
    if duration is None:
        return f"[ERROR] 无法读取时长: {input_path}"

    start = intro
    end = duration - outro

    if end <= start:
        return f"[ERROR] 视频太短无法剪裁: {input_path}"
    
    print(f"[INFO] {input_path} → {target_mp4}, {target_wav}")
    # FFmpeg 命令，非MP4转MP4、视频剪裁、音频提取
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        # 输出 1: 视频处理
        "-ss", str(start), 
        "-i", input_path,
        "-to", str(end),
        "-map_metadata", "-1",
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
        target_mp4,
        # 输出 2: 音频处理
        "-vn", "-c:a", "pcm_s16le", "-f", "wav",
        target_wav
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.abspath(input_path) != os.path.abspath(target_mp4):
            os.remove(input_path)
        return f"[SUCCESS] {input_path} → {target_mp4}, {target_wav}"
    except subprocess.CalledProcessError as e:
        return f"[ERROR] 处理失败 {input_path}: {e}"
    

def collect_tasks(root_dir, intro, outro):
    """扫描目录，按组排序，生成任务列表"""
    all_tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        video_files = sorted([f for f in filenames if f.lower().endswith(VIDEO_EXTENSIONS)])
        
        if not video_files:
            continue
            
        for idx, filename in enumerate(video_files, start=1):
            input_path = os.path.join(dirpath, filename)
            target_mp4 = os.path.join(dirpath, f"{idx:02d}" + ".mp4")
            target_wav = os.path.join(dirpath, f"{idx:02d}" + ".wav")
            
            # 命名冲突检查
            if os.path.abspath(input_path) == os.path.abspath(target_mp4):
                temp_input = os.path.join(dirpath, f"source_{filename}")
                os.rename(input_path, temp_input)
                input_path = temp_input
            
            all_tasks.append((input_path, target_mp4, target_wav, intro, outro))
            
    return all_tasks

def main():
    ensure_dependencies()
    
    parser = argparse.ArgumentParser(description="批量视频标准化：统一格式、统一文件名、视频剪裁、提取音频")
    parser.add_argument("--root", type=str, required=True, help="视频根目录")
    parser.add_argument("--intro", type=int, default=10, help="剪掉片头秒数") # 请自行设定
    parser.add_argument("--outro", type=int, default=10, help="剪掉片尾秒数") # 请自行设定
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="并发数")
    
    args = parser.parse_args()

    print(f"[INFO] 正在扫描: {args.root}")
    tasks = collect_tasks(args.root, args.intro, args.outro)
    
    if not tasks:
        print("[ERROR] 未找到视频文件。")
        return

    print(f"[INFO] 准备处理 {len(tasks)} 个视频")
    with mp.Pool(processes=args.workers) as pool:
        pbar = tqdm(total=len(tasks), desc="Processing")
        for result in pool.imap_unordered(process_video, tasks):
            pbar.write(result)
            pbar.update(1)
        pbar.close()

    print("[SUCCESS] 处理完成！所有源视频已转为裁剪后的标准化 MP4 和 WAV。")

if __name__ == "__main__":
    main()
