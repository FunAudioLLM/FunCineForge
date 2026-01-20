#!/usr/bin/env python3
"""
FunCineForge 项目初始化脚本
请在FunCineForge目录下运行 python setup.py
"""
import sys
import subprocess
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

# 配置信息
TEMP_DIR = "temp_hf_models"
FILE_MAPPINGS = [
    {
        "source": ["speech_campplus", "speech_fsmn_vad", "asd.onnx", "face_recog_ir101.onnx", "fqa.onnx", "fun_2d.pth", "fun_2d.zip", "version-RFB-320.onnx"],
        "dest_dir": "speaker_diarization/pretrained_models",
    },
    {
        "source": ["Qwen2-0.5B-CosyVoice-BlankEN", "speech_tokenizer_v3.onnx"],
        "dest_dir": "tokenizer",
    },
    {
        "source": ["MelBandRoformer.ckpt"],
        "dest_dir": "speech_separation/models/melbandroformer",
    }
]

def print_step(step_num, message):
    print(f"步骤 {step_num}: {message}")

def check_requirements():
    """检查基本要求"""
    print_step(1, "检查环境")
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        return False
    current_dir = Path.cwd()
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"错误: 未在当前目录找到requirements.txt")
        return False
    return True

def install_dependencies():
    """安装Python依赖"""
    print_step(2, "安装依赖")
    try:
        print("正在安装依赖包，这可能需要几分钟...")
        
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        
        subprocess.run(
            pip_cmd,
            check=True,
            text=True,
            timeout=300
        )
        print("依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e.returncode}")
        print(f"\n{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("依赖安装超时，请检查网络连接")
        return False
    except Exception as e:
        print(f"依赖安装失败: {e}")
        return False

def check_git_available():
    """检查git是否可用"""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Git: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("未找到git命令请先安装git")
        print("  - Windows: 从 https://git-scm.com/ 下载并安装")
        print("  - macOS: 运行 'brew install git' 或从官网下载")
        print("  - Ubuntu/Debian: 运行 'sudo apt-get install git'")
        print("  - CentOS/RHEL: 运行 'sudo yum install git'")
        return False

def download_huggingface_models():
    """从HuggingFace下载模型"""
    print_step(3, "下载模型文件")
    
    if not check_git_available():
        return False
    
    temp_path = Path(TEMP_DIR)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(exist_ok=True)
    
    try:
        snapshot_download(
            repo_id="xuan3986/FunCineForge",
            local_dir=str(temp_path),
            resume_download=True,
            force_download=False,
            ignore_patterns=["*.md", ".git*"],
            token=None,
            repo_type="model",
        )   
        print("模型下载完成")
        return True
        
    except Exception as e:
        print(f"下载发生错误: {e}")
        return False

def organize_model_files():
    """整理和移动模型文件"""
    print_step(4, "整理模型文件")
    
    temp_path = Path(TEMP_DIR)
    files_to_move = []
    
    for mapping in FILE_MAPPINGS:
        dest_dir = Path(mapping["dest_dir"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for source_item in mapping["source"]:
            source_path = temp_path / source_item
            if source_path.exists():
                files_to_move.append({
                    "source": source_path,
                    "dest": dest_dir / source_item,
                })

    for file_info in files_to_move:
        try:
            if file_info["dest"].exists():
                print(f"跳过已存在的文件: {file_info['source'].name}")
            else:
                shutil.move(str(file_info["source"]), str(file_info["dest"]))
        except Exception as e:
            print(f"移动 {file_info['source'].name} 时出错: {e}")
            return False
    return True


def cleanup_temp_dir():
    temp_path = Path(TEMP_DIR)
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            print(f"已清理临时目录")
        except Exception as e:
            print(f"清理临时目录出错: {e}")
            return False
    return True

def main():
    print("FunCineForge 初始化脚本")
    
    start_time = time.time()
    
    if not check_requirements():
        sys.exit(1)
    
    if not install_dependencies():
        print("请手动运行: pip install -r requirements.txt")
        response = input("是否继续? (y/N): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    if not download_huggingface_models():
        sys.exit(1)
    
    if not organize_model_files():
        sys.exit(1)
    
    cleanup_temp_dir()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("初始化完成!")
    print(f"总耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()