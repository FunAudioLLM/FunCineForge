# VideoClipper

## Introduction
This recipe provides a method for automatically segmenting long videos and transcribing them into subtitle files. It uses the FunASR series of models for speech activity detection, speech recognition, and punctuation prediction in the video, and then trims long audio files based on terminating punctuation marks. We use a CIF-based model to timestamp long Chinese text for trimming, achieving an accuracy of over 95%. However, due to the difficulty of timestamp prediction for English, the prediction accuracy of the FunASR-based CIF model is relatively low. Therefore, we use Qwen3-ASR to achieve automatic ASR and sentence-by-sentence segmentation for English videos.

This recipe currently supports long videos in both Chinese and English. For batch video content, we have implemented concurrency optimizations and now support high-concurrency CPU and multi-GPU operation.

## Usage
### Quick Start
For high-concurrency batch processing of videos, please place all videos in the input directory according to Illustrate's requirements.

Chinese:
```
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh --device cpu
```

English:
```
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_en --output datasets/clean/en --lang en --device gpu
```

We also provide a FunASR-based English model; if you'd like to try it, you can change the code executed in run.sh to videoclipper_en_funasr.py.

If you want to test a single video sample, you can execute:

Chinese:
```
bash run.sh -stage 1 --stop_stage 2 --input datasets/raw_zh/红楼梦/01.mp4 --output datasets/clean/zh --lang zh --device cpu
```

English:
```
bash run.sh -stage 1 --stop_stage 2 --input datasets/raw_zh/小谢尔顿第一季/01.mp4 --output datasets/clean/zh --lang en --device gpu
```

## Illustrate
- For Chinese videos, if you find that the generated srt file is too fragmented, you need to add "if punc_id > 2:" to line 167 after "sentence_text += punc_list[punc_id - 2]" in funasr 1.2.7 funasr.utils.timestamp_tools file.
  
- For English videos, please execute the following commands in this directory to configure the Qwen3-ASR environment and pre-download the model files to your local machine.
```
pip install -U qwen-asr[vllm]
pip install -U flash-attn --no-build-isolation
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B
```

- We recommend performing the Speech Separation stage first to obtain a clean vocal track. The code will automatically detect whether a vocals folder exists in the same directory as each mp4 video, which can significantly reduce the CER and WER of the ASR model.

- The input directory structure is as follows:
```
datasets/raw_zh/
│
└───film name 1
│   │   01.mp4
│   │   01.wav
│   │   02.mp4
│   │   02.wav
│   │   ...
│   └───vocals
│   │       │   01.wav
│   │       │   02.wav
│   │       │   ...
│   │
│   └───instrumental
│   │       │   01.wav
│   │       │   02.wav
│   │       │   ...
│   
└───film name 2
│   ...
```
- The output directory structure is as follows:
```
datasets/clean/zh
│
└───film name 1
│   │
│   └───01
│   │   │   total.srt
│   │   │   sentences
│   │   │   timestamp
│   │   │   recog_res_raw
│   │   └───clipped
│   │   │   │   01_00_00_00_00.mp4
│   │   │   │   01_00_00_00_00.srt
│   │   │   │   01_00_00_00_00.wav
│   │   │   ...
│   │   │ 
│   │   └───vocals
│   │       │   01_00_00_00_00.wav
│   │       │   01_00_00_08_69.wav
│   │       ...
│   └───02
│   │   │   total.srt
│   │   │   sentences
│   │   │   timestamp
│   │   │   recog_res_raw
│   │   └───clipped
│   │   │   │   02_00_00_00_00.mp4
│   │   │   │   02_00_00_00_00.srt
│   │   │   │   02_00_00_00_00.wav
│   │   │   ...
│   │   │ 
│   │   └───vocals
│   │       │   02_00_00_00_00.wav
│   │       │   02_00_00_08_69.wav
│   │       ...
│   ...
│   
└───film name 2
│   ...
```
