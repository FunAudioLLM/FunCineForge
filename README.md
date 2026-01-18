### <p align="center">「English | [简体中文](./README_zh.md)」</p>

**<p align="center"> 🎬 FunCineForge: A Unified Dataset Pipeline and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes </p>**

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>
</div>

<div align="center">  
<h4><a href="#Dataset&Demo">Dataset & Demo</a>
｜<a href="#Dataset-Pipeline">Dataset Pipeline</a>
｜<a href="#Dubbing-Model">Dubbing Model</a>
｜<a href="#Recent-Updates">Recent Updates</a>
｜<a href="#Comminicate">Comminicate</a>
</h4>
</div>

**FunCineForge** contains an end-to-end dataset pipeline for producing large-scale dubbing datasets and an MLLM-based dubbing model designed for diverse cinematic scenes. Using this pipeline, we constructed the first large-scale Chinese television dubbing dataset, which includes rich annotations and diverse scenes. In monologue, narration, dialogue, and multi-speaker scenes, our dubbing model consistently outperforms state-of-the-art methods in terms of audio quality, lip-sync, timbre transition, and instruction following.

<a name="Dataset&Demo"></a>
## Dataset & Demo 🎬
You can access this [Website]() to get our dataset samples and demo samples. 


<a name="Dataset-Pipeline"></a>
## Dataset Pipeline 🔨

### Environmental Installation

FunCineForge dataset pipeline toolkit only relies on a Python environment to run.
```shell
# Conda
git clone git@github.com:FunAudioLLM/FunCineForge.git
conda create -n FunCineForge python=3.8.20 -y && conda activate FunCineForge
sudo apt-get install ffmpeg
# Initial settings
cd FunCineForge
python setup.py
```

### Data collection
If you want to produce your own data, 
we recommend that you refer to the following requirements to collect the corresponding movies or television series.

1. Video source: TV dramas or movies, non documentaries, with more monologues or dialogue scenes, clear and unobstructed faces (such as without masks and veils).
2. Speech Requirements: Standard pronunciation, clear articulation, prominent human voice. Avoid materials with strong dialects, excessive background noise, or strong colloquialism.
3. Image Requirements: High resolution, clear facial details, sufficient lighting, avoiding extremely dark or strong backlit scenes.

### How to use

- Standardize the video format to mp4, using libx264 & libmp3lame encoding; crop the opening and ending credits of television series (the default is to crop 5 minutes each).
```shell
python normalize_mp4.py --root datasets/raw_zh
python trim_video.py --root datasets/raw_zh
```
- [Video Clip](./video_clip/README.md). For long-sequence video, VAD is used to obtain sentence-level segments, which are then transcribed using ASR to generate subtitle files. The long-sequence video is then cut into segments.
```shell
cd video_clip
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh
```
- Video duration limit and subtitle file cleaning. (Without --execute, only pre-deleted files will be printed. After checking, add --execute to confirm the deletion.)
```shell
python clean_video.py --root datasets/clean/zh --execute
python clean_srt.py --root datasets/clean/zh --execute
```
- [Speech Separation](./speech_separation/README.md). The audio is used to separate the vocals from the instrumental music.
```shell
cd speech_separation
python run.py --root datasets/clean/zh --gpus 0 1 2 3
```
- [Speaker Diarization](./speaker_diarization/README.md). Multimodal active speaker recognition obtains RTTM files; identifies the speaker's facial frames, extracts frame-level speaker face and lip raw data, identifies speaking frames from facial frames, and extracts facial features of speaking frames.
```shell
cd speaker_diarization
bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root datasets/clean/zh --gpus "0 1 2 3"
```
- Based on a mllm, the system uses audio, ASR text, and RTTM files as input. It extracts emotional clues through thought chaining and uses the large model to correct the small model solution to reduce the ASR. It also annotates character age, gender, and timbre information. Experimental results show that the large-model + small-model solution reduces the WER from 3.2% to 0.6% and the speaker ID error rate from 4.3% to 1.2%, achieving quality comparable to or even better than manual transcription. Adding the --resume enables breakpoint COT inference to prevent wasted resources from repeated COT inferences.
```shell
python cot.py --root_dir datasets/clean/zh --provider google --model gemini-2.5-flash --api_key xxx --resume
```
- (Reference) Extract speech tokens based on the CosyVoice3 tokenizer for llm training.
```shell
python speech_tokenizer.py --root datasets/clean/zh
```
- (Reference) COT results are cleaned and corrected; video clip types (monologue, dialogue, multi-person, narration) are determined; training and test sets are split; and indexes are generated.
```shell
python build_datasets.py --root_dir datasets/clean/zh --out_dir datasets/clean --save
```

<a name="Dubbing-Model"></a>
## Dubbing Model ⚙️



<a name="Recent-Updates"></a>
## Recent Updates 🚀
- 2025/12/18: FunCineForge dataset pipeline toolkit is online! 🔥
- 2026/01/17: Demo samples and dataset samples released. 🔥



<a name="Comminicate"></a>
## Comminicate 🍟
The FunCineForge open-source project is developed and maintained by the Tongyi Lab Speech Team and student from the National Engineering Research Center of Speech and Language Information Processing.
We welcome you to participate in discussions on FunCineForge GitHub Issues or contact us for collaborative development.
For any questions, you can contact the [developer](mailto:jxliu@mail.ustc.edu.cn).

⭐ Hope you will support FunCineForge. Thank you.

### Disclaimer

This repository contains research artifacts:

⚠️ Not an official Alibaba product

⚠️ Released for academic/research purposes only

⚠️ FunCineForge is subject to specific license terms