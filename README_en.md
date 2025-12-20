### <p align="center">「[简体中文](./README.md) | English」</p>

**<p align="center"> 🧠 FunCineForge: Exploring the end-to-end production tools for large-scale television multimodal datasets </p>**

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>
</div>

<div align="center">  
<h4><a href="#Quick-Start">Quick Start</a>
｜<a href="#Recent-Updates">Recent Updates</a>
｜<a href="#Comminicate">Comminicate</a>
</h4>
</div>

**FunCineForge** is a fully open-source, locally deployed tool for producing multimodal speech datasets. It integrates batches of film or television data from the source into comprehensive data including text, speech, video, clues, timestamps, and other information for training our VTTS dubbing LLM.
All pre-trained models have been uploaded to [Hugging Face](https://huggingface.co/xuan3986/FunCineForge). In addition, we open source the datasets produced by funcineforge.


<a name="Quick-Start"></a>
## Quick Start 🚀

### Environmental Installation

FunCineForge only relies on a Python environment to run.
```shell
# Conda
git clone git@github.com:xuan3986/FunCineForge.git
conda create -n FunCineForge python=3.8.20 -y && conda activate FunCineForge
sudo apt-get install ffmpeg
# Initial settings
cd FunCineForge
python setup.py
```

### Datasets
You can access [FunCineForge Datasets]() Website to get our dataset. If you want to produce your own data, 
we recommend that you refer to the following requirements to collect the corresponding movies and television series.

1. Video source: Television series or movies, non documentaries, more monologues or dialogue scenes, clear and unobstructed faces (such as without masks and veils).
2. Voice requirements: standard pronunciation, clear articulation and prominent voice. Avoid materials with too much dialect, background noise or strong oral sense.
3. Picture quality: high pixels, clear facial details, sufficient light, avoiding extremely dark or strongly backlit picture scenes.

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
- (Optional) Extract speech tokens based on the CosyVoice3 tokenizer for llm training.
```shell
python speech_tokenizer.py --root datasets/clean/zh
```
- (Reference) COT results are cleaned and corrected; video clip types (monologue, dialogue, multi-person, narration) are determined; training and test sets are split; and indexes are generated.
```shell
python build_datasets.py --root_dir datasets/clean/zh --out_dir datasets/clean --save
```
<a name="Recent-Updates"></a>
## Recent Updates 🔨

- 2025/12/18 FunCineForge source code is online! 🔥
<!-- - 2025/12/19 The dataset is open source! 🔥 -->



<a name="Comminicate"></a>
## Comminicate 🍟
FunCineForge open source project comes from [FunResearch](https://github.com/FunAudioLLM/FunResearch), developed and maintained by the Tongyi Labs Speech Team.
We welcome you to join the Fun Research community for discussions, collaborations, and more.
[FunASR](https://github.com/modelscope/FunASR) is one of Alibaba Tongyi Lab's open source speech toolkits. Everyone is welcome to use it.

For any questions, please contact me.

⭐ Hope you will support FunCineForge. Thank you.

### Disclaimer

This repository contains research artifacts:

⚠️ Not an official Alibaba product

⚠️ Released for academic/research purposes only

⚠️ FunCineForge is subject to specific license terms