### <p align="center">「English | [简体中文](./README_zh.md)」</p>

<p align="center">
<b>🎬 FunCineForge: A Unified Dataset Pipeline and Model for Zero-Shot Movie Dubbing<br>
in Diverse Cinematic Scenes</b>
</p>

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
｜<a href="#Publication">Publication</a>
｜<a href="#Comminicate">Comminicate</a>
</h4>
</div>

**FunCineForge** contains an end-to-end dataset pipeline for producing large-scale dubbing datasets and an MLLM-based dubbing model designed for diverse cinematic scenes. Using this pipeline, we constructed the first large-scale Chinese television dubbing dataset CineDub-CN, which includes rich annotations and diverse scenes. In monologue, narration, dialogue, and multi-speaker scenes, our dubbing model consistently outperforms state-of-the-art methods in terms of audio quality, lip-sync, timbre transition, and instruction following.

<a name="Dataset&Demo"></a>
## Dataset & Demo 🎬
You can access [https://funcineforge.github.io/](https://funcineforge.github.io/) to get our CineDub-CN dataset samples and demo samples. 


<a name="Dataset-Pipeline"></a>
## Dataset Pipeline 🔨

### Environmental Installation

FunCineForge dataset pipeline toolkit only relies on a Python environment to run.
```shell
# Conda
git clone git@github.com:FunAudioLLM/FunResearch.git
conda create -n FunCineForge python=3.10 -y && conda activate FunCineForge
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

- [1] Standardize video format and name; trim the beginning and end of long videos; extract the audio from the trimmed video. (default is to trim 10 seconds from both the beginning and end.)
```shell
python normalize_trim.py --root datasets/raw_zh --intro 10 --outro 10
```

- [2] [Speech Separation](./speech_separation/README.md). The audio is used to separate the vocals from the instrumental music.
```shell
cd speech_separation
python run.py --root datasets/clean/zh --gpus 0 1 2 3
```

- [3] [VideoClipper](./video_clip/README.md). For long videos, VideoClipper is used to obtain sentence-level subtitle files and clip the long video into segments based on timestamps. Now it supports bilingualism in both Chinese and English. Below is an example in Chinese. It is recommended to use gpu acceleration for English.
```shell
cd video_clip
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh --device cpu
```

- Video duration limit and check for cleanup. (Without --execute, only pre-deleted files will be printed. After checking, add --execute to confirm the deletion.)
```shell
python clean_video.py --root datasets/clean/zh
python clean_srt.py --root datasets/clean/zh --lang zh
```

- [4] [Speaker Diarization](./speaker_diarization/README.md). Multimodal active speaker recognition obtains RTTM files; identifies the speaker's facial frames, extracts frame-level speaker face and lip raw data.
```shell
cd speaker_diarization
bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root datasets/clean/zh --gpus "0 1 2 3"
```

- [5] Multimodal CoT Correction. Based on general-purpose MLLMs, the system uses audio, ASR text, and RTTM files as input. It leverages Chain-of-Thought (CoT) reasoning to extract clues and corrects the results of the specialized models. It also annotates character age, gender, and vocal timbre. Experimental results show that this strategy reduces the CER from 4.53% to 0.94% and the speaker diarization error rate from 8.38% to 1.20%, achieving quality comparable to or even better than manual transcription. Adding the --resume enables breakpoint COT inference to prevent wasted resources from repeated COT inferences.
```shell
python cot.py --root_dir datasets/clean/zh --provider google --model gemini-3.0-pro --api_key xxx --resume
python build_datasets.py --root_dir datasets/clean/zh --out_dir datasets/clean --save
```

- (Reference) Extract speech tokens based on the CosyVoice3 tokenizer for llm training.
```shell
python speech_tokenizer.py --root datasets/clean/zh
```

<a name="Dubbing-Model"></a>
## Dubbing Model ⚙️
Please stay tuned.



<a name="Recent-Updates"></a>
## Recent Updates 🚀
- 2025/12/18: FunCineForge dataset pipeline toolkit is online! 🔥
- 2026/01/19: Demo samples and dataset samples released. 🔥
- 2026/01/25: Fix some environmental and operational issues.
- 2026/02/09: Optimized the data pipeline and added support for English videos.

<a name="Publication"></a>
## Publication 📚
If you use our dataset or code, please cite the following paper:
<pre>
@misc{liu2026funcineforgeunifieddatasettoolkit,
    title={FunCineForge: A Unified Dataset Toolkit and Model for Zero-Shot Movie Dubbing in Diverse Cinematic Scenes}, 
    author={Jiaxuan Liu and Yang Xiang and Han Zhao and Xiangang Li and Zhenhua Ling},
    year={2026},
    eprint={2601.14777},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
</pre>

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