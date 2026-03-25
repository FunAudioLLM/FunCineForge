### <p align="center">「[English](./README.md) | 简体中文」</p>

<p align="center">
<b>🎬 Fun-CineForge：一种用于多样化影视场景零样本配音的统一数据集管道和模型</b>
</p>

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>
</div>

<div align="center">
<h4><a href="#开源">开源</a>
｜<a href="#环境安装">环境安装</a>
｜<a href="#数据集管道">数据集管道</a>
｜<a href="#配音模型">配音模型</a>
｜<a href="#近期更新">近期更新</a>
｜<a href="#发表">发表</a>
｜<a href="#社区交流">社区交流</a>
</h4>
</div>

**Fun-CineForge** 包含一个生产大规模配音数据集的端到端数据集管道，和一个基于多模态大模型的配音模型，该模型专为多样的电影场景而设计。利用该管道，我们构建了首个大规模中文电视剧配音数据集 CineDub-CN，该数据集包含丰富的标注和多样化的场景。在独白、旁白、对话和多说话人场景中，我们的配音模型在音频质量、唇形同步、音色转换和指令遵循等方面全部优于最先进的方法。

<a name="开源"></a>
## 开源 🎬
您可以访问此 [https://funcineforge.github.io/](https://funcineforge.github.io/) 获取我们的 CineDub-CN 数据集和 CineDub-EN 数据集样例和演示样例。

ModelScope 链接： [https://www.modelscope.cn/models/FunAudioLLM/Fun-CineForge/](https://www.modelscope.cn/models/FunAudioLLM/Fun-CineForge/)

HuggingFace 链接： [https://huggingface.co/FunAudioLLM/Fun-CineForge/](https://huggingface.co/FunAudioLLM/Fun-CineForge/)

<a name="环境安装"></a>
## 环境安装

Fun-CineForge 依赖 Conda 和 Python 环境。执行 **setup.py** 自动安装整个项目环境和开源模型。

```shell
# Conda
git clone git@github.com:FunAudioLLM/FunCineForge.git
conda create -n FunCineForge python=3.10 -y && conda activate FunCineForge
sudo apt-get install ffmpeg
# 初始化设置
python setup.py
```

<a name="数据集管道"></a>
## 数据集管道 🔨

### 数据收集
如果您想自行生产数据，我们建议您参考下面的要求收集相应的电影或影视剧。

1. 视频来源：电视剧或电影，非纪录片，人物独白或对话场景较多，人脸清晰且无遮挡（如无面罩、面纱）。
2. 语音要求：发音标准，吐字清晰，人声突出。避免方言浓重、背景噪音过大或口语感过强的素材。
3. 图片要求：高分辨率，面部细节清晰，光线充足，避免极端阴暗或强烈逆光的场景。

### 使用方法

- [1] 将视频格式、名称标准化；裁剪长视频的片头片尾；提取裁剪后视频的音频。（默认是从起止各裁剪 10 秒。）
```shell
python normalize_trim.py --root datasets/raw_zh --intro 10 --outro 10
```

- [2] [Speech Separation](./speech_separation/README.md). 音频进行人声乐声分离。
```shell
cd speech_separation
python run.py --root datasets/clean/zh --gpus 0 1 2 3
```

- [3] [VideoClipper](./video_clip/README.md). 对于长视频，使用 VideoClipper 获取句子级别的字幕文件，并根据时间戳将长视频剪辑成片段。现在它支持中英双语。以下是中文示例。英文建议采用 gpu 加速处理。
```shell
cd video_clip
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh --device cpu
```

- 视频时长限制及清理检查。（若不使用--execute参数，则仅打印已预删除的文件。检查后，若需确认删除，请添加--execute参数。）
```shell
python clean_video.py --root datasets/clean/zh
python clean_srt.py --root datasets/clean/zh --lang zh
```

- [4] [Speaker Diarization](./speaker_diarization/README.md). 多模态主动说话人识别，得到 RTTM 文件；识别说话人的面部帧，提取帧级的说话人面部和唇部原始数据，从面部帧中识别说话帧，提取说话帧的面部特征。
```shell
cd speaker_diarization
bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root datasets/clean/zh --gpus "0 1 2 3"
```

- （参考）基于 CosyVoice3 tokenizer 提取 speech tokens 用于大模型训练。
```shell
python speech_tokenizer.py --root datasets/clean/zh
```

- [5] 多模态思维链校正。该系统基于通用多模态大模型，以音频、ASR 抄本和 RTTM 文件为输入，利用思维链推理来提取线索，并校正专用模型的结果，并标注人物年龄、性别和音色。实验结果表明，该策略将词错率从4.53% 降低到 0.94%，说话人识别错误率从 8.38% 降低到 1.20%，其质量可与人工转录相媲美，甚至更优。添加--resume选项可启用断点思维链推理，以避免重复思维链推理造成的资源浪费。现支持中英文。
```shell
python cot.py --root_dir datasets/clean/zh --lang zh --provider google --model gemini-3-pro-preview --api_key xxx --resume
python cot.py --root_dir datasets/clean/en --lang en --provider google --model gemini-3-pro-preview --api_key xxx --resume
```

- 数据集检索文件的构建会读取生产的所有数据，双向校验脚本内容和说话人分离结果。
```shell
python build_datasets.py --root_zh datasets/clean/zh --root_en datasets/clean/en --out_dir datasets/clean --save
```

<a name="Dubbing-Model"></a>
## 配音模型 ⚙️
我们开源了推理代码和 **infer.sh** 脚本，在 data 文件夹中提供了一些测试样例，以供体验。推理需要一张消费级 GPU。按下面的命令运行：

```shell
cd exps
bash infer.sh
```

您现在可以通过访问 [ModelScope Space](https://www.modelscope.cn/studios/FunAudioLLM/Fun-CineForge-Demo/) 或 [HuggingFace Space](https://huggingface.co/spaces/FunAudioLLM/Fun-CineForge-Demo/) 来体验 Fun-CineForge-zh-en-v1-0.5B 模型。

<a name="近期更新"></a>
## 近期更新 🚀
- 2025/12/18：Fun-CineForge 数据集管道工具包上线！🔥
- 2026/01/19：发布中文演示样例和 CineDub-CN 数据集样例。 🔥
- 2026/01/25：修复了一些环境和运行问题。
- 2026/02/09：优化了数据管道，新增支持英文视频的能力。
- 2026/03/05：发布英文演示样例和 CineDub-EN 数据集样例。 🔥
- 2026/03/16：开源推理代码和 checkpoints。 🔥

<a name="发表"></a>
## 发表 📚
如果您使用了我们的数据集或代码，请引用以下论文：
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


<a name="社区交流"></a>
## 社区交流 🍟
Fun-CineForge 开源项目由通义实验室语音团队和中国科学技术大学 NERCSLIP 学生开发并维护，我们欢迎您在 Fun-CineForge [GitHub Issues](https://github.com/FunAudioLLM/FunCineForge/issues) 参与问题讨论，或联系我们合作开发。
有任何问题您可以联系[开发者](mailto:jxliu@mail.ustc.edu.cn)。

⭐ 希望您你支持 Fun-CineForge，谢谢。

### 免责声明

该仓库包含的研究成果：

⚠️ 目前非通义实验室商业化产品

⚠️ 供学术研究/前沿探索用途

⚠️ 数据集样例受特定许可条款约束