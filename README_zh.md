### <p align="center">「[English](./README.md) | 简体中文」</p>

**<p align="center"> 🎬 FunCineForge：一种统一的数据集工具与模型用于多样化影视场景的零样本影视配音 </p>**

<div align="center">

![license](https://img.shields.io/github/license/modelscope/modelscope.svg)
<a href=""><img src="https://img.shields.io/badge/OS-Linux-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Python->=3.8-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Pytorch->=2.1-blue"></a>
</div>

<div align="center">
<h4><a href="#数据集&样例">数据集 & 样例</a>
｜<a href="#数据集工具">数据集工具</a>
｜<a href="#配音模型">配音模型</a>
｜<a href="#近期更新">近期更新</a>
｜<a href="#社区交流">社区交流</a>
</h4>
</div>

**FunCineForge** 包含一个生产大规模配音数据集的端到端数据集工具，和一个基于多模态大模型的配音模型，该模型专为多样的电影场景而设计。利用该工具，我们构建了首个大规模中文电视剧配音数据集，该数据集包含丰富的标注和多样化的场景。在独白、旁白、对话和多说话人场景中，我们的配音模型在音频质量、唇形同步、音色转换和指令遵循等方面全部优于最先进的方法。

<a name="数据集&样例"></a>
## 数据集 & 样例 🎬
您可以访问此 [网站]() 获取我们的数据集样本和演示样本。

<a name="数据集工具"></a>
## 数据集工具 🔨

### 环境安装

FunCineForge 的运行仅依赖于一个 Python 环境。
```shell
# 克隆 FunCineForge 仓库
git clone git@github.com:xuan3986/FunCineForge.git
conda create -n FunCineForge python=3.8.20 -y && conda activate FunCineForge
sudo apt-get install ffmpeg
# 初始化设置
cd FunCineForge
python setup.py
```

### 数据收集
如果您想自行生产数据，我们建议您参考下面的要求收集相应的电影或影视剧。

1. 视频来源：电视剧或电影，非纪录片，人物独白或对话场景较多，人脸清晰且无遮挡（如无面罩、面纱）。
2. 语音要求：发音标准，吐字清晰，人声突出。避免方言浓重、背景噪音过大或口语感过强的素材。
3. 图片要求：高分辨率，面部细节清晰，光线充足，避免极端阴暗或强烈逆光的场景。

### 使用方法

- 规范视频格式为 mp4，libx264 & libmp3lame 编码；裁剪影视剧片头片尾（默认片头片尾各裁剪5分钟）
```shell
python normalize_mp4.py --root datasets/raw_zh
python trim_video.py --root datasets/raw_zh
```
- [Video Clip](./video_clip/README.md). 对长序列视频 VAD，得到句子级的片段，通过 ASR 得到转录文本，生成字幕文件。再将长序列视频剪裁为片段。
```shell
cd video_clip
bash run.sh --stage 1 --stop_stage 2 --input datasets/raw_zh --output datasets/clean/zh --lang zh
```
- 视频时长限制和字幕文件清洗。(不加 --execute 只会打印预删除文件，检查无误后添加 --execute 运行确认删除)
```shell
python clean_video.py --root datasets/clean/zh --execute
python clean_srt.py --root datasets/clean/zh --execute
```
- [Speech Separation](./speech_separation/README.md). 音频进行人声乐声分离。
```shell
cd speech_separation
python run.py --root datasets/clean/zh --gpus 0 1 2 3
```
- [Speaker Diarization](./speaker_diarization/README.md). 多模态主动说话人识别，得到 RTTM 文件；识别说话人的面部帧，提取帧级的说话人面部和唇部原始数据，从面部帧中识别说话帧，提取说话帧的面部特征。
```shell
cd speaker_diarization
bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root datasets/clean/zh --gpus "0 1 2 3"
```
- 基于多模态大模型，输入音频，ASR 文本，RTTM 文件，通过思维链得到情感线索，并采样大模型矫正小模型方案降低 ASR 词错率，同时标注角色年龄性别和音色属性信息。实验验证大模型+小模型方案的 WER 从 3.2% 降低至 0.6%，speaker ID 的错误率从 4.3% 降低至 1.2%，与人工转录质量相当甚至更优。添加 --resume 实现断点 COT，以防止重复文件COT推理浪费资源。
```shell
python cot.py --root_dir datasets/clean/zh --provider google --model gemini-2.5-flash --api_key xxx --resume
```
- （参考）基于 CosyVoice3 tokenizer 提取 speech tokens 用于大模型训练。
```shell
python speech_tokenizer.py --root datasets/clean/zh
```
- （参考）结果清洗并纠正；视频片段类型（独白、对话、多人、旁白）判断；切分训练集和测试集；生成索引。
```shell
python build_datasets.py --root_dir datasets/clean/zh --out_dir datasets/clean --save
```

<a name="Dubbing-Model"></a>
## 配音模型 ⚙️

<a name="近期更新"></a>
## 近期更新 🚀
- 2025/12/18：FunCineForge toolkit 源代码上线！🔥
- 2026/01/17：发布演示样例和数据集样本。 🔥


<a name="社区交流"></a>
## 社区交流 🍟
FunCineForge 开源项目由通义实验室语音团队和语音及语言信息处理国家工程研究中心学生开发并维护，我们欢迎您在 FunCineForge GitHub Issues 参与问题讨论，或联系我们合作开发。
有任何问题您可以联系[开发者](mailto:jxliu@mail.ustc.edu.cn)。

⭐ 希望您你支持 FunCineForge，谢谢。

### 免责声明

本仓库包含研究成果：

⚠️ 非阿里巴巴官方产品

⚠️ 仅供学术/研究用途

⚠️ FunCineForge 受特定许可条款约束