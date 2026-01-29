# Speaker Diarization

## Introduction
This recipe offers speaker diarization methods that address the problem of "who spoke when". It provides multimodal diarization. The audio diarization comprises overlap detection, voice activity detection, speech segmentation, speaker embedding extraction. The video diarization comprises face detection, cctive speaker detection, face recognition, lip recognition.
Then multimodal speaker clustering results are achieved.


The DER results of two diarization pipelines on a multi-person conversation video dataset.
| Pipeline | DER |
|:-----:|:------:|
|Audio-only diarization|5.3%|
|Multimodal diarization|3.7%|

## Usage
### Quick Start

Ensure that ffmpeg is available in your environment.
``` sh
sudo apt-get update
sudo apt-get install ffmpeg
```
The [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) is used as a overlapping speech detection module. Make sure to accept [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) user conditions and create an access token at [hf.co/settings/tokens](https://hf.co/settings/tokens)

- Stage1: Generate video.list and wav.list
- Stage2: Process the wav and use the CAM++ speaker recognition model (Tongyi) to extract speaker embeddings (auditory modality) for each sub-segment of the audio.
  - First, perform speaker overlap detection to obtain overlap.list.
  - Delete speaker overlap samples to obtain clean_wav.list and clean_video.list.
  - Use the [FSMN-Monophone VAD](https://www.modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) model to perform VAD on the audio and perform fine-grained segmentation to obtain json/vad.json.
  - Prepare subsegment information to obtain json/subseg.json.
  - Use [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced) model to extract the speaker embedding of wav audio and save it to embs_wav.
- Stage 3: Process the video and extract the speaker's facial data (visual modality) through a face detection model, an active speaker detection model, a face recognition model, and a facial landmark detection model.
  - For 25fps video, sample one frame every 5 frames (every 0.2 seconds).
  - Detect all faces in the sampled frames using the a lightweight fast [face detection](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) model
  - Score all faces using the [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) model, and use the face with the highest score as the active speaker's face
  - (Optional, but not recommended) Use a [face quality assessment](https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa) model to filter out faces with poor quality.
  - Use the [CurricularFace](https://github.com/HuangYG123/CurricularFace) model to extract the face embedding of the speaker in the active frame.
  - Use the [FAN](https://github.com/1adrianb/face-alignment) model to perform 2D facial key point detection on the speaker's face, obtain the mouth coordinate (relative coordinates) of each face frame and extract the raw face and mouth data.
- Stage 4: Joint cluster the audio and visual embeddings to obtain the multimodal active speaker detection results and save them in RTTM file.




hf_access_token is your access token
``` sh
bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root datasets/clean/zh --gpus "0 1 2 3"
```

To better understand the source code, you can refer to the **sample.mp4** and **run.sh** files in the subfolder **speaker_diarization_sample** to perform single-sample inference.

## Limitations
- It may not perform well when the audio duration is too short and when the number of speakers is too large.
- The final accuracy is highly dependent on the performance of each modules. Among them, the ASD model affects the quality of the results