"""
This script uses pretrained models to perform speaker visual embeddings extracting.
This script use following open source models:
    1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
    3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
    5. Lip detection: https://huggingface.co/pyannote/segmentation-3.0
Processing pipeline: 
    1. Face detection (input: video frames)
    2. Active speaker detection (input: consecutive face frames, audio)
    3. Face quality assessment (input: video frames)
    4. Face recognition (input: video frames)
    5. Lip detection (input: video frames)
"""

import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import time, torch, cv2, pickle, gc, python_speech_features
from scipy import signal


class VisionProcesser():
    def __init__(
        self, 
        video_file_path, 
        audio_file_path, 
        audio_vad, 
        out_feat_path, 
        visual_models, 
        conf=None, 
        out_video_path=None
        ):
        # read audio data and check the samplerate.
        fs, audio = wavfile.read(audio_file_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        duration = audio.shape[0] / fs
        target_length = int(duration * 16000)
        self.audio = signal.resample(audio, target_length)

        # convert time interval to integer sampling point interval.
        audio_vad = [[int(i*16000), int(j*16000)] for (i, j) in audio_vad]
        self.video_path = video_file_path

        # read video data
        self.cap = cv2.VideoCapture(video_file_path)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print('video %s info: w: {}, h: {}, count: {}, fps: {}'.format(w, h, self.count, self.fps) % self.video_path)

        # initial vision models
        self.visual_models = visual_models

        # store facial feats along with the necessary information.
        self.active_facial_embs = {
            'frameI':np.empty((0,), dtype=np.int32),
            'feat':np.empty((0, 512), dtype=np.float32),
            'faceI': np.empty((0,), dtype=np.int32),
            'face': [],
            'face_bbox': np.empty((0, 4), dtype=np.int32),
            'lip': [],
            'lip_bbox': np.empty((0, 4), dtype=np.int32),
        }

        self.audio_vad = audio_vad
        self.out_video_path = out_video_path
        self.out_feat_path = out_feat_path

        self.min_track = conf['min_track']
        self.num_failed_det = conf['num_failed_det']
        self.crop_scale = conf['crop_scale']
        self.min_face_size = conf['min_face_size']
        self.face_det_stride = conf['face_det_stride']
        self.shot_stride = conf['shot_stride']

        if self.out_video_path is not None:
            # save the active face detection results video (for debugging).
            self.v_out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(w), int(h)))

        # record the time spent by each module.
        self.elapsed_time = {'faceTime':[], 'trackTime':[], 'cropTime':[],'asdTime':[], 'featTime':[], 'totalTime':[]}


    def run(self):
        frames, face_det_frames = [], []
        for [audio_sample_st, audio_sample_ed] in self.audio_vad:
            frame_st, frame_ed = int(audio_sample_st/640), int(audio_sample_ed/640) # 16000采样率/640=25fps，转换为视频的25fps帧数
            num_frames = frame_ed - frame_st + 1
            # go to frame 'frame_st'.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_st)
            index = 0
            for _ in range(num_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                if index % self.face_det_stride==0:
                    face_det_frames.append(frame)
                frames.append(frame)
                if (index + 1) % self.shot_stride==0:
                    audio = self.audio[(frame_st + index + 1 - self.shot_stride)*640:(frame_st + index + 1)*640]
                    self.process_one_shot(frames, face_det_frames, audio, frame_st + index + 1 - self.shot_stride)
                    frames, face_det_frames = [], []
                index += 1
            if len(frames) != 0:
                audio = self.audio[(frame_st + index - len(frames))*640:(frame_st + index)*640]
                self.process_one_shot(frames, face_det_frames, audio, frame_st + index - len(frames))
                frames, face_det_frames = [], []

        self.cap.release()
        if self.out_video_path is not None:
            self.v_out.release()

        out_data = {
            'embeddings':self.active_facial_embs['feat'],    # 'times': self.active_facial_embs['frameI']*0.04, # 25 fps
            'frameI': self.active_facial_embs['frameI'], # 说话人活跃的人脸帧索引
            'faceI': self.active_facial_embs['faceI'], # 存在人脸的帧索引
            'face': self.active_facial_embs['face'],
            'face_bbox': self.active_facial_embs['face_bbox'],
            'lip': self.active_facial_embs['lip'],
            'lip_bbox': self.active_facial_embs['lip_bbox'],
        }
        pickle.dump(out_data, open(self.out_feat_path, 'wb'))

        # print elapsed time
        all_elapsed_time = 0
        for k in self.elapsed_time:
            all_elapsed_time += sum(self.elapsed_time[k])
            self.elapsed_time[k] = sum(self.elapsed_time[k])
        elapsed_time_msg = 'The total time for %s is %.2fs, including' % (self.video_path, all_elapsed_time)
        for k in self.elapsed_time:
            elapsed_time_msg += ' %s %.2fs,'%(k, self.elapsed_time[k])
        print(elapsed_time_msg[:-1]+'.')
        try:
            del out_data
        except Exception:
            pass

    def process_one_shot(self, frames, face_det_frames, audio, frame_st=None):
        curTime = time.time()
        dets = self.face_detection(face_det_frames)
        faceTime = time.time()

        allTracks, vidTracks = [], []
        allTracks.extend(self.track_shot(dets))
        trackTime = time.time()

        for ii, track in enumerate(allTracks):
            vidTracks.append(self.crop_video(track, frames, audio))
        cropTime = time.time()

        scores = self.evaluate_asd(vidTracks)
        asdTime = time.time()

        active_facial_embs = self.evaluate_fr(frames, vidTracks, scores)
        self.active_facial_embs['frameI'] = np.append(self.active_facial_embs['frameI'], active_facial_embs['frameI'] + frame_st)
        self.active_facial_embs['feat'] = np.append(self.active_facial_embs['feat'], active_facial_embs['feat'], axis=0)
        self.active_facial_embs['faceI'] = np.append(self.active_facial_embs['faceI'], active_facial_embs['faceI'] + frame_st)
        self.active_facial_embs['face'].extend(active_facial_embs['face'])
        self.active_facial_embs['face_bbox'] = np.vstack([self.active_facial_embs['face_bbox'], active_facial_embs['face_bbox']])
        self.active_facial_embs['lip'].extend(active_facial_embs['lip'])
        self.active_facial_embs['lip_bbox']= np.vstack([self.active_facial_embs['lip_bbox'], active_facial_embs['lip_bbox']])
        
        featTime = time.time()
        if self.out_video_path is not None:
             self.visualization(frames, vidTracks, scores, active_facial_embs)
             
        try:
            del dets, allTracks, vidTracks, active_facial_embs
        except Exception:
            pass

        self.elapsed_time['faceTime'].append(faceTime-curTime)
        self.elapsed_time['trackTime'].append(trackTime-faceTime)
        self.elapsed_time['cropTime'].append(cropTime-trackTime)
        self.elapsed_time['asdTime'].append(asdTime-cropTime)
        self.elapsed_time['featTime'].append(featTime-asdTime)
        self.elapsed_time['totalTime'].append(featTime-curTime)

    def face_detection(self, frames):
        dets = []
        for fidx, image in enumerate(frames):
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, _, probs = self.visual_models.detect_faces(image_input, top_k=10, prob_threshold=0.9)
            bboxes = torch.cat([bboxes, probs.reshape(-1, 1)], dim=-1)
            dets.append([])
            for bbox in bboxes:
                frame_idex = fidx * self.face_det_stride
                dets[-1].append({'frame':frame_idex, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) 
        return dets

    def bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, scene_faces):
        # Face tracking
        tracks = []
        while True:   # continuously search for consecutive faces.
            track = []
            for frame_faces in scene_faces:
                for face in frame_faces:
                    if track == []:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:  # the face does not interrupt for 'num_failed_det' frame.
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        # minimum IOU between consecutive face.
                        if iou > 0.5:
                            track.append(face)
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if track == []:
                break
            elif len(track) > 1 and track[-1]['frame'] - track[0]['frame'] + 1 >= self.min_track:
                frame_num = np.array([ f['frame'] for f in track ])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                frameI = np.arange(frame_num[0], frame_num[-1]+1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn  = interp1d(frame_num, bboxes[:,ij]) # missing boxes can be filled by interpolation.
                    bboxesI.append(interpfn(frameI))
                bboxesI  = np.stack(bboxesI, axis=1)
                if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > self.min_face_size:  # need face size > min_face_size
                    tracks.append({'frame':frameI,'bbox':bboxesI})
        return tracks

    def crop_video(self, track, frames, audio):
        # crop the face clips
        crop_frames = []
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
            dets['y'].append((det[1]+det[3])/2) # crop center x 
            dets['x'].append((det[0]+det[2])/2) # crop center y
        for fidx, frame in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]   # detection box size
            bsi = int(bs * (1 + 2 * cs))  # pad videos by this amount 
            image = frames[frame]
            frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            my  = dets['y'][fidx] + bsi  # BBox center Y
            mx  = dets['x'][fidx] + bsi  # BBox center X
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            crop_frames.append(cv2.resize(face, (224, 224)))
        cropaudio = audio[track['frame'][0]*640:(track['frame'][-1]+1)*640]
        return {'track':track, 'proc_track':dets, 'data':[crop_frames, cropaudio]}

    def evaluate_asd(self, tracks):
        # active speaker detection by pretrained TalkNet
        all_scores = []
        for ins in tracks:
            video, audio = ins['data']
            audio_feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
            video_feature = []
            for frame in video:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h0, w0 = face.shape
                interp = cv2.INTER_CUBIC if (h0 < 224 or w0 < 224) else cv2.INTER_AREA
                face = cv2.resize(face, (224,224), interpolation=interp)
                # face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_feature.append(face)
            video_feature = np.array(video_feature)
            length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100, video_feature.shape[0] / 25)
            audio_feature = audio_feature[:int(round(length * 100)),:]
            video_feature = video_feature[:int(round(length * 25)),:,:]
            audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)
            video_feature = np.expand_dims(video_feature, axis=0).astype(np.float32)
            score = self.visual_models.asd_score(audio_feature, video_feature)
            all_score = np.asarray(score, dtype=np.float32)
            all_scores.append(all_score)
        try:
            del audio_feature, video_feature, score
        except Exception:
            pass
        return all_scores
    
            
    def evaluate_fr(self, frames, tracks, scores):
        SMOOTH_W = 4
        ON_THRESHOLD = 0.0
        OFF_THRESHOLD = -0.5
        QUALITY_HIGH = 0.0
        QUALITY_LOW = -0.3

        # 先平滑每个 track 的 scores
        smooth_scores_all = []
        for score in scores:
            s = np.asarray(score).flatten()
            if s.size == 0:
                smooth_scores_all.append(s)
                continue
            # 中值 + 简单移动平均
            s_med = signal.medfilt(s, kernel_size=5 if len(s)>=5 else 3)
            k = np.ones(5)/5
            s_avg = np.convolve(s_med, k, mode='same')
            smooth_scores_all.append(s_avg)

        # aggregate faces per frame
        faces = [[] for _ in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = smooth_scores_all[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - SMOOTH_W, 0): min(fidx + SMOOTH_W+1, len(score))]
                s = float(np.mean(s))
                bbox = track['track']['bbox'][fidx]
                bbox = bbox.astype(np.int32)
                face = frames[frame][max(bbox[1],0):min(bbox[3],frames[frame].shape[0]),
                                    max(bbox[0],0):min(bbox[2],frames[frame].shape[1])]
                faces[frame].append({'track':tidx, 'score':s, 'facedata':face, 'bbox': bbox})
                
        # per-frame decision
        active_facial_embs = {
            'frameI': [],
            'trackI': [],
            'faceI': [],
            'face': [],
            'face_bbox': [],
            'feat': [],
            'lip': [],
            'lip_bbox': [],
        }
        # 这里做简单 per-frame decision: 选 score 最大的
        for fidx in range(0, len(faces), max(1, self.face_det_stride)):
            if len(faces[fidx]) == 0:
                continue
            # choose best candidate by score
            best = max(faces[fidx], key=lambda x: x['score'])
            res = self.visual_models.detect_lip(best['facedata'])
            # 如果没有检测到嘴唇，跳过，会筛去低质量像素的人脸
            if res is None or res.get('lip_crop') is None:
                continue
            # 只要该帧检测到一张或者多种人脸，就保存一个最有可能是说话人（best['facedata']）的人脸（不管说不说话）
            active_facial_embs['faceI'].append(fidx)
            active_facial_embs['face'].append(best['facedata']) # BGR ndarray
            active_facial_embs['lip'].append(res.get('lip_crop')) # BGR ndarray
            active_facial_embs['face_bbox'].append(best['bbox'])  # 相对于整个一帧图片的脸的位置坐标
            active_facial_embs['lip_bbox'].append(res.get('lip_bbox'))  # 相对于脸框图的位置坐标
            feature = self.visual_models.get_face_embedding(best['facedata'])
            active_facial_embs['feat'].append(feature) # 完整面部特征
            
            
            s = best['score']
            if s < OFF_THRESHOLD:
                continue
            # 人脸质量评估（可选，开启后只会筛选评分更高的人脸帧）
            # face_q_score = self.visual_models.face_quality_score(best['facedata'])
            # if (face_q_score >= QUALITY_HIGH) or (face_q_score >= QUALITY_LOW and s >= ON_THRESHOLD):
            if  s >= OFF_THRESHOLD:
                # feature, feature_normalized = self.visual_models.get_face_embedding(best['facedata']) # 仅保留模型认为在说话帧
                active_facial_embs['frameI'].append(fidx)
                active_facial_embs['trackI'].append(best['track'])

        # 转 numpy
        active_facial_embs['frameI'] = np.array(active_facial_embs['frameI'], dtype=np.int32)
        active_facial_embs['trackI'] = np.array(active_facial_embs['trackI'], dtype=np.int32)
        active_facial_embs['faceI'] = np.array(active_facial_embs['faceI'], dtype=np.int32)
        active_facial_embs['face_bbox'] = np.array(active_facial_embs['face_bbox'], dtype=np.int32) if active_facial_embs['face_bbox'] else np.empty((0,4), np.int32)
        active_facial_embs['lip_bbox']  = np.array(active_facial_embs['lip_bbox'], dtype=np.int32) if active_facial_embs['lip_bbox'] else np.empty((0,4), np.int32)
        active_facial_embs['feat'] = np.vstack(active_facial_embs['feat']) if active_facial_embs['feat'] else np.empty((0,512), np.float32)
        return active_facial_embs


    def visualization(self, frames, tracks, scores, embs=None):
        # 先聚合所有 track 在每帧的 bbox/score 信息（与原实现一致）
        faces = [[] for _ in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score))]  # 注意 len(score) 作为上界
                s = np.mean(s)
                faces[frame].append({'track':tidx, 'score':float(s),'bbox':track['track']['bbox'][fidx]})

        # 构造已保存帧集合（相对于本 shot）
        feat_set = set()
        lip_bbox_dict = {}  # 存储嘴唇边界框的字典
        if embs is not None:
            if 'frameI' in embs and embs['frameI'].size > 0:
                trackI = embs.get('trackI')
                feat_set = set((int(f), int(t)) for f, t in zip(embs['frameI'].tolist(), trackI.tolist()))
            
            if 'lip_bbox' in embs and embs['lip_bbox'].size > 0:
                for i, frame_idx in enumerate(embs['faceI']):
                    lip_bbox_dict[int(frame_idx)] = embs['lip_bbox'][i]  

        for fidx, image in enumerate(frames):
            for face in faces[fidx]:
                bbox = face['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                # lip bbox
                lip_bbox = None
                if fidx in lip_bbox_dict:
                    lip_bbox = lip_bbox_dict[fidx]
                    lip_x1 = x1 + lip_bbox[0]
                    lip_y1 = y1 + lip_bbox[1]
                    lip_x2 = x1 + lip_bbox[2]
                    lip_y2 = y1 + lip_bbox[3]
                if (fidx, face['track']) in feat_set:
                    # 绿色表示已保存, 蓝色表示嘴唇
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if lip_bbox is not None:
                        cv2.rectangle(image, (lip_x1, lip_y1), (lip_x2, lip_y2), (255, 0, 0), 2)
                    txt = round(face['score'], 2)
                    cv2.putText(image, '%s'%(txt), (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    # 红色表示未保存
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    if lip_bbox is not None:
                        cv2.rectangle(image, (lip_x1, lip_y1), (lip_x2, lip_y2), (255, 0, 0), 2)
                    txt = round(face['score'], 2)
                    cv2.putText(image, '%s'%(txt), (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

            # 写入视频
            self.v_out.write(image)
    
    
    def close(self):
        try:
            if hasattr(self, "active_facial_embs"):
                for k, v in self.active_facial_embs.items():
                    if isinstance(v, np.ndarray):
                        del v
                    elif isinstance(v, list):
                        v.clear()
                self.active_facial_embs.clear()
        except Exception as e:
            print(f"[WARN] Error while closing VisionProcesser: {e}")
        gc.collect()

    def __del__(self):
        self.close()