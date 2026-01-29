import os
import cv2
from .api import FaceAlignment, LandmarksType
import numpy as np

class LipDetector:
    """
    在 face crop 上检测唇部位置，
    基于修改的 face_alignment api 调用 FAN 模型实现。
    """

    def __init__(
        self, 
        model_dir=None,
        device='cpu',
        device_id=0,
        landmarks_type=LandmarksType.TWO_D,
    ):
        # 设置设备字符串
        device_str = 'cpu'
        if device == 'cuda':
            device_str = f'cuda:{device_id}'
        
        if model_dir is not None:
            model_path = os.path.join(model_dir, 'fun_2d.pth')
            net_path = os.path.join(model_dir, 'fun_2d.zip') # 使用预下载模型避免长时间下载
            print(f"Loading FAN model from {model_path} on {device_str}...")
        else:
            model_path = None
        self.fa = FaceAlignment(
            landmarks_type = landmarks_type, 
            device = device_str,
            net_path = net_path,
            face_detector_kwargs = {'path_to_detector': model_path},
        )


    def detect_lip(self, face_img):
        """
        face_img: BGR image of the face crop (tight face crop, e.g. 224x224)
        返回 dict:
            { 'lip_bbox': (x1,y1,x2,y2) (relative to face_img) or None,
              'lip_crop': np.ndarray or None,
              'kps': np.ndarray (N,2) or None }
        """
        H, W = face_img.shape[:2]
        
        # 1) 使用 FAN 模型检测关键点
        try:
            # 转换颜色空间 BGR -> RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 预测关键点
            preds = self.fa.get_landmarks(rgb_img)
            
            if preds is not None and len(preds) > 0:
                # 获取第一个检测到的人脸关键点 (68个点)
                kps = preds[0]
                
                # 提取嘴唇关键点 (48-68点)
                mouth_kps = kps[48:68]
                
                # 计算边界框
                min_xy = mouth_kps.min(axis=0)
                max_xy = mouth_kps.max(axis=0)
                
                # 添加 padding
                pad = 0.18 * (max_xy - min_xy)
                x1 = int(max(0, min_xy[0] - pad[0]))
                y1 = int(max(0, min_xy[1] - pad[1]))
                x2 = int(min(W - 1, max_xy[0] + pad[0]))
                y2 = int(min(H - 1, max_xy[1] + pad[1]))
                
                # 裁剪嘴唇区域
                lip_bbox_array = np.array([x1, y1, x2, y2], dtype=np.int32)
                lip_crop = face_img[y1:y2, x1:x2].copy()
                
                return {
                    'lip_bbox': lip_bbox_array, 
                    'lip_crop': lip_crop, 
                    'kps': mouth_kps
                }
                
        except Exception as e:
            print(f"FAN detection failed: {e}")
            return None
        