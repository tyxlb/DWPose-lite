# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import numpy as np
from . import util

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self,device="CPU",det=True):
        """
        det:detect person, recommended to disable it if only one person
        """
        onnx_det = 'ckpts/yolox_l.onnx'
        onnx_pose = 'ckpts/dw-ll_ucoco_384.onnx'
        providers=['CUDAExecutionProvider'] if device=="GPU" else ['CPUExecutionProvider']

        self.det=det
        if self.det:
            self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg:np.ndarray):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        if self.det:
            det_result = inference_detector(self.session_det, oriImg)
        else:
            det_result=[]
        candidate, subset = inference_pose(self.session_pose, det_result, oriImg)
        body = candidate[:,:18].copy()
        score = subset[:,:18]
        score[score<0.3]=-1

        un_visible = subset<0.3
        candidate[un_visible] = -1

        foot = candidate[:,18:24]

        faces = candidate[:,24:92]

        hands = candidate[:,92:113]
        hands = np.vstack([hands, candidate[:,113:]])
        
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)

        return draw_pose(pose, H, W)
