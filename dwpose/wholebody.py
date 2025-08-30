import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    def __init__(self,device="CPU"):
        onnx_det = 'ckpts/yolox_l.onnx'
        onnx_pose = 'ckpts/dw-ll_ucoco_384.onnx'
        providers=['CUDAExecutionProvider'] if device=="GPU" else ['CPUExecutionProvider']

        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg:np.ndarray):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
        return keypoints, scores


