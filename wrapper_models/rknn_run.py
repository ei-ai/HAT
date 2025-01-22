import torch
import numpy as np
from rknnlite.api import RKNNLite


class RKNNLiteRuntime:
    def __init__(self, rknn_path):
        self.rknn_lite = RKNNLite()
        print(f'| --> Load RKNN model')
        ret = self.rknn_lite.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model.")
        print(f'| RKNN model path: {rknn_path}')


    def init_runtime(self): 
        print(f'| --> Init runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print(f'| Init runtime environment failed')
            exit(ret)
            
    
    def run(self, inputs):
        return self.rknn_lite.inference(inputs=inputs)
    

    def release(self):
        return self.rknn_lite.release()
    
    
class WrapperModelRKNN:
    def __init__(self, dataset_name):
        self.full_path = f'rknn_models/{dataset_name}/{dataset_name}.rknn'
        self.full = RKNNLiteRuntime(self.full_path)
    
    
    def init_runtime(self): # init rknn lite runtime
        self.model()


    def run(self, src_tokens, prev_output_tokens):
        src_tokens = src_tokens.numpy()
        prev_output_tokens = prev_output_tokens.numpy()
        src_tokens = np.expand_dims(src_tokens, axis=0)
        prev_output_tokens = np.expand_dims(prev_output_tokens, axis=0)
        inputs = [src_tokens, prev_output_tokens]
        return self.model.inference(inputs=inputs)
    

    def release(self, ):
        self.model.release()
        
