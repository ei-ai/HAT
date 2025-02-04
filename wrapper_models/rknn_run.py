import torch
import numpy as np
from rknnlite.api import RKNNLite

class RKNNLite:
    def __init__(self, model_name, type):
        self.type = type
        self.rknn_path = f'rknn_models/{model_name}/{model_name}_{self.type}.rknn'
        self.rknn_lite = RKNNLite()
        print(f'| --> Load RKNN model')
        ret = self.rknn_lite.load_rknn(self.rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model.")
        print(f'| RKNN model path: {self.rknn_path}')

        print(f'| --> Init runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print(f'| Init runtime environment failed')
            exit(ret)


    def run(self, src_tokens, prev_output_tokens):
        src_tokens = src_tokens.numpy()
        prev_output_tokens = prev_output_tokens.numpy()
        src_tokens = np.expand_dims(src_tokens, axis=0)
        prev_output_tokens = np.expand_dims(prev_output_tokens, axis=0)
        inputs = [src_tokens, prev_output_tokens]
        return self.rknn_lite.inference(inputs=inputs)
    
    def encoder(self, src_tokens):
        src_tokens = src_tokens.numpy()
        # src_tokens = src_tokens.reshape(1, 1, *src_tokens.shape) 
        inputs = [src_tokens]
        return self.rknn_lite.inference(inputs=inputs)
        
        
    def decoder(self, prev_output_tokens, encoder_out):
        prev_output_tokens = prev_output_tokens.numpy()
        prev_output_tokens = prev_output_tokens.reshape(1, 1, *prev_output_tokens.shape)
        encoder_out = encoder_out.numpy()
        encoder_out = encoder_out.reshape(1, *encoder_out.shape)
        inputs = [prev_output_tokens, encoder_out] 
        return self.rknn_lite.inference(inputs=inputs)

    def release(self):
        self.rknn_lite.release()
        
