import torch
import numpy as np
from rknnlite.api import RKNNLite


class RKNNcoder:
    def __init__(self, rknn_path, encoder=True):
        self.rknn_lite = RKNNLite()
        self.encoder = encoder
        coder_type = 'Encoder' if encoder else 'Decoder'
        print(f'| --> Load RKNN {coder_type} model')
        ret = self.rknn_lite.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN {coder_type} model.")
        print(f'| RKNN {coder_type} model path: {rknn_path}')

    def init_runtime(self, encoder): 
        coder_type = 'encoder' if self.encoder else 'decoder'
        print(f'| --> Init {coder_type} runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print(f'| Init {coder_type} runtime environment failed')
            exit(ret)
            
    
    def run(self, inputs):
        return self.rknn_lite.inference(inputs=inputs)
    
    
class WrapperModelRKNN:
    def __init__(self, dataset_name):
        self.rknn_lite = RKNNLite()
        self.rknn_path = f'rknn_models/{dataset_name}/{dataset_name}.rknn'
        self.encoder_rknn_path = f'rknn_models/{dataset_name}/{dataset_name}_enc.rknn'
        self.decoder_rknn_path = f'rknn_models/{dataset_name}/{dataset_name}_dec.rknn'
        
        print('| --> Load RKNN model')
        ret = self.rknn_lite.load_rknn(self.rknn_path) 
        if ret != 0:
            print('| Load RKNN model failed')
            exit(ret)
        print(f'| RKNN model path: {self.rknn_path}')
        
        self.Encoder = RKNNcoder(self.encoder_rknn_path, encoder=True) 
        self.Decoder = RKNNcoder(self.decoder_rknn_path, encoder=False) 
    
    
    def init_runtime(self): # init rknn lite runtime
        print('| --> Init model runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('| Init model runtime environment failed')
            exit(ret)
            
        self.Encoder.init_runtime(encoder=True) 
        self.Decoder.init_runtime(encoder=False) 
    
    
    def encoder(self, src_tokens, src_lengths):
        src_tokens = src_tokens.numpy()
        src_lengths = src_lengths.numpy()  
        src_tokens = np.expand_dims(src_tokens, axis=0)
        src_lengths = np.expand_dims(src_lengths, axis=0)
        inputs = [src_tokens, src_lengths]
        return self.Encoder.run(inputs)

        
    def decoder(self, prev_output_tokens, encoder_out, incremental_state=None):
        prev_output_tokens = prev_output_tokens.numpy()
        prev_output_tokens = np.expand_dims(prev_output_tokens, axis=0)
        encoder_out = encoder_out["encoder_out"].detach().numpy()
        encoder_out = np.expand_dims(encoder_out, axis=0)
        inputs = [prev_output_tokens, encoder_out]
        # inputs = [prev_output_tokens] + encoder_out
        if incremental_state is not None:
            print("Warning: incremental_state is not supported in this implementation.")    
        return self.Decoder.run(inputs)
    
    
    def run(self, src_tokens, prev_output_tokens):
        src_tokens = src_tokens.numpy()
        prev_output_tokens = prev_output_tokens.numpy()
        inputs = [src_tokens, prev_output_tokens]
        return self.rknn_lite.inference(inputs=inputs)
        
        
