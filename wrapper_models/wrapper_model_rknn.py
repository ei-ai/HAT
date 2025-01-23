import torch
import numpy as np
from rknn.api import RKNN
#from rknnlite.api import RKNNLite
# lite 말고 rknn 쓰게 바꿔야 함 

class RKNNLiteRuntime:
    def __init__(self, rknn_path):
        #self.rknn_lite = RKNNLite()
        self.rknn=RKNN()
        print(f'| --> Load RKNN model')
        #ret = self.rknn_lite.load_rknn(rknn_path)
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model.")
        print(f'| RKNN model path: {rknn_path}')


    def init_runtime(self): 
        print(f'| --> Init runtime environment')
        #ret = self.rknn_lite.init_runtime()
        ret = self.rknn.init_runtime(target='RK3588')
        if ret != 0:
            print(f'| Init runtime environment failed')
            exit(ret)
            
    
    def run(self, inputs):
       # return self.rknn_lite.inference(inputs=inputs)
       return self.rknn.inference(inputs=inputs)
    

    def latency(self, encoder=False, decoder=False):
        if encoder:
            # encoder_ffn_embed_dim_avg,encoder_self_attention_heads_avg만 측정
            # 리턴값: 초 단위, dtype: float
            return
        if decoder:
            # decoder_ffn_embed_dim_avg,decoder_self_attention_heads_avg,decoder_ende_attention_heads_avg,decoder_arbitrary_ende_attn_avg
            # 리턴값: 초 단위, dtype: float
            return
    

    def release(self):
       # return self.rknn_lite.release()
       return self.rknn.release()
    
    
class WrapperModelRKNN:
    def __init__(self, dataset_name, full=True, coder=True):
        self.full = full
        self.coder = coder
    
        if full:
            self.full_path = f'rknn_models/{dataset_name}/{dataset_name}.rknn'
            self.full = RKNNLiteRuntime(self.full_path)
        if coder:
            self.encoder_rknn_path = f'rknn_models/{dataset_name}/{dataset_name}_enc.rknn'
            self.decoder_rknn_path = f'rknn_models/{dataset_name}/{dataset_name}_dec.rknn'
            self.Encoder = RKNNLiteRuntime(self.encoder_rknn_path) 
            self.Decoder = RKNNLiteRuntime(self.decoder_rknn_path) 
    
    
    def init_runtime(self, full=True, coder=True): # init rknn lite runtime
        if full:
            self.full()
        if coder:
            self.Encoder.init_runtime() 
            self.Decoder.init_runtime() 
    
    
    def encoder(self, src_tokens, src_lengths):
        src_tokens = src_tokens.numpy() 
        src_tokens = np.expand_dims(src_tokens, axis=0)
        inputs = [src_tokens]
        inputs = np.array(inputs, int)
        print(inputs)
        return self.Encoder.run(inputs)

        
    def decoder(self, prev_output_tokens, encoder_out, incremental_state=None):
        prev_output_tokens = prev_output_tokens.numpy()
        prev_output_tokens = np.expand_dims(prev_output_tokens, axis=0)
        encoder_out = encoder_out.numpy()
        encoder_out = np.expand_dims(encoder_out, axis=0)
        inputs = [prev_output_tokens, encoder_out]
        # inputs = [prev_output_tokens] + encoder_out
        if incremental_state is not None:
            print("Warning: incremental_state is not supported in this implementation.")    
        return self.Decoder.run(inputs)
    
    
    def latency(self, full=False, encoder=False, decoder=False):
        if full:
            self.full.lat(full=True)
        if encoder:
            self.encoder.release(encoder=True)
        if decoder:
            self.decoder.release(decoder=True)
    

    def release(self, full=False, encoder=False, decoder=False):
        if full:
            self.full.release()
        if encoder:
            self.encoder.release()
        if decoder:
            self.decoder.release()
            
        
