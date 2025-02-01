import torch
import numpy as np
from rknn.api import RKNN


class RKNNLiteRuntime:
    def __init__(self, rknn_path):
        self.rknn=RKNN()
        print(f'| --> Load RKNN model')
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model.")
        print(f'| RKNN model path: {rknn_path}')


    def init_runtime(self): 
        print(f'| --> Init runtime environment')
        ret = self.rknn.init_runtime(target='RK3588', perf_debug=True)
        if ret != 0:
            print(f'| Init runtime environment failed')
            exit(ret)
            
    
    def run(self, inputs):
       return self.rknn.inference(inputs=inputs)
    

    def latency(self, loop_cnt=100, encoder=False, decoder=False):
        if encoder:
            # encoder_ffn_embed_dim_avg,encoder_self_attention_heads_avg만 측정
            # 리턴값: 초 단위, dtype: float
            ret = self.rknn.eval_perf(loop_cnt=loop_cnt)
            if ret !=0:
                raise RuntimeError(f"Latency evaluation failed")
            avg_total_time= ret['total_time'] / loop_cnt / 1000.0

            # 레이어별 실행 시간
            layer_times = ret.get('layer_times', [])
            avg_layer_times = [(lt / loop_cnt) * 1000 for lt in layer_times]
            return avg_total_time, avg_layer_times

        if decoder:
            # decoder_ffn_embed_dim_avg,decoder_self_attention_heads_avg,decoder_ende_attention_heads_avg,decoder_arbitrary_ende_attn_avg
            # 리턴값: 초 단위, dtype: float
            ret = self.rknn.eval_perf(loop_cnt=loop_cnt)
            if ret !=0:
                raise RuntimeError(f"Latency evaluation failed")
            avg_total_time= ret['total_time'] / loop_cnt / 1000.0

            # 레이어별 실행 시간
            layer_times = ret.get('layer_times', [])
            avg_layer_times = [(lt / loop_cnt) * 1000 for lt in layer_times]
            return avg_total_time, avg_layer_times
    

    def release(self):
       return self.rknn.release()
    
    
class WrapperModelRKNN:
    def __init__(self, model_name, full=False, coder=False):
        self.full = full
        self.coder = coder
    
        if full:
            self.full_path = f'rknn_models/{model_name}/{model_name}.rknn'
            self.rknn = RKNNLiteRuntime(self.full_path)
        if coder:
            self.encoder_rknn_path = f'rknn_models/{model_name}/{model_name}_enc.rknn'
            self.decoder_rknn_path = f'rknn_models/{model_name}/{model_name}_dec.rknn'
            self.Encoder = RKNNLiteRuntime(self.encoder_rknn_path) 
            self.Decoder = RKNNLiteRuntime(self.decoder_rknn_path) 
    
    
    def init_runtime(self, full=False, coder=False): # init rknn lite runtime
        if full:
            self.rknn.init_runtime()
        if coder:
            self.Encoder.init_runtime() 
            self.Decoder.init_runtime() 
    
    
    def encoder(self, src_tokens):
        src_tokens = src_tokens.numpy()
        src_tokens = src_tokens.reshape(1, 1, *src_tokens.shape) 
        inputs = [src_tokens]
        return self.Encoder.run(inputs)
        
        
    def decoder(self, prev_output_tokens, encoder_out, incremental_state=None):
        prev_output_tokens = prev_output_tokens.numpy()
        prev_output_tokens = prev_output_tokens.reshape(1, 1, *prev_output_tokens.shape)
        encoder_out = encoder_out.numpy()
        encoder_out = encoder_out.reshape(1, *encoder_out.shape)
        inputs = [prev_output_tokens, encoder_out]
        if incremental_state is not None:
            print("Warning: incremental_state is not supported in this implementation.")    
        return self.Decoder.run(inputs)
    
    
    def latency(self, encoder=False, decoder=False):
        if encoder:
            self.encoder.latency(encoder=True)
        if decoder:
            self.decoder.latency(decoder=True)
    

    def release(self, full=False, encoder=False, decoder=False):
        if full:
            self.rknn.release()
        if encoder:
            self.encoder.release()
        if decoder:
            self.decoder.release()