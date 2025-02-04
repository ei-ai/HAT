import torch
import numpy as np
from rknn.api import RKNN


class RKNNLiteRuntime:
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
    
class WrapperModelRKNN:
    def __init__(self, model_name, type=None):
        self.type = type
        self.rknn = RKNN()
        if model_name == 'wmt16_en_de':
            model_name = 'wmt14_en_de'
        self.rknn_path = f'rknn_models/{model_name}/{model_name}_{type}.rknn'
        
        print(f'| --> Load RKNN model')
        ret = self.rknn.load_rknn(self.rknn_path)
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model.")
        print(f'| RKNN model path: {self.rknn_path}')

    
    def init_runtime(self, target=None):
        if target is None:
            print(f'W Rknn type does not specified')
            
        print(f'| --> Init {self.type} runtime environment')
        ret = self.rknn.init_runtime(target, perf_debug=True)
        if ret != 0:
            print(f'| Init runtime environment failed')
            exit(ret)
        print(f'| Init done')
    

    def encoder(self, src_tokens):
        src_tokens = src_tokens.numpy()
        src_tokens = src_tokens.reshape(1, 1, *src_tokens.shape) 
        inputs = [src_tokens]
        return self.rknn.inference(inputs=inputs)
        
        
    def decoder(self, prev_output_tokens, encoder_out):
        prev_output_tokens = prev_output_tokens.numpy()
        prev_output_tokens = prev_output_tokens.reshape(1, 1, *prev_output_tokens.shape)
        encoder_out = encoder_out.numpy()
        encoder_out = encoder_out.reshape(1, *encoder_out.shape)
        inputs = [prev_output_tokens, encoder_out] 
        return self.rknn.inference(inputs=inputs)
    
    # for testing
    def latency(self):
        if self.type=='enc':
            # encoder_ffn_embed_dim_avg,encoder_self_attention_heads_avg만 측정
            # 리턴값: 초 단위, dtype: float
            print('| Encoder')

            perf_datail = self.rknn.eval_perf()
            if ret !=0:
                raise RuntimeError(f"| Encoder latency evaluation failed")

            return perf_datail

        if self.type=='dec':
            # decoder_ffn_embed_dim_avg,decoder_self_attention_heads_avg,decoder_ende_attention_heads_avg,decoder_arbitrary_ende_attn_avg
            # 리턴값: 초 단위, dtype: float
            print('| Decoder')

            ret = self.rknn.eval_perf()
            if ret !=0:
                raise RuntimeError(f"| Decoder latency evaluation failed")

            return perf_datail

    def release(self):
        self.rknn.release()