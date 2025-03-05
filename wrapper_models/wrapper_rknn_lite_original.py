import numpy as np
from rknnlite.api import RKNNLite

class WrapperModelRKNNLite:
    def __init__(self, model_name, type):
        self.type = type
        self.model_name = model_name
        if model_name=='wmt16_en_de':
            model_name = 'wmt14_en_de'
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
        inputs = [src_tokens, prev_output_tokens]
        return self.rknn_lite.inference(inputs=inputs)
    
    def encoder(self, src_tokens):
        src_tokens = src_tokens.numpy()
        inputs = [src_tokens]
        return self.rknn_lite.inference(inputs)
        
        
    def decoder(self, prev_output_tokens, encoder_out):
        prev_output_tokens = prev_output_tokens.numpy()
        encoder_out = encoder_out.numpy()
        if 'iwslt' in self.model_name:
            Concat_6=encoder_out.copy()
            inputs = [prev_output_tokens, encoder_out, Concat_6]
        elif 'wmt' in self.model_name:
            Concat_5=encoder_out.copy()
            Concat_6=encoder_out.copy()
            inputs = [prev_output_tokens, encoder_out, Concat_5, Concat_6] 
        return self.rknn_lite.inference(inputs=inputs)

    def release(self):
        self.rknn_lite.release()
        

