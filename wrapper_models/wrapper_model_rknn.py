import torch
from rknn.api import RKNNLite

# 인코더랑 디코더 인퍼런스 함수랑 시간 측정 함수만 만들면 끝난드아....

class RKNNEncoder:
    def __init__(self, encoder_rknn_path): #모델 로드
        self.encoder_rknn = RKNNLite()
        print('--> Load RKNN Encoder model')
        ret = self.encoder_rknn.load_rknn(encoder_rknn_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN encoder model.")

    def init_runtime(self): # 모델 초기화
        ret = self.rknn_lite.rknn.init_runtime(target='rk3588')
        if ret != 0:
            print('Init encoder runtime failed')
            exit(ret)

    def forward(self, src_tokens, src_lengths):
        inputs = self._prepare_encoder_inputs(src_tokens, src_lengths)
        outputs = self.encoder_rknn.inference(inputs=inputs)
        return torch.tensor(outputs[0])

    def reorder_encoder_out(self, encoder_out, new_order): # fairseq/models/transformer_super.py에서 복붙
        # 뭐야 얘도 안쓰여(latency_dataset.py)
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)

        if 'encoder_out_all' in encoder_out.keys():
            new_encoder_out_all = []
            for encoder_out_one_layer in encoder_out['encoder_out_all']:
                new_encoder_out_all.append(encoder_out_one_layer.index_select(1, new_order))
            encoder_out['encoder_out_all'] = new_encoder_out_all

        return encoder_out
    
    def _prepare_encoder_inputs(self, src_tokens, src_lengths):
        return [src_tokens.numpy(), src_lengths.numpy()]


class RKNNDecoder:
    def __init__(self, decoder_rknn_path): #모델 로드
        self.decoder_rknn = RKNNLite()
        print('--> Load RKNN Decoder model')
        ret = self.decoder_rknn.load_rknn(decoder_rknn_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN decoder model.")

    def init_runtime(self): # 모델 초기화
        ret = self.rknn_lite.rknn.init_runtime(target='rk3588')
        if ret != 0:
            print('Init dncoder runtime failed')
            exit(ret)
    
    def forward(self, prev_output_tokens, encoder_outputs, incremental_state=None):
        if incremental_state is not None: # 실제 인퍼런스
            inputs = self._prepare_decoder_inputs_with_state(prev_output_tokens, encoder_outputs, incremental_state)
        else: # 드라이 런
            inputs = self._prepare_decoder_inputs(prev_output_tokens, encoder_outputs)
        
        outputs = self.decoder_rknn.inference(inputs=inputs)
        return torch.tensor(outputs[0])
    
    def _prepare_decoder_inputs_with_state(self, prev_output_tokens, encoder_outputs, incremental_state):
        prepared_inputs = [
            prev_output_tokens.numpy(),
            encoder_outputs.numpy(),
            incremental_state
        ]
        return prepared_inputs

    def _prepare_decoder_inputs(self, prev_output_tokens, encoder_outputs):
        return [prev_output_tokens.numpy(), encoder_outputs.numpy()]


class WrapperModelRKNN(torch.nn.Module):
    def __init__(self, model, datset_name):
        super(WrapperModelRKNN, self).__init__()
        self.model = model
        self.rknn_lite = RKNNLite()
        self.rknn_path = f'rknn_models/{datset_name}/{datset_name}.rknn'
        self.encoder_rknn_path = f'rknn_models/{datset_name}/{datset_name}_enc.rknn'
        self.decoder_rknn_path = f'rknn_models/{datset_name}/{datset_name}_dec.rknn'
        
        ret = self.rknn_lite.rknn.load_rknn(self.rknn_path) #모델 로드
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('--> Load RKNN model')

        self.encoder = RKNNEncoder(self.encoder_rknn_path) # 인코더 로드
        self.decoder = RKNNDecoder(self.decoder_rknn_path) # 디코더 로드
    
    def npu(self): # 모델 초기화
        ret = self.rknn_lite.rknn.init_runtime(target='rk3588')
        if ret != 0:
            print('Init runtime failed')
            exit(ret)
        self.encoder.init_runtime() # 인코더 초기화
        self.decoder.init_runtime() # 디코더 초기화
    
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        # input = rknn 모델 입력으로 변환 필요
        # 일단 현재 작업 중인 latency_dataset.py에서는 안 쓰여서 나중에
        outputs = self.rknn_lite.inference(inputs=inputs)
        return torch.tensor(outputs[0])

