import torch
from rknn.api import RKNNLite

class RKNNEncoder:
    def __init__(self, encoder_rknn_path):
        self.encoder_rknn = RKNNLite()
        print('--> Load RKNN Encoder model')
        ret = self.encoder_rknn.load_rknn(encoder_rknn_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN encoder model.")

    def forward(self, src_tokens, src_lengths):
        inputs = self._prepare_encoder_inputs(src_tokens, src_lengths)
        outputs = self.encoder_rknn.inference(inputs=inputs)
        return torch.tensor(outputs[0])

    def reorder_encoder_out(self, encoder_out, new_order):
        # 동일 기능 구현
        return encoder_out.index_select(0, new_order)

    def _prepare_encoder_inputs(self, src_tokens, src_lengths):
        return [src_tokens.numpy(), src_lengths.numpy()]


class RKNNDecoder:
    def __init__(self, decoder_rknn_path):
        self.decoder_rknn = RKNNLite()
        print('--> Load RKNN Decoder model')
        ret = self.decoder_rknn.load_rknn(decoder_rknn_path)
        if ret != 0:
            raise RuntimeError("Failed to load RKNN decoder model.")

    def forward(self, prev_output_tokens, encoder_outputs):
        inputs = self._prepare_decoder_inputs(prev_output_tokens, encoder_outputs)
        outputs = self.decoder_rknn.inference(inputs=inputs)
        return torch.tensor(outputs[0])

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
        
        ret = self.rknn_lite.rknn.load_rknn(self.rknn_path)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('--> Load RKNN model')

        self.encoder = RKNNEncoder(self.encoder_rknn_path)
        self.decoder = RKNNDecoder(self.decoder_rknn_path)
    
    def npu(self): # 모델 초기화, 사실상 돌리는건 인코더 디코더라 걔네를 초기화 시켜야 하는거 아닌가
        ret = self.rknn_lite.rknn.init_runtime()
        if ret != 0:
            print('Init runtime failed')
            exit(ret)
        
    def set_sample_config(config_sam):
        # 동일 기능 구현 
        print()
    
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        # input = rknn 모델 입력으로 변환 필요 
        outputs = self.rknn_lite.inference(inputs=inputs)
        return torch.tensor(outputs[0])

