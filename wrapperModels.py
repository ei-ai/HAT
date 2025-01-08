import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import fairseq_model
from fairseq.models.transformer_super import TransformerSuperModel

class WrapperModelONNX(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelONNX, self).__init__()
        self.model = model

    def prepare_for_onnx_export(self):
        if hasattr(self.model, "prepare_for_onnx_export_"):
            self.model.prepare_for_onnx_export_()

        for param in self.model.parameters():
            param.requires_grad = False

        for module in self.model.modules():
            module.onnx_trace = True


    