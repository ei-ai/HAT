import torch

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
            if hasattr(module, 'onnx_trace'):
                module.onnx_trace = True
                
            if hasattr(module, 'weights') and hasattr(module, 'padding_idx'):
                module.weights = torch.nn.Parameter(module.weights)
                
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)