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
    
    """
    def make_positions(tensor, padding_idx, onnx_trace=False):
        mask = tensor.ne(padding_idx).int() 
        cumsum_result = mask.clone()   
        for i in range(1, mask.size(1)): 
            cumsum_result[:, i] += cumsum_result[:, i - 1]  

        return (
            cumsum_result.type_as(mask) * mask
        ).long() + padding_idx
    """
        
