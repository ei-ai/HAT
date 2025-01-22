# Contents
* [wmt14ende](#wmt14ende)
* [wmt14enfr](#wmt14enfr)
* [wmt19ende](#wmt19ende)
* [iwslt14deen](#iwslt14deen)

# wmt14ende
`python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/super.yml`
`python convert2onnx.py --configs=configs/wmt14.en-de/convert_onnx/super.yml > ./errors/onnx/convert_onnx_wmt14.en-de.txt 2>&1`
    
[맨위로](#contents)
    
# wmt14enfr
`python convert2onnx.py --configs=configs/wmt14.en-fr/convert_onnx/super.yml`
`python convert2onnx.py --configs=configs/wmt14.en-fr/convert_onnx/super.yml > ./errors/onnx/convert_onnx_wmt14.en-fr.txt 2>&1`

[맨위로](#contents)
    
# wmt19ende
`python convert2onnx.py --configs=configs/wmt19.en-de/convert_onnx/super.yml`
`python convert2onnx.py --configs=configs/wmt19.en-de/convert_onnx/super.yml > ./errors/onnx/convert_onnx_wmt19.en-de.txt 2>&1`

    
[맨위로](#contents)
    
# iwslt14deen
`python convert2onnx.py --configs=configs/iwslt14.de-en/convert_onnx/super.yml`
`python convert2onnx.py --configs=configs/iwslt14.de-en/convert_onnx/super.yml > ./errors/onnx/convert_onnx_iwslt14.de-en.txt 2>&1`

1. fairseq/modules/sinusoidal_positional_embedding.py:60: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  if self.weights is None or max_pos > self.weights.size(0):  
2. /home/mk/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:62: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  new_max_pos = max(max_pos, self.weights.size(0) if self.weights is not None else 0)  
3. fairseq/models/transformer_super.py:394: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  if not encoder_padding_mask.any():  
4. fairseq/modules/multihead_attention_super.py:265: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]  
5. fairseq/modules/multihead_attention_super.py:295: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]  
6. fairseq/modules/multihead_attention_super.py:297: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  if (self.onnx_trace and attn.size(1) == 1):  
7. fairseq/models/transformer_super.py:735: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
  if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:  

      
[맨위로](#contents)
    