# wmt19ende
`python convert_onnx.py --dataset-name=wmt19ende --configs=configs/wmt19.en-de/convert_onnx/space0.yml`

/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:57: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:59: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.weights is None or max_pos > self.weights.size(0):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not encoder_padding_mask.any():
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert embed_dim == self.embed_dim, (embed_dim, self.embed_dim)
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:109: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(query.size()) == [tgt_len, bsz, embed_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:253: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if (self.onnx_trace and attn.size(1) == 1):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:480: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:

# wmt14ende
`python convert_onnx.py --dataset-name=wmt14ende --configs=configs/wmt14.en-de/convert_onnx/space0.yml`

/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:57: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:59: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.weights is None or max_pos > self.weights.size(0):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not encoder_padding_mask.any():
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert embed_dim == self.embed_dim, (embed_dim, self.embed_dim)
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:109: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(query.size()) == [tgt_len, bsz, embed_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:253: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if (self.onnx_trace and attn.size(1) == 1):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:480: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:

# wmt14enfr
`python convert_onnx.py --dataset-name=wmt14enfr --configs=configs/wmt14.en-fr/convert_onnx/space0.yml`

Traceback (most recent call last):
  File "/Users/minseokim/Documents/git/HAT/convert_onnx.py", line 89, in <module>
    main()
  File "/Users/minseokim/Documents/git/HAT/convert_onnx.py", line 85, in main
    export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name)
  File "/Users/minseokim/Documents/git/HAT/convert_onnx.py", line 33, in export_to_onnx
    torch.onnx.export(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/__init__.py", line 375, in export
    export(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/utils.py", line 502, in export
    _export(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/utils.py", line 1564, in _export
    graph, params_dict, torch_out = _model_to_graph(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/utils.py", line 1113, in _model_to_graph
    graph, params, torch_out, module = _create_jit_graph(model, args)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/utils.py", line 997, in _create_jit_graph
    graph, torch_out = _trace_and_get_graph_from_model(model, args)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/onnx/utils.py", line 904, in _trace_and_get_graph_from_model
    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/jit/_trace.py", line 1500, in _get_trace_graph
    outs = ONNXTracedModule(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/jit/_trace.py", line 139, in forward
    graph, out = torch._C._create_graph_by_tracing(
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/jit/_trace.py", line 130, in wrapper
    outs.append(self.inner(*trace_inputs))
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1726, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/Users/minseokim/Documents/git/HAT/convert_onnx.py", line 18, in forward
    return self.model(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=prev_output_tokens)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1726, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/Users/minseokim/Documents/git/HAT/fairseq/models/fairseq_model.py", line 222, in forward
    encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1726, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/Users/minseokim/Documents/git/HAT/fairseq/models/transformer_super.py", line 381, in forward
    x = self.sample_embed_scale * self.embed_tokens(src_tokens, part='encoder')
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/minseokim/miniconda/envs/hat/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1726, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/Users/minseokim/Documents/git/HAT/fairseq/modules/embedding_super.py", line 53, in forward
    return F.embedding(input, self.sampled_weight(part), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
  File "/Users/minseokim/Documents/git/HAT/fairseq/modules/embedding_super.py", line 50, in sampled_weight
    return self.sample_parameters(part)[part]['weight']
KeyError: 'weight'


# iwslt14deen
`python convert_onnx.py --dataset-name=iwslt14deen --configs=configs/iwslt14.de-en/convert_onnx/space1.yml`

/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:57: TracerWarning: Iterating over a tensor might cause the trace to be incorrect. Passing a tensor of different shape won't change the number of iterations executed (and might lead to errors or silently give incorrect results).
  bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
/Users/minseokim/Documents/git/HAT/fairseq/modules/sinusoidal_positional_embedding.py:59: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if self.weights is None or max_pos > self.weights.size(0):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not encoder_padding_mask.any():
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert embed_dim == self.embed_dim, (embed_dim, self.embed_dim)
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:109: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(query.size()) == [tgt_len, bsz, embed_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:223: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:253: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
/Users/minseokim/Documents/git/HAT/fairseq/modules/multihead_attention.py:254: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if (self.onnx_trace and attn.size(1) == 1):
/Users/minseokim/Documents/git/HAT/fairseq/models/transformer.py:480: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim: