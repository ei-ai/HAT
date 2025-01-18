1. ImportError: It imports rknnlite from rknn.api
* It should be rknnlite.api, not rknn.api
```sh 
(rknn) radxa@rock-5b:~/git/HAT$ python latency_dataset.py --latnpu --configs=configs/iwslt14.de-en/latency_dataset/npu.yml
In file included from /home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/numpy/core/include/numpy/ndarraytypes.h:1929,
                 from /home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/numpy/core/include/numpy/ndarrayobject.h:12,
                 from /home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/numpy/core/include/numpy/arrayobject.h:5,
                 from /home/radxa/.pyxbld/temp.linux-aarch64-cpython-39/home/radxa/git/HAT/fairseq/data/data_utils_fast.c:1240:
/home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:17:2: warning: #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
   17 | #warning "Using deprecated NumPy API, disable it with " \
      |  ^~~~~~~
Traceback (most recent call last):
  File "/home/radxa/git/HAT/latency_dataset.py", line 17, in <module>
    from wrapper_models import wrapper_model_rknn
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 2, in <module>
    from rknn.api import RKNNLite
ImportError: cannot import name 'RKNNLite' from 'rknn.api' (/home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/rknn/api/__init__.py)
```

2. AttrivuteError: 'RKNNLite' object has  no attribute 'rknn' 
* I changed `wrapper_model_rknn.py`(maybe line 2) as `from rknnlite.api...` 
* & I got this message below
* I think that code was written assuming that rknn.api would be used
* so i reverted my change and write this 
```sh
(rknn) radxa@rock-5b:~/git/HAT$ python latency_dataset.py --latnpu --configs=configs/iwslt14.de-en/latency_dataset/npu.yml
Namespace(configs='configs/iwslt14.de-en/latency_dataset/npu.yml', pdb=False, no_progress_bar=False, log_interval=1000, log_format=None, tensorboard_logdir='', tbmf_wrapper=False, seed=1, cpu=False, fp16=False, memory_efficient_fp16=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir=None, criterion='cross_entropy', optimizer='nag', lr_scheduler='fixed', task='translation', num_workers=10, skip_invalid_size_inputs_valid_test=False, max_tokens=4096, max_sentences=None, required_batch_size_multiple=8, dataset_impl=None, train_subset='train', valid_subset='valid', validate_interval=1, disable_validation=False, max_tokens_valid=4096, max_sentences_valid=None, curriculum=0, distributed_world_size=1, distributed_rank=0, distributed_backend='nccl', distributed_init_method=None, distributed_port=-1, device_id=0, distributed_no_spawn=False, ddp_backend='c10d', bucket_cap_mb=25, fix_batches_to_gpus=False, find_unused_parameters=False, arch='transformersuper_iwslt_de_en', max_epoch=0, max_update=0, clip_norm=25, sentence_avg=False, update_freq=[1], lr=[0.25], min_lr=-1, use_bmuf=False, save_dir=None, restore_file='checkpoint_last.pt', reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, optimizer_overrides='{}', save_interval=1, save_interval_updates=0, keep_interval_updates=-1, keep_last_epochs=-1, no_save=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_save_optimizer_state=False, best_checkpoint_metric='loss', maximize_best_checkpoint_metric=False, latnpu=True, latgpu=True, latcpu=False, latiter=20, latsilent=True, lat_dataset_path='./latency_dataset/iwslt14deen_npu.csv', lat_dataset_size=2000, path=None, remove_bpe=None, quiet=False, model_overrides='{}', results_path=None, beam=5, nbest=1, max_len_a=0, max_len_b=200, min_len=1, match_source_len=False, no_early_stop=False, unnormalized=False, no_beamable_mm=False, lenpen=1, unkpen=0, replace_unk=None, sacrebleu=False, score_reference=False, prefix_size=0, no_repeat_ngram_size=0, sampling=False, sampling_topk=-1, sampling_topp=-1.0, temperature=1.0, diverse_beam_groups=-1, diverse_beam_strength=0.5, print_alignment=False, profile_latency=False, no_token_positional_embeddings=False, get_attn=False, encoder_embed_choice=[640, 512], decoder_embed_choice=[640, 512], encoder_layer_num_choice=[6], decoder_layer_num_choice=[6, 5, 4, 3, 2, 1], encoder_ffn_embed_dim_choice=[3072, 2048, 1024, 512], decoder_ffn_embed_dim_choice=[3072, 2048, 1024, 512], encoder_self_attention_heads_choice=[8, 4, 2], decoder_self_attention_heads_choice=[8, 4, 2], decoder_ende_attention_heads_choice=[8, 4, 2], qkv_dim=512, decoder_arbitrary_ende_attn_choice=[-1, 1, 2], vocab_original_scaling=False, encoder_embed_dim_subtransformer=None, decoder_embed_dim_subtransformer=None, encoder_ffn_embed_dim_all_subtransformer=None, decoder_ffn_embed_dim_all_subtransformer=None, encoder_self_attention_heads_all_subtransformer=None, decoder_self_attention_heads_all_subtransformer=None, decoder_ende_attention_heads_all_subtransformer=None, decoder_arbitrary_ende_attn_all_subtransformer=None, momentum=0.99, weight_decay=0.0, force_anneal=None, lr_shrink=0.1, warmup_updates=0, data='data/binary/iwslt14_de_en', source_lang='de', target_lang='en', lazy_load=False, raw_text=False, left_pad_source='True', left_pad_target='False', max_source_positions=1024, max_target_positions=1024, upsample_primary=1, encoder_embed_dim=640, decoder_embed_dim=640, encoder_ffn_embed_dim=3072, decoder_ffn_embed_dim=3072, encoder_layers=6, decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, encoder_embed_path=None, encoder_normalize_before=False, encoder_learned_pos=False, decoder_embed_path=None, decoder_normalize_before=False, decoder_learned_pos=False, attention_dropout=0.0, activation_dropout=0.0, activation_fn='relu', dropout=0.1, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, share_decoder_input_output_embed=False, share_all_embeddings=False, adaptive_input=False, decoder_output_dim=640, decoder_input_dim=640)
| [de] dictionary: 8848 types
| [en] dictionary: 6632 types
| Fallback to xavier initializer
W rknn-toolkit-lite2 version: 2.3.0
Traceback (most recent call last):
  File "/home/radxa/git/HAT/latency_dataset.py", line 213, in <module>
    cli_main()
  File "/home/radxa/git/HAT/latency_dataset.py", line 210, in cli_main
    main(args)
  File "/home/radxa/git/HAT/latency_dataset.py", line 40, in main
    model = wrapper_model_rknn.WrapperModelRKNN(model, args.data.removeprefix('/data/binary/'))
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 90, in __init__
    ret = self.rknn_lite.rknn.load_rknn(self.rknn_path) #모델 로드
AttributeError: 'RKNNLite' object has no attribute 'rknn'
```

3. inputerror
```sh
I RKNN: [10:23:26.831] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [10:23:26.833] RKNN Driver Information, version: 0.9.6
I RKNN: [10:23:26.839] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [10:23:26.841] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [10:23:26.965] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_ops or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
E RKNN: [10:23:26.965] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_ops or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [10:23:26.965] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
I RKNN: [10:23:27.017] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [10:23:27.017] RKNN Driver Information, version: 0.9.6
I RKNN: [10:23:27.017] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [10:23:27.017] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [10:23:27.084] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_ops or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [10:23:27.084] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
I RKNN: [10:23:27.143] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [10:23:27.143] RKNN Driver Information, version: 0.9.6
I RKNN: [10:23:27.144] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), target: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [10:23:27.144] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [10:23:27.235] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_ops or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [10:23:27.235] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
0
Enc Input :  [array([[2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        7]]), array([23])]
Enc Input shpe :  (1, 23)
Enc Input dtype :  int64
Traceback (most recent call last):
  File "/home/radxa/git/HAT/latency_dataset.py", line 213, in <module>
    cli_main()
  File "/home/radxa/git/HAT/latency_dataset.py", line 210, in cli_main
    main(args)
  File "/home/radxa/git/HAT/latency_dataset.py", line 98, in main
    encoder_out_test = model.encoder(src_tokens=src_tokens_test, src_lengths=src_lengths_test)
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 37, in __call__
    return self.forward(src_tokens, src_lengths)
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 26, in forward
    inputs_info = self.encoder_rknn.inputs
AttributeError: 'RKNNLite' object has no attribute 'inputs'
```
test