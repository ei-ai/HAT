1. train.py iwslt
```
| Measuring model latency on NPU...
W inference: The 'data_format' is not set, and its default value is 'nhwc'!
E RKNN: [10:03:04.261] rknn_inputs_set, param input size(184) < model input size(200)
E inference: Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line 344, in rknn.api.rknn_log.error_catch_decorator.error_catch_wrapper
  File "rknn/api/rknn_base.py", line 2678, in rknn.api.rknn_base.RKNNBase.inference
  File "rknn/api/rknn_runtime.py", line 582, in rknn.api.rknn_runtime.RKNNRuntime.set_inputs
Exception: Set inputs failed. error code: RKNN_ERR_PARAM_INVALID

W inference: ===================== WARN(1) =====================
E rknn-toolkit2 version: 2.3.0
Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line 344, in rknn.api.rknn_log.error_catch_decorator.error_catch_wrapper
  File "rknn/api/rknn_base.py", line 2678, in rknn.api.rknn_base.RKNNBase.inference
  File "rknn/api/rknn_runtime.py", line 582, in rknn.api.rknn_runtime.RKNNRuntime.set_inputs
Exception: Set inputs failed. error code: RKNN_ERR_PARAM_INVALID

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/radxa/git/HAT/train.py", line 470, in <module>
    cli_main()
  File "/home/radxa/git/HAT/train.py", line 424, in cli_main
    latency_npu(args)
  File "/home/radxa/git/HAT/train.py", line 344, in latency_npu
    model.encoder(src_tokens=src_tokens_test)
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 87, in encoder
    return self.Encoder.run(inputs)
  File "/home/radxa/git/HAT/wrapper_models/wrapper_model_rknn.py", line 25, in run
    return self.rknn.inference(inputs=inputs)
  File "/home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/rknn/api/rknn.py", line 308, in inference
    return self.rknn_base.inference(inputs=inputs, data_format=data_format,
  File "rknn/api/rknn_log.py", line 349, in rknn.api.rknn_log.error_catch_decorator.error_catch_wrapper
  File "rknn/api/rknn_log.py", line 95, in rknn.api.rknn_log.RKNNLog.e
ValueError: Traceback (most recent call last):
  File "rknn/api/rknn_log.py", line이
```
(rknn) radxa@rock-5b:~/git/HAT$ python latency_dataset_test.py --latnpu --configs=configs/wmt14.en-de/latency_dataset/npu.yml
| [en] dictionary: 32768 types
| [de] dictionary: 32768 types
Measuring model latency on NPU for dataset generation...
W rknn-toolkit-lite2 version: 2.3.0
| --> Load RKNN model
| RKNN model path: rknn_models/wmt14_en_de/wmt14_en_de_enc.rknn
| --> Init runtime environment
I RKNN: [07:45:06.415] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [07:45:06.415] RKNN Driver Information, version: 0.9.6
I RKNN: [07:45:06.416] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), targ
et: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [07:45:06.416] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [07:45:06.493] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_o
ps or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [07:45:06.493] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_sha
pes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning me
ssage.)
W rknn-toolkit-lite2 version: 2.3.0
| --> Load RKNN model
| RKNN model path: rknn_models/wmt14_en_de/wmt14_en_de_dec.rknn
| --> Init runtime environment
I RKNN: [07:45:07.462] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [07:45:07.462] RKNN Driver Information, version: 0.9.6
I RKNN: [07:45:07.462] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), targ
et: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [07:45:07.462] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [07:45:07.578] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_o
ps or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [07:45:07.578] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_sha
pes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning me
ssage.)
0
Segmentation fault

(rknn) radxa@rock-5b:~/git/HAT$ python latency_dataset_test.py --latnpu --configs=configs/iwslt14.de-en/latency_dataset/npu.yml
| [de] dictionary: 8848 types
| [en] dictionary: 6632 types
Measuring model latency on NPU for dataset generation...
W rknn-toolkit-lite2 version: 2.3.0
| --> Load RKNN model
| RKNN model path: rknn_models/iwslt14_de_en/iwslt14_de_en_enc.rknn
| --> Init runtime environment
I RKNN: [07:47:03.861] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [07:47:03.862] RKNN Driver Information, version: 0.9.6
I RKNN: [07:47:03.862] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), targ
et: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [07:47:03.862] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [07:47:03.909] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_o
ps or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [07:47:03.910] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_sha
pes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning me
ssage.)
W rknn-toolkit-lite2 version: 2.3.0
| --> Load RKNN model
| RKNN model path: rknn_models/iwslt14_de_en/iwslt14_de_en_dec.rknn
| --> Init runtime environment
I RKNN: [07:47:04.039] RKNN Runtime Information, librknnrt version: 2.0.0b0 (35a6907d79@2024-03-24T10:31:14)
I RKNN: [07:47:04.039] RKNN Driver Information, version: 0.9.6
I RKNN: [07:47:04.039] RKNN Model Information, version: 6, toolkit version: 2.3.0(compiler version: 2.3.0 (@2024-11-07T08:11:34)), targ
et: RKNPU v2, target platform: rk3588, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [07:47:04.039] RKNN Model version: 2.3.0 not match with rknn runtime version: 2.0.0
E RKNN: [07:47:04.112] Unsupport CPU op: CumSum in this librknnrt.so, please try to register custom op by callingrknn_register_custom_o
ps or please try updating to the latest version of the toolkit2 and runtime from: https://console.zbox.filez.com/l/I00fc3 (PWD: rknn)
W RKNN: [07:47:04.113] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_sha
pes
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning me
ssage.)
0
E RKNN: [07:47:04.114] rknn_inputs_set, param input size(184) < model input size(200)
E Catch exception when setting inputs.
E Traceback (most recent call last):
 File "/home/radxa/miniconda3/envs/rknn/lib/python3.9/site-packages/rknnlite/api/rknn_lite.py", line 209, in inference
   self.rknn_runtime.set_inputs(inputs, data_type, data_format, inputs_pass_through=inputs_pass_through)
 File "rknnlite/api/rknn_runtime.py", line 1164, in rknnlite.api.rknn_runtime.RKNNRuntime.set_inputs
Exception: Set inputs failed. error code: RKNN_ERR_PARAM_INVALID
```
