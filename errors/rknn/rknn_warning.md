# Contents
* [wmt16_en_de](#wmt16_en_de)
* [wmt14_en_fr](#wmt14_en_fr)
* [wmt19_en_de](#wmt19_en_de)
* [iwslt14_de_en](#iwslt14_de_en)

```sh
python convert2rknn.py --onnx-name=wmt14_en_de \
    > ./errors/rknn/convert_rknn_wmt14.en_de.txt 2>&1
python convert2rknn.py --onnx-name=wmt14_en_fr \
    > ./errors/rknn/convert_rknn_wmt14_en_fr.txt 2>&1
python convert2rknn.py --onnx-name=wmt19_en_de \
    > ./errors/rknn/convert_rknn_wmt19_en_de.txt 2>&1
python convert2rknn.py --onnx-name=iwslt14_de_en \
    > ./errors/rknn/without_cumsum/convert_rknn_iwslt14_de_en.txt 2>&1

python convert2rknn.py --onnx-name=wmt14_en_de --enc \
    > ./errors/rknn/convert_rknn_wmt14.en_de_enc.txt 2>&1
python convert2rknn.py --onnx-name=wmt14_en_fr --enc \
    > ./errors/rknn/convert_rknn_wmt14_en_fr.txt 2>&1
python convert2rknn.py --onnx-name=wmt19_en_de --enc \
    > ./errors/rknn/convert_rknn_wmt19_en_de_enc.txt 2>&1
python convert2rknn.py --onnx-name=iwslt14_de_en --enc \
    > ./errors/rknn/convert_rknn_iwslt14_de_en_enc.txt 2>&1

python convert2rknn.py --onnx-name=wmt14_en_de --dec \
    > ./errors/rknn/small/convert_rknn_wmt14.en_de_dec.txt 2>&1
python convert2rknn.py --onnx-name=wmt14_en_fr --dec \
    > ./errors/rknn/small/convert_rknn_wmt14.en_fr_dec.txt 2>&1
python convert2rknn.py --onnx-name=wmt19_en_de --dec \
    > ./errors/rknn/small/convert_rknn_wmt19_en_de_dec.txt 2>&1
python convert2rknn.py --onnx-name=iwslt14_de_en --dec \
    > ./errors/rknn/small/convert_rknn_iwslt14_de_en_dec.txt 2>&1

python convert2rknn.py --onnx-name=iwslt14_de_en --enc \
    > ./errors/rknn/small/convert_rknn_iwslt14_de_en_enc.txt 2>&1
python convert2rknn.py --onnx-name=iwslt14_de_en --dec \
    > ./errors/rknn/small/convert_rknn_iwslt14_de_en_dec.txt 2>&1    
```

# wmt16_en_de
`python convert2rknn.py --dataset-name=wmt14_en_de`
`python convert2rknn.py --dataset-name=wmt14_en_de > ./errors/rknn/convert_rknn_wmt14.en-de.txt 2>&1`  
  
I Loading : 100%|█████████████████████████████████████████████████| 24/24 [00:00<00:00, 1717.07it/s]
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
| done
| --> Building RKNN model
I rknn building ...
E RKNN: [22:20:30.187] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:30.187] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:30.187] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:30.187] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:30.201] dataconvert type -1 is unsupport in current!
E RKNN: [22:20:30.201] dataconvert type -1 is unsupport in current!
E RKNN: [22:20:31.306] channel is too large, may produce thousands of regtask, fallback to cpu!
I rknn building done.
No lowering found for: /encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.

[맨위로](#contents)
    

# wmt14_en_fr
`python convert2rknn.py --dataset-name=wmt14_en_fr`
`python convert2rknn.py --dataset-name=wmt14_en_fr > ./errors/rknn/convert_rknn_wmt14.en-fr.txt 2>&1`  
  
I Loading : 100%|█████████████████████████████████████████████████| 16/16 [00:00<00:00, 2253.41it/s]
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
| done
| --> Building RKNN model
I rknn building ...
E RKNN: [22:20:45.940] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:45.940] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:45.940] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:45.940] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:20:45.949] dataconvert type -1 is unsupport in current!
E RKNN: [22:20:45.949] dataconvert type -1 is unsupport in current!
E RKNN: [22:20:46.363] channel is too large, may produce thousands of regtask, fallback to cpu!
I rknn building done.
No lowering found for: /encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.

[맨위로](#contents)
    

# wmt19_en_de
`python convert2rknn.py --dataset-name=wmt19_en_de`
`python convert2rknn.py --dataset-name=wmt19_en_de > ./errors/rknn/convert_rknn_wmt19.en-de.txt 2>&1`  
  
I Loading : 100%|█████████████████████████████████████████████████| 24/24 [00:00<00:00, 2950.79it/s]
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
| done
I rknn building ...
E RKNN: [22:21:04.010] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:21:04.011] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:21:04.011] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:21:04.011] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [22:21:05.039] dataconvert type -1 is unsupport in current!
E RKNN: [22:21:05.039] dataconvert type -1 is unsupport in current!
E RKNN: [22:21:05.893] channel is too large, may produce thousands of regtask, fallback to cpu!
I rknn building done.
No lowering found for: /encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
    
[맨위로](#contents)
    

# iwslt14_de_en
`python convert2rknn.py --dataset-name=iwslt14_de_en`
`python convert2rknn.py --dataset-name=iwslt14_de_en > ./errors/rknn/convert_rknn_iwslt14.de-en.txt 2>&1`

* super
```sh
W RKNN: [18:00:50.471] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:00:50.471] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
# 채널 문제 

...

D RKNN: [18:00:50.576] >>>>>> start: rknn::RKNNLayoutMatchPass
W RKNN: [18:00:50.577] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:00:50.577] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:00:50.577] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:00:50.577] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
D RKNN: [18:00:50.577] <<<<<<<< end: rknn::RKNNLayoutMatchPass
# onnxsim 안 쓴 모델 사용해도 같은 에러가 뜨는 걸로 보아 아마 연산자 미지원이 원인일 것 같다. 

...

D RKNN: [18:38:50.176] >>>>>> start: OpEmit
D RKNN: [18:38:50.176] finish initComputeZoneMapByStepsVector
D RKNN: [18:38:50.176] finish initComputeZoneMapByStepsVector
W RKNN: [18:38:50.177] cast tensor failed
W RKNN: [18:38:50.177] emit dataconvert failed
D RKNN: [18:38:50.177] finish initComputeZoneMapByStepsVector
D RKNN: [18:38:50.177] finish initComputeZoneMapByStepsVector
W RKNN: [18:38:50.177] cast tensor failed
W RKNN: [18:38:50.177] emit dataconvert failed
D RKNN: [18:38:50.177] finish initComputeZoneMapByStepsVector
...
D RKNN: [18:38:50.906] <<<<<<<< end: OpEmit
# ?
# 왜 이런 문제가 생기는 지는 확인이 필요할 것 같다.

W RKNN: [18:00:51.432] Warning: Tensor /model/encoder/embed_positions/Constant_4_output_0 need parameter qtype, type is set to float16 by default!
# iwslt 제외하고는 fp16으로 quantization되게끔 설정되어 있으니까 냅둬도 될 것 같다.
# 다만 iwslt 모델에서 quantization으로 인해 정확도 하락이 발생할 수 있다는 점은 염두에 둬야 함 

```

* enc
```sh
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:01:37.634] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.

...

W RKNN: [18:01:37.635] cast tensor failed
W RKNN: [18:01:37.635] emit dataconvert failed

...

W RKNN: [18:01:37.820] Warning: Tensor /model/embed_positions/Constant_4_output_0 need parameter qtype, type is set to float16 by default!
```

* dec
```sh
D RKNN: [18:02:28.767] >>>>>> start: OpEmit
W RKNN: [18:02:28.768] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.768] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.768] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.769] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.769] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.769] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.770] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:02:28.771] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.772] emitUnpackRegtasks: equiv_channels(E RKNN: [18:02:28.980] dataconvert type -1 is unsupport in current!
25600) > limitations(8192)
W RKNN: [18:02:28.772] emitUnpackRegtasks: equiv_channels(25600) > limitations(8192)
W RKNN: [18:02:28.772] emitUnpackRegtasks: equiv_channels(25600) > limitations(8192)
W RKNN: [18:02:28.772] emitUnpackRegtasks: equiv_channels(25600) > limitations(8192)
W RKNN: [18:02:28.773] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:02:28.773] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.804] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:02:28.804] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.830] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:02:28.831] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)
W RKNN: [18:02:28.858] emitUnpackRegtasks: equiv_channels(38400) > limitations(8192)
W RKNN: [18:02:28.858] emitUnpackRegtasks: equiv_channels(12800) > limitations(8192)

...

D RKNN: [18:02:28.979] >>>>>> start: rknn::RKNNLayoutMatchPass
W RKNN: [18:02:28.979] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:02:28.979] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:02:28.979] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.
W RKNN: [18:02:28.979] LayoutMatchManager: recursion_depth=3, Logic is Dangerous, Will Force layout to native.

...

D RKNN: [18:02:28.980] finish initComputeZoneMapByStepsVector
W RKNN: [18:02:28.980] cast tensor failed
W RKNN: [18:02:28.980] emit dataconvert failed

...

W RKNN: [18:02:30.584] Warning: Tensor /model/embed_positions/Constant_5_output_0 need parameter qtype, type is set to float16 by default!

```

[맨위로](#contents)
    

