# Contents
* [wmt16_en_de](#wmt16_en_de)
* [wmt14_en_fr](#wmt14_en_fr)
* [wmt19_en_de](#wmt19_en_de)
* [iwslt14_de_en](#iwslt14_de_en)

W load_onnx: The config.mean_values is None, zeros will be set for input 1!
W load_onnx: The config.std_values is None, ones will be set for input 1!
W load_onnx: The config.mean_values is None, zeros will be set for input 2!
W load_onnx: The config.std_values is None, ones will be set for input 2!
W load_onnx: The config.mean_values is None, zeros will be set for input 3!
W load_onnx: The config.std_values is None, ones will be set for input 3!

# wmt16_en_de
`python convert_rknn.py --dataset-name=wmt14_en_de`
`python convert_rknn.py --dataset-name=wmt14_en_de > ./errors/rknn/convert_rknn_wmt14.en-de.txt 2>&1`  
  
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
`python convert_rknn.py --dataset-name=wmt14_en_fr`
`python convert_rknn.py --dataset-name=wmt14_en_fr > ./errors/rknn/convert_rknn_wmt14.en-fr.txt 2>&1`  
  
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
`python convert_rknn.py --dataset-name=wmt19_en_de`
`python convert_rknn.py --dataset-name=wmt19_en_de > ./errors/rknn/convert_rknn_wmt19.en-de.txt 2>&1`  
  
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
`python convert_rknn.py --dataset-name=iwslt14_de_en`
`python convert_rknn.py --dataset-name=iwslt14_de_en > ./errors/rknn/convert_rknn_iwslt14.de-en.txt 2>&1`
  
I Loading : 100%|█████████████████████████████████████████████████| 29/29 [00:00<00:00, 2495.69it/s]
[1;33mW[0m [1;33mbuild: For tensor ['/model/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/model/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/model/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/model/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
[1;33mW[0m [1;33mbuild: For tensor ['/model/decoder/layers.4/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.[0m
| done
I rknn building ...
E RKNN: [17:58:14.673] dataconvert type -1 is unsupport in current!
E RKNN: [17:58:14.673] dataconvert type -1 is unsupport in current!
I rknn building done.
No lowering found for: /model/encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /model/decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.

[맨위로](#contents)
    