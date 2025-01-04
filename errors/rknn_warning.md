# Contents
* [wmt14ende](#wmt14ende)
* [wmt14enfr](#wmt14enfr)
* [wmt19ende](#wmt19ende)
* [iwslt14deen](#iwslt14deen)

# wmt14ende
`python convert_rknn.py --dataset-name=wmt14ende`
| --> Building RKNN model
W build: For tensor ['/model/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.4/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.5/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
I OpFusing 0: 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 131.73it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 51.32it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.43it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.28it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 25.93it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 25.27it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 25.04it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:04<00:00, 21.03it/s]
I rknn building ...
No lowering found for: /model/encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /model/decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
E RKNN: [19:02:58.658] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:02:58.658] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:02:58.658] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:02:58.658] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:02:58.663] dataconvert type -1 is unsupport in current!
E RKNN: [19:02:58.663] dataconvert type -1 is unsupport in current!
E RKNN: [19:02:58.757] channel is too large, may produce thousands of regtask, fallback to cpu!
I rknn building done.
    
[맨위로](#contents)
    

# wmt14enfr
`python convert_rknn.py --dataset-name=wmt14enfr`
    
[맨위로](#contents)
    

# wmt19ende
`python convert_rknn.py --dataset-name=wmt19ende`
| --> Building RKNN model
W build: For tensor ['/model/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.4/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.5/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
I OpFusing 0: 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 141.20it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 54.35it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 27.23it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 27.08it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.70it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.04it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 25.82it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:04<00:00, 22.51it/s]
I rknn building ...
No lowering found for: /model/encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /model/decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
E RKNN: [19:06:00.179] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:06:00.179] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:06:00.179] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:06:00.179] channel is too large, may produce thousands of regtask, fallback to cpu!
E RKNN: [19:06:00.183] dataconvert type -1 is unsupport in current!
E RKNN: [19:06:00.184] dataconvert type -1 is unsupport in current!
E RKNN: [19:06:00.274] channel is too large, may produce thousands of regtask, fallback to cpu!
    
[맨위로](#contents)
    

# iwslt14deen
`python convert_rknn.py --dataset-name=iwslt14deen`
| --> Building RKNN model
W build: For tensor ['/model/decoder/layers.0/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.1/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.2/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.3/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.4/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
W build: For tensor ['/model/decoder/layers.5/self_attn/Tile_output_0'], the value smaller than -3e+38 has been corrected to -10000. Set opt_level to 2 or lower to disable this correction.
I OpFusing 0: 100%|██████████████████████████████████████████████| 100/100 [00:00<00:00, 148.35it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:01<00:00, 54.87it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 27.99it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 27.83it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 27.44it/s]
I OpFusing 0 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.77it/s]
I OpFusing 1 : 100%|██████████████████████████████████████████████| 100/100 [00:03<00:00, 26.55it/s]
I OpFusing 2 : 100%|██████████████████████████████████████████████| 100/100 [00:04<00:00, 24.07it/s]
I rknn building ...
No lowering found for: /model/encoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
No lowering found for: /model/decoder/embed_positions/CumSum, node type = CumSum, use CustomOperatorLower instead.
E RKNN: [19:06:45.033] dataconvert type -1 is unsupport in current!
E RKNN: [19:06:45.033] dataconvert type -1 is unsupport in current!
I rknn building done.
    
[맨위로](#contents)
    