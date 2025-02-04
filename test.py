from rknn.api import RKNN


target = 'RK3588'
rknn_model = model_name = 'iwslt14_de_en'


print('| Measuring model latency on NPU...')
enc = RKNN()
rknn_path = f'rknn_models/{model_name}/{model_name}.rknn'
print(f'| --> Load RKNN model')
ret = enc.load_rknn(rknn_path)
if ret != 0:
    raise RuntimeError(f"Failed to load RKNN model.")

print(f'| --> Init {target} runtime environment')
ret = enc.init_runtime(target, perf_debug=True)
if ret != 0:
    print(f'| Init runtime environment failed')
    exit(ret)
print(f'| Init done')

# encoder_latencies = []
print('| Measuring encoder...')
latency = enc.eval_perf(is_print=True)
print('/n/n| Encoder one run on NPU: /n')
print(latency)

enc.release()
