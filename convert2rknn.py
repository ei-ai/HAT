import os
import argparse
from rknn.api import RKNN
# https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V1.6.0_EN.pdf
# 25페이지부터 나오는 방식 사용(3.1.1~3.1.5), 3.1.6도 시도해 봤는데 쉽지 않아서 그냥 이 방법 사용 


def export_to_rknn(onnx_model_path, rknn_model_path, onnx_name):
    os.makedirs(os.path.dirname(rknn_model_path), exist_ok=True)
    rknn = RKNN(verbose=False, verbose_file=f'./rknn_models/{onnx_name}/rknn_build_{onnx_name}.log')

    print('| --> Configuring RKNN model')
    rknn.config(target_platform='rk3588')
    print('| done')

    print('| --> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print('| Load model failed!')
        exit(ret)
    print('| done')

    print('| --> Building RKNN model')
    ret = rknn.build(do_quantization=False, rknn_batch_size=1)
    if ret != 0:
        print('| Build model failed!')
        exit(ret)
    print('| done')

    print('| --> Exporting RKNN model')
    ret = rknn.export_rknn(export_path=rknn_model_path)
    if ret != 0:
        print('| Export RKNN model failed!')
        exit(ret)
    print('| done')

    print(f"| RKNN model has been successfully exported to {rknn_model_path}")

    rknn.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-name", required=True, help="ONNX model name")
    args = parser.parse_args()

    onnx_model_path = f"./onnx_models/{args.onnx_name}.onnx"
    if args.onnx_name == 'wmt14_en_de':
        onnx_model_path = f"./onnx_models/wmt16_en_de.onnx"
    rknn_model_path = f"./rknn_models/{args.onnx_name}/{args.onnx_name}.rknn"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}. Please ensure the ONNX model is available.")
        exit(1)

    print(f"| Using ONNX model from {onnx_model_path}...")

    print("| Exporting model to RKNN...")
    export_to_rknn(onnx_model_path, rknn_model_path, args.onnx_name)
    print("| all set!")


if __name__ == "__main__":
    main()
