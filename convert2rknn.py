import os
import argparse
from rknn.api import RKNN
# https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V1.6.0_EN.pdf
# 25페이지부터 나오는 방식 사용(3.1.1~3.1.5)


def export_to_rknn(onnx_model_path, rknn_model_path):
    os.makedirs(os.path.dirname(rknn_model_path), exist_ok=True)
    rknn = RKNN(verbose=True, verbose_file=f'{rknn_model_path[0:-5]}.log')

    print('| --> Configuring RKNN model')
    rknn.config(target_platform='rk3588', optimization_level=1)
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
    parser.add_argument('--enc', action='store_true', help='Convert encoder only')
    parser.add_argument('--dec', action='store_true', help='Convert decoder only')
    args = parser.parse_args()

    onnx_model_path = f"./onnx_models/{args.onnx_name}"
    if args.onnx_name == 'wmt14_en_de':
        onnx_model_path = f"./onnx_models/wmt16_en_de"
    rknn_model_path = f"./rknn_models/{args.onnx_name}/{args.onnx_name}"

    if args.enc:
        print("| Encoder only")
        onnx_model_path = onnx_model_path + "_enc_sim.onnx"
        rknn_model_path = rknn_model_path + "_enc_sim.rknn"
    elif args.dec:
        print("| Decoder only")
        onnx_model_path = onnx_model_path + "_dec_sim.onnx"
        rknn_model_path = rknn_model_path + "_dec_sim.rknn"
    else:
        onnx_model_path = onnx_model_path + "_sim.onnx"
        rknn_model_path = rknn_model_path + "_sim.rknn"


    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}. Please ensure the ONNX model is available.")
        exit(1)

    print(f"| Using ONNX model from {onnx_model_path}...")

    print("| Exporting model to RKNN...")
    export_to_rknn(onnx_model_path, rknn_model_path)
    print("| all set!")


if __name__ == "__main__":
    main()
