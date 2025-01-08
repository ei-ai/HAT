import os
import argparse
from rknn.api import RKNN


def export_to_rknn(onnx_model_path, rknn_model_path, dataset_name):
    os.makedirs(os.path.dirname(rknn_model_path), exist_ok=True)
    rknn = RKNN(verbose=True, verbose_file=f'./rknn_models/{dataset_name}/rknn_build_{dataset_name}.log')

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, help="Dataset name: [iwslt14deen|wmt14ende|wmt14enfr|wmt19ende]")
    args = parser.parse_args()

    onnx_model_path = f"./onnx_models/{args.dataset_name}.onnx"
    if args.dataset_name == 'wmt14_en_de':
        onnx_model_path = f"./onnx_models/wmt16_en_de.onnx"
    rknn_model_path = f"./rknn_models/{args.dataset_name}/{args.dataset_name}.rknn"

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}. Please ensure the ONNX model is available.")
        exit(1)

    print(f"| Using ONNX model from {onnx_model_path}...")

    print("| Exporting model to RKNN...")
    export_to_rknn(onnx_model_path, rknn_model_path, args.dataset_name)
    print("| all set!")


if __name__ == "__main__":
    main()
