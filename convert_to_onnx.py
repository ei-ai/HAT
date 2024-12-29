import os
import torch
import argparse
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq import tasks

def export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name):
    dummy_src = torch.randint(0, src_vocab_size, (1, 10))  
    dummy_tgt = torch.randint(0, tgt_vocab_size, (1, 10))  

    onnx_file_path = f"./onnx_models/{dataset_name}.onnx"
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_src, dummy_tgt),
        onnx_file_path,
        opset_version=11,
        input_names=["src_tokens", "tgt_tokens"],
        output_names=["output"],
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "sequence_length"},
            "tgt_tokens": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}
        }
    )
    print(f"ONNX model exported to {onnx_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert HAT model to ONNX format")
    parser.add_argument("--dataset-name", required=True, help="--dataset-name=[iwslt14deen|wmt14ende|wmt14enfr|wmt19ende]")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if dataset_name=="iwslt14deen":
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space1.pt"
    else:
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space0.pt"

    print(f"Loading model from {model_path}...")
    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    # 수정중 
    task = tasks.setup_task(args)
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(f"| Model: {args.arch} \n| Criterion: {criterion.__class__.__name__}")
    model.eval()

    # 필요시 수정 
    src_vocab_size = 32000  
    tgt_vocab_size = 32000  

    print("Exporting model to ONNX...")
    export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name)
    print("all set")
    

if __name__ == "__main__":
    main()