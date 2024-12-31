import os
import torch
import argparse
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq import tasks, options

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
    print(f"| ONNX model exported to {onnx_file_path}")

def main():
    parser = options.get_converting_parser()
    parser.add_argument("--dataset-name", required=True, help="Dataset Name: [iwslt14deen|wmt14ende|wmt14enfr|wmt19ende]")
    # known_args, _ = parser.parse_known_args()
    # args = options.parse_args_and_arch(parser, input_args=known_args)
    args = options.parse_args_and_arch(parser)

    dataset_name = args.dataset_name
    if dataset_name=="iwslt14deen":
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space1.pt"
    else:
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space0.pt"

    print(f"| Loading model from {model_path}...")
    
    if dataset_name == "iwslt14deen":
        dataset_name = "iwslt14_de_en"  
        args.arch = "transformer_iwslt_de_en"    
    elif dataset_name == "wmt14enfr":
        dataset_name = "wmt14_en_fr"
        args.arch = "transformersuper_wmt_en_fr"
    elif dataset_name == "wmt14ende":
        dataset_name = "wmt16_en_de"
        args.arch = "transformer_wmt_en_de"
    else:
        dataset_name = "wmt19_en_de"
        args.arch = "transformer_wmt_en_de"

    args.data = f"./data/binary/{dataset_name}" 

    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.eval()

    src_vocab_size = len(task.source_dictionary)  
    tgt_vocab_size = len(task.target_dictionary) 

    print("| Exporting model to ONNX...")
    export_to_onnx(model, src_vocab_size, tgt_vocab_size, dataset_name)
    print("all set")
    

if __name__ == "__main__":
    main()