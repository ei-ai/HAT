import os
import torch
import argparse
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq import tasks, options
from fairseq.data import Dictionary, indexed_dataset
from fairseq.models import fairseq_model

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def prepare_for_onnx_export(self):
        if hasattr(self.model, "prepare_for_onnx_export_"):
            self.model.prepare_for_onnx_export_()

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        return self.model(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=prev_output_tokens)

def load_dataset_as_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = [line.strip().split(' ', 1) for line in f]
        dataset = [(word, int(freq)) for word, freq in dataset]
    return dataset

def generate_dummy_data(src_dict_path, tgt_dict_path, batch_size, task):
    src_dataset = load_dataset_as_list(src_dict_path)
    tgt_dataset = load_dataset_as_list(tgt_dict_path)

    src_pad_idx = task.source_dictionary.pad()
    tgt_pad_idx = task.target_dictionary.pad()

    dummy_src_tokens = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([task.source_dictionary.index(word)]) for word, _ in src_dataset], 
        batch_first=True,
        padding_value=src_pad_idx
    )

    dummy_tgt_tokens = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor([task.target_dictionary.index(word)]) for word, _ in tgt_dataset],  
        batch_first=True,
        padding_value=tgt_pad_idx
    )

    dummy_src_lengths = torch.tensor([len(x) for x in dummy_src_tokens])

    return dummy_src_tokens, dummy_src_lengths, dummy_tgt_tokens


def export_to_onnx(model, src_tokens, src_lengths, dummy_tgt_tokens, dataset_name):
    wrapper_model = WrapperModel(model)
    wrapper_model.prepare_for_onnx_export()

    onnx_file_path = f"./onnx_models/{dataset_name}.onnx"
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    torch.onnx.export(
        wrapper_model,
        (src_tokens, src_lengths, dummy_tgt_tokens),
        onnx_file_path,
        opset_version=14,
        input_names=["src_tokens", "src_lengths", "tgt_tokens"],
        output_names=["output"],
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "sequence_length"},
            "tgt_tokens": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
    )
    print(f"| Saved  \n| ONNX model path: {onnx_file_path}")

def main():
    parser = options.get_converting_parser()
    args = options.parse_args_and_arch(parser)
    print(f"| Configs: {args}")

    print(f"| Loading model from {args.model_path}...")
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.eval()

    print("| Generating dummy data from preprocessed files...")
    dummy_src_tokens, dummy_src_lengths, dummy_tgt_tokens = generate_dummy_data(
        args.src_dict_path,
        args.tgt_dict_path, 
        args.required_batch_size_multiple, 
        task
    )

    print("| Exporting model to ONNX...")
    export_to_onnx(
        model, 
        dummy_src_tokens, 
        dummy_src_lengths, 
        dummy_tgt_tokens, 
        args.data[12:])
    
    print("| All set!")

if __name__ == "__main__":
    main()
