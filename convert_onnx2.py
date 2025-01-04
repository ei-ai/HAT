import os
import torch
import argparse
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq import tasks, options
from fairseq.data import Dictionary, indexed_dataset
from fairseq.models import fairseq_model
# 에러 발생지점 
# line 92, 93부분(언어 불러오는 부분)
# generate_dummy_data() 데이터셋 불러오는 부분(로드는 성공, 인덱싱에서 오류 생김)

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def prepare_for_onnx_export(self):
        if hasattr(self.model, "prepare_for_onnx_export_"):
            self.model.prepare_for_onnx_export_()

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        return self.model(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=prev_output_tokens)

def generate_dummy_data(task, split, batch_size):
    task.load_dataset(split=split)
    dataset = task.dataset(split)

    src_dataset = dataset.src
    tgt_dataset = dataset.tgt

    dataset_size = len(src_dataset)
    print(f"| Dataset size: {dataset_size}")
    try:
        dummy_src_tokens = [src_dataset[i] for i in range(batch_size)]
        dummy_prev_output_tokens = [tgt_dataset[i] for i in range(batch_size)]
    except KeyError as e:
        print(f"KeyError while accessing dataset: {e}")
        raise
    if batch_size > dataset_size:
        print(f"| Warning: batch_size ({batch_size}) > dataset_size ({dataset_size}). Adjusting batch_size.")
        batch_size = dataset_size


    dummy_src_tokens = [src_dataset[i] for i in range(batch_size)]
    dummy_prev_output_tokens = [tgt_dataset[i] for i in range(batch_size)]

    src_pad_idx = task.source_dictionary.pad()
    tgt_pad_idx = task.target_dictionary.pad()

    dummy_src_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in dummy_src_tokens], batch_first=True, padding_value=src_pad_idx)
    dummy_prev_output_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in dummy_prev_output_tokens], batch_first=True, padding_value=tgt_pad_idx)

    dummy_src_lengths = torch.tensor([len(x) for x in dummy_src_tokens])

    return dummy_src_tokens, dummy_src_lengths, dummy_prev_output_tokens


def export_to_onnx(model, src_tokens, src_lengths, prev_output_tokens, dataset_name):
    wrapper_model = WrapperModel(model)

    wrapper_model.prepare_for_onnx_export()

    onnx_file_path = f"./onnx_models/{dataset_name}.onnx"
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    torch.onnx.export(
        wrapper_model,
        (src_tokens, src_lengths, prev_output_tokens),
        onnx_file_path,
        opset_version=14,
        input_names=["src_tokens", "src_lengths", "prev_output_tokens"],
        output_names=["output"],
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "sequence_length"},
            "prev_output_tokens": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
    )
    print(f"| Saved  \n| ONNX model path: {onnx_file_path}")

def main():
    parser = options.get_converting_parser()
    parser.add_argument("--dataset-name", required=True, help="Dataset Name: [iwslt14deen|wmt14ende|wmt14enfr|wmt19ende]")
    args = options.parse_args_and_arch(parser)

    dataset_name = args.dataset_name
    src_lang = args.source_lang
    tgt_lang = args.target_lang
    batch_size = args.required_batch_size_multiple
    print(f"| src_lang: {src_lang}")
    print(f"| tgt_lang: {tgt_lang}")
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
    # src_lang = "en"
    # tgt_lang = "de"
    args.data = f"./data/binary/{dataset_name}"
    src_data_path = f"./data/binary/{dataset_name}/valid.{src_lang}-{tgt_lang}.{src_lang}.bin"
    tgt_data_path = f"./data/binary/{dataset_name}/valid.{src_lang}-{tgt_lang}.{tgt_lang}.bin"
    src_dict_path = f"./data/binary/{dataset_name}/dict.{src_lang}.txt"
    tgt_dict_path = f"./data/binary/{dataset_name}/dict.{tgt_lang}.txt"
    print(f"| Configs: {args}")

    if dataset_name == "iwslt14deen":
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space1.pt"
    else:
        model_path = f"./downloaded_models/HAT_{dataset_name}_super_space0.pt"
        
    # ---------------------여기까진 cli_main()으로 설정하는 게 코드상 깔끔할 것 같다. 나중에 수정

    print(f"| Loading model from {model_path}...")
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.eval()

    print("| Generating dummy data from preprocessed files...")
    split = "valid"
    dummy_src_tokens, dummy_src_lengths, dummy_prev_output_tokens = generate_dummy_data(
        task, split, batch_size
    )

    print("| Exporting model to ONNX...")
    export_to_onnx(model, dummy_src_tokens, dummy_src_lengths, dummy_prev_output_tokens, dataset_name)
    print("| All set!")

if __name__ == "__main__":
    main()
