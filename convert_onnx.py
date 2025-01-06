import os
import torch
import argparse
from fairseq.models.transformer_super import TransformerSuperModel
from fairseq import tasks, options, utils
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

def generate_dummy_data(args):
    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 23, 'wmt': 30}
    if 'iwslt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['iwslt']
    elif 'wmt' in args.arch:
        dummy_sentence_length = dummy_sentence_length_dict['wmt']
    else:
        raise NotImplementedError

    dummy_src_tokens = [2] + [7] * (dummy_sentence_length - 1)
    dummy_prev = [7] * (dummy_sentence_length - 1) + [2]

    src_tokens_test = torch.tensor([dummy_src_tokens], dtype=torch.long)
    src_lengths_test = torch.tensor([dummy_sentence_length])
    prev_output_tokens_test_with_beam = torch.tensor([dummy_prev] * args.beam, dtype=torch.long)

    return src_tokens_test, src_lengths_test, prev_output_tokens_test_with_beam


def export_to_onnx(model, src_tokens, src_lengths, prev_output_tokens, dataset_name):
    wrapper_model = WrapperModel(model)
    wrapper_model.prepare_for_onnx_export()

    onnx_file_path = f"./onnx_models/{dataset_name}.onnx"
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    torch.onnx.export(
        model,
        (src_tokens, src_lengths, prev_output_tokens), 
        onnx_file_path,
        opset_version=14,
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["src_tokens", "src_lengths", "prev_output_tokens"],
        output_names=["output"],
        dynamic_axes={
            "src_tokens": {0: "batch_size", 1: "sequence_length"},
            "prev_output_tokens": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"},
        }
    )
    print(f"| Saved \n| ONNX model path: {onnx_file_path}")


def main():
    parser = options.get_converting_parser()
    args = options.parse_args_and_arch(parser)
    print(f"| Configs: {args}")

    print(f"| Buildng model {args.arch}...")
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.eval()

    src_tokens, src_lengths, prev_output_tokens = generate_dummy_data(args)

    config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)
    model.set_sample_config(config_sam)

    print("| Exporting model to ONNX...")
    export_to_onnx(
        model, 
        src_tokens, 
        src_lengths, 
        prev_output_tokens,
        args.data.strip("data/binary/"))
    
    print("| All set!")

if __name__ == "__main__":
    main()
