import os
import torch
from fairseq import tasks, options, utils
from wrapper_models import wrapper_model_onnx

def generate_dummy_data(args):
    # specify the length of the dummy input for profile
    # for iwslt, the average length is 23, for wmt, that is 30
    dummy_sentence_length_dict = {'iwslt': 25, 'wmt': 30}
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


def export_to_onnx(model, args):
    wrapper_model = wrapper_model_onnx.WrapperModelONNX(model)
    wrapper_model.prepare_for_onnx_export()

    src_tokens, src_lengths, prev_output_tokens = generate_dummy_data(args)

    onnx_file_path = f"./onnx_models/{args.data.removeprefix('data/binary/')}.onnx"
    os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

    torch.onnx.export(
        wrapper_model,
        (src_tokens, src_lengths, prev_output_tokens), 
        onnx_file_path,
        opset_version=14,
        export_params=True,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=["src_tokens", "src_lengths", "prev_output_tokens"],
        output_names=["output"],
    )
    print(f"| Saved \n| ONNX model path: {onnx_file_path}")


def main():
    parser = options.get_converting_parser()
    args = options.parse_args_and_arch(parser)
    print(f"| Configs: {args}")

    print(f"| Buildng model {args.arch}...")
    task = tasks.setup_task(args)
    model = task.build_model(args)
    with torch.no_grad():
        config_sam = utils.sample_configs(utils.get_all_choices(args), reset_rand_seed=False, super_decoder_num_layer=args.decoder_layers)
        model.set_sample_config(config_sam)
    model.eval()

    print("| Exporting model to ONNX...")
    export_to_onnx(model, args)
    
    print("| All set!")


if __name__ == "__main__":
    main()