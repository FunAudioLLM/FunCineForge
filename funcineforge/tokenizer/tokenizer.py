from funcineforge.register import tables


@tables.register("tokenizer_classes", "FunCineForgeTokenizer")
def FunCineForgeTokenizer(init_param_path, **kwargs):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(init_param_path)
    special_tokens = {
        'eos_token': '<|endoftext|>',
        'pad_token': '<|endoftext|>',
        'additional_special_tokens': [
            '<|im_start|>', '<|im_end|>',
            '<|startofclue|>', '<|endofclue|>', '<|endofprompt|>',
            '[breath]', '<strong>', '</strong>', '[noise]',
            '[laughter]', '[cough]', '[clucking]', '[accent]',
            '[quick_breath]',
            "<laughter>", "</laughter>",
            "[hissing]", "[sigh]", "[vocalized-noise]",
            "[lipsmack]", "[mn]", "<|endofsystem|>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer