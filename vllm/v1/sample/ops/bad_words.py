from typing import List, Optional

import torch

from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer

_SMALLEST_LOGIT = float("-inf")


def _get_bad_words_ids(bad_words: List[str], tokenizer: AnyTokenizer) -> List[int]:
    bad_words_ids: List[List[int]] = []

    for bad_word in bad_words:
        # To prohibit words both at the beginning
        # and in the middle of text
        # (related to add_prefix_space tokenizer parameter)
        for add_prefix_space in [False, True]:
            prefix = " " if add_prefix_space else ""
            prompt = prefix + bad_word.lstrip()

            if isinstance(tokenizer, MistralTokenizer):
                # Mistral tokenizers should not add special tokens
                prompt_token_ids = tokenizer.encode(text=prompt)
            else:
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

            # If no space at the beginning
            # or if prefix space produces a new word token
            if (not add_prefix_space) or (
                add_prefix_space
                and prompt_token_ids[0] != bad_words_ids[-1][0]
                and len(prompt_token_ids) == len(bad_words_ids[-1])
            ):
                bad_words_ids.append(prompt_token_ids)

    return bad_words_ids


def _check_token_ids_bounds(bad_words_ids: List[int], vocab_size: int) -> None:
    invalid_token_ids = []

    for bad_word_ids in bad_words_ids:
        for token_id in bad_word_ids:
            if token_id < 0 or token_id >= vocab_size:
                invalid_token_ids.append(token_id)

    if len(invalid_token_ids) > 0:
        raise ValueError(
            f"The model vocabulary size is {vocab_size},"
            f" but the following tokens"
            f" were specified as bad: {invalid_token_ids}."
            f" All token id values should be integers satisfying:"
            f" 0 <= token_id < {vocab_size}."
        )


def _init_word_bias(
    bad_words_ids: List[int], logits: torch.FloatTensor
) -> torch.FloatTensor:
    vocab_size = logits.shape[-1]
    _check_token_ids_bounds(bad_words_ids=bad_words_ids, vocab_size=vocab_size)
    word_bias = torch.zeros((vocab_size,), dtype=torch.float, device=logits.device)
    for bad_word_ids in bad_words_ids:
        if len(bad_word_ids) == 1:
            bad_word_id = bad_word_ids[0]
            word_bias[bad_word_id] = _SMALLEST_LOGIT
    return word_bias


def apply_bad_words(
    logits: torch.Tensor,
    bad_words: Optional[List[str]],
    tokenizer: AnyTokenizer,
    past_tokens_ids,
) -> torch.Tensor:
    if bad_words is None or len(bad_words) == 0:
        return logits
    bad_words_ids = _get_bad_words_ids(bad_words=bad_words, tokenizer=tokenizer)
    word_bias = _init_word_bias(bad_words_ids, logits)
    last_token_bias = torch.zeros_like(logits)

    for bad_word_ids in bad_words_ids:
        if len(bad_word_ids) == 1:  # 1-token words already processed
            continue

        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        actual_prefix = past_tokens_ids[-prefix_length:]
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if tuple(actual_prefix) == tuple(expected_prefix):
            last_token_bias[last_token_id] = _SMALLEST_LOGIT

    logits = logits + word_bias + last_token_bias

    return logits
