# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# hf download Qwen/Qwen3-0.6B --local-dir /data/local/models/qwen3_06b
# rm /data/local/models/qwen3_06b/tokenizer*

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.sampling_params import RequestOutputKind

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=500,
    stop="is",
    n=2,
    output_kind=RequestOutputKind.CUMULATIVE,
    # stop_token_ids=[5],
)
tokens_prompt = TokensPrompt(prompt_token_ids=[2, 3, 4], )

if __name__ == "__main__":
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
        # skip_tokenizer_init=True,
        gpu_memory_utilization=0.8,
        disable_log_stats=False,
    )
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    for output in outputs:
        prompt = output.prompt_token_ids
        generated_token_ids = output.outputs[0].token_ids
        print(f"Prompt: {prompt!r}, Generated tokens: {generated_token_ids!r}")
