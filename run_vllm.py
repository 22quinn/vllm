#!/usr/bin/env python

from vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, top_p=1.0)
# sampling_params = SamplingParams(temperature=0, top_p=1.0, bad_words=["the University", "Philippines"])
sampling_params = SamplingParams(temperature=0, top_p=1.0, bad_words=["at the", "school"])
llm = LLM(model="facebook/opt-125m", enforce_eager=True)
outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
