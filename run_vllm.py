#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ad_inst_lora_path = snapshot_download(
    repo_id="isbondarev/advertisment_instraction_mistral_lora")
llm = LLM(model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
          enable_lora=True,
          quantization="fp8")

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
)

prompts = [
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
    "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
]

outputs = llm.generate(prompts,
                       sampling_params,
                       lora_request=LoRARequest("sql_adapter", 1,
                                                ad_inst_lora_path))

exit(0)

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, top_p=1.0)
# sampling_params = SamplingParams(temperature=0, top_p=1.0, bad_words=["the University", "Philippines"])
sampling_params = SamplingParams(temperature=0,
                                 top_p=1.0,
                                 bad_words=["at the", "school"])
# sampling_params = SamplingParams(temperature=0, top_p=1.0, bad_words=["at the"])
# llm = LLM(model="facebook/opt-125m", enforce_eager=True)
# outputs = llm.generate(prompts, sampling_params)

# for i, output in enumerate(outputs):
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}")
#     print(f"Generated text: {generated_text!r}")

MODEL = "22quinn/Llama-3.2-1B-1Label-dummy"
PROMPTS = ["Hello my name is Robert", "ok I got it"]
model = LLM(MODEL,
            task="classify",
            enforce_eager=True,
            enable_prefix_caching=True)
outputs = model.classify(PROMPTS)
for output in outputs:
    print(output.outputs.probs)
