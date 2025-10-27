# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
experimental support for data-parallel inference with torchrun
Note the data load balancing and distribution is done out of the vllm engine,
no internal lb supported in external_launcher mode.

To run this example:
```bash
$ torchrun --nproc-per-node=2 examples/offline_inference/torchrun_dp_example.py
```
"""

import os
import random

from vllm import SamplingParams
from vllm.distributed import (
    get_tensor_model_parallel_rank,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.v1.engine.llm_engine import LLMEngine

# Create prompts, the same across all ranks
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create sampling parameters, the same across all ranks
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1000000)

# Use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.
# it is important to set an explicit seed to make sure that
# all ranks have the same random seed, so that sampling can be
# deterministic across ranks.
engine_args = EngineArgs(
    model="microsoft/Phi-mini-MoE-instruct",
    tensor_parallel_size=2,
    data_parallel_size=4,
    pipeline_parallel_size=1,
    enable_expert_parallel=True,
    distributed_executor_backend="external_launcher",
    max_model_len=4096,
    gpu_memory_utilization=0.6,
    seed=1,
    enforce_eager=True,
    async_scheduling=True,
)
llm_engine = LLMEngine.from_engine_args(engine_args)

dp_rank = llm_engine.vllm_config.parallel_config.data_parallel_rank
dp_size = llm_engine.vllm_config.parallel_config.data_parallel_size
tp_rank = get_tensor_model_parallel_rank()
torch_rank = int(os.environ["RANK"])

prompts = prompts[dp_rank]
print(f"Rank: {torch_rank}, {prompts=}")
request_id: int = random.randint(0, 100000000)
llm_engine.add_request(str(request_id), prompts, sampling_params)

max_steps = 10
for step in range(max_steps):
    if not llm_engine.has_unfinished_requests():
        break
    outputs = llm_engine.step()
    # if not dp_rank == 0:
    #     continue
    active_bs = len(outputs)
    print(f"[step_{step}] Rank: {torch_rank}, {active_bs=}")
    for output in outputs:
        assert isinstance(output, RequestOutput)
        prompt = output.prompt
        for one_sample in output.outputs:
            # if (
            #     one_sample.finish_reason is not None
            #     or one_sample.stop_reason is not None
            # ):
            generated_text = one_sample.text
            print(
                f"[step_{step}] Rank: {torch_rank}, DP Rank: {dp_rank}, TP rank: {tp_rank}; Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n"
            )

"""
Further tips:

1. to communicate control messages across all ranks, use the cpu group,
a PyTorch ProcessGroup with GLOO backend.

```python
from vllm.distributed.parallel_state import get_world_group
cpu_group = get_world_group().cpu_group
torch_rank = dist.get_rank(group=cpu_group)
if torch_rank == 0:
    # do something for rank 0, e.g. saving the results to disk.
```

2. to communicate data across all ranks, use the model's device group,
a PyTorch ProcessGroup with NCCL backend.
```python
from vllm.distributed.parallel_state import get_world_group
device_group = get_world_group().device_group
```

3. to access the model directly in every rank, use the following code:
```python
llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
```
"""
