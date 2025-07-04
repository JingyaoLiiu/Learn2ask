# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import requests
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][
        0
    ]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(
    value: Union[torch.Tensor, np.ndarray], repeats: int
) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class L2AvLLMRollout(BaseRollout):
    def __init__(
        self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (
            not config.enforce_eager and config.free_cache_engine
        ), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size,
                    num_tp_per_train_tp=num_tp_per_train_tp,
                )
            else:
                vllm_ps.initialize_model_parallel(
                    tensor_model_parallel_size=tensor_parallel_size
                )

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.llm_config.max_position_embeddings
                )
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = (
                    model_hf_config.text_config.max_position_embeddings
                )
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert (
                max_position_embeddings >= config.prompt_length + config.response_length
            ), "model context length should be greater than total sequence length"

        max_model_len = int(
            config.max_model_len or config.prompt_length + config.response_length
        )

        if (
            max_num_batched_tokens < max_model_len
            and self.config.enable_chunked_prefill
        ):
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = (
            "dummy" if config.load_format.startswith("dummy") else config.load_format
        )

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {
            key: val for key, val in engine_kwargs.items() if val is not None
        }
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # print(prompts)

        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)

        # 获取tokenizer类型
        tokenizer_class_str = str(self.tokenizer.__class__).lower()

        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="/simon-stub-path",
                    )
                ] * batch_size

        if self.sampling_params.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.sampling_params.n)
            attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
            position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
            batch_size = batch_size * self.sampling_params.n
            new_vllm_inputs = []

            # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
            if "tools_kwargs" in non_tensor_batch.keys():
                non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                    non_tensor_batch["tools_kwargs"], self.sampling_params.n
                )

            if "origin_ques" in non_tensor_batch.keys():
                non_tensor_batch["origin_ques"] = _repeat_interleave(
                    non_tensor_batch["origin_ques"], self.sampling_params.n
                )

            for input_data in vllm_inputs:
                new_vllm_inputs += [input_data] * self.sampling_params.n
            vllm_inputs = new_vllm_inputs
            kwargs["n"] = 1

        if "qwen" in tokenizer_class_str :
            self.tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

        user_prompt = "Answer to your question: [Answer]\n\nWith these information, answer the previous question."

        import re

        def extract_model_input(prompt,tokenizer_class_str):
            if "qwen" in tokenizer_class_str:
                return re.search(
                    r"<\|im_start\|>user\n(.*)?<\|im_end\|>\n<\|im_start\|>assistant\n",
                    prompt,
                    re.DOTALL,
                ).group(1)
            else:
                return re.search(
                    r"<\|start_header_id\|>user\n(.*)?<\|end_header_id\|>\n(.*?)<\|eot_id\|>",
                        prompt,
                        re.DOTALL,
                ).group(1)

        def call_user_agent(response_text, context):
            PROMPT_TEMPLATE = '## Role\nYou are a **Math Question Analyzer**, a specialized AI assistant designed to extract and provide specific information from given math problems based on student queries.\n\n## Capabilities\n- Analyze the content of the provided math question with precision  \n- Identify and extract requested information if relevant parts present in the question  \n\n## Knowledge Base\n- Mathematical terminology and problem structures  \n- Information extraction techniques  \n- Contextual understanding of student inquiries  \n\n## Instructions\n1. **Input Format**:  \n- Math question: "[math_question]"\n- Student\'s query: "[student_question]"\n\n2. **Processing Rules**:  \n- Output **only the combined relevant parts about the requested information** (no explanations).  \n- Output "None" If the requested information is not mentioned in the math problem.\n\n3. **Constraints**:  \n- Never infer or calculate missing information.  \n- Never add commentary, examples, or supplemental text.  \n- Prioritize brevity and accuracy.  \n\n## Output Example\n**Math question**: "James earns $20 an hour while working at his main job. He earns 20% less while working his second job. He works 30 hours at his main job and half that much at his second job. How much does he earn per week?"\n**Student\'s query**: "What is James\'s hourly wage at his second job?"\n**Reply**: "20% less than his main job"\n\n## Your Turn\n**Math question**: "[Context]"\n**Student\'s query**: "[Question]"\n**Reply**:'

            extracted = extract_text(extract_boxed(response_text))
            prompt = PROMPT_TEMPLATE.replace("[Context]", context).replace(
                "[Question]", extracted
            )
            if extracted is None:
                return user_prompt.replace("[Answer]", "None")
            try:
                payload = {
                    "model": "Qwen3-14B",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                res = requests.post(
                    os.environ["USER_AGENT"] + "/v1/chat/completions",
                    json=payload,
                    timeout=60,
                )
                if res.status_code == 200:
                    result = res.json()
                    content = result["choices"][0]["message"]["content"]
                    return user_prompt.replace("[Answer]", content.strip())
                else:
                    print(f"[UserAgent Error] Status {res.status_code}: {res.text}")
                    return user_prompt.replace("[Answer]", "None")
            except Exception as e:
                print(f"[UserAgent Exception] {e}")
                return user_prompt.replace("[Answer]", "None")

        def add_qwen3_no_think_prefix(text):
            if not text.startswith("<think>"):
                return "<think>\n\n</think>\n\n" + text
            else:
                return text

        def update_messages(messages, next_turn):
            messages.append(
                {
                    "role": "assistant" if messages[-1]["role"] == "user" else "user",
                    "content": next_turn,
                }
            )

        def fix_messages(messages):
            qwen3_messages = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "assistant":
                    content = add_qwen3_no_think_prefix(content)
                qwen3_messages.append({"role": role, "content": content})
            return qwen3_messages

        def return_response_and_mask(tokenizer, messages):
            return_response, return_label_mask = [], []
            session = []
            session.append(messages[0])
            last_token_ids = tokenizer.apply_chat_template(
                fix_messages(session),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for message in messages[1:]:
                session.append(message)
                token_ids = tokenizer.apply_chat_template(
                    fix_messages(session),
                    tokenize=True,
                    add_generation_prompt=message["role"] == "user",
                    enable_thinking=False,
                )

                new_tokens = token_ids[len(last_token_ids) :]
                return_response += new_tokens

                # print(f"[DEBUG] new tokens: {new_tokens}")
                # print(f"[DEBUG] token count: {len(new_tokens)}")
                # print(f"[DEBUG] tokens as text: {tokenizer.decode(new_tokens)}")

                if message["role"] == "assistant":
                    return_label_mask += [1] * (
                        len(token_ids) - len(last_token_ids) - 1
                    ) + [0]
                else:
                    return_label_mask += [0] * (len(token_ids) - len(last_token_ids))
                last_token_ids = token_ids
            return (
                return_response[:-1],
                return_label_mask[:-1],
            )  # final token is always \n

        def extract_text(content):
            if content is None:
                return content
            i = 0
            origin = content
            while "\\text{" in content:
                i += 1
                start_index = content.rfind("\\text{")
                start_index += len("\\text{")
                if start_index == -1:
                    return None  # No opening brace found

                brace_count = 1
                end_index = start_index

                while end_index < len(content) and brace_count > 0:
                    end_index += 1
                    if content[end_index] == "{":
                        brace_count += 1
                    elif content[end_index] == "}":
                        brace_count -= 1
                if brace_count != 0:
                    return None  # Unbalanced braces

                content = (
                    content[: start_index - len("\\text{")]
                    + content[start_index:end_index]
                    + content[end_index + 1 :]
                )
            content = re.sub(r"\s+", " ", content)
            return content

        def extract_boxed(content):
            """提取 \\boxed{} 中的内容"""
            start_index = content.rfind("\\boxed{")
            if start_index == -1:
                return None

            start_index += len("\\boxed{")
            brace_count = 1
            end_index = start_index

            while end_index < len(content) and brace_count > 0:
                if content[end_index] == "{":
                    brace_count += 1
                elif content[end_index] == "}":
                    brace_count -= 1
                end_index += 1

            if brace_count != 0:
                return None

            return content[start_index : end_index - 1].strip()

        messages_list = [
            [
                {
                    "role": "user",
                    "content": extract_model_input(
                        self.tokenizer.decode(vllm_input["prompt_token_ids"]),
                        tokenizer_class_str
                    ),
                }
            ]
            for vllm_input in vllm_inputs
        ]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # print(f"Sampling n set to {self.sampling_params.n}")

            self.sampling_params.detokenize = True

            # print("Fitst Assistant Generation...")
            print("Fitst Assistant Generation...", flush=True)

            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            responses = [output.outputs[0].text for output in outputs]
            [
                update_messages(messages, next_turn)
                for messages, next_turn in zip(messages_list, responses)
            ]

            tmp_message_list = []
            tmp_origin_ques_list = []

            # for messages in messages_list:
            for messages, origin_ques in zip(
                messages_list, non_tensor_batch["origin_ques"]
            ):
                final_turn_content = messages[-1]["content"]
                boxed = extract_text(extract_boxed(final_turn_content))
                if boxed and (
                    "?" in boxed[-3:] and len(boxed.split()) > 3
                ):  # is a question
                    tmp_message_list.append(messages)
                    tmp_origin_ques_list.append(origin_ques)

            if tmp_message_list:
                # TODO: multi-thread implement
                # user_replies = [call_user_agent(messages[0]["content"], response,non_tensor_batch["origin_ques"]) for messages, response in zip(tmp_message_list, responses)]

                print("Second User Generation(Multi Thread)...")
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [
                        executor.submit(call_user_agent, messages[-1]["content"], origin_ques)
                        for messages, origin_ques in zip(tmp_message_list, tmp_origin_ques_list)
                    ]
                    user_replies = [future.result() for future in futures]
       
                # user_replies = [
                #     call_user_agent(messages[-1]["content"], origin_ques)
                #     for messages, origin_ques in zip(
                #         tmp_message_list, tmp_origin_ques_list
                #     )
                # ]
                [
                    update_messages(messages, next_turn)
                    for messages, next_turn in zip(tmp_message_list, user_replies)
                ]

            tmp_message_list = []
            for messages in messages_list:
                if messages[-1]["role"] == "user":
                    final_turn_content = messages[-1]["content"]
                    if "None\n\n" not in final_turn_content:  # has valid reply
                        tmp_message_list.append(messages)

            if tmp_message_list:
                vllm_inputs = [
                    {
                        "prompt_token_ids": self.tokenizer.apply_chat_template(
                            fix_messages(messages),
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    }
                    for messages in tmp_message_list
                ]

                print("Third Assistant Generation...")
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

                responses = [output.outputs[0].text for output in outputs]
                [
                    update_messages(messages, next_turn)
                    for messages, next_turn in zip(tmp_message_list, responses)
                ]

            print("Generation 1 2 3 OK")
            response, label_mask = [], []
            for messages in messages_list:
                return_response, return_label_mask = return_response_and_mask(
                    self.tokenizer, messages
                )
                response.append(return_response)
                label_mask.append(return_label_mask)

            # # TODO(sgm): disable logprob when recompute_log_prob is enable
            # # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            # response = []
            # rollout_log_probs = []
            # for output in outputs:
            #     for sample_id in range(len(output.outputs)):
            #         response_ids = output.outputs[sample_id].token_ids
            #         response.append(response_ids)
            #         curr_log_prob = []
            #         for i, logprob in enumerate(output.outputs[sample_id].logprobs):
            #             curr_log_prob.append(logprob[response_ids[i]].logprob)
            #         rollout_log_probs.append(curr_log_prob)

            # rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
            # rollout_log_probs = rollout_log_probs.to(torch.float32)

            # 3 times of response length is safer, but it consumes more memory. If unfortunate, truncate it.
            response = pad_2d_list_to_length(
                response, self.pad_token_id, max_length=self.config.response_length * 2
            ).to(idx.device)
            label_mask = pad_2d_list_to_length(
                label_mask, 0, max_length=self.config.response_length * 2
            ).to(idx.device)
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, 3, -1
            )

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=self.pad_token_id,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        loss_mask = torch.cat((torch.zeros_like(idx), label_mask), dim=-1)

        # # all the tp ranks should contain the same data here. data in all ranks are valid
        # batch = TensorDict(
        #     {
        #         "prompts": idx,
        #         "responses": response,
        #         "input_ids": seq,  # here input_ids become the whole sentences
        #         'rollout_log_probs': rollout_log_probs, # we will recompute old log prob with actor
        #         "attention_mask": attention_mask,
        #         "position_ids": position_ids,
        #     },
        #     batch_size=batch_size,
        # )

        # # check alignment
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     result_list = []
        #     for token_id, token, mask in zip(seq[-1], self.tokenizer.convert_ids_to_tokens(seq[-1]), loss_mask[-1]):
        #         if token != self.tokenizer.pad_token:
        #             result_list.append((token_id.item(), token, mask.item()))
        #     print(result_list)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "loss_mask": loss_mask,  # mask user agent parts
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
