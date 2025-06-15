import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from typing import Dict
from tqdm import tqdm
import json
import wandb

@dataclass
class ScriptConfig(SFTConfig):
    data_fpath: str = field(default=None)
    model_fpath: str = field(default=None)
    response_template: str = field(default=None)
    instruction_template: str = field(default=None)


def format_dataset(examples, tokenizer=None):
    if isinstance(examples["messages"][0], list):
        output_texts = []
        for i in range(len(examples["messages"])):
            output_texts.append(tokenizer.apply_chat_template(examples["messages"][i], tokenize=False))
        return {"text": output_texts}
    else:
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}


def create_data_collator(args, tokenizer) -> DataCollatorForCompletionOnlyLM:
    response_tokens = tokenizer.tokenize(
        args.response_template.replace("\\n", "\n"), 
    )
    instruction_tokens = tokenizer.tokenize(
        args.instruction_template.replace("\\n", "\n"), 
    )
    response_ids = tokenizer.convert_tokens_to_ids(response_tokens)
    instruction_ids = tokenizer.convert_tokens_to_ids(instruction_tokens)
    # print("Response Tokens:", response_tokens, "Response Token IDs:", response_ids)
    # print("instruction Tokens:", instruction_tokens, "instruction Token IDs:", instruction_ids)
    return DataCollatorForCompletionOnlyLM(response_template=args.response_template,instruction_template=args.instruction_template,tokenizer=tokenizer)


def main():

    parser = HfArgumentParser(ScriptConfig)
    args = parser.parse_args_into_dataclasses()[0]
    
    config = AutoConfig.from_pretrained(args.model_fpath)

    train_dataset = load_dataset("json", data_files=args.data_fpath, split="train")
    

    if args.report_to == "wandb":
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "sft-project"),
            name=os.getenv("WANDB_NAME", "sft-run"),
            dir=os.getenv("WANDB_DIR", "./wandb_logs"),
            mode=os.getenv("WANDB_MODE", "offline"),
        )


    # print(f"Dataset features: {dataset.features}")
    # print(f"First sample: {dataset[0]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_fpath, config=config, padding_side="left")
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set content = message.content %}\n        {%- set reasoning_content = '' %}\n        {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

    formatted_dataset = train_dataset.map(format_dataset,fn_kwargs={"tokenizer": tokenizer})

    # TEST: Tokenize one sample
    # print(f"sample 1:{formatted_dataset['text'][0]}")

    # tokenized_sample = tokenizer(formatted_dataset['text'][0], padding=True, truncation=True, return_special_tokens_mask=True)

    data_collator = create_data_collator(args,tokenizer)
    # print(data_collator.torch_call([tokenized_sample]))


    trainer = SFTTrainer(
        model=args.model_fpath,
        args=args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # train_dataset = trainer.train_dataset
    # for i, example in enumerate(train_dataset):
    #     print(f"Example {i}: {example}")
    #     if i == 1:
    #         break

    print("Start train...")
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()