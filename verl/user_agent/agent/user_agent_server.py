from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

app = FastAPI()


class USER_AGENT_INPUT(BaseModel):
    context: str
    question: str


PROMPT_TEMPLATE = """
## Role
You are a **Math Question Analyzer**, a specialized AI assistant designed to extract and provide specific information from given math problems based on student queries.

## Capabilities
- Analyze the content of the provided math question with precision  
- Identify and extract requested information if relevant parts present in the question  

## Knowledge Base
- Mathematical terminology and problem structures  
- Information extraction techniques  
- Contextual understanding of student inquiries  

## Instructions
1. **Input Format**:  
   - Math question: "[math_question]"
   - Student's query: "[student_question]"

2. **Processing Rules**:  
   - Output **only the combined relevant parts about the requested information** (no explanations).  
   - Output "None" If the requested information is not mentioned in the math problem.

3. **Constraints**:  
   - Never infer or calculate missing information.  
   - Never add commentary, examples, or supplemental text.  
   - Prioritize brevity and accuracy.  

## Output Example
**Math question**: "James earns $20 an hour while working at his main job. He earns 20% less while working his second job. He works 30 hours at his main job and half that much at his second job. How much does he earn per week?"
**Student's query**: "What is James's hourly wage at his second job?"
**Reply**: "20% less than his main job"

## Your Turn
**Math question**: "[Context]"
**Student's query**: "[Question]"
**Reply**:
"""


tokenizer = None
model = None


def init_model(args):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

import re

import re

def extract_reply(decoded: str) -> str:
    reply_split = decoded.split("**Reply**:")
    if len(reply_split) < 2:
        return decoded.strip()  
    reply_part = reply_split[-1].strip()

    lines = reply_part.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip() not in ("assistant", "<think>", "</think>", "")]
    
    if cleaned_lines:
        return cleaned_lines[-1]  
    else:
        return "None"


def generate_reply(context: str, question: str, enable_thinking: bool = False) -> str:
    prompt = PROMPT_TEMPLATE.replace("[Context]", context).replace("[Question]", question)
    prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking  
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # print("Prompt:\n", prompt)
    # print("Input length:", inputs["input_ids"].shape[1])

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # return decoded
    # print(f'decoded: {decoded}')
    return extract_reply(decoded)

@app.post("/reply")
async def get_reply(data: USER_AGENT_INPUT):
    reply = generate_reply(data.context, data.question)
    #return {"reply": reply}
    return reply

@app.get("/health", response_class=PlainTextResponse)
async def health_check():
    return "iam healthy"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the service on")
    args = parser.parse_args()

    init_model(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)