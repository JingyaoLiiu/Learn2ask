import sys
import jsonlines
from transformers import AutoTokenizer
import copy

MODEL = sys.argv[1]
DATA_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
ENABLE_THINKING = False

agent_prompt = """
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
**Reply**: ?
""".strip()

def extract_boxed(content):
    """提取 \\boxed{} 中的内容"""
    start_index = content.rfind('\\boxed{')
    if start_index == -1:
        return None
    
    start_index += len('\\boxed{')
    brace_count = 1
    end_index = start_index
    
    while end_index < len(content) and brace_count > 0:
        if content[end_index] == '{':
            brace_count += 1
        elif content[end_index] == '}':
            brace_count -= 1
        end_index += 1
    
    if brace_count != 0:
        return None
    
    return content[start_index:end_index-1].strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

output = []
for instance in dataset:
   # instance.pop("query")
   # for generated_text in instance.pop("generated_text"):
   #      hyp = extract_boxed(generated_text)
   #      if hyp:
   #          if (len(hyp.split()) > 4 or "?" in hyp[-5:]):
   #             copied = copy.deepcopy(instance)
   #             query = agent_prompt.replace("[Context]", copied["origin"].replace("[", "").replace("]", "")).replace("[Question]", hyp)
   #             copied["query"] = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
   #             output.append(copied)
   query = agent_prompt.replace("[Context]", instance["origin"].replace("[", "").replace("]", "")).replace("[Question]", instance["user_query"])
   instance["query"] = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
   output.append(instance)


print(output[0]["query"])
print("Save", len(output))

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(output)
