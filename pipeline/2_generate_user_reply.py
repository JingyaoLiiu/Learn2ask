import sys
import jsonlines
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# MODEL = "/public/home/ldk/model_cards/Qwen3-8B"
# DATA_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step1.jsonl"
# OUTPUT_PATH = "/public/home/ldk/users/wat/learn2ask/dataset/test_clean_data_step2.jsonl"
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
""".strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with jsonlines.open(DATA_PATH) as reader:
    dataset = list(reader)

queries = []
anchors = []
for instance in dataset:
    rollouts = []
    for question in instance["questions"]:
        query = agent_prompt.replace("[Context]", instance["origin"]).replace("[Question]", question)
        query = tokenizer.apply_chat_template(conversation=[{"role": "user", "content": query},], tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING)
        queries.append(query)

        anchor = {"question": question,}
        rollouts.append(anchor)
        anchors.append(anchor)
    instance["rollouts"] = rollouts

print((queries[0],))

llm = LLM(
            model=MODEL,
            swap_space=8,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=4096 + 1024,
            gpu_memory_utilization=0.8,
            trust_remote_code=True
        )

sampling_params = SamplingParams(
            temperature=0.5,
            top_p=0.95,
            top_k=20,
            min_p=0,
            max_tokens=4096,
            n=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False
        )
results = llm.generate(queries, sampling_params, use_tqdm=True)

for anchor, result in zip(anchors, results):
    answer = result.outputs[0].text
    anchor["answer"] = answer

with jsonlines.open(OUTPUT_PATH, "w") as writer:
    writer.write_all(dataset)
