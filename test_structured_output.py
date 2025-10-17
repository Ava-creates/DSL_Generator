from vllm import LLM, SamplingParams

# Initialize your GPT-OSS model
llm = LLM(
    model="/scratch/avani/gpt",    # path to local GPT-OSS model
    tensor_parallel_size=4         # number of GPUs or tensor parallel units
)

# Configure generation parameters
params = SamplingParams(
    temperature=0.7,
    max_tokens=5000
)

prompt = "Write a Python function to compute the Fibonacci sequence."

outputs = llm.generate([prompt], params)
result_text = outputs[0].outputs[0].text
print(result_text)
