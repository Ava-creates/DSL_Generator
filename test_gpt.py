cfrom vllm import LLM, SamplingParams
#from vllm import LLMEngineArgs

llm = LLM(model="/scratch/avani/gpt",     tensor_parallel_size=4 )


params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["Hello! What is GPT-OSS-120?", "whatevgefv"], params)

for out in outputs:
    print(out.outputs[0].text)
print(llm.generate("Hello! What is GPT-OSS-120?", params))
