from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0)

llm = LLM(
    model="/media/model-space/Llama-2-7b-hf",
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    instance_type="decode_instance",
)


llm.start_decode(sampling_params)
output = llm._run_engine(use_tqdm=False)
print(output)
