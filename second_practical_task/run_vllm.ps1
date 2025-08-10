docker run --gpus all -p 8000:8000 --ipc=host `
  -v D:\hf_cache:/root/.cache/huggingface `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --dtype auto --api-key 123 `
  --gpu-memory-utilization 0.82 `
  --max-model-len 2048
