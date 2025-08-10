from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Hi!!"}],
    temperature=0.3,
)
print(resp.choices[0].message.content)
