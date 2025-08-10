from openai import OpenAI

with open('prompts.txt', 'r', encoding='utf-8') as f:
    prompts_data = f.read()

prompts = prompts_data.split(';')

client = OpenAI(base_url="http://localhost:8000/v1", api_key="123")

results = []

for prompt in prompts:
    prompt = prompt.strip()

    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = resp.choices[0].message.content.strip()
    results.append(f"Prompt: {prompt}\nAnswer: {answer}\n<")


with open("answers.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))






