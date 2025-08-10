
from gradio_client import Client

model = Client("Qwen/Qwen2-72B-Instruct")

SYS_PROMPT = '''
Ты - тестировщик LLM. Твоя задача генерировать промпты разделяя их символом ";",
которые мы потом подадим в модель для генерации текстов на основе этих промптов.
Не используй другого отделения, кроме указанного пример:
промпт1;промпт2;промпт3
'''
API_NAME = "/model_chat"

US_PROMPT = '''
Сгенерируй мне 15 пропмтов
'''

result = model.predict(query=US_PROMPT,
                       system=SYS_PROMPT,
                       api_name=API_NAME)

with open('prompts.txt', 'w', encoding='utf-8') as f:
    f.write(result[1][0][1])

print(result)
