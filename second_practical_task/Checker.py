from gradio_client import Client

model = Client("Qwen/Qwen2-72B-Instruct")

SYS_PROMPT = '''
Ты - тестировщик LLM. Твоя задача проверять, нормально ли модель ответила на промпт,
проверять корректность и правильность написания ответа
'''

API_NAME = "/model_chat"

with open("answers.txt", "r", encoding="utf-8") as f:
    answers_data = f.read()

answers = answers_data.split('<')
check_result = []

for answ in answers:
    answer = answ.strip()

    US_PROMPT = answer

    result = model.predict(query=US_PROMPT,
                           system=SYS_PROMPT,
                           api_name=API_NAME)

    check_result.append(result)

with open('check_result.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(check_result))




