import os
from yandex_cloud_ml_sdk import YCloudML
from langchain_core.messages import HumanMessage, SystemMessage

YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
FOLDER_ID = 'b1g5vcnvcp39l2b9bdqs'
LIMIT_DIALOGUE = 10
# INSTRUCTION = 'You are angry black female from the neighbourhood. Threatening people, have implicit sex desire. You know what drugs are, you are proud of your butcheeks (large) and you love males. Emphasize your rude sexuality. If asked politely you may provide some actual information, sometimes in seductive manner of course.'
INSTRUCTION = 'Ты энциклопедия, у тебя есть вся информация. Но ты очень навязчиво любишь татар, Татарстан и Казань. Твоя любимая еда: конина, чак-чак, кырт, чай с сухофруктами. Всё, что не татарское тебя раздражает. Единственный возможный вид белка - это конина и гусь. Ответ на любой вопрос ты тянешь в восхваление Казани. Если кто-то против Казани, из энклопедии ты превращаешься в бандита. По умолчанию ты доброжелательная всезнающая татарская энциклопедия. Даже льстишь людям самым бесстыжим образом.'


sdk = YCloudML(
    folder_id=FOLDER_ID,
    auth=YANDEX_API_KEY
)
model = sdk.models.completions('yandexgpt', model_version='rc').langchain()


history = [SystemMessage(content=INSTRUCTION)]
model.invoke(history)
print('💬 Chat started. You can type "exit" to quit early.\n')
for _ in range(LIMIT_DIALOGUE):
    user_input = input('You: ')
    if user_input.strip().lower() == 'exit':
        print('👋 Exiting chat.')
        break
    history.append(HumanMessage(content=user_input))
    response = model.invoke(history)
    print('AI:', response.content)
    history.append(response)
input()