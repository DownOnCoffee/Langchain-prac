from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage

model = OllamaLLM(model = 'llama3.1:8b')
print('To exit , type "exit"')

chat_history = []

template = ChatPromptTemplate.from_messages([
    ('system','You are a helpful AI tutor that helps in teaching.'),
    MessagesPlaceholder(variable_name="chat_history"),
])

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        print(chat_history, 'chat history')
        break
    chat_history.append(HumanMessage(content=user_input))
    prompt = template.format_messages(chat_history=chat_history)
    res = model.invoke(prompt)
    chat_history.append(AIMessage(content=res))
    print(f'AI: {res}')




