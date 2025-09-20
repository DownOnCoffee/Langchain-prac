from langchain_ollama.llms import OllamaLLM

# For prompts/templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import PydanticOutputParser

# For structured outputs
from pydantic import BaseModel, Field

class Answer(BaseModel):
    summary: str = Field(description='A brief summary of the explaination given out by you.')
    topic: str = Field(description="The Main topic of the explaination.")
    Source: list[str] = Field(description='Sources which you used to find the answer.')
    subject: str = Field(description="Which subject did the user ask the question from for example : Maths,English,Science,Social science etc.")

parser = PydanticOutputParser(pydantic_object=Answer)

model = OllamaLLM(model = 'llama3.1:8b', num_predict=200)
# structured_model = model.with_structured_output(Answer)  Ollama doesn't support structured response.

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
    instructions = parser.get_format_instructions()
    chat_history.append(HumanMessage(content=f'{user_input} \n {instructions}'))
    prompt = template.format_messages(chat_history=chat_history)
    res = model.invoke(prompt)
    chat_history.append(AIMessage(content=res))
    print(f'AI: {res}')




