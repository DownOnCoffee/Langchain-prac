from langchain_ollama.llms import OllamaLLM

# For prompts/templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda

# For structured outputs
from pydantic import BaseModel, Field

class Review(BaseModel) :
    category : str = Field(description='One of the three categories that the review has been classified to.')

strparser = StrOutputParser()
pydanticparser = PydanticOutputParser(pydantic_object=Review)

model = OllamaLLM(model='llama3.1:8b',num_predict=300)

instructions = pydanticparser.get_format_instructions()

prompt1 = PromptTemplate(
    template = 'Analyze the customer review and classify into 3 categories: Quality issue,Positove feedback and Delivery issue. review -> {review} \n {ins}  Return a JSON object with a single string for category not a list just a single string and no extra text.',
    input_variables=['review'],
    partial_variables={'ins':instructions}
)

prompt2 = PromptTemplate(
    template="Write a statement about feeling sorry for the quality issue. It should be a two liner strictly.",
)

prompt3 = PromptTemplate(
    template="Write a thank you note for buying from us.It should be a two liner strictly."
)
prompt4 = PromptTemplate(
    template="Write a statement asking the customer do they want refund for their order or wait for the order to get delivered.  "
)



chain1 = prompt1 | model | pydanticparser

conditional_chain = RunnableBranch(
    (lambda x:x.category == 'Quality issue',RunnableLambda(lambda x: {})| prompt2|model|strparser ), # RunnableLambda(lambda x: {}) is needed as PromptTemplate in second chain expects dict and not a pydantic object thus we are sending an empty dict.
    (lambda x:x.category == 'Positive feedback',RunnableLambda(lambda x: {})|prompt3|model|strparser),
    (lambda x:x.category == 'Delivery issue',RunnableLambda(lambda x: {})|prompt4|model|strparser),
    RunnableLambda(lambda x: "could not classify the review into a category.")
)

merged_chain = chain1 | conditional_chain


text = """
 My order is too late . I dont want't it now!
"""

res = merged_chain.invoke({'review':text})
print(res, 'result')
